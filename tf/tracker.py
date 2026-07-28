"""The tracker: one scan in, a set of confirmed tracks out.

Scan cycle, in order, because the order is load-bearing:

    predict -> gate -> score -> associate -> update -> manage -> initiate

Initiation goes last and only sees measurements that association didn't claim.
Doing it earlier is a classic way to build a tracker that spawns a duplicate
track on every clutter point that happens to land near a real target.
"""

import numpy as np

from .assoc import gnn, jpda
from .gating import Gate
from .models import imm_modes, position_measurement
from .track import CONFIRMED, DELETED, Track

ASSOCIATORS = {"jpda": jpda.associate, "gnn": gnn.associate}


class Config:
    def __init__(self, dt=1.0, sigma=20.0, Pd=0.9, clutter_density=1e-6,
                 v_max=200.0, gate_prob=0.99, p_false_confirm=1e-3,
                 p_miss_track=0.05,
                 max_misses=4, init_threshold=0.3, event_cap=20000,
                 q_quiet=0.05, q_manoeuvre=2.0, turn_rate_deg=6.0,
                 mode_switch=0.05):
        self.dt = dt
        self.sigma = sigma
        self.Pd = Pd
        self.clutter_density = clutter_density
        self.v_max = v_max
        self.gate_prob = gate_prob
        # Wald sequential test bounds, from the two error rates you actually
        # care about: how often a clutter track gets confirmed, and how often
        # a real one gets thrown away.
        self.p_false_confirm = p_false_confirm
        self.p_miss_track = p_miss_track
        self.confirm_score = np.log((1.0 - p_miss_track) / p_false_confirm)
        self.delete_score = np.log(p_miss_track / (1.0 - p_false_confirm))
        self.max_misses = max_misses
        self.init_threshold = init_threshold
        self.event_cap = event_cap
        self.q_quiet = q_quiet
        self.q_manoeuvre = q_manoeuvre
        self.turn_rate_deg = turn_rate_deg
        self.mode_switch = mode_switch


def imm_transition(n_modes, switch):
    """Sojourn-time transition matrix: stay put with high probability, and
    spread the rest evenly. `switch` is roughly the per-scan probability of
    starting or ending a manoeuvre; too low and the IMM lags into turns, too
    high and it stays diffuse on straight legs."""
    Pi = np.full((n_modes, n_modes), switch / (n_modes - 1))
    np.fill_diagonal(Pi, 1.0 - switch)
    return Pi


class Tracker:
    def __init__(self, cfg=None, associator="jpda", single_model=False):
        self.cfg = cfg or Config()
        self.associate = ASSOCIATORS[associator]
        self.associator_name = associator
        self.single_model = single_model
        self.H, self.R = position_measurement(self.cfg.sigma)
        self.gate = Gate(dim=2, prob=self.cfg.gate_prob)
        self.tracks = []
        self.frame = 0
        self.stats = {"clusters": 0, "events": 0, "max_cluster": 0,
                      "truncated": 0, "fallbacks": 0, "scans": 0}

    # -- filter bank -------------------------------------------------------

    def _modes(self):
        c = self.cfg
        if self.single_model:
            # One CV mode with process noise between quiet and manoeuvring --
            # the compromise a non-IMM tracker is forced into.
            from .models import Mode, cv, q_dwna
            q = 0.5 * (c.q_quiet + c.q_manoeuvre)
            return [Mode("cv", cv(c.dt), q_dwna(q, c.dt))]
        return imm_modes(c.dt, c.q_quiet, c.q_manoeuvre,
                         np.deg2rad(c.turn_rate_deg))

    def _transition(self):
        modes = self._modes()
        if len(modes) == 1:
            return np.ones((1, 1))
        return imm_transition(len(modes), self.cfg.mode_switch)

    # -- main loop ---------------------------------------------------------

    def step(self, Z):
        """Process one scan. Z is (m, 2) of Cartesian position measurements."""
        cfg = self.cfg
        Z = np.asarray(Z, dtype=float).reshape(-1, 2)
        self.frame += 1
        self.stats["scans"] += 1

        for tr in self.tracks:
            tr.predict()

        track_gates, likelihoods = [], []
        for tr in self.tracks:
            idx, _S = tr.gate(Z, self.H, self.R, self.gate)
            track_gates.append(idx)
            L = tr.likelihoods(Z, self.H, self.R) if len(Z) else np.zeros(0)
            likelihoods.append(L)

        betas, beta0, st = self.associate(
            track_gates, likelihoods, cfg.Pd, cfg.clutter_density, len(Z),
            Pg=self.gate.Pg, cap=cfg.event_cap)
        for k in ("clusters", "events", "truncated", "fallbacks"):
            self.stats[k] += st[k]
        self.stats["max_cluster"] = max(self.stats["max_cluster"],
                                        st["max_cluster"])

        for tr, b, b0, gated, L in zip(self.tracks, betas, beta0,
                                       track_gates, likelihoods):
            tr.update(Z, b, b0, self.H, self.R, cfg.Pd, self.gate.Pg,
                      cfg.clutter_density, self.frame,
                      gated_likelihood=float(sum(L[j] for j in gated)),
                      n_gated=len(gated))
            tr.promote(cfg.confirm_score, cfg.delete_score, cfg.max_misses)

        self.tracks = [t for t in self.tracks if t.status != DELETED]

        # A measurement is "claimed" if the *total* association weight over all
        # tracks is high enough. Because joint events are exclusive, sum_t
        # beta[t][j] is exactly P(measurement j came from some existing track),
        # so this is the quantity the decision should be made on.
        #
        # Taking a per-track max instead is wrong specifically for JPDA: soft
        # association can split a measurement three ways at 0.3 each, no single
        # track clears the threshold, and the tracker spawns a duplicate on top
        # of targets it is already holding. GNN never showed the bug because
        # its weights are one-hot.
        claim = np.zeros(len(Z))
        for b in betas:
            if len(b):
                claim += b

        modes, Pi = self._modes(), self._transition()
        for i in range(len(Z)):
            if claim[i] <= cfg.init_threshold:
                self.tracks.append(Track.from_measurement(
                    Z[i], modes, Pi, cfg.sigma, cfg.v_max, self.frame))

        return self.confirmed()

    def confirmed(self):
        return [t for t in self.tracks if t.status == CONFIRMED]

    def run(self, scans):
        """Convenience: process a whole scenario, returning confirmed track
        positions per scan."""
        out = []
        for Z in scans:
            tracks = self.step(Z)
            out.append({t.id: t.position for t in tracks})
        return out
