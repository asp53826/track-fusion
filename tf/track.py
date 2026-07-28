"""Track objects and their lifecycle.

Track management is the part of a tracker that nobody writes papers about and
everybody's numbers depend on. A tracker that reports beautiful position RMSE
on confirmed tracks while dropping a third of the targets is not tracking; the
OSPA metric in `metrics.py` exists to stop that from looking like success.
"""

import numpy as np

from . import kalman as kf
from .imm import IMM
from .kalman import Gaussian

TENTATIVE = "tentative"
CONFIRMED = "confirmed"
DELETED = "deleted"


class Track:
    _next_id = 1

    def __init__(self, imm, frame):
        self.id = Track._next_id
        Track._next_id += 1
        self.imm = imm
        self.status = TENTATIVE
        self.history = []          # per scan: 1 if the gate held anything
        self.misses = 0
        self.born = frame
        self.last_update = frame
        self.score = 0.0

    @classmethod
    def reset_ids(cls):
        cls._next_id = 1

    @classmethod
    def from_measurement(cls, z, modes, transition, sigma, v_max, frame):
        """Single-point initiation.

        Position comes from the measurement; velocity is unknown, so it gets a
        covariance wide enough to cover the fastest target we expect. Two-point
        initiation would give a real velocity estimate, but it needs the
        association to already be right across two scans, which in clutter is
        the problem you are trying to solve.
        """
        x = np.array([z[0], 0.0, z[1], 0.0])
        P = np.diag([sigma ** 2, v_max ** 2, sigma ** 2, v_max ** 2])
        imm = IMM(modes, transition, init=Gaussian(x, P))
        return cls(imm, frame)

    # -- prediction / gating ----------------------------------------------

    def predict(self):
        self.imm.step_prior()

    def gate(self, Z, H, R, gate):
        """Indices of measurements inside the validation gate."""
        if len(Z) == 0:
            return [], None
        zhat, S, chol = self.imm.combined_measurement(H, R)
        keep = []
        for i, z in enumerate(Z):
            if gate.contains(kf.mahalanobis_sq(z - zhat, chol)):
                keep.append(i)
        return keep, S

    def likelihoods(self, Z, H, R):
        return self.imm.measurement_likelihoods(Z, H, R)

    # -- update / lifecycle -----------------------------------------------

    def update(self, Z, betas, beta0, H, R, Pd, Pg, clutter_density, frame,
               gated_likelihood=0.0, n_gated=0):
        self.imm.update_pda(Z, betas, beta0, H, R, Pd, Pg, clutter_density)

        # Sequential log-likelihood ratio: target-present against
        # everything-in-the-gate-is-clutter.
        #
        #     dS = ln[ (1 - Pd*Pg) + (Pd/lambda) * sum_j L_j ]
        #
        # This is what separates a real track from one built out of clutter,
        # and counting detections cannot. A freshly initiated track has an
        # enormous covariance because its velocity is unknown, so its gate is
        # huge and it will nearly always contain *something* -- M-of-N sees a
        # detection and confirms. The score sees that the predicted density is
        # spread thin over a large gate, so each hit is only weak evidence,
        # and misses charge ln(1 - Pd*Pg) against it.
        ratio = (1.0 - Pd * Pg) + (Pd / max(clutter_density, 1e-300)) * gated_likelihood
        self.score += np.log(max(ratio, 1e-300))

        # `misses` counts scans where the gate was *empty* -- nothing to
        # associate with at all. It deliberately does not mean "beta0 was
        # high".
        #
        # Using beta0 > 0.5 as a miss silently destroys JPDA in clutter: soft
        # association spreads weight across every gated measurement, so in
        # dense clutter the true measurement's share falls below a half, every
        # scan is scored a miss, and max_misses deletes tracks that are in fact
        # being updated correctly. Track quality is the score's job; this
        # counter only handles the case where there is no data.
        self.history.append(1 if n_gated else 0)
        if n_gated:
            self.misses = 0
            self.last_update = frame
        else:
            self.misses += 1

    def promote(self, confirm_score, delete_score, max_misses):
        """Wald sequential test on the track score, plus a staleness cutoff."""
        if self.status == TENTATIVE and self.score >= confirm_score:
            self.status = CONFIRMED
        if self.score <= delete_score or self.misses >= max_misses:
            self.status = DELETED

    @property
    def estimate(self):
        return self.imm.estimate()

    @property
    def position(self):
        x = self.imm.estimate().x
        return np.array([x[0], x[2]])
