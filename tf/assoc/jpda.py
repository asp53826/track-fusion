"""Joint Probabilistic Data Association.

The idea: don't pick an assignment, average over all of them. For each track
you get beta[t][j], the probability that measurement j came from track t,
marginalised over every consistent joint interpretation of the scan. "Consistent"
is doing real work there — it enforces that one measurement has one source and
one target produces at most one detection, which is what separates JPDA from
running independent PDA filters that both happily eat the same measurement.

The cost is that the number of joint events is combinatorial in the size of a
cluster of mutually-gating tracks. Everything here is either about computing
that sum exactly, or about noticing when it's about to get out of hand.
"""

import numpy as np


def cluster(track_gates, n_tracks):
    """Group tracks that share at least one gated measurement.

    Tracks in different clusters can't influence each other's association
    probabilities, so the joint sum factorises over clusters. This is the
    only thing that makes JPDA tractable in practice: 20 isolated tracks is
    20 trivial problems, 20 mutually-gating tracks is not.
    """
    parent = list(range(n_tracks))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owners = {}
    for t, gated in enumerate(track_gates):
        for j in gated:
            if j in owners:
                union(owners[j], t)
            else:
                owners[j] = t

    groups = {}
    for t in range(n_tracks):
        groups.setdefault(find(t), []).append(t)
    return list(groups.values())


def _events(tracks, track_gates, used, idx, current, out, cap):
    """Depth-first enumeration of feasible joint association events."""
    if len(out) >= cap:
        return False
    if idx == len(tracks):
        out.append(dict(current))
        return True

    t = tracks[idx]
    # option 1: this track was not detected this scan
    current[t] = None
    if not _events(tracks, track_gates, used, idx + 1, current, out, cap):
        return False
    # option 2: it produced one of its gated measurements, if still unclaimed
    for j in track_gates[t]:
        if j in used:
            continue
        used.add(j)
        current[t] = j
        ok = _events(tracks, track_gates, used, idx + 1, current, out, cap)
        used.discard(j)
        if not ok:
            return False
    current.pop(t, None)
    return True


def enumerate_events(tracks, track_gates, cap=20000):
    """All feasible joint events for one cluster.

    Returns (events, complete). `complete=False` means the cap was hit and the
    enumeration is a truncated subset — the caller has to decide what to do
    about it rather than silently getting a wrong normalisation.
    """
    out = []
    complete = _events(tracks, track_gates, set(), 0, {}, out, cap)
    return out, complete


def _hard_fallback(group, track_gates, likelihoods, Pd, Pg, clutter_density,
                   n_meas, betas, beta0):
    """Hard assignment for a cluster too big to enumerate.

    An exact JPDA over a large cluster is not merely slow, it is unbounded:
    the event count grows super-exponentially in the number of mutually-gating
    tracks. Every real system draws a line somewhere. Drawing it here, and
    counting how often it is hit, is better than an event cap alone -- a cap
    truncates the sum and silently renormalises over whichever events happened
    to be enumerated first, which is a biased answer presented as an exact one.
    """
    from scipy.optimize import linear_sum_assignment

    BIG = 1e9
    idx = {t: i for i, t in enumerate(group)}
    C = np.full((len(group), n_meas + len(group)), BIG)
    for t in group:
        for j in track_gates[t]:
            L = likelihoods[t][j]
            if L > 0.0:
                C[idx[t], j] = -np.log(Pd * L / max(clutter_density, 1e-300))
        C[idx[t], n_meas + idx[t]] = -np.log(max(1.0 - Pd * Pg, 1e-300))

    rows, cols = linear_sum_assignment(C)
    for r, cc in zip(rows, cols):
        if cc < n_meas and C[r, cc] < BIG:
            betas[group[r]][cc] = 1.0
            beta0[group[r]] = 0.0


def associate(track_gates, likelihoods, Pd, clutter_density, n_meas,
              Pg=1.0, cap=20000, max_cluster=7):
    """Marginal association probabilities.

    likelihoods[t][j] is the predictive density of measurement j under track t.
    Returns (betas, beta0, stats) where betas[t][j] is the association
    probability and beta0[t] is P(track t was not detected).

    Event weight, parametric JPDA with Poisson clutter of intensity lambda:

        P(theta) ~ prod_{t detected} [Pd * L_t(z_jt) / lambda]
                 * prod_{t missed}   [1 - Pd * Pg]

    The lambda^(number of false alarms) factor that appears in the textbook
    form is constant across events with the same measurement count, so it
    cancels in the normalisation and only the per-detection 1/lambda survives.

    The missed term is (1 - Pd*Pg), not (1 - Pd): a track can be detected and
    still contribute no gated measurement, because the detection landed
    outside the gate. This has to match the mode likelihood in IMM.update_pda
    or the association weights and the mode probabilities are answering
    subtly different questions.
    """
    miss = 1.0 - Pd * Pg
    n_tracks = len(track_gates)
    betas = [np.zeros(n_meas) for _ in range(n_tracks)]
    beta0 = np.ones(n_tracks)
    stats = {"clusters": 0, "events": 0, "max_cluster": 0, "truncated": 0,
             "fallbacks": 0}

    for group in cluster(track_gates, n_tracks):
        stats["clusters"] += 1
        stats["max_cluster"] = max(stats["max_cluster"], len(group))

        if len(group) > max_cluster:
            stats["fallbacks"] += 1
            _hard_fallback(group, track_gates, likelihoods, Pd, Pg,
                           clutter_density, n_meas, betas, beta0)
            continue

        events, complete = enumerate_events(group, track_gates, cap)
        stats["events"] += len(events)
        if not complete:
            stats["truncated"] += 1

        weights = np.empty(len(events))
        for e, ev in enumerate(events):
            w = 1.0
            for t in group:
                j = ev[t]
                if j is None:
                    w *= miss
                else:
                    w *= Pd * likelihoods[t][j] / max(clutter_density, 1e-300)
            weights[e] = w

        total = weights.sum()
        if total <= 0.0:
            # every interpretation is vanishingly unlikely; treat as all-missed
            continue
        weights /= total

        for w, ev in zip(weights, events):
            for t in group:
                j = ev[t]
                if j is None:
                    continue
                betas[t][j] += w
        for t in group:
            beta0[t] = max(0.0, 1.0 - betas[t].sum())

    return betas, beta0, stats
