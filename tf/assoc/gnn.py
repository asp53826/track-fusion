"""Global nearest neighbour association.

The baseline JPDA has to beat. GNN commits: it solves one assignment problem
and each track gets exactly one measurement or none. That is optimal if the
assignment is right and unrecoverable if it isn't, which is the whole argument
for soft association.

Deliberately emits the same (betas, beta0) interface as JPDA, with the betas
being one-hot. That way the tracker has a single update path and the only
variable between the two configurations is how association weight is assigned
— not the filter, not the gate, not the track management.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def associate(track_gates, likelihoods, Pd, clutter_density, n_meas, Pg=1.0,
              **_):
    n_tracks = len(track_gates)
    betas = [np.zeros(n_meas) for _ in range(n_tracks)]
    beta0 = np.ones(n_tracks)
    stats = {"clusters": 0, "events": 0, "max_cluster": 0, "truncated": 0,
             "fallbacks": 0}

    if n_tracks == 0:
        return betas, beta0, stats

    # Square cost matrix: real measurements on the left, one dummy "missed"
    # column per track on the right, so every track can always be assigned.
    BIG = 1e9
    C = np.full((n_tracks, n_meas + n_tracks), BIG)
    for t, gated in enumerate(track_gates):
        for j in gated:
            L = likelihoods[t][j]
            if L <= 0.0:
                continue
            C[t, j] = -np.log(Pd * L / max(clutter_density, 1e-300))
        C[t, n_meas + t] = -np.log(max(1.0 - Pd * Pg, 1e-300))

    rows, cols = linear_sum_assignment(C)
    for t, j in zip(rows, cols):
        if j < n_meas and C[t, j] < BIG:
            betas[t][j] = 1.0
            beta0[t] = 0.0

    return betas, beta0, stats
