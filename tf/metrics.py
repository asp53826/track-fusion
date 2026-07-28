"""Scoring.

RMSE over matched tracks is the metric that flatters a tracker: drop the hard
targets and your RMSE improves. OSPA fixes that by charging you for cardinality
errors in the same units as position errors, so losing a target and misplacing
one are on a common scale and you can't trade one for the other silently.

The decomposition matters as much as the total. OSPA going up tells you the
tracker got worse; the localisation/cardinality split tells you whether it
started missing targets or just got sloppier about where they are, and those
have completely different fixes.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def ospa(X, Y, c=100.0, p=2.0):
    """Optimal SubPattern Assignment distance between two point sets.

    Returns (total, localisation, cardinality). X and Y are (n, d) arrays;
    order doesn't matter and neither does which is truth.
    """
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    Y = np.asarray(Y, dtype=float).reshape(-1, 2)
    m, n = len(X), len(Y)

    if m == 0 and n == 0:
        return 0.0, 0.0, 0.0
    if m > n:
        X, Y, m, n = Y, X, n, m
    if m == 0:
        return c, 0.0, c

    D = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    D = np.minimum(D, c) ** p
    rows, cols = linear_sum_assignment(D)
    matched = D[rows, cols].sum()

    loc = (matched / n) ** (1.0 / p)
    card = ((c ** p) * (n - m) / n) ** (1.0 / p)
    total = ((matched + (c ** p) * (n - m)) / n) ** (1.0 / p)
    return total, loc, card


def ospa_sequence(truth, estimates, c=100.0, p=2.0):
    """Mean OSPA over a run. `truth` and `estimates` are lists of dicts
    mapping id -> position, one per scan."""
    tot = np.zeros(3)
    for t, e in zip(truth, estimates):
        tot += np.array(ospa(list(t.values()), list(e.values()), c, p))
    n = max(len(truth), 1)
    return tuple(tot / n)


def assignment_history(truth, estimates, radius=100.0):
    """Greedy per-scan matching of estimated tracks to true targets.

    Used only for the identity metrics below. It is deliberately not how OSPA
    scores anything — OSPA is identity-blind, and that's the point of having
    both: OSPA says how good the picture is, these say whether the labels on
    it stayed put.
    """
    out = []
    for t, e in zip(truth, estimates):
        tid, tpos = list(t.keys()), list(t.values())
        eid, epos = list(e.keys()), list(e.values())
        pairs = {}
        if tid and eid:
            D = np.linalg.norm(np.asarray(tpos)[:, None, :]
                               - np.asarray(epos)[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(D)
            for r, cc in zip(rows, cols):
                if D[r, cc] <= radius:
                    pairs[tid[r]] = eid[cc]
        out.append(pairs)
    return out


def identity_metrics(truth, estimates, radius=100.0):
    """Track-level quality: swaps, fragmentation, coverage, false tracks."""
    pairs = assignment_history(truth, estimates, radius)

    last = {}
    swaps = 0
    frags = 0
    covered = 0
    total_truth = 0

    for scan in pairs:
        for tgt, est in scan.items():
            if tgt in last:
                if last[tgt] != est:
                    swaps += 1
            last[tgt] = est

    for scan_truth, scan in zip(truth, pairs):
        total_truth += len(scan_truth)
        covered += len(scan)

    # a fragmentation is a target that was held, lost, then held again
    held = {}
    for scan_truth, scan in zip(truth, pairs):
        for tgt in scan_truth:
            on = tgt in scan
            prev = held.get(tgt)
            if prev is False and on:
                frags += 1
            if prev is not None or on:
                held[tgt] = on

    false_tracks = 0
    for scan_est, scan in zip(estimates, pairs):
        false_tracks += max(0, len(scan_est) - len(scan))

    return {
        "id_swaps": swaps,
        "fragmentations": frags,
        "coverage": covered / max(total_truth, 1),
        "false_tracks_per_scan": false_tracks / max(len(estimates), 1),
    }


def rmse_position(truth, estimates, radius=100.0):
    """Position RMSE over matched pairs only. Reported alongside OSPA, never
    instead of it — see the module docstring."""
    pairs = assignment_history(truth, estimates, radius)
    errs = []
    for t, e, scan in zip(truth, estimates, pairs):
        for tgt, est in scan.items():
            errs.append(np.linalg.norm(np.asarray(t[tgt]) - np.asarray(e[est])))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
