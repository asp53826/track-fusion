import itertools

import numpy as np
import pytest

from tf.assoc import gnn, jpda


def test_betas_and_beta0_form_a_distribution():
    """For every track, sum_j beta_j + beta_0 == 1. This is the invariant that
    catches a bad normalisation, and a bad normalisation silently rescales the
    Kalman gain."""
    gates = [[0, 1], [1, 2]]
    L = [np.array([0.4, 0.2, 0.0]), np.array([0.0, 0.3, 0.5])]
    betas, beta0, _ = jpda.associate(gates, L, 0.9, 1e-4, 3, Pg=0.99)
    for b, b0 in zip(betas, beta0):
        assert b.sum() + b0 == pytest.approx(1.0)


def test_single_track_single_measurement_matches_the_closed_form():
    """One track, one measurement: JPDA collapses to PDA and the answer is
    writable by hand."""
    Pd, Pg, lam, L = 0.9, 0.99, 1e-4, 0.05
    betas, beta0, _ = jpda.associate([[0]], [np.array([L])], Pd, lam, 1, Pg=Pg)
    num = Pd * L / lam
    want = num / (num + (1.0 - Pd * Pg))
    assert betas[0][0] == pytest.approx(want)
    assert beta0[0] == pytest.approx(1.0 - want)


def test_exclusivity_two_tracks_cannot_both_own_one_measurement():
    """The defining property of *joint* association. Two tracks, one shared
    measurement: their betas for it must sum to at most 1. Independent PDA
    filters would each take it with high probability."""
    gates = [[0], [0]]
    L = [np.array([0.5]), np.array([0.5])]
    betas, _, _ = jpda.associate(gates, L, 0.9, 1e-4, 1, Pg=0.99)
    assert betas[0][0] + betas[1][0] <= 1.0 + 1e-12
    assert betas[0][0] == pytest.approx(betas[1][0])   # symmetric by symmetry


def test_a_contested_measurement_is_shared_not_doubled():
    """Same measurement, one track fits it much better -- it should take most
    of the weight, and the other should be left mostly undetected."""
    gates = [[0], [0]]
    L = [np.array([1.0]), np.array([0.01])]
    betas, beta0, _ = jpda.associate(gates, L, 0.9, 1e-4, 1, Pg=0.99)
    assert betas[0][0] > betas[1][0]
    assert beta0[1] > beta0[0]


def test_clustering_splits_independent_tracks():
    gates = [[0], [1], [2]]
    groups = jpda.cluster(gates, 3)
    assert sorted(len(g) for g in groups) == [1, 1, 1]


def test_clustering_joins_tracks_through_a_shared_measurement():
    # 0 and 1 share measurement 1; 2 is on its own
    gates = [[0, 1], [1], [5]]
    groups = jpda.cluster(gates, 3)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]


def test_clustering_is_transitive():
    """0-1 share a measurement, 1-2 share a different one, so all three are one
    cluster even though 0 and 2 share nothing."""
    gates = [[0], [0, 1], [1]]
    groups = jpda.cluster(gates, 3)
    assert len(groups) == 1 and len(groups[0]) == 3


def test_independent_clusters_give_the_same_answer_as_solving_them_apart():
    """Factorisation check: two non-overlapping pairs solved together must
    equal each pair solved alone."""
    gates = [[0, 1], [0, 1], [2, 3], [2, 3]]
    L = [np.array([0.4, 0.1, 0.0, 0.0]), np.array([0.2, 0.5, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.3, 0.2]), np.array([0.0, 0.0, 0.1, 0.6])]
    joint, joint0, _ = jpda.associate(gates, L, 0.9, 1e-4, 4, Pg=0.99)

    sub, sub0, _ = jpda.associate([[0, 1], [0, 1]], [L[0][:2], L[1][:2]],
                                  0.9, 1e-4, 2, Pg=0.99)
    assert np.allclose(joint[0][:2], sub[0])
    assert joint0[0] == pytest.approx(sub0[0])


def test_enumeration_counts_match_the_combinatorics():
    """Two tracks each gating both measurements: every track independently
    takes nothing / m0 / m1, minus the two events where both take the same
    one. 3*3 - 2 = 7."""
    events, complete = jpda.enumerate_events([0, 1], [[0, 1], [0, 1]])
    assert complete and len(events) == 7
    for ev in events:
        assigned = [v for v in ev.values() if v is not None]
        assert len(assigned) == len(set(assigned))     # no measurement reused


def test_enumeration_respects_the_cap():
    """Six tracks all gating six measurements is thousands of events; with a
    small cap the enumeration must report itself incomplete rather than
    silently returning a wrong normalisation."""
    gates = [list(range(6)) for _ in range(6)]
    events, complete = jpda.enumerate_events(list(range(6)), gates, cap=50)
    assert not complete
    assert len(events) <= 50 + 6


def test_truncation_is_reported_in_stats():
    gates = [list(range(6)) for _ in range(6)]
    L = [np.full(6, 0.1) for _ in range(6)]
    _, _, stats = jpda.associate(gates, L, 0.9, 1e-4, 6, Pg=0.99, cap=50)
    assert stats["truncated"] == 1
    assert stats["max_cluster"] == 6


def test_brute_force_agreement_on_a_small_problem():
    """Independent re-derivation: enumerate every assignment by hand and
    marginalise, then compare against the implementation."""
    Pd, Pg, lam = 0.85, 1.0, 2e-4
    gates = [[0, 1], [1, 2]]
    L = [{0: 0.30, 1: 0.10}, {1: 0.20, 2: 0.40}]
    Lv = [np.array([0.30, 0.10, 0.0]), np.array([0.0, 0.20, 0.40])]

    total = 0.0
    acc = [np.zeros(3), np.zeros(3)]
    miss = [0.0, 0.0]
    for a in [None, 0, 1]:
        for b in [None, 1, 2]:
            if a is not None and a == b:
                continue
            w = 1.0
            for t, j in ((0, a), (1, b)):
                w *= (1 - Pd * Pg) if j is None else Pd * L[t][j] / lam
            total += w
            for t, j in ((0, a), (1, b)):
                if j is None:
                    miss[t] += w
                else:
                    acc[t][j] += w

    betas, beta0, _ = jpda.associate(gates, Lv, Pd, lam, 3, Pg=Pg)
    for t in range(2):
        assert np.allclose(betas[t], acc[t] / total)
        assert beta0[t] == pytest.approx(miss[t] / total)


# -- GNN ------------------------------------------------------------------

def test_gnn_is_one_hot():
    gates = [[0, 1], [0, 1]]
    L = [np.array([0.5, 0.1]), np.array([0.2, 0.6])]
    betas, beta0, _ = gnn.associate(gates, L, 0.9, 1e-4, 2, Pg=0.99)
    for b in betas:
        assert set(np.unique(b)) <= {0.0, 1.0}
        assert b.sum() <= 1.0


def test_gnn_picks_the_globally_best_assignment_not_the_greedy_one():
    """Track 0 slightly prefers m0, but giving m0 to track 1 and m1 to track 0
    is better overall. A greedy nearest-neighbour gets this wrong."""
    gates = [[0, 1], [0]]
    L = [np.array([0.50, 0.45]), np.array([0.90, 0.0])]
    betas, _, _ = gnn.associate(gates, L, 0.9, 1e-4, 2, Pg=0.99)
    assert betas[0][1] == 1.0
    assert betas[1][0] == 1.0


def test_gnn_leaves_a_track_unassigned_when_nothing_gates():
    betas, beta0, _ = gnn.associate([[]], [np.zeros(2)], 0.9, 1e-4, 2, Pg=0.99)
    assert betas[0].sum() == 0.0
    assert beta0[0] == 1.0


def test_gnn_and_jpda_agree_when_the_answer_is_unambiguous():
    """One track, one measurement, essentially no clutter: soft and hard
    association should reach the same conclusion."""
    gates = [[0]]
    L = [np.array([1.0])]
    bj, b0j, _ = jpda.associate(gates, L, 0.99, 1e-9, 1, Pg=1.0)
    bg, b0g, _ = gnn.associate(gates, L, 0.99, 1e-9, 1, Pg=1.0)
    assert bj[0][0] == pytest.approx(bg[0][0], abs=1e-6)
