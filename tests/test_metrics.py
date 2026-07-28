import numpy as np
import pytest

from tf.metrics import identity_metrics, ospa, ospa_sequence, rmse_position


def test_identical_sets_score_zero():
    X = [[0.0, 0.0], [10.0, 10.0]]
    assert ospa(X, X) == (0.0, 0.0, 0.0)


def test_both_empty_scores_zero():
    assert ospa([], []) == (0.0, 0.0, 0.0)


def test_missing_everything_costs_the_cutoff():
    total, loc, card = ospa([], [[0.0, 0.0], [5.0, 5.0]], c=100.0)
    assert total == pytest.approx(100.0)
    assert card == pytest.approx(100.0)
    assert loc == 0.0


def test_symmetric_in_its_arguments():
    X = [[0.0, 0.0], [10.0, 0.0]]
    Y = [[1.0, 0.0]]
    assert ospa(X, Y) == pytest.approx(ospa(Y, X))


def test_localisation_component_is_the_matched_error():
    """Two perfectly matched pairs offset by 3 and 4 -> RMS of 3 and 4."""
    X = [[0.0, 0.0], [100.0, 0.0]]
    Y = [[3.0, 0.0], [104.0, 0.0]]
    total, loc, card = ospa(X, Y, c=100.0, p=2.0)
    assert loc == pytest.approx(np.sqrt((9 + 16) / 2))
    assert card == 0.0
    assert total == pytest.approx(loc)


def test_cardinality_penalty_uses_the_larger_set_as_denominator():
    """One matched pair, one unmatched truth: cardinality term is c^p * 1 / 2."""
    X = [[0.0, 0.0]]
    Y = [[0.0, 0.0], [500.0, 500.0]]
    total, loc, card = ospa(X, Y, c=100.0, p=2.0)
    assert card == pytest.approx(np.sqrt(100.0 ** 2 / 2))
    assert loc == 0.0
    assert total == pytest.approx(card)


def test_distance_is_capped_at_c():
    """A wildly misplaced estimate costs the same as a missing one -- that cap
    is what stops one outlier from dominating the score."""
    near = ospa([[0.0, 0.0]], [[1e6, 0.0]], c=100.0)[0]
    missing = ospa([], [[0.0, 0.0]], c=100.0)[0]
    assert near == pytest.approx(missing)


def test_extra_false_tracks_are_penalised_like_missing_ones():
    """A spurious track costs exactly what a dropped target costs.

    The comparison has to hold cardinality fixed. OSPA normalises by the
    *larger* of the two sets, so (1 truth, 0 est) and (1 truth, 2 est) are not
    comparable -- their denominators are 1 and 2. Both cases here are
    one-matched-one-unmatched out of two, which is the like-for-like test.
    """
    matched = [0.0, 0.0]
    dropped = ospa([matched, [900.0, 900.0]], [matched], c=100.0)[0]
    spurious = ospa([matched], [matched, [900.0, 900.0]], c=100.0)[0]
    assert dropped == pytest.approx(spurious)
    assert dropped == pytest.approx(100.0 / np.sqrt(2))


def test_sequence_average():
    truth = [{0: np.array([0.0, 0.0])}, {0: np.array([0.0, 0.0])}]
    est = [{7: np.array([0.0, 0.0])}, {7: np.array([6.0, 8.0])}]
    total, loc, card = ospa_sequence(truth, est, c=100.0)
    assert loc == pytest.approx(5.0)     # (0 + 10) / 2
    assert card == 0.0


def test_identity_metrics_count_a_swap():
    truth = [{0: np.array([0.0, 0.0]), 1: np.array([100.0, 0.0])} for _ in range(3)]
    est = [
        {10: np.array([0.0, 0.0]), 11: np.array([100.0, 0.0])},
        {10: np.array([0.0, 0.0]), 11: np.array([100.0, 0.0])},
        {11: np.array([0.0, 0.0]), 10: np.array([100.0, 0.0])},   # labels swap
    ]
    m = identity_metrics(truth, est, radius=50.0)
    assert m["id_swaps"] == 2
    assert m["coverage"] == pytest.approx(1.0)


def test_identity_metrics_count_coverage_and_false_tracks():
    truth = [{0: np.array([0.0, 0.0])} for _ in range(2)]
    est = [{5: np.array([0.0, 0.0]), 6: np.array([900.0, 900.0])}, {}]
    m = identity_metrics(truth, est, radius=50.0)
    assert m["coverage"] == pytest.approx(0.5)
    assert m["false_tracks_per_scan"] == pytest.approx(0.5)


def test_rmse_ignores_unmatched_and_ospa_does_not():
    """The point of having both. A tracker that reports only its one good
    track gets a perfect RMSE and a bad OSPA."""
    truth = [{0: np.array([0.0, 0.0]), 1: np.array([500.0, 0.0])}]
    est = [{9: np.array([0.0, 0.0])}]
    assert rmse_position(truth, est, radius=50.0) == pytest.approx(0.0)
    assert ospa_sequence(truth, est, c=100.0)[0] > 50.0
