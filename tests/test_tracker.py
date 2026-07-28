import numpy as np
import pytest

from tf import Config, Tracker
from tf.metrics import identity_metrics, ospa_sequence
from tf.scenarios import crossing, dense, manoeuvre
from tf.track import CONFIRMED, DELETED, TENTATIVE, Track


@pytest.fixture(autouse=True)
def _reset_ids():
    Track.reset_ids()


def clean_config(**kw):
    base = dict(sigma=15.0, Pd=1.0, clutter_density=1e-8, gate_prob=0.999)
    base.update(kw)
    return Config(**base)


def test_no_measurements_means_no_tracks():
    t = Tracker(clean_config())
    for _ in range(5):
        assert t.step(np.zeros((0, 2))) == []
    assert t.tracks == []


def test_a_steady_target_gets_confirmed_and_held():
    t = Tracker(clean_config())
    pos = np.array([0.0, 0.0])
    vel = np.array([30.0, 20.0])
    confirmed_at = None
    for k in range(20):
        pos = pos + vel
        out = t.step(pos.reshape(1, 2))
        if out and confirmed_at is None:
            confirmed_at = k
    assert confirmed_at is not None
    assert confirmed_at <= 5              # M-of-N = 3-of-5
    assert len(t.confirmed()) == 1
    assert np.linalg.norm(t.confirmed()[0].position - pos) < 40.0


def test_confirmation_needs_m_of_n():
    """A track fed one measurement then nothing must never confirm."""
    t = Tracker(clean_config(max_misses=99))
    t.step(np.array([[0.0, 0.0]]))
    for _ in range(6):
        t.step(np.zeros((0, 2)))
    assert all(tr.status != CONFIRMED for tr in t.tracks)


def test_a_track_is_deleted_after_enough_misses():
    t = Tracker(clean_config(max_misses=3))
    pos = np.array([0.0, 0.0])
    for _ in range(8):
        pos = pos + np.array([30.0, 0.0])
        t.step(pos.reshape(1, 2))
    assert len(t.confirmed()) == 1
    for _ in range(6):
        t.step(np.zeros((0, 2)))
    assert t.confirmed() == []


def test_clutter_alone_does_not_build_confirmed_tracks():
    """Uniform clutter with no target. A few tentative tracks are expected;
    confirmed ones mean the initiation logic is too eager."""
    rng = np.random.default_rng(4)
    t = Tracker(Config(sigma=20.0, Pd=0.9, clutter_density=6.0 / (4000 * 4000)))
    n_confirmed = 0
    for _ in range(40):
        Z = rng.uniform(-2000, 2000, (rng.poisson(6.0), 2))
        n_confirmed += len(t.step(Z))
    assert n_confirmed <= 4


def test_two_targets_stay_two_tracks():
    t = Tracker(clean_config())
    a = np.array([-1000.0, 0.0])
    b = np.array([1000.0, 0.0])
    for _ in range(15):
        a = a + np.array([40.0, 10.0])
        b = b + np.array([-40.0, 10.0])
        out = t.step(np.vstack([a, b]))
    assert len(out) == 2


def test_soft_association_is_more_accurate_through_a_crossing():
    """Same scenario, same filter bank, same track management -- the only
    variable is soft versus hard association.

    JPDA wins on *localisation*, consistently: near the crossing both
    measurements are plausible for both tracks, and averaging over that
    ambiguity beats committing to one answer and being wrong half the time.

    It is deliberately not asserted that JPDA wins overall. It does not here.
    See test_hard_association_produces_fewer_false_tracks below and the
    clutter sweep in bench/benchmark.py -- JPDA's total OSPA is worse at low
    clutter because it sustains more false tracks, and only overtakes GNN once
    the clutter is heavy enough that hard assignment starts committing to
    genuinely wrong measurements.
    """
    gnn_loc, jpda_loc = [], []
    cfg = Config(sigma=20.0, Pd=0.95, clutter_density=3.0 / (8000 * 6000))
    for seed in range(6):
        sc = crossing(seed=seed, clutter=3.0)
        for name, acc in (("gnn", gnn_loc), ("jpda", jpda_loc)):
            Track.reset_ids()
            est = Tracker(cfg, name).run(sc.scans)
            acc.append(ospa_sequence(sc.truth, est)[1])
    assert np.mean(jpda_loc) < np.mean(gnn_loc)


def test_hard_association_produces_fewer_false_tracks():
    """The other side of the same coin, and the reason JPDA loses on total
    OSPA in light clutter. Soft weight keeps marginal tracks alive."""
    false_tracks = {}
    cfg = Config(sigma=20.0, Pd=0.95, clutter_density=3.0 / (8000 * 6000))
    for name in ("gnn", "jpda"):
        tot = 0.0
        for seed in range(6):
            sc = crossing(seed=seed, clutter=3.0)
            Track.reset_ids()
            est = Tracker(cfg, name).run(sc.scans)
            tot += identity_metrics(sc.truth, est)["false_tracks_per_scan"]
        false_tracks[name] = tot / 6
    assert false_tracks["gnn"] < false_tracks["jpda"]


def test_large_clusters_fall_back_instead_of_enumerating_forever():
    """JPDA's event count is super-exponential in cluster size. Past the limit
    the cluster must be solved by hard assignment and the fallback counted,
    rather than truncating the joint sum and renormalising over whichever
    events happened to come first."""
    from tf.assoc import jpda
    gates = [list(range(9)) for _ in range(9)]
    L = [np.full(9, 0.2) for _ in range(9)]
    betas, beta0, stats = jpda.associate(gates, L, 0.9, 1e-4, 9, Pg=0.99,
                                         max_cluster=7)
    assert stats["fallbacks"] == 1
    assert stats["events"] == 0
    for b in betas:
        assert set(np.unique(b)) <= {0.0, 1.0}


def test_imm_beats_a_single_model_on_a_manoeuvring_target():
    ospas = {}
    for single in (True, False):
        tot = 0.0
        for seed in range(5):
            sc = manoeuvre(seed=seed)
            Track.reset_ids()
            cfg = Config(sigma=20.0, Pd=0.98,
                         clutter_density=0.5 / (8000 * 5000))
            est = Tracker(cfg, "jpda", single_model=single).run(sc.scans)
            tot += ospa_sequence(sc.truth, est)[0]
        ospas[single] = tot / 5
    assert ospas[False] < ospas[True]


def test_tracker_survives_a_dense_scenario_and_reports_its_cost():
    sc = dense(n_scans=25, n_targets=5, clutter=8.0, seed=1)
    cfg = Config(sigma=20.0, Pd=0.9, clutter_density=8.0 / (6000 * 6000))
    t = Tracker(cfg, "jpda")
    est = t.run(sc.scans)
    assert t.stats["max_cluster"] >= 2          # gates really do overlap
    assert identity_metrics(sc.truth, est)["coverage"] > 0.5


def test_event_cap_degrades_rather_than_hangs():
    """With a tiny cap the tracker must still complete a dense scan."""
    sc = dense(n_scans=10, n_targets=6, clutter=15.0, seed=2)
    cfg = Config(sigma=20.0, Pd=0.9, clutter_density=15.0 / (6000 * 6000),
                 event_cap=25)
    t = Tracker(cfg, "jpda")
    t.run(sc.scans)
    assert t.stats["truncated"] > 0


def test_track_ids_are_unique():
    sc = dense(n_scans=20, n_targets=4, seed=3)
    cfg = Config(sigma=20.0, Pd=0.9, clutter_density=10.0 / (6000 * 6000))
    t = Tracker(cfg, "jpda")
    est = t.run(sc.scans)
    for scan in est:
        assert len(scan) == len(set(scan.keys()))
