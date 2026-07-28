import numpy as np
import pytest

from tf.models import cv, ct, q_dwna, imm_modes


def test_cv_propagates_position_by_velocity():
    F = cv(2.0)
    x = np.array([0.0, 3.0, 0.0, -4.0])
    assert np.allclose(F @ x, [6.0, 3.0, -8.0, -4.0])


def test_ct_reduces_to_cv_at_zero_turn_rate():
    # the closed form is 0/0 here, so this is really testing the Taylor branch
    assert np.allclose(ct(0.0, 1.0), cv(1.0))
    assert np.allclose(ct(1e-9, 0.5), cv(0.5), atol=1e-9)


def test_ct_matches_naive_closed_form_where_the_naive_form_is_reliable():
    """At a normal turn rate both formulations agree to machine precision."""
    w, T = np.deg2rad(6.0), 1.0
    wT = w * T
    naive = np.array([
        [1.0, np.sin(wT) / w, 0.0, -(1 - np.cos(wT)) / w],
        [0.0, np.cos(wT), 0.0, -np.sin(wT)],
        [0.0, (1 - np.cos(wT)) / w, 1.0, np.sin(wT) / w],
        [0.0, np.sin(wT), 0.0, np.cos(wT)],
    ])
    assert np.allclose(ct(w, T), naive, rtol=1e-12, atol=0.0)


def test_ct_stays_accurate_where_the_naive_closed_form_falls_apart():
    """(1 - cos wT) cancels catastrophically for tiny wT.

    The exact value of the (1-cos(wT))/w term is w*T^2/2 to well within
    double precision at these rates, so we can check against it directly.
    The naive form is off by ~1e-4 relative here; sinc is exact.
    """
    T = 1.0
    for w in (1e-7, 1e-9, 0.0):
        exact_c = 0.5 * w * T * T
        assert ct(w, T)[2, 1] == pytest.approx(exact_c, rel=1e-12, abs=1e-30)
        assert ct(w, T)[0, 1] == pytest.approx(T, rel=1e-12)


def test_ct_has_no_discontinuity_across_the_rate_range():
    """Sweep through zero and look for a *step*, not for smallness.

    Consecutive matrices differ by roughly (derivative * step) everywhere, so
    the absolute jump size means nothing on its own. What a thresholded
    implementation leaves behind is one jump much larger than its neighbours,
    right at the cutoff -- so compare the worst jump to the typical one.
    """
    ws = np.linspace(-1e-5, 1e-5, 4001)
    mats = np.array([ct(w, 1.0) for w in ws])
    jumps = np.abs(np.diff(mats, axis=0)).max(axis=(1, 2))
    assert jumps.max() < 3.0 * np.median(jumps)


def test_ct_preserves_speed():
    """A coordinated turn rotates the velocity vector; it must not scale it."""
    F = ct(np.deg2rad(9.0), 1.0)
    x = np.array([100.0, 60.0, -50.0, 80.0])
    speed_before = np.hypot(x[1], x[3])
    for _ in range(40):
        x = F @ x
        assert np.isclose(np.hypot(x[1], x[3]), speed_before)


def test_ct_turns_the_right_way():
    """Positive omega is counter-clockwise: +x velocity rotates toward +y."""
    F = ct(np.deg2rad(30.0), 1.0)
    v = F @ np.array([0.0, 100.0, 0.0, 0.0])
    assert v[1] == pytest.approx(100.0 * np.cos(np.deg2rad(30.0)))
    assert v[3] == pytest.approx(100.0 * np.sin(np.deg2rad(30.0)))


def test_ct_closes_a_full_circle():
    """360 degrees of turn returns to the starting state."""
    rate = np.deg2rad(10.0)
    F = ct(rate, 1.0)
    x0 = np.array([500.0, 80.0, -200.0, 0.0])
    x = x0.copy()
    for _ in range(36):          # 36 steps * 10 deg = 360
        x = F @ x
    assert np.allclose(x, x0, atol=1e-6)


def test_process_noise_is_psd_and_scales_with_q():
    Q = q_dwna(0.7, 1.3)
    assert np.allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-12)
    assert np.allclose(q_dwna(1.4, 1.3), 2.0 * Q)


def test_process_noise_couples_position_and_velocity():
    """Position and velocity share the same acceleration, so the off-diagonal
    term must be there. A diagonal Q is a common and quietly wrong shortcut."""
    Q = q_dwna(1.0, 1.0)
    assert Q[0, 1] > 0.0 and Q[2, 3] > 0.0
    # but x and y are driven independently
    assert Q[0, 2] == 0.0 and Q[1, 3] == 0.0


def test_ct_inverse_is_the_same_turn_run_backwards():
    """F(w, T)^-1 == F(w, -T). Note it is *not* F(-w, T) -- reversing the turn
    direction is a different motion from reversing time."""
    w, T = np.deg2rad(6.0), 1.0
    assert np.allclose(ct(w, T) @ ct(w, -T), np.eye(4))
    assert not np.allclose(ct(w, T) @ ct(-w, T), np.eye(4))


def test_imm_bank_has_one_straight_and_two_mirrored_turns():
    """The two turn modes are mirror images under y -> -y, which is what makes
    the bank symmetric about a straight leg."""
    modes = imm_modes(1.0)
    assert [m.name for m in modes] == ["cv", "ct+", "ct-"]
    M = np.diag([1.0, 1.0, -1.0, -1.0])
    assert np.allclose(M @ modes[1].F @ M, modes[2].F)
