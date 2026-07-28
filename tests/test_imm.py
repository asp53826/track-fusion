import numpy as np
import pytest

from tf import kalman as kf
from tf.imm import IMM
from tf.kalman import Gaussian
from tf.models import Mode, cv, ct, q_dwna, position_measurement, imm_modes
from tf.tracker import imm_transition


def bank(n, T=1.0, q=0.1):
    """n identical CV modes."""
    return [Mode(f"cv{i}", cv(T), q_dwna(q, T)) for i in range(n)]


def test_mode_probabilities_stay_normalised():
    H, R = position_measurement(10.0)
    imm = IMM(imm_modes(1.0), imm_transition(3, 0.05),
              init=Gaussian(np.zeros(4), np.eye(4) * 100))
    rng = np.random.default_rng(0)
    for _ in range(50):
        imm.step_prior()
        Z = rng.normal(0, 50, (2, 2))
        L = imm.measurement_likelihoods(Z, H, R)
        b = L / L.sum() if L.sum() > 0 else np.array([0.5, 0.5])
        imm.update_pda(Z, b * 0.9, 0.1, H, R, 0.9, 0.99, 1e-6)
        assert imm.mu.sum() == pytest.approx(1.0)
        assert np.all(imm.mu >= 0.0)


def test_identical_modes_reproduce_a_single_kalman_filter():
    """The strongest invariant available: if every mode is the same model, the
    IMM's mixing and combination must be the identity, and the output has to
    match a plain KF step for step. Anything wrong in mix() or estimate()
    shows up here."""
    T, q, sigma = 1.0, 0.1, 15.0
    H, R = position_measurement(sigma)
    F, Q = cv(T), q_dwna(q, T)

    x0 = np.array([10.0, 5.0, -20.0, 2.0])
    P0 = np.diag([100.0, 400.0, 100.0, 400.0])

    imm = IMM(bank(3, T, q), imm_transition(3, 0.05), init=Gaussian(x0, P0))
    g = Gaussian(x0.copy(), P0.copy())

    rng = np.random.default_rng(7)
    for _ in range(30):
        z = rng.normal(0.0, 100.0, 2)

        imm.step_prior()
        imm.update_pda(np.array([z]), np.array([1.0]), 0.0,
                       H, R, 0.9, 0.99, 1e-6)

        g = kf.predict(g, F, Q)
        _, S, chol = kf.innovation(g, H, R)
        K = kf.gain(g, H, chol)
        g = kf.update(g, K, z - H @ g.x, H)

        est = imm.estimate()
        assert np.allclose(est.x, g.x, atol=1e-9)
        assert np.allclose(est.P, g.P, atol=1e-8)


def test_mixing_is_a_no_op_when_all_modes_agree():
    imm = IMM(bank(3), imm_transition(3, 0.1),
              init=Gaussian(np.array([1.0, 2.0, 3.0, 4.0]), np.eye(4)))
    imm.mix()
    for g in imm.states:
        assert np.allclose(g.x, [1.0, 2.0, 3.0, 4.0])
        assert np.allclose(g.P, np.eye(4))


def test_mixing_inflates_covariance_when_modes_disagree():
    """The spread between mode means has to end up in the mixed covariance,
    otherwise the bank becomes overconfident the moment it has an opinion."""
    modes = bank(2)
    imm = IMM(modes, np.array([[0.5, 0.5], [0.5, 0.5]]),
              init=Gaussian(np.zeros(4), np.eye(4)))
    imm.states[0].x = np.array([100.0, 0.0, 0.0, 0.0])
    imm.states[1].x = np.array([-100.0, 0.0, 0.0, 0.0])
    imm.mix()
    assert imm.states[0].P[0, 0] > 1.0 + 9000.0


def test_combined_gate_covariance_is_conservative():
    """Moment-matched S must dominate every individual mode's S, or gating on
    it could reject a measurement the mixture would have accepted."""
    H, R = position_measurement(10.0)
    imm = IMM(imm_modes(1.0), imm_transition(3, 0.05),
              init=Gaussian(np.array([0.0, 150.0, 0.0, 0.0]),
                            np.diag([100.0, 2500.0, 100.0, 2500.0])))
    imm.step_prior()
    _, S, _ = imm.combined_measurement(H, R)
    for _z, Sj, _c in imm.predicted_measurements(H, R):
        # S - Sj need not be PSD in general, but the determinant (gate volume)
        # must not shrink below any single mode's
        assert np.linalg.det(S) >= np.linalg.det(Sj) - 1e-9


def test_mixture_likelihood_is_a_proper_density():
    """Integrates to 1 over the plane, and is not the same as the likelihood
    under the moment-matched Gaussian."""
    H, R = position_measurement(20.0)
    imm = IMM(imm_modes(1.0), imm_transition(3, 0.05),
              init=Gaussian(np.array([0.0, 200.0, 0.0, 0.0]),
                            np.diag([100.0, 1e4, 100.0, 1e4])))
    imm.step_prior()
    imm.mu = np.array([0.2, 0.5, 0.3])

    g = np.linspace(-1500, 1500, 601)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    dens = imm.measurement_likelihoods(pts, H, R)
    cell = (g[1] - g[0]) ** 2
    assert dens.sum() * cell == pytest.approx(1.0, rel=2e-3)


def test_imm_beats_a_single_filter_through_a_turn():
    """The whole reason the bank exists. Same data, same measurement noise;
    the only difference is one CV filter versus the three-mode bank."""
    T, sigma = 1.0, 20.0
    H, R = position_measurement(sigma)
    rng = np.random.default_rng(11)

    truth = np.array([-2000.0, 120.0, 0.0, 0.0])
    F_turn = ct(np.deg2rad(6.0), T)
    F_straight = cv(T)

    single = IMM([Mode("cv", cv(T), q_dwna(1.0, T))], np.ones((1, 1)),
                 init=Gaussian(np.array([-2000.0, 120.0, 0.0, 0.0]),
                               np.diag([400.0, 1e4, 400.0, 1e4])))
    multi = IMM(imm_modes(T), imm_transition(3, 0.05),
                init=Gaussian(np.array([-2000.0, 120.0, 0.0, 0.0]),
                              np.diag([400.0, 1e4, 400.0, 1e4])))

    err_s, err_m = [], []
    for k in range(60):
        truth = (F_straight if k < 20 else F_turn) @ truth
        z = H @ truth + rng.normal(0.0, sigma, 2)
        for f, err in ((single, err_s), (multi, err_m)):
            f.step_prior()
            f.update_pda(np.array([z]), np.array([1.0]), 0.0,
                         H, R, 0.95, 0.99, 1e-6)
            if k >= 20:
                err.append(np.linalg.norm(H @ f.estimate().x - H @ truth))

    assert np.mean(err_m) < np.mean(err_s)


def test_transition_matrix_is_validated():
    with pytest.raises(ValueError):
        IMM(bank(2), np.array([[0.5, 0.9], [0.1, 0.9]]),
            init=Gaussian(np.zeros(4), np.eye(4)))
    with pytest.raises(ValueError):
        IMM(bank(3), np.eye(2), init=Gaussian(np.zeros(4), np.eye(4)))
