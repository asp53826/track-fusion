import numpy as np
import pytest
from scipy.stats import multivariate_normal

from tf import kalman as kf
from tf.kalman import Gaussian
from tf.models import cv, q_dwna, position_measurement


def test_log_likelihood_matches_scipy():
    S = np.array([[4.0, 1.2], [1.2, 9.0]])
    nu = np.array([0.7, -1.9])
    chol = kf.innovation(Gaussian(np.zeros(4), np.zeros((4, 4))),
                         np.zeros((2, 4)), S)[2]
    got = kf.log_likelihood(nu, chol)
    want = multivariate_normal(np.zeros(2), S).logpdf(nu)
    assert got == pytest.approx(want)


def test_mahalanobis_matches_explicit_inverse():
    S = np.array([[5.0, -2.0], [-2.0, 3.0]])
    nu = np.array([1.0, 2.0])
    chol = kf.innovation(Gaussian(np.zeros(4), np.zeros((4, 4))),
                         np.zeros((2, 4)), S)[2]
    assert kf.mahalanobis_sq(nu, chol) == pytest.approx(
        nu @ np.linalg.inv(S) @ nu)


def test_update_reduces_covariance():
    H, R = position_measurement(10.0)
    g = Gaussian(np.array([0.0, 0.0, 0.0, 0.0]), np.eye(4) * 100.0)
    zhat, S, chol = kf.innovation(g, H, R)
    K = kf.gain(g, H, chol)
    out = kf.update(g, K, np.array([5.0, -5.0]), H)
    assert np.trace(out.P) < np.trace(g.P)
    assert np.allclose(out.P, out.P.T)


def test_gain_equals_textbook_form():
    H, R = position_measurement(7.0)
    P = np.diag([50.0, 20.0, 40.0, 15.0])
    g = Gaussian(np.zeros(4), P)
    _, S, chol = kf.innovation(g, H, R)
    assert np.allclose(kf.gain(g, H, chol), P @ H.T @ np.linalg.inv(S))


def test_filter_converges_on_a_straight_target():
    """The estimate should end up closer to truth than a single measurement."""
    rng = np.random.default_rng(3)
    sigma = 25.0
    H, R = position_measurement(sigma)
    F, Q = cv(1.0), q_dwna(0.01, 1.0)

    truth = np.array([0.0, 40.0, 0.0, -25.0])
    g = Gaussian(np.array([0.0, 0.0, 0.0, 0.0]), np.diag([sigma**2, 1e4,
                                                          sigma**2, 1e4]))
    err = []
    for _ in range(60):
        truth = F @ truth
        z = H @ truth + rng.normal(0.0, sigma, 2)
        g = kf.predict(g, F, Q)
        _, S, chol = kf.innovation(g, H, R)
        K = kf.gain(g, H, chol)
        g = kf.update(g, K, z - H @ g.x, H)
        err.append(np.linalg.norm(H @ g.x - H @ truth))

    # steady-state error well under the per-measurement noise
    assert np.mean(err[-20:]) < sigma * 0.6


def test_covariance_stays_symmetric_over_many_steps():
    F, Q = cv(1.0), q_dwna(0.1, 1.0)
    H, R = position_measurement(10.0)
    g = Gaussian(np.zeros(4), np.eye(4) * 500.0)
    for _ in range(200):
        g = kf.predict(g, F, Q)
        _, S, chol = kf.innovation(g, H, R)
        K = kf.gain(g, H, chol)
        g = kf.update(g, K, np.zeros(2), H)
    assert np.allclose(g.P, g.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(g.P) > 0)
