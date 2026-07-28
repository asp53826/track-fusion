"""Linear Kalman filter, plus the pieces association needs.

Nothing exotic here, but two details matter downstream:

  * `innovation` returns S and its Cholesky factor, because gating and the
    Gaussian likelihood both want S^-1 and neither should be forming it with
    an explicit inverse.
  * `update` is split from `predict` and takes an already-computed gain, so
    PDA can reuse the same gain across a weighted sum of innovations instead
    of re-deriving it per measurement.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class Gaussian:
    __slots__ = ("x", "P")

    def __init__(self, x, P):
        self.x = np.asarray(x, dtype=float)
        self.P = np.asarray(P, dtype=float)

    def copy(self):
        return Gaussian(self.x.copy(), self.P.copy())


def predict(g, F, Q):
    return Gaussian(F @ g.x, F @ g.P @ F.T + Q)


def innovation(g, H, R):
    """Predicted measurement, innovation covariance, and its Cholesky factor."""
    zhat = H @ g.x
    S = H @ g.P @ H.T + R
    S = 0.5 * (S + S.T)          # keep it symmetric; drift here breaks cho_factor
    return zhat, S, cho_factor(S, lower=True)


def gain(g, H, S_chol):
    """K = P H' S^-1, solved rather than inverted."""
    return cho_solve(S_chol, (g.P @ H.T).T).T


def update(g, K, nu, H):
    """Joseph-free covariance update.

    P = (I - KH) P is fine here because K is the optimal gain and everything
    stays symmetric; the Joseph form only earns its cost with a suboptimal K.
    PDA needs a different covariance anyway and computes its own.
    """
    x = g.x + K @ nu
    P = g.P - K @ H @ g.P
    return Gaussian(x, 0.5 * (P + P.T))


def log_likelihood(nu, S_chol):
    """log N(nu; 0, S). Uses the Cholesky factor for both the quadratic form
    and the determinant, so no inverse and no det() call."""
    L = S_chol[0]
    alpha = cho_solve(S_chol, nu)
    log_det = 2.0 * np.sum(np.log(np.abs(np.diag(L))))
    return -0.5 * (nu @ alpha + log_det + len(nu) * np.log(2.0 * np.pi))


def mahalanobis_sq(nu, S_chol):
    return float(nu @ cho_solve(S_chol, nu))
