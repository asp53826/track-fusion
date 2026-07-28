"""Ellipsoidal gating.

Gating is a pure cost optimisation — it throws away pairings that association
would have scored near zero anyway. The threshold is a chi-square quantile
because the squared Mahalanobis distance of a correct measurement is
chi-square with dim(z) degrees of freedom, assuming the filter is consistent.
That assumption is worth remembering: an inconsistent filter has an optimistic
S, which shrinks the gate, which drops the true measurement, which makes the
filter more inconsistent. Gates are where a diverging tracker dies.
"""

import numpy as np
from scipy.stats import chi2


class Gate:
    def __init__(self, dim=2, prob=0.99):
        self.dim = dim
        self.prob = prob
        self.threshold = chi2.ppf(prob, dim)

    @property
    def Pg(self):
        """Probability the true measurement falls inside the gate."""
        return self.prob

    def contains(self, d2):
        return d2 <= self.threshold

    def volume(self, S):
        """Gate volume, needed by non-parametric clutter estimation.

        c_d * threshold^(d/2) * sqrt(det S), with c_d the unit-hypersphere
        volume. Only the 2D case is used here.
        """
        if self.dim != 2:
            raise NotImplementedError("volume() is only derived for dim=2")
        return np.pi * self.threshold * np.sqrt(np.linalg.det(S))
