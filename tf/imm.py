"""Interacting Multiple Model estimator.

The point of an IMM is that a single Kalman filter has to pick one process
noise, and that choice is a straight trade: tight noise tracks a straight leg
accurately and then lags badly through a turn, loose noise survives the turn
and is noisy everywhere else. The IMM runs both and lets the measurements
decide the weighting, re-mixing the estimates each step so a mode that has
been idle doesn't drift off and become useless when it's needed.

Association is handled outside this class, but it can't be handled
independently of it: the predictive density of a manoeuvring track is a
*mixture*, not a Gaussian, so `measurement_likelihoods` returns the mixture
value rather than the likelihood under the combined moment-matched Gaussian.
Using the latter is a common shortcut and it under-weights measurements out
on the turning modes, which is exactly when you need association to work.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import kalman as kf
from .kalman import Gaussian


class IMM:
    def __init__(self, modes, transition, mode_probs=None, init=None):
        self.modes = modes
        self.Pi = np.asarray(transition, dtype=float)
        r = len(modes)
        if self.Pi.shape != (r, r):
            raise ValueError(f"transition must be {r}x{r}, got {self.Pi.shape}")
        if not np.allclose(self.Pi.sum(axis=1), 1.0):
            raise ValueError("transition rows must sum to 1")

        self.mu = (np.full(r, 1.0 / r) if mode_probs is None
                   else np.asarray(mode_probs, dtype=float).copy())
        self.states = [init.copy() for _ in modes] if init is not None else None

    # -- IMM cycle ---------------------------------------------------------

    def mix(self):
        """Step 1-2: mixing probabilities, then a moment-matched mixed prior
        for each mode. This is the whole trick — without it each mode would
        evolve in isolation and the bank would degenerate."""
        r = len(self.modes)
        cbar = self.Pi.T @ self.mu                      # normaliser per target mode
        cbar = np.maximum(cbar, 1e-300)
        w = (self.Pi * self.mu[:, None]) / cbar[None, :]  # w[i, j] = mu_{i|j}

        mixed = []
        for j in range(r):
            x = sum(w[i, j] * self.states[i].x for i in range(r))
            P = np.zeros_like(self.states[0].P)
            for i in range(r):
                d = (self.states[i].x - x).reshape(-1, 1)
                P += w[i, j] * (self.states[i].P + d @ d.T)
            mixed.append(Gaussian(x, P))

        self.states = mixed
        self._cbar = cbar

    def predict(self):
        self.states = [kf.predict(g, m.F, m.Q)
                       for g, m in zip(self.states, self.modes)]

    def step_prior(self):
        self.mix()
        self.predict()

    # -- association interface --------------------------------------------

    def predicted_measurements(self, H, R):
        """Per-mode (zhat, S, chol). Association needs all of them."""
        return [kf.innovation(g, H, R) for g in self.states]

    def combined_measurement(self, H, R):
        """Moment-matched predicted measurement, for gating only.

        Gating wants one ellipsoid, not a mixture, and the moment-matched
        covariance is conservative relative to the mixture — it is the mixture
        covariance plus the spread of the mode means — so gating on it cannot
        drop a measurement the mixture would have accepted.
        """
        preds = [(H @ g.x, H @ g.P @ H.T + R) for g in self.states]
        z = sum(m * (zh) for m, (zh, _) in zip(self.mu, preds))
        S = np.zeros_like(preds[0][1])
        for m, (zh, Sj) in zip(self.mu, preds):
            d = (zh - z).reshape(-1, 1)
            S += m * (Sj + d @ d.T)
        S = 0.5 * (S + S.T)
        return z, S, cho_factor(S, lower=True)

    def measurement_likelihoods(self, Z, H, R):
        """Mixture likelihood of each measurement: sum_j mu_j N(z; zhat_j, S_j).

        Vectorised over measurements. This is called once per track per scan
        with every gated measurement, so the per-measurement Python loop it
        replaced dominated the whole tracker at high clutter densities.
        """
        Z = np.asarray(Z, dtype=float).reshape(-1, H.shape[0])
        out = np.zeros(len(Z))
        if len(Z) == 0:
            return out

        for j, (zhat, _S, chol) in enumerate(self.predicted_measurements(H, R)):
            if self.mu[j] <= 0.0:
                continue
            nu = Z - zhat                              # (m, d)
            alpha = cho_solve(chol, nu.T)              # (d, m)
            quad = np.einsum("md,dm->m", nu, alpha)
            log_det = 2.0 * np.sum(np.log(np.abs(np.diag(chol[0]))))
            out += self.mu[j] * np.exp(
                -0.5 * (quad + log_det + Z.shape[1] * np.log(2.0 * np.pi)))
        return out

    # -- update ------------------------------------------------------------

    def update_pda(self, Z, betas, beta0, H, R, Pd, Pg, clutter_density):
        """PDA update applied per mode, with association weights shared.

        `betas[i]` is P(measurement i originated from this track) and `beta0`
        is P(none did). The weights come from the joint association step, so
        every mode is updated against the same soft assignment — the modes
        disagree about the *state*, not about which measurements are the
        target's.
        """
        r = len(self.modes)
        lam = np.zeros(r)

        for j in range(r):
            g = self.states[j]
            zhat, S, chol = kf.innovation(g, H, R)
            K = kf.gain(g, H, chol)

            if len(Z):
                nus = np.asarray([z - zhat for z in Z])
                nu_bar = betas @ nus
                # spread of the innovations, weighted -- this is the term that
                # makes a PDA covariance larger than a Kalman one, and it is
                # the honest cost of not knowing which measurement was real
                spread = sum(b * np.outer(n, n) for b, n in zip(betas, nus))
                spread -= np.outer(nu_bar, nu_bar)
                like = np.exp([kf.log_likelihood(n, chol) for n in nus])
            else:
                nu_bar = np.zeros(H.shape[0])
                spread = np.zeros((H.shape[0], H.shape[0]))
                like = np.zeros(0)

            Pc = g.P - K @ H @ g.P
            P = (beta0 * g.P
                 + (1.0 - beta0) * Pc
                 + K @ spread @ K.T)
            self.states[j] = Gaussian(g.x + K @ nu_bar, 0.5 * (P + P.T))

            # mode likelihood, PDAF form: miss term plus detected terms
            lam[j] = (1.0 - Pd * Pg)
            if len(like):
                lam[j] += (Pd / max(clutter_density, 1e-300)) * like.sum()

        post = self._cbar * lam
        total = post.sum()
        self.mu = post / total if total > 0 else np.full(r, 1.0 / r)

    def update_missed(self, H, R, Pd, Pg):
        """No measurements in the gate at all."""
        self.update_pda(np.zeros((0, H.shape[0])), np.zeros(0), 1.0,
                        H, R, Pd, Pg, 1.0)

    # -- output ------------------------------------------------------------

    def estimate(self):
        """Moment-matched combination across modes."""
        x = sum(m * g.x for m, g in zip(self.mu, self.states))
        P = np.zeros_like(self.states[0].P)
        for m, g in zip(self.mu, self.states):
            d = (g.x - x).reshape(-1, 1)
            P += m * (g.P + d @ d.T)
        return Gaussian(x, 0.5 * (P + P.T))
