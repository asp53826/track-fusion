"""Motion and measurement models.

State is [x, vx, y, vy] throughout. Keeping every mode on the same state vector
is what lets the IMM mix them without any padding games, and it's why the turn
models here take a *fixed* rate instead of estimating one — a state-augmented
CT model would be nonlinear and force an EKF into what is otherwise an
entirely linear filter bank.
"""

import numpy as np

POS = [0, 2]  # indices of x and y in the state vector


def cv(T):
    """Constant velocity."""
    return np.array([
        [1.0, T, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, T],
        [0.0, 0.0, 0.0, 1.0],
    ])


def ct(omega, T):
    """Coordinated turn at a known rate, radians/sec, counter-clockwise.

    Velocity rotates by omega*T over the step and position is the integral of
    that rotation. The two position terms are sin(wT)/w and (1-cos(wT))/w,
    both 0/0 as w -> 0, and the naive forms are bad well before that: at
    wT = 1e-7, `1 - cos(wT)` cancels down to about two significant digits.

    Writing them through sinc removes the singularity and the cancellation
    together, so there's no threshold to tune and no branch to get wrong:

        sin(wT)/w      = T * sinc(wT/pi)
        (1-cos(wT))/w  = 2 sin^2(wT/2)/w = (w T^2/2) * sinc(wT/2pi)^2

    Both are exact at w = 0, where this reduces to cv(T).
    """
    wT = omega * T
    s_w = T * np.sinc(wT / np.pi)
    c_w = 0.5 * omega * T * T * np.sinc(wT / (2.0 * np.pi)) ** 2

    return np.array([
        [1.0, s_w, 0.0, -c_w],
        [0.0, np.cos(wT), 0.0, -np.sin(wT)],
        [0.0, c_w, 1.0, s_w],
        [0.0, np.sin(wT), 0.0, np.cos(wT)],
    ])


def q_dwna(q, T):
    """Discrete white-noise acceleration process noise.

    The usual piecewise-constant-acceleration form. `q` is the acceleration
    PSD in (m/s^2)^2; it's the one knob that says how hard the target is
    allowed to manoeuvre, so the IMM modes differ mostly in this value.
    """
    T2, T3, T4 = T * T, T ** 3, T ** 4
    blk = np.array([[T4 / 4.0, T3 / 2.0],
                    [T3 / 2.0, T2]]) * q
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 1], [0, 1])] = blk
    Q[np.ix_([2, 3], [2, 3])] = blk
    return Q


def position_measurement(sigma):
    """Cartesian position measurement: H picks x and y out of the state."""
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
    R = np.eye(2) * sigma ** 2
    return H, R


class Mode:
    """One motion hypothesis in the filter bank."""

    def __init__(self, name, F, Q):
        self.name = name
        self.F = F
        self.Q = Q


def imm_modes(T, q_quiet=0.05, q_manoeuvre=2.0, turn_rate=np.deg2rad(6.0)):
    """The standard manoeuvring-target bank: straight, left turn, right turn.

    Three modes is the usual sweet spot. Adding more turn rates mostly buys
    you a slower filter, because the mixing step is O(modes^2) and the extra
    hypotheses split probability mass rather than sharpening it.
    """
    return [
        Mode("cv", cv(T), q_dwna(q_quiet, T)),
        Mode("ct+", ct(turn_rate, T), q_dwna(q_manoeuvre, T)),
        Mode("ct-", ct(-turn_rate, T), q_dwna(q_manoeuvre, T)),
    ]
