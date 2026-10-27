"""State-space realizations of the time kernels (generalizes createDiscreteTimeSys.m).

A stationary time kernel gamma(tau) with a rational power spectrum is the
covariance of the output c z(t) of the linear SDE

    dz/dt = F z + G w(t),    w white noise with unit intensity,

with F in companion form (last row -den) and G = e_n. Sampled every Ts, it is
the discrete-time model

    z_{k+1} = A z_k + w_k,   w_k ~ N(0, Q),   z_0 ~ N(0, V),

with A = expm(F Ts), V the stationary covariance (F V + V F' + G G' = 0) and
Q = V - A V A', so that gamma(k Ts) = c A^k V c'.

Supported kernels (parameter dicts as for `kernel_function`), with lam = 1/std
for "matern" nu = 1/2 and lam = sqrt(2 nu) / std otherwise:

    matern nu = 1/2 : den = [lam],                   num = [sqrt(2 scale lam)]
    matern nu = 3/2 : den = [lam^2, 2 lam],          num = [sqrt(4 scale lam^3)]
    matern nu = 5/2 : den = [lam^3, 3 lam^2, 3 lam], num = [sqrt(16/3 scale lam^5)]
    periodic        : den = [w2, 2 / std],           num = sqrt(2 scale / std) [sqrt(w2), 1],
                      with w2 = 1 / std^2 + (2 pi frequency)^2
    gaussian        : rational approximation of exp(-tau^2 / 2) precomputed for
                      scale = std = 1 (Code/data), rescaled in time by std and in
                      amplitude by sqrt(scale). "ssDim" (1 to 6, default 6) sets
                      its state dimension.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg

GAUSSIAN_APPROX_DIR = Path(__file__).resolve().parent.parent / "data" / "gaussian_time_kernel_approximations"


def _load_gaussian_time_approximation(ss_dim):
    """Return (num, den) of the precomputed Gaussian time-kernel approximation."""
    path = GAUSSIAN_APPROX_DIR / f"ssDim={ss_dim}_for_scale=1_std=1.csv"
    coeffs = pd.read_csv(path, index_col="coefficient")
    return coeffs["num"].to_numpy(dtype=np.float64), coeffs["den"].to_numpy(dtype=np.float64)


def create_discrete_time_sys(num_coeff, den_coeff, Ts):
    """Return (A, C, V, Q) of the companion-form realization sampled every Ts.

    Parameters
    ----------
    num_coeff, den_coeff : array_like
        Numerator and denominator coefficients (lowest order first).
    Ts : float
        Sampling time.

    Returns
    -------
    A : numpy.ndarray, shape (s, s)
    C : numpy.ndarray, shape (1, s)
    V : numpy.ndarray, shape (s, s)
        Stationary state covariance.
    Q : numpy.ndarray, shape (s, s)
        Discrete-time process noise covariance, V - A V A'. MATLAB integrates
        it numerically; for the stationary process this is the exact value.
    """
    num_coeff = np.atleast_1d(np.asarray(num_coeff, dtype=np.float64)).ravel()
    den_coeff = np.atleast_1d(np.asarray(den_coeff, dtype=np.float64)).ravel()
    state_dim = den_coeff.size

    F = np.diag(np.ones(state_dim - 1), 1)
    F[-1, :] = -den_coeff
    GGt = np.zeros((state_dim, state_dim))
    GGt[-1, -1] = 1.0
    C = np.zeros((1, state_dim))
    C[0, : num_coeff.size] = num_coeff

    A = scipy.linalg.expm(F * Ts)
    V = scipy.linalg.solve_continuous_lyapunov(F, -GGt)
    V = 0.5 * (V + V.T)
    Q = V - A @ V @ A.T
    return A, C, V, 0.5 * (Q + Q.T)


def time_kernel_coefficients(time_par):
    """Return (num, den, time_scale) for the companion-form realization of `time_par`.

    The realization with coefficients (num, den) is sampled at Ts / time_scale.
    This is 1 except for the Gaussian kernel, whose approximation is only
    available for std = 1 and is rescaled in time instead.
    """
    kernel_type = time_par["type"]
    scale, std = float(time_par["scale"]), float(time_par["std"])

    if kernel_type == "matern":
        nu = float(time_par["nu"])
        if nu == 0.5:
            lam = 1.0 / std
            return np.array([np.sqrt(2 * scale * lam)]), np.array([lam]), 1.0
        lam = np.sqrt(2 * nu) / std
        if nu == 1.5:
            return np.array([np.sqrt(4 * scale * lam**3)]), np.array([lam**2, 2 * lam]), 1.0
        if nu == 2.5:
            return np.array([np.sqrt(16 / 3 * scale * lam**5)]), np.array([lam**3, 3 * lam**2, 3 * lam]), 1.0
        raise ValueError(f"time_kernel_coefficients: Matern nu must be 0.5, 1.5 or 2.5, got {nu}")

    if kernel_type == "periodic":
        omega_sq = (1 / std) ** 2 + (2 * np.pi * time_par["frequency"]) ** 2
        num = np.sqrt(2 * scale / std) * np.array([np.sqrt(omega_sq), 1.0])
        return num, np.array([omega_sq, 2 / std]), 1.0

    if kernel_type == "gaussian":
        num, den = _load_gaussian_time_approximation(int(time_par.get("ssDim", 6)))
        return np.sqrt(scale) * num, den, std

    raise ValueError(f"time_kernel_coefficients: no state-space form for time kernel '{kernel_type}'")


def time_state_space(time_par, Ts):
    """Return (A, C, V, Q) of the time kernel `time_par` sampled every Ts.

    C A^k V C' is exactly equal to gamma(k Ts) for "matern" and "periodic" kernels,
    approximately equal for the "gaussian" kernel.

    Parameters
    ----------
    time_par : dict
        Parameters of temporal kernel
    Ts : float
        Sampling time.

    Returns
    -------
    A : numpy.ndarray, shape (s, s)
    C : numpy.ndarray, shape (1, s)
    V : numpy.ndarray, shape (s, s)
        Stationary state covariance.
    Q : numpy.ndarray, shape (s, s)
        Discrete-time process noise covariance, V - A V A'. MATLAB integrates
        it numerically; for the stationary process this is the exact value.
    """
    num, den, time_scale = time_kernel_coefficients(time_par)
    return create_discrete_time_sys(num, den, Ts / time_scale)
