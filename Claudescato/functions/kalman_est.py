"""Port of kalmanEst.m."""

import time

import numpy as np
import scipy.linalg


def kalman_est(A, C, Q, V0, meas, noise_var):
    """Standard Kalman filter with missing measurements.

    Parameters
    ----------
    A, C, Q : numpy.ndarray
        State matrix (s, s), output matrix (n, s) and process noise
        covariance (s, s).
    V0 : numpy.ndarray, shape (s, s)
        Initial state covariance. The initial state mean is zero.
    meas : numpy.ndarray, shape (n, T)
        Measurements, NaN where missing.
    noise_var : numpy.ndarray, shape (n, T)
        Measurement noise variances. MATLAB passes the diagonal matrices
        R(:,:,t) = diag(noiseVar(:,t)); only their diagonals are needed here.

    Returns
    -------
    x : numpy.ndarray, shape (s, T)
        Filtered (corrected) state means.
    V : numpy.ndarray, shape (T, s, s)
        Filtered state covariances.
    xp : numpy.ndarray, shape (s, T)
        One-step predicted state means.
    Vp : numpy.ndarray, shape (T, s, s)
        One-step predicted state covariances.
    exeTimePerIter : float
        Average execution time per Kalman iteration, in seconds.
    negLogMarginal : float
        Negative marginal log-likelihood of the observed data. MATLAB calls
        this value logMarginal, but it accumulates +0.5 (m log 2 pi +
        log det S + e' S^-1 e), which is the negative log-likelihood.
    """
    state_dim = A.shape[0]
    num_time_instants = meas.shape[1]
    I = np.eye(state_dim)
    times = np.zeros(num_time_instants)
    neg_log_marginal = 0.0

    # initialization
    xt = np.zeros(state_dim)
    Vt = V0.copy()
    xp = np.zeros((state_dim, num_time_instants))
    Vp = np.zeros((num_time_instants, state_dim, state_dim))
    x = np.zeros((state_dim, num_time_instants))
    V = np.zeros((num_time_instants, state_dim, state_dim))

    for t in range(num_time_instants):
        tstart = time.perf_counter()
        # prediction
        xpt = A @ xt
        Vpt = A @ Vt @ A.T + Q

        # correction (only with the available measurements)
        observed = ~np.isnan(meas[:, t])
        if observed.any():
            Ct = C[observed, :]
            Rt = np.diag(noise_var[observed, t])
            innovation = meas[observed, t] - Ct @ xpt
            innov_var = Ct @ Vpt @ Ct.T + Rt
            innov_chol = scipy.linalg.cho_factor(innov_var, lower=True)
            K = scipy.linalg.cho_solve(innov_chol, Ct @ Vpt).T   # Kalman gain, Vpt Ct' S^-1
            xt = xpt + K @ innovation
            IKC = I - K @ Ct
            Vt = IKC @ Vpt @ IKC.T + K @ Rt @ K.T                 # Joseph form
        else:
            innovation = None
            xt, Vt = xpt, Vpt
        times[t] = time.perf_counter() - tstart

        # save values
        xp[:, t] = xpt
        Vp[t] = Vpt
        x[:, t] = xt
        V[t] = Vt

        # computations for the marginal likelihood
        if innovation is not None:
            log_det = 2.0 * np.sum(np.log(np.diag(innov_chol[0])))
            quad = innovation @ scipy.linalg.cho_solve(innov_chol, innovation)
            neg_log_marginal += 0.5 * (innovation.size * np.log(2 * np.pi) + log_det + quad)

    return x, V, xp, Vp, times.mean(), neg_log_marginal
