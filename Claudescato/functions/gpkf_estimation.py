"""Port of gpkfEstimation.m (the proposed GPKF estimator)."""

import numpy as np

from .create_discrete_time_sys import create_discrete_time_sys
from .kalman_est import kalman_est
from .kernel_function import kernel_function
from .kernel_sampled import kernel_sampled


def gpkf_estimation(param_data, param_gpkf, meas, noise_var):
    """GPKF estimate at the measured locations.

    The time kernel is realized as a linear state-space model and the space
    kernel enters through the output matrix, C = chol(Ks)' kron(I, c).
    This is a filter: the estimate at time t uses measurements up to time t.

    Parameters
    ----------
    param_data : dict
        Data parameters ("spaceLocsMeas", "samplingTime").
    param_gpkf : dict
        GPKF parameters. param_gpkf["kernel"]["time"] must hold the state-space
        coefficients "num" and "den" (see load_parameters).
    meas : numpy.ndarray, shape (numMeasLocs, numTimeInstants)
        Measurements, NaN where missing.
    noise_var : numpy.ndarray, same shape as meas

    Returns
    -------
    posteriorMean : numpy.ndarray, shape (numMeasLocs, numTimeInstants)
    posteriorCov : numpy.ndarray, shape (numTimeInstants, numMeasLocs, numMeasLocs)
    exeTime : float
        Average execution time per Kalman iteration, in seconds.
    negLogMarginal : float
        Negative marginal log-likelihood (called logMarginal in MATLAB).
    """
    num_space_locs = meas.shape[0]
    time_kernel = param_gpkf["kernel"]["time"]

    # create DT state space model
    a, c, v0, q = create_discrete_time_sys(time_kernel["num"], time_kernel["den"], param_data["samplingTime"])

    # create space kernel
    kernel_space = kernel_function(param_gpkf["kernel"]["space"])
    Ks = kernel_sampled(param_data["spaceLocsMeas"], param_data["spaceLocsMeas"], kernel_space)
    Ks_chol = np.linalg.cholesky(Ks)   # lower triangular, as chol(Ks)' in MATLAB

    # quantities needed for kalman estimation
    I = np.eye(num_space_locs)
    A = np.kron(I, a)
    C = Ks_chol @ np.kron(I, c)
    V0 = np.kron(I, v0)
    Q = np.kron(I, q)

    # compute kalman estimate
    x, V, _, _, exe_time, neg_log_marginal = kalman_est(A, C, Q, V0, meas, noise_var)

    # output function and posterior variance
    posterior_mean = C @ x
    posterior_cov = C @ V @ C.T   # broadcasts over the time axis
    return posterior_mean, posterior_cov, exe_time, neg_log_marginal
