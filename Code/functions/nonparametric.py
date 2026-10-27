"""Multi-output batch GP (generalizes nonparametricEstimation.m and nonparametricPrediction.m).

The latent field is conditioned on every observed measurement at once, so the
estimate at time t uses all time instants (a smoother). The training set is the
grid of measured locations x outputs x time instants, ordered as
Y.ravel(order="F"), and the covariances are built from the Kronecker form of
`multioutput_kernel_sampled`. The cost is O((numMeasLocs * D * T)^3).
"""

import time

import numpy as np
import scipy.linalg

from .generate_multioutput_data import multioutput_kernel_sampled


def batch_gp_posterior(kernel, locs_meas, time_instants, meas, noise_var, locs_test):
    """Condition the GP on the observed measurements and return per-time posteriors.

    Parameters
    ----------
    kernel : dict
        Multi-output kernel (see `multioutput_kernel`).
    locs_meas : array_like, shape (numMeasLocs, p)
    time_instants : array_like, shape (T,)
    meas : numpy.ndarray, shape (numMeasLocs, D, T)
        Measurements, NaN where missing (these are dropped).
    noise_var : numpy.ndarray, same shape as meas
    locs_test : array_like, shape (numTestLocs, p)
        Locations at which the latent field is returned, at every time instant.

    Returns
    -------
    mean : numpy.ndarray, shape (numTestLocs, D, T)
    cov : numpy.ndarray, shape (T, numTestLocs * D, numTestLocs * D)
        Joint posterior covariance of all locations and outputs at each time.
    """
    num_outputs, num_time_inst = meas.shape[1], meas.shape[2]
    num_test = len(locs_test)
    block = num_test * num_outputs

    y = meas.ravel(order="F")
    observed = ~np.isnan(y)

    # training covariance of the observed measurements
    K_train = multioutput_kernel_sampled(locs_meas, locs_meas, time_instants, time_instants, kernel)
    K_obs = K_train[np.ix_(observed, observed)] + np.diag(noise_var.ravel(order="F")[observed])
    chol = scipy.linalg.cho_factor(K_obs, lower=True)
    alpha = scipy.linalg.cho_solve(chol, y[observed])

    # cross covariance between every test input and the observed measurements
    K_cross = multioutput_kernel_sampled(locs_test, locs_meas, time_instants, time_instants, kernel)[:, observed]
    mean = (K_cross @ alpha).reshape(num_test, num_outputs, num_time_inst, order="F")

    # prior covariance at one time instant (the kernel is stationary in time)
    K_test = multioutput_kernel_sampled(locs_test, locs_test, time_instants[:1], time_instants[:1], kernel)
    cov = np.zeros((num_time_inst, block, block))
    for t in range(num_time_inst):
        W = scipy.linalg.solve_triangular(chol[0], K_cross[t * block:(t + 1) * block].T, lower=True)
        cov[t] = K_test - W.T @ W
    return mean, cov


def nonparametric_estimation(param_data, param_np, meas, noise_var):
    """Batch multi-output GP estimate at the measured locations.

    Parameters
    ----------
    param_data : dict
        Data parameters ("spaceLocsMeas", "timeInstants").
    param_np : dict
        Nonparametric parameters, with a multi-output "kernel".
    meas : numpy.ndarray, shape (numMeasLocs, D, numTimeInstants)
        Measurements, NaN where missing.
    noise_var : numpy.ndarray, same shape as meas

    Returns
    -------
    posteriorMean : numpy.ndarray, shape (numMeasLocs, D, numTimeInstants)
    posteriorCov : numpy.ndarray, shape (numTimeInstants, numMeasLocs * D, numMeasLocs * D)
    exeTime : float
        Total execution time in seconds.
    """
    start = time.perf_counter()
    mean, cov = batch_gp_posterior(param_np["kernel"], param_data["spaceLocsMeas"], param_data["timeInstants"],
                                   meas, noise_var, param_data["spaceLocsMeas"])
    return mean, cov, time.perf_counter() - start


def nonparametric_prediction(param_data, param_np, meas, noise_var):
    """Batch multi-output GP prediction at the unmeasured locations.

    Arguments are as in `nonparametric_estimation`; "spaceLocsPred" is also
    required in `param_data`.

    Returns
    -------
    predictedMean : numpy.ndarray, shape (numPredLocs, D, numTimeInstants)
    predictedCov : numpy.ndarray, shape (numTimeInstants, numPredLocs * D, numPredLocs * D)
    exeTime : float
        Total execution time in seconds.
    """
    start = time.perf_counter()
    mean, cov = batch_gp_posterior(param_np["kernel"], param_data["spaceLocsMeas"], param_data["timeInstants"],
                                   meas, noise_var, param_data["spaceLocsPred"])
    return mean, cov, time.perf_counter() - start
