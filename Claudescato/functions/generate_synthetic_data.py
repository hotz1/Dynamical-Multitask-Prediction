"""Port of generateSyntheticData.m."""

import numpy as np

from .kernel_function import kernel_function
from .kernel_sampled import kernel_sampled


def _psd_sqrt(K):
    """Return L with L L' = K, clipping tiny negative eigenvalues to zero."""
    eigvals, eigvecs = np.linalg.eigh(0.5 * (K + K.T))
    return eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))


def generate_synthetic_data(param, rng=None):
    """Sample a spatio-temporal GP and noisy, partially missing measurements.

    Parameters
    ----------
    param : dict
        Data parameters ("spaceLocs", "spaceLocsMeasIdx", "timeInstants",
        "noiseStd", "kernel").
    rng : numpy.random.Generator, optional

    Returns
    -------
    F : numpy.ndarray, shape (numLocs, numTimeInstants)
        True (zero-mean) GP sample; rows are locations, columns are times.
    Y : numpy.ndarray, shape (numMeasLocs, numTimeInstants)
        Noisy measurements at the measured locations, NaN where missing.
    noiseVar : numpy.ndarray, same shape as Y
        Measurement noise variances, +inf where missing.

    Notes
    -----
    MATLAB draws f ~ N(0, kron(Kt, Ks)) with mvnrnd. Here the same
    distribution is sampled as F = Ls Z Lt', with Ks = Ls Ls', Kt = Lt Lt' and
    Z standard normal. This avoids factorizing the full space-time matrix.
    """
    rng = np.random.default_rng() if rng is None else rng

    kernel_space = kernel_function(param["kernel"]["space"])
    kernel_time = kernel_function(param["kernel"]["time"])
    Ks = kernel_sampled(param["spaceLocs"], param["spaceLocs"], kernel_space)
    Kt = kernel_sampled(param["timeInstants"], param["timeInstants"], kernel_time)

    num_space_locs = Ks.shape[0]
    num_time_inst = Kt.shape[0]

    # sample "true" (zero mean) GP in matrix form (row: space, column: time)
    Z = rng.standard_normal((num_space_locs, num_time_inst))
    F = _psd_sqrt(Ks) @ Z @ _psd_sqrt(Kt).T

    # create measurements
    meas_idx = param["spaceLocsMeasIdx"]
    num_space_locs_meas = len(meas_idx)
    Y = F[meas_idx, :] + param["noiseStd"] * rng.standard_normal((num_space_locs_meas, num_time_inst))

    # delete (randomly) some measurements and build the noise variance matrix
    noise_var = param["noiseStd"] ** 2 * np.ones((num_space_locs_meas, num_time_inst))
    for t in range(num_time_inst):
        num_missing = rng.integers(1, num_space_locs_meas + 1)
        idx = np.sort(rng.choice(num_space_locs_meas, size=num_missing, replace=False))
        Y[idx, t] = np.nan
        noise_var[idx, t] = np.inf

    return F, Y, noise_var
