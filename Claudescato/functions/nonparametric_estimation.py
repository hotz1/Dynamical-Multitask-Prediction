"""Port of nonparametricEstimation.m."""

import numpy as np

from ._space_time import space_locs_2d, space_time_grid, space_time_inputs
from .kernel_function import kernel_function
from .np_est import np_est


def nonparametric_estimation(param_data, param_np, meas, noise_var):
    """Nonparametric (batch GP) estimate at the measured locations.

    This is a smoother: every estimate is conditioned on all time instants.

    Parameters
    ----------
    param_data : dict
        Data parameters ("spaceLocsMeas", "timeInstants").
    param_np : dict
        Nonparametric parameters, with a "kernel" dict holding "space" and
        "time" kernel parameters.
    meas : numpy.ndarray, shape (numMeasLocs, numTimeInstants)
        Measurements, NaN where missing.
    noise_var : numpy.ndarray, same shape as meas

    Returns
    -------
    posteriorMean : numpy.ndarray, shape (numMeasLocs, numTimeInstants)
    posteriorCov : numpy.ndarray, shape (numTimeInstants, numMeasLocs, numMeasLocs)
        Spatial posterior covariance at each time instant.
    exeTime : float
        Total execution time in seconds.
    """
    space_locs = space_locs_2d(param_data["spaceLocsMeas"])
    time_instants = param_data["timeInstants"]
    kernel = kernel_function({"type": "separable", **param_np["kernel"]}, space_dim=space_locs.shape[1])

    train_inputs = space_time_grid(space_locs, time_instants)
    test_blocks = [space_time_inputs(space_locs, t) for t in time_instants]
    # column-major vectorization, as meas(:) in MATLAB
    means, covs, exe_time = np_est(kernel, train_inputs, meas.ravel(order="F"),
                                   noise_var.ravel(order="F"), test_blocks)
    return np.column_stack(means), np.stack(covs), exe_time
