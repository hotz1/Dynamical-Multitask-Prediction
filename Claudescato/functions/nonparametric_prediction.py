"""Port of nonparametricPrediction.m."""

import numpy as np

from ._space_time import space_locs_2d, space_time_grid, space_time_inputs
from .kernel_function import kernel_function
from .np_pred import np_pred


def nonparametric_prediction(param_data, param_np, meas, noise_var):
    """Nonparametric (batch GP) prediction at the unmeasured locations.

    Arguments are as in `nonparametric_estimation`. "spaceLocsPred" is also
    required in `param_data`.

    Returns
    -------
    predictedMean : numpy.ndarray, shape (numPredLocs, numTimeInstants)
    predictedCov : numpy.ndarray, shape (numTimeInstants, numPredLocs, numPredLocs)
    exeTime : float
        Total execution time in seconds.
    """
    space_locs_meas = space_locs_2d(param_data["spaceLocsMeas"])
    space_locs_pred = space_locs_2d(param_data["spaceLocsPred"])
    time_instants = param_data["timeInstants"]
    kernel = kernel_function({"type": "separable", **param_np["kernel"]}, space_dim=space_locs_meas.shape[1])

    train_inputs = space_time_grid(space_locs_meas, time_instants)
    test_blocks = [space_time_inputs(space_locs_pred, t) for t in time_instants]
    means, covs, exe_time = np_pred(kernel, train_inputs, meas.ravel(order="F"),
                                    noise_var.ravel(order="F"), test_blocks)
    return np.column_stack(means), np.stack(covs), exe_time
