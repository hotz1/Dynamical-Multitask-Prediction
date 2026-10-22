"""Port of gpkfPrediction.m."""

import time

import numpy as np
import scipy.linalg

from .gpkf_estimation import gpkf_estimation
from .kernel_function import kernel_function
from .kernel_sampled import kernel_sampled


def gpkf_prediction(param_data, param_gpkf, meas, noise_var, estimate=None):
    """GPKF prediction at the unmeasured locations.

    The GPKF estimate at the measured locations is propagated to the
    prediction locations with the space kernel.

    Parameters
    ----------
    param_data, param_gpkf, meas, noise_var :
        As in `gpkf_estimation`. "spaceLocsPred" and "timeInstants" are also
        required in `param_data`.
    estimate : tuple, optional
        Output of `gpkf_estimation` for the same inputs. MATLAB always reruns
        the estimation here; passing it avoids repeating that work.

    Returns
    -------
    predictedMean : numpy.ndarray, shape (numPredLocs, numTimeInstants)
    predictedCov : numpy.ndarray, shape (numTimeInstants, numPredLocs, numPredLocs)
    exeTime : float
        Average execution time per time instant, in seconds.
    """
    # first run the GPKF estimation
    if estimate is None:
        estimate = gpkf_estimation(param_data, param_gpkf, meas, noise_var)
    post_mean, post_cov = estimate[0], estimate[1]

    kernel_space = kernel_function(param_gpkf["kernel"]["space"])
    kernel_section = kernel_sampled(param_data["spaceLocsPred"], param_data["spaceLocsMeas"], kernel_space)
    kernel_prediction = kernel_sampled(param_data["spaceLocsPred"], param_data["spaceLocsPred"], kernel_space)
    Ks = kernel_sampled(param_data["spaceLocsMeas"], param_data["spaceLocsMeas"], kernel_space)
    Ks_inv = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Ks), np.eye(Ks.shape[0]))

    num_space_locs_pred = kernel_prediction.shape[0]
    num_time_insts = len(param_data["timeInstants"])
    predicted_cov = np.zeros((num_time_insts, num_space_locs_pred, num_space_locs_pred))
    scale = param_gpkf["kernel"]["time"]["scale"]

    t_start = time.perf_counter()
    predicted_mean = kernel_section @ (Ks_inv @ post_mean)
    for t in range(num_time_insts):
        W = Ks_inv @ (Ks - post_cov[t] / scale) @ Ks_inv
        predicted_cov[t] = scale * (kernel_prediction - kernel_section @ W @ kernel_section.T)
    exe_time = (time.perf_counter() - t_start) / num_time_insts

    return predicted_mean, predicted_cov, exe_time
