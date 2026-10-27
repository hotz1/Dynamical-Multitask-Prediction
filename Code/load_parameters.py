"""Simulation parameters for the multi-output extension (generalizes loadParameters.m).

Builds nested dicts that mirror MATLAB's `Params` struct (Params.data,
Params.np, Params.gpkf) for synthetic multi-output data. Scalar outputs are the
case D = 1. Location indices are 0-based.

The estimators use the kernel that generated the data (oracle setting), but
"np" and "gpkf" hold their own copies so they can be changed independently.
"""

import copy

import numpy as np

from functions.generate_multioutput_data import example_multioutput_kernel
from functions.time_state_space import time_kernel_coefficients


def load_parameters(model="icm", num_outputs=3, num_components=2, kernel=None, rng=None,
                    num_locs=100, meas_fraction=0.8, sampling_time=0.5, start_time=0.0, end_time=10.0,
                    noise_std=0.5, missing_prob=0.5):
    """Return the simulation parameters.

    Parameters
    ----------
    model : {"independent", "icm", "lmc", "lmc_time"}, optional
        Model used to build an example kernel when `kernel` is not given.
    num_outputs : int, optional
        Number of outputs D (1 for scalar outputs).
    num_components : int, optional
        Number of terms Q for the LMC models.
    kernel : dict, optional
        Multi-output kernel from `multioutput_kernel`. Overrides `model`,
        `num_outputs` and `num_components`.
    rng : numpy.random.Generator, optional
        Random generator for the example kernel and the measured locations.
    num_locs, meas_fraction : int, float, optional
        Number of locations on the grid 1, ..., num_locs, and the fraction of
        them that is measured (the rest are prediction locations).
    sampling_time, start_time, end_time : float, optional
        Time grid.
    noise_std : float or array_like, shape (D,), optional
        Measurement noise standard deviation(s).
    missing_prob : float, optional
        Probability that a measurement vector is unrecorded.

    Returns
    -------
    dict with keys "data", "np" and "gpkf".
    """
    rng = np.random.default_rng() if rng is None else rng
    if kernel is None:
        kernel = example_multioutput_kernel(model, num_outputs=num_outputs, num_components=num_components, rng=rng)

    # every time kernel must have a state-space form for the GPKF
    for comp in kernel["components"]:
        time_kernel_coefficients(comp["time_par"])

    data = {
        "type": "synthetic",
        "model": kernel["model"],
        "numOutputs": np.shape(kernel["components"][0]["B"])[0],
        "numLocs": num_locs,
        "spaceLocsIdx": np.arange(num_locs),
        "spaceLocs": np.arange(1, num_locs + 1, dtype=np.float64)[:, None],
        "samplingTime": sampling_time,
        "startTime": start_time,
        "endTime": end_time,
        "noiseStd": noise_std,
        "missingProb": missing_prob,
        "kernel": kernel,
    }

    # measured and prediction locations
    num_meas = round(meas_fraction * num_locs)
    data["spaceLocsMeasIdx"] = np.sort(rng.choice(data["spaceLocsIdx"], size=num_meas, replace=False))
    data["spaceLocsMeas"] = data["spaceLocs"][data["spaceLocsMeasIdx"], :]
    data["spaceLocsPredIdx"] = np.setdiff1d(data["spaceLocsIdx"], data["spaceLocsMeasIdx"])
    data["spaceLocsPred"] = data["spaceLocs"][data["spaceLocsPredIdx"], :]
    num_times = int(round((end_time - start_time) / sampling_time)) + 1
    data["timeInstants"] = start_time + sampling_time * np.arange(num_times)

    return {"data": data, "np": {"kernel": copy.deepcopy(kernel)}, "gpkf": {"kernel": copy.deepcopy(kernel)}}
