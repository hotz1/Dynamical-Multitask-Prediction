"""Port of loadParameters.m.

Builds the simulation parameters as nested dicts that mirror MATLAB's
`Params` struct (Params.data, Params.np, Params.gpkf).

Differences from MATLAB:
  * location indices are 0-based;
  * "timeInstants" for the Colorado data are still the 1-based month indices
    used by MATLAB, since they are also the kernel's time inputs.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from functions.load_colorado_data import read_colorado_csv

# Code/data holds the CSV versions of Todescato et al.'s data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLORADO_PATH = DATA_DIR / "datasets" / "colorado.csv"
GAUSSIAN_APPROX_DIR = DATA_DIR / "gaussian_time_kernel_approximations"


def _load_gaussian_time_approximation(ss_dim):
    """Return (num, den) of the precomputed Gaussian time-kernel approximation."""
    path = GAUSSIAN_APPROX_DIR / f"ssDim={ss_dim}_for_scale=1_std=1.csv"
    coeffs = pd.read_csv(path, index_col="coefficient")
    return coeffs.loc["num"].to_numpy(dtype=np.float64), coeffs.loc["den"].to_numpy(dtype=np.float64)


def load_parameters(data_type="synthetic", rng=None, gaussian_ss_dim=6):
    """Return the simulation parameters.

    Parameters
    ----------
    data_type : {"synthetic", "colorado"}
        Data set to use.
    rng : numpy.random.Generator, optional
        Random generator for the choice of measured locations.
    gaussian_ss_dim : int, optional
        State dimension (1 to 6) of the Gaussian time-kernel approximation.

    Returns
    -------
    dict with keys "data", "np" and "gpkf".
    """
    rng = np.random.default_rng() if rng is None else rng
    params = {"data": {"type": data_type}, "np": {}, "gpkf": {}}
    data = params["data"]

    if data_type == "synthetic":
        # DATA parameters
        data["numLocs"] = 100
        data["spaceLocsIdx"] = np.arange(data["numLocs"])
        data["spaceLocs"] = np.arange(1, data["numLocs"] + 1, dtype=np.float64)[:, None]
        data["samplingTime"] = 0.5
        data["startTime"] = 0.0
        data["endTime"] = 10.0
        data["noiseStd"] = 0.5
        data["kernel"] = {
            "space": {"type": "gaussian", "scale": 1.0, "std": 1.0},
            # time type: "exponential", "gaussian" or "periodic".
            # NOTE: to use the gaussian kernel with GPKF, scale and std must be 1.
            "time": {"type": "exponential", "scale": 1.0, "std": 1.0, "frequency": 1.0},
        }
        # NONPARAMETRIC KERNEL parameters
        params["np"]["kernel"] = copy.deepcopy(data["kernel"])
        # GPKF parameters
        params["gpkf"]["kernel"] = copy.deepcopy(data["kernel"])

    elif data_type == "colorado":
        data["path"] = COLORADO_PATH
        locations, _ = read_colorado_csv(COLORADO_PATH)
        data["numLocs"] = locations.shape[0]
        data["spaceLocsIdx"] = np.arange(data["numLocs"])
        data["spaceLocs"] = locations
        data["samplingTime"] = 1.0
        data["startYear"] = 102   # in [1, 103]
        data["endYear"] = 103     # in [1, 103]
        data["startTime"] = (data["startYear"] - 1) * 12 + 1
        data["endTime"] = data["endYear"] * 12
        data["noiseStd"] = 0.05

        time_type = "periodic"    # or "exponential" (gaussian is not available here)
        if time_type == "exponential":
            time_kernel = {"type": "exponential", "scale": 1100.0, "std": 1 / 1e-2}
        else:
            time_kernel = {"type": "periodic", "scale": 1338.7, "std": 1 / 1.1122, "frequency": 1 / 12}
        params["np"]["kernel"] = {
            "space": {"type": "exponential", "scale": 1.0, "std": 1 / 0.5853},
            "time": time_kernel,
        }
        params["gpkf"]["kernel"] = copy.deepcopy(params["np"]["kernel"])

    else:
        raise ValueError(f"load_parameters: unknown data type '{data_type}'")

    # compute additional (common) parameters
    num_meas = round(0.8 * data["numLocs"])
    data["spaceLocsMeasIdx"] = np.sort(rng.choice(data["spaceLocsIdx"], size=num_meas, replace=False))
    data["spaceLocsMeas"] = data["spaceLocs"][data["spaceLocsMeasIdx"], :]
    data["spaceLocsPredIdx"] = np.setdiff1d(data["spaceLocsIdx"], data["spaceLocsMeasIdx"])
    data["spaceLocsPred"] = data["spaceLocs"][data["spaceLocsPredIdx"], :]
    num_times = int(round((data["endTime"] - data["startTime"]) / data["samplingTime"])) + 1
    data["timeInstants"] = data["startTime"] + data["samplingTime"] * np.arange(num_times)

    # state space realization for the gpkf time kernel
    tk = params["gpkf"]["kernel"]["time"]
    if tk["type"] == "exponential":
        tk["num"] = np.array([np.sqrt(2 * tk["scale"] / tk["std"])])
        tk["den"] = np.array([1 / tk["std"]])
    elif tk["type"] == "gaussian":
        tk["ssDim"] = gaussian_ss_dim
        tk["num"], tk["den"] = _load_gaussian_time_approximation(gaussian_ss_dim)
    elif tk["type"] == "periodic":
        omega_sq = (1 / tk["std"]) ** 2 + (2 * np.pi * tk["frequency"]) ** 2
        tk["num"] = np.sqrt(2 * tk["scale"] / tk["std"]) * np.array([np.sqrt(omega_sq), 1.0])
        tk["den"] = np.array([omega_sq, 2 / tk["std"]])
    else:
        raise ValueError("load_parameters: not admissible kernel type")

    return params
