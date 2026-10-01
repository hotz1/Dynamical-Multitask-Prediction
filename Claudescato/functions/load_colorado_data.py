import numpy as np
import pandas as pd

# columns preceding the monthly measurements in colorado.csv
_META_COLUMNS = ["Longitude", "Latitude", "ID"]


def read_colorado_csv(path):
    """Return (station locations, monthly measurements) from colorado.csv.

    Locations have shape (numStations, 2) as (longitude, latitude).
    Measurements have shape (numStations, numMonths).
    """
    data = pd.read_csv(path, dtype={"ID": str})
    locations = data[["Longitude", "Latitude"]].to_numpy(dtype=np.float64)
    measurements = data.drop(columns=_META_COLUMNS).to_numpy(dtype=np.float64)
    return locations, measurements


def load_colorado_data(param):
    """Load the Colorado rainfall measurements for the selected time window.

    Parameters
    ----------
    param : dict
        Data parameters ("path", "timeInstants", "spaceLocsMeasIdx",
        "noiseStd"). "timeInstants" holds 1-based month indices, as in MATLAB.

    Returns
    -------
    F : numpy.ndarray, shape (numStations, numTimeInstants)
        All measurements, used as ground truth since none is available.
    Y : numpy.ndarray, shape (numMeasStations, numTimeInstants)
        Measurements at the measured stations (NaN where missing).
    noiseVar : numpy.ndarray, same shape as Y
        (noiseStd * |Y|)^2, +inf where missing and noiseStd^2 where Y == 0.
    """
    _, measurements = read_colorado_csv(param["path"])
    cols = np.asarray(param["timeInstants"], dtype=int) - 1   # 1-based -> 0-based
    F = measurements[:, cols]
    Y = F[param["spaceLocsMeasIdx"], :]
    noise_var = (param["noiseStd"] * np.abs(Y)) ** 2
    noise_var[np.isnan(noise_var)] = np.inf
    noise_var[Y == 0] = param["noiseStd"] ** 2
    return F, Y, noise_var
