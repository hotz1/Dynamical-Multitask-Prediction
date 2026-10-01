"""Port of loadDataSet.m."""

from .generate_synthetic_data import generate_synthetic_data
from .load_colorado_data import load_colorado_data


def load_data_set(param, rng=None):
    """Return (F, Y, noiseVar) for the data set named by param["type"].

    F and Y hold samples of the true GP and the measurements. noiseVar holds
    the measurement variances and has the same shape as Y.
    """
    if param["type"] == "synthetic":
        return generate_synthetic_data(param, rng)
    if param["type"] == "colorado":
        return load_colorado_data(param)
    raise ValueError(f"load_data_set: data set '{param['type']}' does not exist")
