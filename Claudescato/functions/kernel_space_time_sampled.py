"""Port of kernelSpaceTimeSampled.m."""

import numpy as np

from .kernel_function import kernel_function
from .kernel_sampled import kernel_sampled


def kernel_space_time_sampled(space_locs1, space_locs2, time_instants1, time_instants2, param):
    """Return the separable space-time kernel sampled on the given grids.

    The ordering matches MATLAB's column-major vectorization of a
    (space x time) matrix: space varies fastest, so K = kron(K_time, K_space).

    Parameters
    ----------
    space_locs1, space_locs2 : array_like, shape (N1, d) and (N2, d)
    time_instants1, time_instants2 : array_like, shape (T1,) and (T2,)
    param : dict
        Dict with "space" and "time" kernel parameter sub-dicts.

    Returns
    -------
    numpy.ndarray, shape (T1 * N1, T2 * N2)
    """
    kernel_space = kernel_function(param["space"])
    kernel_time = kernel_function(param["time"])
    Ks = kernel_sampled(space_locs1, space_locs2, kernel_space)
    Kt = kernel_sampled(time_instants1, time_instants2, kernel_time)
    return np.kron(Kt, Ks)
