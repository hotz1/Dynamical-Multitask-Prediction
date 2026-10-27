"""Port of kernelSampled.m.

Evaluates a kernel on every pair of rows from two input sets. GPyTorch computes
the full Gram matrix at once, replacing MATLAB's double loop.
"""

import numpy as np
import torch

from .kernel_function import DTYPE


def _as_2d_tensor(x):
    """Convert an input set to an (N, d) float64 tensor."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return torch.as_tensor(x, dtype=DTYPE)


def multioutput_inputs(space_locs, num_outputs, time_instants):
    """Return the inputs [t, s, d] for every (location, output, time) triple.

    Location varies fastest, then output, then time, matching the
    column-major vectorization F.ravel(order="F") of an (N, D, T) array, so
    that kernel_sampled on these inputs gives sum_q kron(Gamma_q, B_q, K_q).

    Returns
    -------
    numpy.ndarray, shape (T * D * N, 1 + d + 1)
    """
    space_locs = np.asarray(space_locs, dtype=np.float64)
    space_locs = space_locs[:, None] if space_locs.ndim == 1 else space_locs
    num_locs = space_locs.shape[0]
    rows = [
        np.hstack([np.full((num_locs, 1), float(t)), space_locs, np.full((num_locs, 1), float(d))])
        for t in np.asarray(time_instants, dtype=np.float64)
        for d in range(num_outputs)
    ]
    return np.vstack(rows)


def kernel_sampled(input_set_1, input_set_2, kernel_func):
    """Return the (N1, N2) matrix K[i, j] = k(input_set_1[i], input_set_2[j]).

    Parameters
    ----------
    input_set_1, input_set_2 : array_like, shape (N1, d) and (N2, d)
        Input locations, one per row. 1-D arrays are treated as (N, 1).
        For a multi-output kernel, build them with `multioutput_inputs`.
    kernel_func : gpytorch.kernels.Kernel
        Kernel returned by `kernel_function` or `multioutput_kernel_function`.

    Returns
    -------
    numpy.ndarray, shape (N1, N2)
    """
    x1 = _as_2d_tensor(input_set_1)
    x2 = _as_2d_tensor(input_set_2)
    with torch.no_grad():
        return kernel_func(x1, x2).to_dense().numpy()
