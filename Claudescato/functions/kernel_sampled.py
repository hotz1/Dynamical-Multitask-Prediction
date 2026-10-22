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


def kernel_sampled(input_set_1, input_set_2, kernel_func):
    """Return the (N1, N2) matrix K[i, j] = k(input_set_1[i], input_set_2[j]).

    Parameters
    ----------
    input_set_1, input_set_2 : array_like, shape (N1, d) and (N2, d)
        Input locations, one per row. 1-D arrays are treated as (N, 1).
    kernel_func : gpytorch.kernels.Kernel
        Kernel returned by `kernel_function`.

    Returns
    -------
    numpy.ndarray, shape (N1, N2)
    """
    x1 = _as_2d_tensor(input_set_1)
    x2 = _as_2d_tensor(input_set_2)
    with torch.no_grad():
        return kernel_func(x1, x2).to_dense().numpy()
