"""Port of kernelFunction.m.

Builds the requested kernel as a GPyTorch kernel module (in float64) instead of
a MATLAB anonymous function. The kernels are parameterized exactly as in the
MATLAB code, with r = ||x1 - x2||:

    exponential : scale * exp(-r / std)                      (Matern-1/2)
    gaussian    : scale * exp(-r^2 / (2 std^2))              (RBF)
    periodic    : scale * cos(2 pi f r) * exp(-r / std)      (cosine x Matern-1/2)
    separable   : k_time(x1[0], x2[0]) * k_space(x1[1:], x2[1:])
"""

import gpytorch
import torch

DTYPE = torch.float64


def _f64(value):
    """Return `value` as a float64 tensor.

    GPyTorch's hyperparameter setters turn Python floats into float32 tensors,
    which would round the values, so they are given float64 tensors instead.
    """
    return torch.as_tensor(value, dtype=DTYPE)


def _scaled(base_kernel, scale, active_dims=None):
    """Wrap `base_kernel` in a ScaleKernel with a fixed outputscale.

    Only this outer kernel receives `active_dims`. GPyTorch slices the inputs
    once here, so the inner kernels must act on all of the (sliced) columns.
    """
    kernel = gpytorch.kernels.ScaleKernel(base_kernel, active_dims=active_dims).to(DTYPE)
    kernel.outputscale = _f64(scale)
    return kernel


def kernel_function(params, active_dims=None, space_dim=1):
    """Return a GPyTorch kernel described by the MATLAB-style parameter dict.

    Parameters
    ----------
    params : dict
        Kernel parameters. Must contain "type" and the parameters specific to
        that type ("scale", "std", and "frequency" for the periodic kernel).
        For "separable", must contain "space" and "time" sub-dicts instead.
    active_dims : sequence of int, optional
        Input columns the kernel acts on (used internally for "separable").
    space_dim : int, optional
        Number of spatial coordinates. Only used for "separable".

    Returns
    -------
    gpytorch.kernels.Kernel
        Kernel module in float64 with all hyperparameters frozen.
    """
    kernel_type = params["type"]

    if kernel_type == "separable":
        # Inputs are [time, space_1, ..., space_d]. The time kernel acts on the
        # first column and the space kernel on the remaining columns.
        kt = kernel_function(params["time"], active_dims=[0])
        ks = kernel_function(params["space"], active_dims=list(range(1, 1 + space_dim)))
        kernel = kt * ks

    elif kernel_type == "exponential":
        base = gpytorch.kernels.MaternKernel(nu=0.5).to(DTYPE)
        base.lengthscale = _f64(params["std"])
        kernel = _scaled(base, params["scale"], active_dims)

    elif kernel_type == "gaussian":
        base = gpytorch.kernels.RBFKernel().to(DTYPE)
        base.lengthscale = _f64(params["std"])
        kernel = _scaled(base, params["scale"], active_dims)

    elif kernel_type == "periodic":
        # GPyTorch's CosineKernel is cos(pi r / p), so cos(2 pi f r) needs p = 1 / (2 f).
        cosine = gpytorch.kernels.CosineKernel().to(DTYPE)
        cosine.period_length = _f64(1.0 / (2.0 * params["frequency"]))
        matern = gpytorch.kernels.MaternKernel(nu=0.5).to(DTYPE)
        matern.lengthscale = _f64(params["std"])
        kernel = _scaled(gpytorch.kernels.ProductKernel(cosine, matern), params["scale"], active_dims)

    else:
        raise ValueError(f"kernel_function: unknown type of kernel '{kernel_type}'")

    kernel = kernel.to(DTYPE)
    for p in kernel.parameters():
        p.requires_grad_(False)
    return kernel

