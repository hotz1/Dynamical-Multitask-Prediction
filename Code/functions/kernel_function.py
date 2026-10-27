"""Port of kernelFunction.m, extended to multi-output kernels.

Builds the requested kernel as a GPyTorch kernel module (in float64) instead of
a MATLAB anonymous function. `kernel_function` builds the scalar kernels,
parameterized exactly as in the MATLAB code, with r = ||x1 - x2||:

    matern      : scale * M_nu(r / std)                      (Matern, nu in {1/2, 3/2, 5/2})
    gaussian    : scale * exp(-r^2 / (2 std^2))              (RBF)
    periodic    : scale * cos(2 pi f r) * exp(-r / std)      (cosine x Matern-1/2)

with M_nu(u) = 2^(1 - nu) / Gamma(nu) * (sqrt(2 nu) u)^nu * K_nu(sqrt(2 nu) u):

    nu = 1/2 : exp(-u)                                       (MATLAB's "exponential")
    nu = 3/2 : (1 + sqrt(3) u) exp(-sqrt(3) u)
    nu = 5/2 : (1 + sqrt(5) u + 5 u^2 / 3) exp(-sqrt(5) u)

`multioutput_kernel_function` combines them into the space-time kernel on
inputs [t, x_1, ..., x_p, d], where d in {0, ..., D-1} is the output index:

    sum_q k_time_q(t, t') * k_space_q(x, x') * B_q[d, d']

A univariate (scalar-output) process is the case D = 1 with B = [[1]], which
recovers MATLAB's separable kernel k_time(t, t') * k_space(x, x').
"""

import gpytorch
import torch

DTYPE = torch.float64


class CoregionalizationKernel(gpytorch.kernels.Kernel):
    """Fixed output covariance k(d, d') = B[d, d'] on an integer index column.

    GPyTorch's IndexKernel parameterizes B = W W' + diag(v) with v > 0, so it
    cannot hold an arbitrary (e.g. diagonal or low-rank) B exactly. This kernel
    stores B as a buffer instead.
    """

    def __init__(self, B, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("B", torch.as_tensor(B, dtype=DTYPE))

    def forward(self, x1, x2, diag=False, **params):
        i1 = x1[..., 0].long()
        i2 = x2[..., 0].long()
        if diag:
            return self.B[i1, i2]
        return self.B[i1.unsqueeze(-1), i2.unsqueeze(-2)]


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


def kernel_function(params, active_dims=None):
    """Return the scalar GPyTorch kernel described by the MATLAB-style parameter dict.

    Parameters
    ----------
    params : dict
        Kernel parameters. Must contain "type" ("matern", "gaussian" or
        "periodic") and the parameters specific to that type.
            Matern kernel needs ("scale", "std", "nu")
            Gaussian kernel needs ("scale", "std", "nu")
            Periodic kernel needs ("scale", "std", "frequency")
    active_dims : sequence of int, optional
        Input columns the kernel acts on (used by `multioutput_kernel_function`).

    Returns
    -------
    gpytorch.kernels.Kernel
        Kernel module in float64 with all hyperparameters frozen.
    """
    kernel_type = params["type"]

    if kernel_type == "matern":
        # GPyTorch only implements the half-integer cases with closed forms.
        nu = float(params["nu"])
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"kernel_function: Matern nu must be 0.5, 1.5 or 2.5, got {params['nu']}")
        base = gpytorch.kernels.MaternKernel(nu=nu).to(DTYPE)
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


def multioutput_kernel_function(params, space_dim=1):
    """Return the multi-output space-time GPyTorch kernel.

    Parameters
    ----------
    params : dict
        Multi-output kernel from `multioutput_kernel`: a dict whose
        "components" is a list of dicts with "space_par", "time_par" (scalar
        kernel parameters for `kernel_function`) and "B" (a (D, D) matrix).
        For a univariate process use D = 1 and B = [[1]].
    space_dim : int, optional
        Number of spatial coordinates p.

    Returns
    -------
    gpytorch.kernels.Kernel
        Kernel module in float64 with all hyperparameters frozen, acting on
        inputs [t, x_1, ..., x_p, d] (see `multioutput_inputs`).
    """
    # Each term is a separable space-time kernel times its output covariance B_q.
    space_dims = list(range(1, 1 + space_dim))
    output_dim = [1 + space_dim]
    terms = [
        kernel_function(comp["time_par"], active_dims=[0])
        * kernel_function(comp["space_par"], active_dims=space_dims)
        * CoregionalizationKernel(comp["B"], active_dims=output_dim)
        for comp in params["components"]
    ]
    kernel = gpytorch.kernels.AdditiveKernel(*terms).to(DTYPE)
    for p in kernel.parameters():
        p.requires_grad_(False)
    return kernel
