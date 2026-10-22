"""Port of npEst.m (classical nonparametric GP estimation), using GPyTorch."""

import contextlib
import time

import gpytorch
import numpy as np
import torch

from .kernel_function import DTYPE


class _ExactGPModel(gpytorch.models.ExactGP):
    """Zero-mean exact GP with a fixed (non-trained) covariance kernel."""

    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _exact_settings():
    """GPyTorch settings that force dense Cholesky solves (no CG or Lanczos)."""
    stack = contextlib.ExitStack()
    stack.enter_context(torch.no_grad())
    stack.enter_context(gpytorch.settings.fast_computations(
        covar_root_decomposition=False, log_prob=False, solves=False))
    stack.enter_context(gpytorch.settings.max_cholesky_size(10**9))
    stack.enter_context(gpytorch.settings.fast_pred_var(False))
    stack.enter_context(gpytorch.settings.skip_posterior_variances(False))
    return stack


def gp_posterior_blocks(kernel, train_inputs, meas, noise_var, test_input_blocks, max_test_points=2000):
    """Condition a zero-mean GP on noisy data and return per-block posteriors.

    Parameters
    ----------
    kernel : gpytorch.kernels.Kernel
        Prior covariance of the latent GP.
    train_inputs : numpy.ndarray, shape (N, D)
        Inputs of all measurements, including missing ones.
    meas : numpy.ndarray, shape (N,)
        Measurements, NaN where missing (these are dropped).
    noise_var : numpy.ndarray, shape (N,)
        Noise variance of each measurement.
    test_input_blocks : list of numpy.ndarray, each of shape (n_b, D)
        Test input sets. The joint posterior is returned within each block,
        which avoids forming the full (N x N) posterior covariance.
    max_test_points : int, optional
        Maximum number of test inputs sent to GPyTorch in one call.

    Returns
    -------
    means : list of numpy.ndarray, shape (n_b,)
    covs : list of numpy.ndarray, shape (n_b, n_b)
        Posterior mean and covariance of the latent GP on each block.
    """
    observed = ~np.isnan(meas)
    train_x = torch.as_tensor(train_inputs[observed], dtype=DTYPE)
    train_y = torch.as_tensor(meas[observed], dtype=DTYPE)
    train_noise = torch.as_tensor(noise_var[observed], dtype=DTYPE)

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_noise, learn_additional_noise=False)
    model = _ExactGPModel(train_x, train_y, likelihood, kernel).to(DTYPE)
    model.eval()
    likelihood.eval()

    means, covs = [], []
    with _exact_settings():
        for group in _group_blocks(test_input_blocks, max_test_points):
            # calling the model (not the likelihood) gives the latent f posterior
            posterior = model(torch.as_tensor(np.vstack(group), dtype=DTYPE))
            mean = posterior.mean.numpy()
            cov = posterior.covariance_matrix.numpy()
            start = 0
            for block in group:
                stop = start + block.shape[0]
                means.append(mean[start:stop].copy())
                covs.append(cov[start:stop, start:stop].copy())
                start = stop
    return means, covs


def _group_blocks(blocks, max_points):
    """Split consecutive blocks into groups of at most `max_points` rows.

    Each GPyTorch call refactorizes the training covariance, so querying
    several blocks per call is much faster. Capping the group size bounds the
    memory used by the joint test covariance of each call.
    """
    group, size = [], 0
    for block in blocks:
        if group and size + block.shape[0] > max_points:
            yield group
            group, size = [], 0
        group.append(block)
        size += block.shape[0]
    if group:
        yield group


def np_est(kernel, train_inputs, meas, noise_var, test_input_blocks):
    """Nonparametric estimate: GP posterior conditioned on all of the data.

    See `gp_posterior_blocks` for the arguments. For estimation, the test
    blocks are the measured locations at each time instant.

    Returns
    -------
    posteriorMean : list of numpy.ndarray
    posteriorCov : list of numpy.ndarray
    exeTime : float
        Total execution time in seconds.
    """
    start = time.perf_counter()
    means, covs = gp_posterior_blocks(kernel, train_inputs, meas, noise_var, test_input_blocks)
    return means, covs, time.perf_counter() - start
