"""Port of npPred.m (classical nonparametric GP prediction), using GPyTorch."""

import time

from .np_est import gp_posterior_blocks


def np_pred(kernel, train_inputs, meas, noise_var, test_input_blocks):
    """Nonparametric prediction at unmeasured inputs.

    Identical to `np_est` except that the test blocks are the unmeasured
    (prediction) locations at each time instant.

    Returns
    -------
    predictedMean : list of numpy.ndarray
    predictedCov : list of numpy.ndarray
    exeTime : float
        Total execution time in seconds.
    """
    start = time.perf_counter()
    means, covs = gp_posterior_blocks(kernel, train_inputs, meas, noise_var, test_input_blocks)
    return means, covs, time.perf_counter() - start
