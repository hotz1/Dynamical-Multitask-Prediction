"""Accuracy and calibration metrics for multi-output posteriors."""

import numpy as np
from scipy.stats import norm


def marginal_variances(cov, num_locs, num_outputs):
    """Return the posterior variances of each (location, output, time).

    Parameters
    ----------
    cov : numpy.ndarray, shape (T, num_locs * D, num_locs * D)
        Per-time joint covariances, location fastest then output.

    Returns
    -------
    numpy.ndarray, shape (num_locs, D, T)
    """
    diag = np.diagonal(cov, axis1=1, axis2=2)                 # (T, num_locs * D), index n + num_locs * d
    return diag.reshape(-1, num_outputs, num_locs).transpose(2, 1, 0)


def evaluate(truth, mean, var, level=0.95):
    """Compare a posterior with the true latent field, per output and pooled.

    Parameters
    ----------
    truth, mean, var : numpy.ndarray, shape (num_locs, D, T)
        True latent values, posterior means and posterior variances.
    level : float, optional
        Nominal level of the central credible intervals for "coverage".

    Returns
    -------
    list of dict
        One dict per output d (with "output" = d) and one pooled over outputs
        (with "output" = "all"), each holding
          rmse     : root mean squared error of the posterior mean;
          coverage : fraction of true values inside the central `level` interval;
          mean_var : average posterior variance;
          nlpd     : average negative log posterior density of the true values.
    """
    var = np.clip(var, 1e-12, None)   # guards tiny negative variances from round-off
    z = norm.ppf(0.5 + level / 2)
    err = truth - mean
    inside = np.abs(err) <= z * np.sqrt(var)
    nlpd = 0.5 * (np.log(2 * np.pi * var) + err**2 / var)

    def summary(sl, label):
        return {"output": label, "rmse": np.sqrt(np.mean(err[sl] ** 2)), "coverage": inside[sl].mean(),
                "mean_var": var[sl].mean(), "nlpd": nlpd[sl].mean()}

    rows = [summary((slice(None), d), d) for d in range(truth.shape[1])]
    rows.append(summary(slice(None), "all"))
    return rows
