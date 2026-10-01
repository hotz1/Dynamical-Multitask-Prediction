"""Port of plotResults.m.

Produces the same four figures as MATLAB: estimation and prediction, each
shown across space (at the final time) and across time (at a random location).
Shaded bands are +/- 3 posterior standard deviations.
"""

import matplotlib.pyplot as plt
import numpy as np

TRUE_COLOR = "#222222"   # neutral ink for the ground truth
NP_COLOR = "#2a78d6"     # blue: nonparametric (batch) GP
KF_COLOR = "#eb6834"     # orange: GPKF


def _band(ax, x, mean, var, color, label):
    """Plot a +/- 3 std band (abs() guards tiny negative variances, as in MATLAB)."""
    std3 = 3 * np.sqrt(np.abs(var))
    ax.fill_between(x, mean - std3, mean + std3, color=color, alpha=0.2, linewidth=0, label=label)


def _panel(ax, x, truth, np_mean, np_var, kf_mean, kf_var, np_name, kf_name, xlabel, title):
    ax.plot(x, truth, color=TRUE_COLOR, linewidth=1.5, label="true GP")
    ax.plot(x, np_mean, color=NP_COLOR, linestyle=":", linewidth=2, label=f"np {np_name} mean (smoothing)")
    _band(ax, x, np_mean, np_var, NP_COLOR, f"np {np_name} ±3 std")
    ax.plot(x, kf_mean, color=KF_COLOR, linestyle="--", linewidth=2, label=f"gpkf {kf_name} mean")
    _band(ax, x, kf_mean, kf_var, KF_COLOR, f"gpkf {kf_name} ±3 std")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize="small", frameon=False)


def plot_results(params, F, results, rng=None):
    """Draw the four comparison figures and return them.

    Parameters
    ----------
    params : dict
        Output of `load_parameters`.
    F : numpy.ndarray, shape (numLocs, numTimeInstants)
        Ground truth.
    results : dict
        Results dict produced by `main.run`.
    rng : numpy.random.Generator, optional
        Used to pick the location shown in the time plots.
    """
    rng = np.random.default_rng() if rng is None else rng
    data = params["data"]
    meas_idx, pred_idx = data["spaceLocsMeasIdx"], data["spaceLocsPredIdx"]
    t_axis = np.arange(1, len(data["timeInstants"]) + 1)
    r = results
    figs = []

    def new_axes():
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        figs.append(fig)
        return ax

    # ESTIMATION: space (final time instant)
    x = np.arange(1, len(meas_idx) + 1)
    _panel(new_axes(), x, F[meas_idx, -1],
           r["postMeanNp"][:, -1], np.diagonal(r["postCovNp"][-1]),
           r["postMeanKf"][:, -1], np.diagonal(r["postCovKf"][-1]),
           "post", "post", "measured space locations index", "Space estimation")

    # ESTIMATION: time (random measured location)
    i = rng.integers(len(meas_idx))
    _panel(new_axes(), t_axis, F[meas_idx[i], :],
           r["postMeanNp"][i, :], r["postCovNp"][:, i, i],
           r["postMeanKf"][i, :], r["postCovKf"][:, i, i],
           "post", "post", "time instants", "Time evolution (estimation)")

    # PREDICTION: space (final time instant)
    x = np.arange(1, len(pred_idx) + 1)
    _panel(new_axes(), x, F[pred_idx, -1],
           r["predMeanNp"][:, -1], np.diagonal(r["predCovNp"][-1]),
           r["predMeanKf"][:, -1], np.diagonal(r["predCovKf"][-1]),
           "pred", "pred", "predicted space locations index", "Space prediction")

    # PREDICTION: time (random unmeasured location)
    i = rng.integers(len(pred_idx))
    _panel(new_axes(), t_axis, F[pred_idx[i], :],
           r["predMeanNp"][i, :], r["predCovNp"][:, i, i],
           r["predMeanKf"][i, :], r["predCovKf"][:, i, i],
           "pred", "pred", "time instants", "Time evolution (prediction)")

    return figs
