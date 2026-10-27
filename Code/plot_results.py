"""Multi-output version of plotResults.m.

Produces the same four figures as MATLAB (estimation and prediction, each shown
across space at the final time and across time at a random location), with one
panel per output. Shaded bands are +/- 3 posterior standard deviations.
"""

import matplotlib.pyplot as plt
import numpy as np

from functions.metrics import marginal_variances

TRUE_COLOR = "#222222"   # neutral ink for the ground truth
NP_COLOR = "#2a78d6"     # blue: batch GP (smoother)
KF_COLOR = "#eb6834"     # orange: GPKF (filter)
METHODS = (("Np", "batch GP (smoothing)", NP_COLOR, ":"), ("Kf", "GPKF (filtering)", KF_COLOR, "--"))


def _style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _figure(x, truth, curves, xlabel, title):
    """One figure with a panel per output; `curves` maps method suffix -> (mean, var)."""
    num_outputs = truth.shape[0]
    fig, axes = plt.subplots(num_outputs, 1, figsize=(9, 2.6 * num_outputs + 0.6), sharex=True,
                             layout="constrained", squeeze=False)
    for d, ax in enumerate(axes[:, 0]):
        ax.plot(x, truth[d], color=TRUE_COLOR, linewidth=1.5, label="true GP")
        for suffix, name, color, style in METHODS:
            if suffix not in curves:
                continue
            mean, var = curves[suffix][0][d], curves[suffix][1][d]
            std3 = 3 * np.sqrt(np.abs(var))   # abs() guards tiny negative variances, as in MATLAB
            ax.plot(x, mean, color=color, linestyle=style, linewidth=2, label=f"{name} mean")
            ax.fill_between(x, mean - std3, mean + std3, color=color, alpha=0.2, linewidth=0, label=f"{name} ±3 std")
        ax.set_ylabel(f"output {d + 1}")
        _style(ax)
    fig.suptitle(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize="small", frameon=False)
    axes[-1, 0].set_xlabel(xlabel)
    return fig


def plot_results(params, F, results, rng=None):
    """Draw the four comparison figures and return them.

    Parameters
    ----------
    params : dict
        Output of `load_parameters`.
    F : numpy.ndarray, shape (numLocs, D, numTimeInstants)
        Ground truth.
    results : dict
        Results dict produced by `main.run_estimators`.
    rng : numpy.random.Generator, optional
        Used to pick the location shown in the time plots.
    """
    rng = np.random.default_rng() if rng is None else rng
    data = params["data"]
    num_outputs = F.shape[1]
    t_axis = np.arange(1, len(data["timeInstants"]) + 1)
    figs = []

    for task, prefix, idx in (("estimation", "post", data["spaceLocsMeasIdx"]),
                              ("prediction", "pred", data["spaceLocsPredIdx"])):
        # posterior means and marginal variances, each (num_locs, D, T)
        curves = {suffix: (results[f"{prefix}Mean{suffix}"],
                           marginal_variances(results[f"{prefix}Cov{suffix}"], len(idx), num_outputs))
                  for suffix, *_ in METHODS if f"{prefix}Mean{suffix}" in results}
        which = "measured" if task == "estimation" else "unmeasured"

        # space (final time instant)
        x = np.arange(1, len(idx) + 1)
        figs.append(_figure(x, F[idx, :, -1].T, {s: (m[:, :, -1].T, v[:, :, -1].T) for s, (m, v) in curves.items()},
                            f"{which} space locations index", f"Space {task}"))

        # time (random location)
        i = rng.integers(len(idx))
        figs.append(_figure(t_axis, F[idx[i]], {s: (m[i], v[i]) for s, (m, v) in curves.items()},
                            "time instants", f"Time evolution ({task}, {which} location {i + 1})"))
    return figs
