"""Simulation study for the multi-output extension.

For every setting in GRID, draws a kernel (and the measured locations) once,
then generates `--reps` independent data sets from it and runs the batch GP and
the GPKF on each, with the true kernel (oracle setting). Metrics (see
functions.metrics.evaluate) are written to a tidy CSV with one row per
(setting, replication, method, task, output), where output "all" pools the
outputs. The CSV is appended after every replication, so an interrupted run
keeps its results; rerunning overwrites it.

Usage:
    python simulation_study.py [--reps R] [--locs N] [--end-time E] [--time-step TS] [--seed SEED] [--out PATH] [--no-batch]
                               [--redraw-kernel] [--no-plots] [--plot-only]
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from functions import generate_multioutput_data
from load_parameters import load_parameters
from main import evaluate_results, run_estimators

# settings of the study: every combination is run. "model" and "num_outputs" are required.
# all other keys are passed to load_parameters (e.g. "missing_frac", "meas_fraction") 
# anything set from the command-line (e.g. --time-step, --end-time) is constant across the study
GRID = {
    "model": ["independent", "icm", "lmc", "lmc_time"],
    "num_outputs": [1, 2, 3, 5],
    "missing_frac": [0, 0.2, 0.5]
}
NUM_COMPONENTS = 3          # Q for the LMC models
LOAD_PARAMETERS_KWARGS = {}  # load_parameters arguments fixed across the study, e.g. {"noise_std": 0.3}
PLOT_AXES = ("model", "num_outputs")   # x-axis and panel columns of the summary figures

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METHOD_STYLE = {"batch": ("batch GP (smoothing)", "#2a78d6", "o"), "gpkf": ("GPKF (filtering)", "#eb6834", "s")}


def run_study(reps, seed, out_path, num_locs=100, end_time=10.0, time_step=0.5, batch=True, redraw_kernel=False):
    """Run every setting of GRID `reps` times and write the metrics to `out_path`.

    Every setting uses `num_locs` locations and time instants 0, `time_step`, ..., `end_time`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    settings = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    for setting_id, setting in enumerate(settings):
        for rep in range(reps):
            # one stream for the kernel and locations, one per replication for the data
            if rep == 0 or redraw_kernel:
                param_rng = np.random.default_rng([seed, setting_id, rep if redraw_kernel else 0, 0])
                extra = {k: v for k, v in setting.items() if k not in PLOT_AXES}
                params = load_parameters(setting["model"], setting["num_outputs"], NUM_COMPONENTS, rng=param_rng,
                                         **{**LOAD_PARAMETERS_KWARGS, "num_locs": num_locs, "end_time": end_time,
                                            "sampling_time": time_step, **extra})
            data_rng = np.random.default_rng([seed, setting_id, rep, 1])
            F, Y, noise_var = generate_multioutput_data(params["data"], data_rng,
                                                        missing_frac=params["data"]["missingFrac"])

            results = run_estimators(params, Y, noise_var, batch=batch)
            rows = pd.DataFrame(evaluate_results(params, F, results))
            meta = {"setting_id": setting_id, "num_locs": num_locs, "end_time": end_time, "time_step": time_step,
                    **setting,
                    "num_components": NUM_COMPONENTS, "rep": rep}
            rows = pd.concat([pd.DataFrame([meta] * len(rows)), rows], axis=1)
            rows.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

            pooled = rows[rows["output"] == "all"].set_index(["method", "task"])["rmse"]
            print(f"[{setting_id + 1}/{len(settings)}] {setting} rep {rep + 1}/{reps}: "
                  + ", ".join(f"{m}/{t} RMSE {v:.3f}" for (m, t), v in pooled.items()))
    return out_path


def plot_summary(csv_path, out_dir=None):
    """Plot the mean (+/- 1.96 standard errors) of RMSE, coverage and wall time per setting.

    One figure per metric and per combination of the other GRID factors (e.g.
    missing fraction), with rows for the task
    (estimation / prediction), columns for the number of outputs D and the
    model on the x-axis. Figures are saved next to the CSV and returned.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df = df[df["output"].astype(str) == "all"]
    out_dir = Path(csv_path).parent if out_dir is None else Path(out_dir)
    models = [m for m in GRID["model"] if m in set(df["model"])]
    tasks = ["estimation", "prediction"]
    metrics = {"rmse": "RMSE", 
               "mae": "MAE",
               "coverage": "95% coverage", 
               "wall_time": "wall time [s]"}
    others = [k for k in GRID if k not in PLOT_AXES and k in df.columns]
    grid_label = f"{df['num_locs'].iloc[0]} locations, t = 0:{df['time_step'].iloc[0]:g}:{df['end_time'].iloc[0]:g}"

    stats = (df.groupby(others + ["num_outputs", "task", "method", "model"])[list(metrics)]
             .agg(["mean", "sem"]))
    figs = []
    for values, sub in (df.groupby(others) if others else [((), df)]):
        values = values if isinstance(values, tuple) else (values,)
        dims = sorted(sub["num_outputs"].unique())
        title = ", ".join(f"{k} = {v}" for k, v in zip(others, values))
        suffix = "".join(f"_{k}={v}" for k, v in zip(others, values))
        for metric, label in metrics.items():
            fig, axes = plt.subplots(len(tasks), len(dims), figsize=(3.2 * len(dims) + 1, 5.5),
                                     sharex=True, sharey="row", layout="constrained", squeeze=False)
            for (i, task), (j, D) in itertools.product(enumerate(tasks), enumerate(dims)):
                ax = axes[i, j]
                for k, (method, (name, color, marker)) in enumerate(METHOD_STYLE.items()):
                    key = (*values, D, task, method)
                    if key not in stats.index.droplevel("model"):
                        continue
                    s = stats.loc[key].reindex(models)
                    x = np.arange(len(models)) + (k - 0.5) * 0.2
                    ax.errorbar(x, s[(metric, "mean")], yerr=1.96 * s[(metric, "sem")].fillna(0), color=color,
                                marker=marker, markersize=6, linestyle="none", capsize=3, linewidth=1.5, label=name)
                if metric == "coverage":
                    ax.axhline(0.95, color="#888888", linewidth=1, linestyle="--", label="nominal 0.95")
                if metric == "wall_time":
                    ax.set_yscale("log")
                ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.set_xticks(np.arange(len(models)), models, rotation=30)
                if i == 0:
                    ax.set_title(f"D = {D}")
                if j == 0:
                    ax.set_ylabel(f"{task}\n{label}")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
            fig.suptitle(", ".join([label, title, grid_label] if title else [label, grid_label]))
            fig.savefig(out_dir / f"{metric}{suffix}.png", dpi=120)
            figs.append(fig)
    return figs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=10, help="replications per setting")
    parser.add_argument("--locs", type=int, default=100, help="number of locations (80%% are measured)")
    parser.add_argument("--end-time", "--end_time", type=float, default=10.0, help="last time instant")
    parser.add_argument("--time-step", "--time_step", type=float, default=0.5,
                        help="sampling time Ts (time instants 0, Ts, ..., end time)")
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument("--out", default=RESULTS_DIR / "simulation_results.csv", help="output CSV path")
    parser.add_argument("--no-batch", action="store_true", help="skip the batch GP (slow for large problems)")
    parser.add_argument("--redraw-kernel", action="store_true",
                        help="draw a new kernel and measured locations for every replication")
    parser.add_argument("--no-plots", action="store_true", help="skip the summary figures")
    parser.add_argument("--plot-only", action="store_true", help="only plot an existing CSV")
    args = parser.parse_args()

    if not args.plot_only:
        run_study(args.reps, args.seed, args.out, args.locs, args.end_time, args.time_step, batch=not args.no_batch,
                  redraw_kernel=args.redraw_kernel)
    if not args.no_plots:
        plot_summary(args.out)
        print(f"Summary figures saved in {Path(args.out).parent}")


if __name__ == "__main__":
    main()
