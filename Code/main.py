"""Multi-output spatio-temporal GP regression via Kalman filtering: one simulation run.

Multi-output version of main.m from Todescato et al.'s MATLAB code: generate
one synthetic data set, run the batch GP and the GPKF (estimation at measured
locations, prediction at unmeasured ones) and plot the results. Scalar outputs
are the case D = 1.

Usage:
    python main.py [--model {independent,icm,lmc,lmc_time}] [--outputs D] [--components Q]
                   [--missing-prob P] [--seed SEED] [--no-batch] [--no-plots]
"""

import argparse
import time

import numpy as np

from functions import (evaluate, generate_multioutput_data, gpkf_estimation, gpkf_prediction,
                       marginal_variances, nonparametric_estimation, nonparametric_prediction)
from load_parameters import load_parameters


def run_estimators(params, Y, noise_var, batch=True):
    """Run the GPKF (and optionally the batch GP) and return the results dict.

    Means have shape (numLocs, D, T) and covariances (T, numLocs * D, numLocs * D).
    "wallTime*" entries are the total wall-clock time of each step, in seconds.
    """
    data = params["data"]
    results = {}

    if batch:
        # nonparametric posterior and predicted GP
        start = time.perf_counter()
        results["postMeanNp"], results["postCovNp"], results["exeTimeNp"] = \
            nonparametric_estimation(data, params["np"], Y, noise_var)
        results["wallTimeNp"] = time.perf_counter() - start
        start = time.perf_counter()
        results["predMeanNp"], results["predCovNp"], results["exeTimeNpPred"] = \
            nonparametric_prediction(data, params["np"], Y, noise_var)
        results["wallTimeNpPred"] = time.perf_counter() - start

    # GPKF (gaussian process kalman filter) estimate and prediction
    start = time.perf_counter()
    gpkf_est = gpkf_estimation(data, params["gpkf"], Y, noise_var)
    results["wallTimeKf"] = time.perf_counter() - start
    results["postMeanKf"], results["postCovKf"], results["exeTimeKf"], results["negLogMarginalKf"], _ = gpkf_est
    start = time.perf_counter()
    results["predMeanKf"], results["predCovKf"], results["exeTimeKfPred"] = \
        gpkf_prediction(data, params["gpkf"], Y, noise_var, estimate=gpkf_est)
    results["wallTimeKfPred"] = time.perf_counter() - start
    return results


def evaluate_results(params, F, results):
    """Return metric rows (see `evaluate`) for every method and task in `results`."""
    data = params["data"]
    num_outputs = F.shape[1]
    tasks = {"estimation": ("post", data["spaceLocsMeasIdx"], ""), "prediction": ("pred", data["spaceLocsPredIdx"], "Pred")}
    rows = []
    for method, suffix in (("batch", "Np"), ("gpkf", "Kf")):
        for task, (prefix, idx, time_suffix) in tasks.items():
            if f"{prefix}Mean{suffix}" not in results:
                continue
            var = marginal_variances(results[f"{prefix}Cov{suffix}"], len(idx), num_outputs)
            for row in evaluate(F[idx], results[f"{prefix}Mean{suffix}"], var):
                rows.append({"method": method, "task": task, **row,
                             "wall_time": results[f"wallTime{suffix}{time_suffix}"]})
    return rows


def run(model="icm", num_outputs=3, num_components=2, missing_prob=0.5, seed=None, batch=True):
    """Generate one data set, run the estimators and return params, data and results."""
    rng = np.random.default_rng(seed)

    # load necessary parameters
    params = load_parameters(model, num_outputs, num_components, rng=rng, missing_prob=missing_prob)

    # generate data
    F, Y, noise_var = generate_multioutput_data(params["data"], rng, missing_prob=params["data"]["missingProb"])

    return params, F, Y, noise_var, run_estimators(params, Y, noise_var, batch=batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["independent", "icm", "lmc", "lmc_time"], default="icm")
    parser.add_argument("--outputs", type=int, default=3, help="number of outputs D")
    parser.add_argument("--components", type=int, default=2, help="number of LMC terms Q")
    parser.add_argument("--missing-prob", type=float, default=0.5, help="probability a measurement vector is missing")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--no-batch", action="store_true", help="skip the batch GP (slow for large problems)")
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    args = parser.parse_args()

    params, F, Y, noise_var, results = run(args.model, args.outputs, args.components, args.missing_prob,
                                           args.seed, batch=not args.no_batch)

    data = params["data"]
    print(f"Model: {data['model']}, D = {data['numOutputs']}, "
          f"{len(data['spaceLocsMeasIdx'])} measured / {len(data['spaceLocsPredIdx'])} unmeasured locations, "
          f"T = {len(data['timeInstants'])}, missing fraction = {np.isnan(Y[:, 0, :]).mean():.2f}")
    print(f"  GPKF negative log marginal lik.: {results['negLogMarginalKf']:.4f}")
    print(f"  {'method':<6} {'task':<11} {'RMSE':>8} {'cover95':>8} {'NLPD':>8} {'time [s]':>9}")
    for row in evaluate_results(params, F, results):
        if row["output"] == "all":
            print(f"  {row['method']:<6} {row['task']:<11} {row['rmse']:8.4f} {row['coverage']:8.3f} "
                  f"{row['nlpd']:8.4f} {row['wall_time']:9.3f}")

    # plotting some results
    if not args.no_plots:
        import matplotlib.pyplot as plt

        from plot_results import plot_results
        plot_results(params, F, results, rng=np.random.default_rng(args.seed))
        plt.show()


if __name__ == "__main__":
    main()
