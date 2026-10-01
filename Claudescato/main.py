"""Efficient Spatio-Temporal Gaussian Process learning via Kalman Filtering.

Python port of main.m from Todescato et al.'s MATLAB code
(https://github.com/MarcoTodescato/Efficient-GP-Regression-via-Kalman-Filtering).

Usage:
    python main.py [--data {synthetic,colorado}] [--seed SEED] [--no-plots]
"""

import argparse

import numpy as np

from functions import (gpkf_estimation, gpkf_prediction, load_data_set,
                       nonparametric_estimation, nonparametric_prediction)
from load_parameters import load_parameters


def run(data_type="synthetic", seed=None):
    """Run all four estimators and return the parameters, data and results."""
    rng = np.random.default_rng(seed)

    # load necessary parameters
    params = load_parameters(data_type, rng)

    # generate data
    F, Y, noise_var = load_data_set(params["data"], rng)

    results = {}
    # nonparametric posterior GP
    results["postMeanNp"], results["postCovNp"], results["exeTimeNp"] = \
        nonparametric_estimation(params["data"], params["np"], Y, noise_var)

    # nonparametric predicted GP
    results["predMeanNp"], results["predCovNp"], results["exeTimeNpPred"] = \
        nonparametric_prediction(params["data"], params["np"], Y, noise_var)

    # GPKF (gaussian process kalman filter) estimate
    gpkf_est = gpkf_estimation(params["data"], params["gpkf"], Y, noise_var)
    results["postMeanKf"], results["postCovKf"], results["exeTimeKf"], results["negLogMarginalKf"] = gpkf_est

    # GPKF (gaussian process kalman filter) prediction
    results["predMeanKf"], results["predCovKf"], results["exeTimeKfPred"] = \
        gpkf_prediction(params["data"], params["gpkf"], Y, noise_var, estimate=gpkf_est)

    return params, F, Y, noise_var, results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["synthetic", "colorado"], default="synthetic")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    args = parser.parse_args()

    params, F, Y, noise_var, results = run(args.data, args.seed)

    print(f"Data set: {args.data}")
    print(f"  NP estimation time (total):         {results['exeTimeNp']:.4f} s")
    print(f"  NP prediction time (total):         {results['exeTimeNpPred']:.4f} s")
    print(f"  GPKF estimation time (per step):    {results['exeTimeKf']:.6f} s")
    print(f"  GPKF prediction time (per step):    {results['exeTimeKfPred']:.6f} s")
    print(f"  GPKF negative log marginal lik.:    {results['negLogMarginalKf']:.4f}")

    # plotting some results
    if not args.no_plots:
        import matplotlib.pyplot as plt

        from plot_results import plot_results
        plot_results(params, F, results, rng=np.random.default_rng(args.seed))
        plt.show()


if __name__ == "__main__":
    main()
