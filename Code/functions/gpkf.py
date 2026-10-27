"""Multi-output GPKF (generalizes gpkfEstimation.m and gpkfPrediction.m).

The latent field is f = sum_q f_q with cov(f_q) = Gamma_q (x) B_q (x) K_q
(see generate_multioutput_data). Each term is written as

    f_q(., d, t) = sum_r M_q[d, r] u_{q,r}(., t),    B_q = M_q M_q',

where the u_{q,r} (r = 1, ..., R_q = rank B_q) are independent scalar
space-time GPs with covariance Gamma_q (x) K_q. As in Todescato et al., each
u_{q,r} at the measured locations is L_q (I (x) c_q) z_{q,r}(t), where
K_q = L_q L_q' and z_{q,r} stacks independent copies of the state-space model
of gamma_q. The full state stacks every z_{q,r}, so it has dimension
sum_q R_q * rank(K_q) * s_q.

Prediction at unmeasured locations uses that, for a separable kernel,
u(pred, t) = K_pm K_mm^+ u(meas, t) + e(t) with e independent of u(meas, .)
and cov(e(t)) = gamma(0) (K_pp - K_pm K_mm^+ K_mp). The prediction is then
another output matrix on the same state plus this residual covariance.

Vectors at one time instant are ordered location fastest, then output, i.e.
Y[:, :, t].ravel(order="F").
"""

import time

import numpy as np
import scipy.linalg

from .kalman_est import kalman_est
from .kernel_function import kernel_function
from .kernel_sampled import kernel_sampled
from .time_state_space import time_state_space


def _psd_factor(K, rtol=1e-10):
    """Return (U, lam) with K = U diag(lam) U', keeping eigenvalues above rtol * max."""
    eigvals, eigvecs = np.linalg.eigh(0.5 * (K + K.T))
    keep = eigvals > rtol * max(eigvals.max(), 0.0)
    return eigvecs[:, keep], eigvals[keep]


def _gpkf_components(param_data, kernel):
    """Return the per-term pieces of the GPKF model on the measured locations."""
    components = []
    for comp in kernel["components"]:
        a, c, v, q = time_state_space(comp["time_par"], param_data["samplingTime"])
        U_B, lam_B = _psd_factor(np.asarray(comp["B"], dtype=np.float64))
        Ks = kernel_sampled(param_data["spaceLocsMeas"], param_data["spaceLocsMeas"], kernel_function(comp["space_par"]))
        U_K, lam_K = _psd_factor(Ks)
        components.append({
            "a": a, "c": c, "v": v, "q": q,
            "M": U_B * np.sqrt(lam_B),          # B_q = M M'
            "U_K": U_K, "lam_K": lam_K,         # K_q = U_K diag(lam_K) U_K'
            "gamma0": (c @ v @ c.T).item(),     # gamma_q(0) of the state-space model
            "space_par": comp["space_par"],
        })
    return components


def _gpkf_model(components):
    """Assemble (A, C, Q, V0) from the per-term pieces."""
    A, C, Q, V0 = [], [], [], []
    for comp in components:
        num_channels = comp["M"].shape[1] * comp["U_K"].shape[1]
        I = np.eye(num_channels)
        L = comp["U_K"] * np.sqrt(comp["lam_K"])
        A.append(np.kron(I, comp["a"]))
        Q.append(np.kron(I, comp["q"]))
        V0.append(np.kron(I, comp["v"]))
        C.append(np.kron(comp["M"], L @ np.kron(np.eye(L.shape[1]), comp["c"])))
    return scipy.linalg.block_diag(*A), np.hstack(C), scipy.linalg.block_diag(*Q), scipy.linalg.block_diag(*V0)


def gpkf_estimation(param_data, param_gpkf, meas, noise_var):
    """Multi-output GPKF estimate at the measured locations.

    This is a filter: the estimate at time t uses measurements up to time t.

    Parameters
    ----------
    param_data : dict
        Data parameters ("spaceLocsMeas", "samplingTime").
    param_gpkf : dict
        GPKF parameters, with a multi-output "kernel" (see `multioutput_kernel`).
        Every time kernel must have a state-space form (see time_state_space).
    meas : numpy.ndarray, shape (numMeasLocs, D, numTimeInstants)
        Measurements, NaN where missing.
    noise_var : numpy.ndarray, same shape as meas

    Returns
    -------
    posteriorMean : numpy.ndarray, shape (numMeasLocs, D, numTimeInstants)
    posteriorCov : numpy.ndarray, shape (numTimeInstants, numMeasLocs * D, numMeasLocs * D)
    exeTime : float
        Average execution time per Kalman iteration, in seconds.
    negLogMarginal : float
        Negative marginal log-likelihood of the measurements.
    filtered : dict
        Filtered states ("x", "V") and model pieces, reused by `gpkf_prediction`.
    """
    num_locs, num_outputs, num_time_inst = meas.shape
    components = _gpkf_components(param_data, param_gpkf["kernel"])
    A, C, Q, V0 = _gpkf_model(components)

    # stack outputs: row n + numMeasLocs * d of the vectorized measurements
    meas_vec = meas.reshape(num_locs * num_outputs, num_time_inst, order="F")
    noise_vec = noise_var.reshape(num_locs * num_outputs, num_time_inst, order="F")
    x, V, _, _, exe_time, neg_log_marginal = kalman_est(A, C, Q, V0, meas_vec, noise_vec)

    # output function and posterior variance
    posterior_mean = (C @ x).reshape(num_locs, num_outputs, num_time_inst, order="F")
    posterior_cov = C @ V @ C.T   # broadcasts over the time axis
    filtered = {"x": x, "V": V, "components": components}
    return posterior_mean, posterior_cov, exe_time, neg_log_marginal, filtered


def gpkf_prediction(param_data, param_gpkf, meas, noise_var, estimate=None):
    """Multi-output GPKF prediction at the unmeasured locations.

    Parameters
    ----------
    param_data, param_gpkf, meas, noise_var :
        As in `gpkf_estimation`. "spaceLocsPred" is also required in `param_data`.
    estimate : tuple, optional
        Output of `gpkf_estimation` for the same inputs; it is rerun if omitted.

    Returns
    -------
    predictedMean : numpy.ndarray, shape (numPredLocs, D, numTimeInstants)
    predictedCov : numpy.ndarray, shape (numTimeInstants, numPredLocs * D, numPredLocs * D)
    exeTime : float
        Average execution time per time instant, in seconds.
    """
    if estimate is None:
        estimate = gpkf_estimation(param_data, param_gpkf, meas, noise_var)
    filtered = estimate[4]
    x, V, components = filtered["x"], filtered["V"], filtered["components"]
    num_outputs, num_time_inst = meas.shape[1], meas.shape[2]
    locs_pred, locs_meas = param_data["spaceLocsPred"], param_data["spaceLocsMeas"]

    t_start = time.perf_counter()
    C_pred, residual_cov = [], 0.0
    for comp in components:
        kernel_space = kernel_function(comp["space_par"])
        kernel_section = kernel_sampled(locs_pred, locs_meas, kernel_space)
        kernel_prediction = kernel_sampled(locs_pred, locs_pred, kernel_space)
        # K_pm K_mm^+ L = K_pm U diag(lam^-1/2), with K_mm = L L'
        H = kernel_section @ (comp["U_K"] / np.sqrt(comp["lam_K"]))
        C_pred.append(np.kron(comp["M"], H @ np.kron(np.eye(H.shape[1]), comp["c"])))
        kriging_cov = kernel_prediction - H @ H.T
        residual_cov = residual_cov + comp["gamma0"] * np.kron(comp["M"] @ comp["M"].T, kriging_cov)
    C_pred = np.hstack(C_pred)

    num_locs_pred = len(locs_pred)
    predicted_mean = (C_pred @ x).reshape(num_locs_pred, num_outputs, num_time_inst, order="F")
    predicted_cov = C_pred @ V @ C_pred.T + residual_cov
    exe_time = (time.perf_counter() - t_start) / num_time_inst
    return predicted_mean, predicted_cov, exe_time
