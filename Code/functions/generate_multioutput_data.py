"""Multi-output (vector-valued) extension of generateSyntheticData.m.

The latent field f(x, t) takes values in R^D and is a zero-mean GP whose kernel
is a sum of Q separable space-output-time terms:
    K((x, d, t), (x', d', t')) = sum_q K_q(x, x') * B_q(d, d') * gamma_q(|t - t'|)

Four models are available, each representing a special case of this sum:
    independent : Q = 1, B diagonal                    (outputs are all independent)
    icm         : Q = 1, B a general covariance matrix (intrinsic coregionalization model)
    lmc         : Q >= 1, one time kernel gamma_q shared by every term
    lmc_time    : Q >= 1, a separate time kernel gamma_q for each term

A multi-output kernel is described by a dict

    {"model": "icm",
     "components": [{"space": {...}, "time": {...}, "B": (D, D) array}, ...]}

where "space" and "time" are kernel parameter dicts as used by
`kernel_function`. `multioutput_kernel` builds such a dict and checks that it
matches the requested model.

Ordering: samples are returned as arrays of shape (N, D, T) (location, output,
time). Vectorizing them in column-major order (F.ravel(order="F")) puts
location fastest, then output, then time, so the covariance of the vector is
sum_q kron(Gamma_q, B_q, K_q). With D = 1 this reduces to the
kron(K_time, K_space) ordering used in the rest of the port.
"""

import numpy as np

from .kernel_function import kernel_function
from .kernel_sampled import kernel_sampled

MODELS = ("independent", "icm", "lmc", "lmc_time")


def _psd_sqrt(K):
    """Return L with LL' = K, clipping tiny negative eigenvalues to zero for stability."""
    eigvals, eigvecs = np.linalg.eigh(0.5 * (K + K.T))
    return eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))


def random_coregionalization_matrix(num_outputs, rank=None, kappa=1.0e-4, scale=1.0, rng=None):
    """Returns a random output covariance B = AA' + diag(kappa).

    Parameters
    ----------
    num_outputs : int
        Number of outputs D.
    rank : int, optional
        Number of columns of A (defaults to D). Minimum is 1, maximum is D. 
        Low rank and small kappa gives strongly correlated outputs.
    kappa : float or array_like, shape (D,), optional
        Extra diagonal term (nugget) ensuring B is positive definite.
    scale : float or array_like, shape (D,), optional
        Standard deviation of the entries of A, either shared or one per row.
    rng : numpy.random.Generator, optional

    Returns
    -------
    numpy.ndarray, shape (D, D)
    """
    rng = np.random.default_rng() if rng is None else rng
    rank = num_outputs if rank is None else rank
    if not (1 <= rank <= num_outputs):
        raise ValueError(f"random_coregionalization_matrix: rank should be integer between 1 and {num_outputs}")

    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), (num_outputs,))
    A = scale[:, None] * rng.standard_normal((num_outputs, rank))
    return A @ A.T + np.diag(np.broadcast_to(np.asarray(kappa, dtype=np.float64), (num_outputs,)))


def multioutput_kernel(model, B, space_par, time_par):
    """
    Returns a multi-output kernel dict, after checking it is valid for the chosen `model`.

    Parameters
    ----------
    model : {"independent", "icm", "lmc", "lmc_time"}
    B : array_like, shape (D, D), or sequence of Q such arrays
        Output covariance matrices B_q. For "independent", a vector of length
        D is also accepted and taken as the diagonal of B.
    space_par : dict or sequence of Q dicts
        Spatial kernel parameters K_q. A single dict is shared by every term in the summation.
    time_par : dict or sequence of Q dicts
        Time kernel parameters gamma_q. A single dict is shared by every term in the summation.
        For "lmc" it must be a single dict.

    Returns
    -------
    dict with keys "model" and "components".
    """
    if model not in MODELS:
        raise ValueError(f"multioutput_kernel: unknown model '{model}', expected one of {MODELS}")

    if model == "independent" and np.ndim(B) == 1:
        B = np.diag(B)
    Bs = [np.asarray(B, dtype=np.float64)] if np.ndim(B) == 2 else [np.asarray(b, dtype=np.float64) for b in B]
    num_comp = len(Bs)

    if model in ("independent", "icm") and num_comp != 1:
        raise ValueError(f"multioutput_kernel: model '{model}' has a single term, got {num_comp} B matrices")
    if model == "lmc" and not isinstance(time_par, dict):
        raise ValueError("multioutput_kernel: model 'lmc' shares one time kernel, give a single dict")

    spaces = [space_par] * num_comp if isinstance(space_par, dict) else list(space_par)
    times = [time_par] * num_comp if isinstance(time_par, dict) else list(time_par)
    if len(spaces) != num_comp or len(times) != num_comp:
        raise ValueError("multioutput_kernel: need one space and one time kernel per B matrix")

    num_outputs = Bs[0].shape[0]
    for q, b in enumerate(Bs):
        if b.shape != (num_outputs, num_outputs):
            raise ValueError(f"multioutput_kernel: B[{q}] has shape {b.shape}, expected {(num_outputs, num_outputs)}")
        if not np.allclose(b, b.T):
            raise ValueError(f"multioutput_kernel: B[{q}] is not symmetric")
        if np.linalg.eigvalsh(b).min() < -1e-10 * max(1.0, np.abs(b).max()):
            raise ValueError(f"multioutput_kernel: B[{q}] is not positive semidefinite")
    if model == "independent" and np.any(Bs[0] != np.diag(np.diag(Bs[0]))):
        raise ValueError("multioutput_kernel: model 'independent' needs a diagonal B")

    components = [{"space_par": s, "time_par": t, "B": b} for s, t, b in zip(spaces, times, Bs)]
    return {"model": model, "components": components}


def _component_matrices(space_locs1, space_locs2, time_instants1, time_instants2, component):
    """Return (K_q, B_q, Gamma_q) sampled on the given grids."""
    Ks = kernel_sampled(space_locs1, space_locs2, kernel_function(component["space_par"]))
    Kt = kernel_sampled(time_instants1, time_instants2, kernel_function(component["time_par"]))
    return Ks, component["B"], Kt


def multioutput_kernel_sampled(space_locs1, space_locs2, time_instants1, time_instants2, kernel):
    """Return the multi-output space-time kernel sampled on the given grids.

    Parameters
    ----------
    space_locs1, space_locs2 : array_like, shape (N1, d) and (N2, d)
    time_instants1, time_instants2 : array_like, shape (T1,) and (T2,)
    kernel : dict
        Multi-output kernel from `multioutput_kernel`.

    Returns
    -------
    numpy.ndarray, shape (T1 * D * N1, T2 * D * N2)
        sum_q kron(Gamma_q, B_q, K_q); location varies fastest, then output,
        then time.
    """
    K = 0.0
    for component in kernel["components"]:
        Ks, B, Kt = _component_matrices(space_locs1, space_locs2, time_instants1, time_instants2, component)
        K = K + np.kron(Kt, np.kron(B, Ks))
    return K


def sample_multioutput_gp(space_locs, time_instants, kernel, rng=None):
    """Draws one sample of the given latent multi-output GP on a space-time grid.

    Each term q is sampled as F_q = Z x_1 L_q x_2 M_q x_3 G_q, with
    K_q = L_q L_q', B_q = M_q M_q', Gamma_q = G_q G_q' and Z an (N, D, T)
    standard normal array, which has covariance kron(Gamma_q, B_q, K_q). The
    terms are independent, so their sum has the covariance of `kernel`.

    Returns
    -------
    numpy.ndarray, shape (N, D, T)
    """
    rng = np.random.default_rng() if rng is None else rng
    F = 0.0
    for component in kernel["components"]:
        Ks, B, Kt = _component_matrices(space_locs, space_locs, time_instants, time_instants, component)
        Z = rng.standard_normal((Ks.shape[0], B.shape[0], Kt.shape[0]))
        F = F + np.einsum("ia,jb,kc,abc->ijk", _psd_sqrt(Ks), _psd_sqrt(B), _psd_sqrt(Kt), Z, optimize=True)
    return F


def generate_multioutput_data(param, rng=None, missing="joint"):
    """Sample a multi-output spatio-temporal GP and noisy, partially missing measurements.

    Parameters
    ----------
    param : dict
        Data parameters as for `generate_synthetic_data` ("spaceLocs",
        "spaceLocsMeasIdx", "timeInstants", "noiseStd"), except that
        "kernel" is a multi-output kernel from `multioutput_kernel`.
        "noiseStd" may be a scalar or a length-D array (one per output).
    rng : numpy.random.Generator, optional
    missing : {"joint", "independent", "none"}, optional
        How measurements are deleted at each time instant, following
        generateSyntheticData.m (a random number between 1 and the number of
        measured locations is removed):
          "joint"       : the same locations are missing for every output;
          "independent" : missing locations are drawn separately per output;
          "none"        : no measurements are removed.

    Returns
    -------
    F : numpy.ndarray, shape (numLocs, D, numTimeInstants)
        True (zero-mean) GP sample.
    Y : numpy.ndarray, shape (numMeasLocs, D, numTimeInstants)
        Noisy measurements at the measured locations, NaN where missing.
    noiseVar : numpy.ndarray, same shape as Y
        Measurement noise variances, +inf where missing.
    """
    rng = np.random.default_rng() if rng is None else rng

    # sample "true" (zero mean) GP (axes: space, output, time)
    F = sample_multioutput_gp(param["spaceLocs"], param["timeInstants"], param["kernel"], rng)
    _, num_outputs, num_time_inst = F.shape

    # create measurements
    meas_idx = param["spaceLocsMeasIdx"]
    num_space_locs_meas = len(meas_idx)
    noise_std = np.broadcast_to(np.asarray(param["noiseStd"], dtype=np.float64), (num_outputs,))
    noise_var = np.broadcast_to((noise_std**2)[None, :, None], (num_space_locs_meas, num_outputs, num_time_inst)).copy()
    Y = F[meas_idx] + np.sqrt(noise_var) * rng.standard_normal(noise_var.shape)

    # delete (randomly) some measurements and update the noise variance
    if missing not in ("joint", "independent", "none"):
        raise ValueError(f"generate_multioutput_data: unknown missing pattern '{missing}'")
    if missing != "none":
        outputs = [slice(None)] if missing == "joint" else range(num_outputs)
        for t in range(num_time_inst):
            for d in outputs:
                num_missing = rng.integers(1, num_space_locs_meas + 1)
                idx = np.sort(rng.choice(num_space_locs_meas, size=num_missing, replace=False))
                Y[idx, d, t] = np.nan
                noise_var[idx, d, t] = np.inf

    return F, Y, noise_var


def example_multioutput_kernel(model, num_outputs=3, num_components=2, rng=None):
    """Returns an example multi-output kernel for `model`, for quick simulations.

    The spatial and time kernels use the synthetic settings of
    `load_parameters` (gaussian space, exponential time). For the LMC models
    the q-th term has spatial length-scale q + 1 (and, for "lmc_time", the
    time kernel alternates between exponential and periodic), so the terms act
    on different scales.
    """
    rng = np.random.default_rng() if rng is None else rng
    space = {"type": "gaussian", "scale": 1.0, "std": 1.0}
    time = {"type": "exponential", "scale": 1.0, "std": 1.0}

    if model == "independent":
        return multioutput_kernel(model, rng.uniform(0.5, 1.5, num_outputs), space, time)
    if model == "icm":
        return multioutput_kernel(model, random_coregionalization_matrix(num_outputs, rng=rng), space, time)

    Bs = [random_coregionalization_matrix(num_outputs, rank=1, kappa=0.05, rng=rng) for _ in range(num_components)]
    spaces = [{"type": "gaussian", "scale": 1.0, "std": float(q + 1)} for q in range(num_components)]
    if model == "lmc":
        return multioutput_kernel(model, Bs, spaces, time)
    times = [
        {"type": "exponential", "scale": 1.0, "std": float(q + 1)} if q % 2 == 0
        else {"type": "periodic", "scale": 1.0, "std": float(q + 1), "frequency": 0.25}
        for q in range(num_components)
    ]
    return multioutput_kernel(model, Bs, spaces, times)
