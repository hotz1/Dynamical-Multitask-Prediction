import numpy as np
import scipy.linalg

def create_discrete_time_sys(num_coeff, den_coeff, Ts):
    """
    Build the discrete-time state-space model of a temporal kernel in the canonical form
    based on the numerator(s) and denominator(s) of the companion form.

    Parameters
    ----------
    num_coeff : array_like
        Coefficients for the numerator of the rational power spectrum.
    den_coeff : array_like
        Coefficients for the denominator of the rational power spectrum.
    Ts : float
        Sampling time used for the discretization.

    Returns
    -------
    A : numpy.ndarray, shape (n, n)
        Discrete-time state matrix, expm(F * Ts).
    C : numpy.ndarray, shape (1, n)
        Output matrix.
    V : numpy.ndarray, shape (n, n)
        Stationary state covariance matrix solving Lyapunov equation.
    Q : numpy.ndarray, shape (n, n)
        Discrete-time measurement variance matrix.
    """

    # Coefficients and dimension
    num_coeff = np.atleast_1d(np.asarray(num_coeff, dtype=np.float64)).ravel()
    den_coeff = np.atleast_1d(np.asarray(den_coeff, dtype=np.float64)).ravel()
    state_dim = den_coeff.size

    # Construct state matrix
    F = np.diag(np.ones(state_dim - 1), 1)
    F[-1, :] = -den_coeff

    # Construct input matrix
    G = np.zeros((state_dim, 1))
    G[-1, 0] = 1.0
    GGt = G @ G.T

    # Construct output matrix
    C = np.zeros((1, state_dim))
    C[0, : num_coeff.size] = num_coeff

    # Discretize w.r.t. time
    A = scipy.linalg.expm(F * Ts)

    # Solve Lyapunov equation
    V = scipy.linalg.solve_continuous_lyapunov(F, -GGt)
    # V = 0.5 * (V + V.T)

    # Discretize the noise matrix
    Q = np.zeros((state_dim, state_dim))
    Ns = 10000
    t = Ts/Ns
    for n in np.linspace(t, Ts, Ns):
        E = scipy.linalg.expm(F * n)
        Q += t * E @ GGt @ E.T

    return A, C, V, Q
