"""Multi-output extension of Todescato et al.'s spatio-temporal GP code."""

from .generate_multioutput_data import (
    example_multioutput_kernel, 
    generate_multioutput_data,
    multioutput_kernel,
    multioutput_kernel_sampled,
    random_coregionalization_matrix,
    sample_multioutput_gp
)
from .gpkf import gpkf_estimation, gpkf_prediction
from .kalman_est import kalman_est
from .kernel_function import kernel_function, multioutput_kernel_function
from .kernel_sampled import kernel_sampled, multioutput_inputs
from .metrics import evaluate, marginal_variances
from .nonparametric import (
    batch_gp_posterior, 
    nonparametric_estimation, 
    nonparametric_prediction
)
from .time_state_space import (
    create_discrete_time_sys, 
    time_kernel_coefficients, 
    time_state_space
)

__all__ = [
    "batch_gp_posterior",
    "create_discrete_time_sys",
    "evaluate",
    "example_multioutput_kernel",
    "generate_multioutput_data",
    "gpkf_estimation",
    "gpkf_prediction",
    "kalman_est",
    "kernel_function",
    "kernel_sampled",
    "marginal_variances",
    "multioutput_inputs",
    "multioutput_kernel",
    "multioutput_kernel_function",
    "multioutput_kernel_sampled",
    "nonparametric_estimation",
    "nonparametric_prediction",
    "random_coregionalization_matrix",
    "sample_multioutput_gp",
    "time_kernel_coefficients",
    "time_state_space",
]
