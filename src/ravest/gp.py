"""Gaussian Process kernel management for radial velocity fitting."""
# gp.py
from typing import Dict, List

import jax.numpy as jnp
import numpy as np
from tinygp import kernels

from ravest.param import Parameter

SUPPORTED_KERNELS = ["Quasiperiodic"]

class GPKernel:
    """Gaussian Process kernel management class for RV fitting.

    Handles kernel type selection, hyperparameter validation, and kernel
    construction.
    """

    def __init__(self, kernel_type: str) -> None:
        """Initialize GP kernel.

        Parameters
        ----------
        kernel_type : str
            Type of kernel. Supported: "quasiperiodic"

        Raises
        ------
        ValueError
            If kernel_type is not supported
        """
        self.kernel_type = kernel_type

        # Set expected hyperparameters based on kernel type
        if self.kernel_type == "Quasiperiodic":
            self.expected_hyperparams = ["gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}. "
                            f"Supported kernels: {SUPPORTED_KERNELS}")

    def get_expected_hyperparams(self) -> List[str]:
        """Get list of expected hyperparameter names for this kernel.

        Returns
        -------
        List[str]
            Names of required hyperparameters
        """
        return self.expected_hyperparams.copy()

    def validate_hyperparams(self, hyperparams: Dict[str, Parameter]) -> None:
        """Validate hyperparameters for this kernel type.

        Parameters
        ----------
        hyperparams : dict
            Dictionary of hyperparameter values

        Raises
        ------
        ValueError
            If hyperparameters are invalid or missing
        """
        # Check all required hyperparams are present
        provided_params = set(hyperparams.keys())
        expected_params = set(self.expected_hyperparams)

        missing_params = expected_params - provided_params
        if missing_params:
            raise ValueError(f"Missing required hyperparameters: {missing_params}")

        unexpected_params = provided_params - expected_params
        if unexpected_params:
            raise ValueError(f"Unexpected hyperparameters: {unexpected_params}")

        # Extract values from Parameter objects and validate them
        hyperparams_values = {name: param.value for name, param in hyperparams.items()}
        self._validate_hyperparams_values(hyperparams_values)


    def _validate_hyperparams_values(self, hyperparams_values: Dict[str, float]) -> None:
        """Validate that hyperparameter values are physically reasonable.

        This is the internal version that works with raw float values.
        Used by run_mcmc and other internal functions.

        Parameters
        ----------
        hyperparams_values : dict
            Dictionary mapping hyperparameter names to float values

        Raises
        ------
        ValueError
            If hyperparameter values are invalid
        """
        # All hyperparameters must be finite
        for key in self.expected_hyperparams:
            if not np.isfinite(hyperparams_values[key]):
                raise ValueError(f"Non-finite hyperparameter found in: {hyperparams_values}")

        # Kernel-specific checks
        if self.kernel_type == "Quasiperiodic":
            # Positive-valued hyperparameters
            for key in self.expected_hyperparams:
                if hyperparams_values[key] <= 0:
                    raise ValueError(f"{key} must be positive, got {hyperparams_values[key]}")

            # TODO: disabled for now. It would be good to have this enabled/disabled via argument
            # # Ensure there is at least one non-trivial turning point in the GP
            # # see Equation 15 in Rajpaul et al. 2021 https://doi.org/10.1093/mnras/stab2192
            # # LaTeX: \lambda_e^2 > \frac{3}{2\pi} P_{GP}^2 \lambda_p^2
            # LHS = jnp.square(hyperparams_values["gp_lambda_e"])
            # RHS = (3 / (2 * jnp.pi)) * jnp.square(hyperparams_values["gp_period"] * hyperparams_values["gp_lambda_p"])

            # if not LHS > RHS:
            #     raise ValueError(
            #         f"gp_lambda_e^2 must be greater than (3/(2pi)) * gp_period^2 * gp_lambda_p^2 "
            #         f"to ensure at least one non-trivial turning point in the GP. Requires LHS > RHS, but got: "
            #         f"LHS: {LHS:.3f}, RHS: {RHS:.3f}"
            #     )



    def build_kernel(self, hyperparams: Dict[str, float]) -> kernels.Kernel:
        """Build tinygp kernel object with specified hyperparameters.

        Parameters
        ----------
        hyperparams : dict
            Dictionary of hyperparameter values

        Returns
        -------
        tinygp kernel object
            Configured kernel (not full GaussianProcess)
        """
        if self.kernel_type == "Quasiperiodic":
            gp_amp = hyperparams["gp_amp"]
            gp_lambda_e = hyperparams["gp_lambda_e"]
            gp_lambda_p = hyperparams["gp_lambda_p"]
            gp_period = hyperparams["gp_period"]

            # LaTeX: exp{ - \frac{(x_i - x_j)^2}{2 {l}^2}}
            # where tinygp's "scale" l = our "gp_lambda_e"
            exp_squared = kernels.ExpSquared(scale=gp_lambda_e)

            # LaTeX: exp{ - \Gamma \sin^2{\pi \frac{x_i-x_j}{P}}}
            # where tinygp's "scale" P = our "gp_period" P_GP
            # where tinygp's "gamma" = 1 / (2 {\lambda_p}^2)
            gamma = 1 / (2 * jnp.square(gp_lambda_p))
            exp_sine_squared = kernels.ExpSineSquared(scale=gp_period, gamma=gamma)

            # LaTeX A^2 * \exp{ - \frac{(x_i - x_j)^2}{2 {\lambda_e}^2}} * exp{ - \frac{\sin^2{\pi \frac{x_i-x_j}{P}}} {2 {\lambda_p}^2}}
            return jnp.square(gp_amp) * exp_sine_squared * exp_squared
