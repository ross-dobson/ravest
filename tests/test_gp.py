import numpy as np
import pytest

from ravest.gp import GPKernel
from ravest.param import Parameter


class TestGPKernel:
    """Tests for the GPKernel class."""

    def test_gpkernel_init_valid(self) -> None:
        """Test GPKernel initialization with valid kernel type."""
        kernel = GPKernel("Quasiperiodic")
        assert kernel.kernel_type == "Quasiperiodic"
        assert kernel.expected_hyperparams == ["gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]

    def test_gpkernel_init_invalid(self) -> None:
        """Test GPKernel initialization with invalid kernel type."""
        with pytest.raises(ValueError, match="Unsupported kernel type.*invalid_kernel"):
            GPKernel("invalid_kernel")

    def test_get_expected_hyperparams(self) -> None:
        """Test getting expected hyperparameters."""
        kernel = GPKernel("Quasiperiodic")
        hyperparams = kernel.get_expected_hyperparams()
        assert hyperparams == ["gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]
        # Verify it returns a copy
        hyperparams.append("extra")
        assert kernel.expected_hyperparams == ["gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]

    def test_validate_hyperparams_valid(self) -> None:
        """Test validating valid hyperparameters."""
        kernel = GPKernel("Quasiperiodic")
        hyperparams = {
            "gp_amp": Parameter(1.0, "m/s"),
            "gp_lambda_e": Parameter(50.0, "d"),
            "gp_lambda_p": Parameter(0.5, ""),
            "gp_period": Parameter(10.0, "d"),
        }
        # Should not raise
        kernel.validate_hyperparams(hyperparams)

    def test_validate_hyperparams_missing(self) -> None:
        """Test error when required hyperparameters are missing."""
        kernel = GPKernel("Quasiperiodic")
        hyperparams = {
            "gp_amp": Parameter(1.0, "m/s"),
            "gp_lambda_e": Parameter(50.0, "d"),
            # Missing gp_lambda_p and gp_period
        }
        with pytest.raises(ValueError, match="Missing required hyperparameters"):
            kernel.validate_hyperparams(hyperparams)

    def test_validate_hyperparams_unexpected(self) -> None:
        """Test error when unexpected hyperparameters are provided."""
        kernel = GPKernel("Quasiperiodic")
        hyperparams = {
            "gp_amp": Parameter(1.0, "m/s"),
            "gp_lambda_e": Parameter(50.0, "d"),
            "gp_lambda_p": Parameter(0.5, ""),
            "gp_period": Parameter(10.0, "d"),
            "extra_param": Parameter(5.0, ""),
        }
        with pytest.raises(ValueError, match="Unexpected hyperparameters"):
            kernel.validate_hyperparams(hyperparams)

    def test_validate_hyperparams_values_non_finite(self) -> None:
        """Test error for non-finite hyperparameter values."""
        kernel = GPKernel("Quasiperiodic")
        hyperparams_values = {
            "gp_amp": np.nan,
            "gp_lambda_e": 50.0,
            "gp_lambda_p": 0.5,
            "gp_period": 10.0,
        }
        with pytest.raises(ValueError, match="Non-finite hyperparameter"):
            kernel._validate_hyperparams_values(hyperparams_values)

    def test_validate_hyperparams_values_negative(self) -> None:
        """Test error for negative hyperparameter values."""
        kernel = GPKernel("Quasiperiodic")
        hyperparams_values = {
            "gp_amp": 1.0,
            "gp_lambda_e": -50.0,  # Invalid negative value
            "gp_lambda_p": 0.5,
            "gp_period": 10.0,
        }
        with pytest.raises(ValueError, match="must be positive"):
            kernel._validate_hyperparams_values(hyperparams_values)

    def test_validate_hyperparams_values_zero(self) -> None:
        """Test error for zero hyperparameter values."""
        kernel = GPKernel("Quasiperiodic")
        hyperparams_values = {
            "gp_amp": 1.0,
            "gp_lambda_e": 50.0,
            "gp_lambda_p": 0.0,  # Invalid zero value
            "gp_period": 10.0,
        }
        with pytest.raises(ValueError, match="must be positive"):
            kernel._validate_hyperparams_values(hyperparams_values)

    def test_build_kernel_quasiperiodic(self) -> None:
        """Test building a Quasiperiodic kernel."""
        kernel = GPKernel("Quasiperiodic")
        hyperparams_values = {
            "gp_amp": 1.0,
            "gp_lambda_e": 50.0,
            "gp_lambda_p": 0.5,
            "gp_period": 10.0,
        }
        tinygp_kernel = kernel.build_kernel(hyperparams_values)
        # Check that it returns a tinygp kernel object
        assert tinygp_kernel is not None
        # The kernel should be a product of scaled kernels
        # We can't check much more without importing tinygp internals
