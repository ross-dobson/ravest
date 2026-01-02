import jax.numpy as jnp
import numpy as np
import pytest

import ravest.prior
from ravest.fit import GPFitter, GPLogLikelihood, GPLogPosterior
from ravest.gp import GPKernel
from ravest.param import Parameter, Parameterisation


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


@pytest.fixture
def test_gp_data():
    """Simple synthetic RV data with GP-like correlation for testing."""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    vel = np.array([5.0, -2.0, -5.0, 2.0, 3.0, -1.0])
    verr = np.array([1.0, 1.1, 0.9, 0.85, 1.5, 1.0])
    return time, vel, verr


@pytest.fixture
def test_gp_circular_params():
    """Simple circular orbit parameters for GP testing."""
    return {
        "P_b": Parameter(2.0, "d", fixed=True),
        "K_b": Parameter(5.0, "m/s", fixed=False),
        "e_b": Parameter(0.0, "", fixed=True),
        "w_b": Parameter(np.pi/2, "rad", fixed=True),
        "Tc_b": Parameter(0.0, "d", fixed=True),
        "g": Parameter(0.0, "m/s", fixed=True),
        "gd": Parameter(0.0, "m/s/day", fixed=True),
        "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
        "jit": Parameter(1.0, "m/s", fixed=False),
    }


@pytest.fixture
def test_gp_hyperparams():
    """Simple GP hyperparameters for testing."""
    return {
        "gp_amp": Parameter(1.0, "m/s", fixed=False),
        "gp_lambda_e": Parameter(50.0, "d", fixed=False),
        "gp_lambda_p": Parameter(0.5, "", fixed=False),
        "gp_period": Parameter(10.0, "d", fixed=False),
    }


@pytest.fixture
def test_gp_priors():
    """Simple priors for GP testing."""
    return {
        "K_b": ravest.prior.Uniform(0, 20),
        "jit": ravest.prior.Uniform(0, 5),
    }


@pytest.fixture
def test_gp_hyperpriors():
    """Simple hyperpriors for GP testing."""
    return {
        "gp_amp": ravest.prior.Uniform(0, 10),
        "gp_lambda_e": ravest.prior.Uniform(1, 100),
        "gp_lambda_p": ravest.prior.Uniform(0.1, 2.0),
        "gp_period": ravest.prior.Uniform(1, 50),
    }


class TestGPLogLikelihood:
    """Tests for the GPLogLikelihood class."""

    def test_gploglikelihood_init(self, test_gp_data) -> None:
        """Test GPLogLikelihood initialization."""
        time, vel, verr = test_gp_data
        gp_kernel = GPKernel("Quasiperiodic")
        ll = GPLogLikelihood(
            time=time, vel=vel, verr=verr, t0=2.0,
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel
        )

        np.testing.assert_array_equal(ll.time, time)
        np.testing.assert_array_equal(ll.vel, vel)
        np.testing.assert_array_equal(ll.verr, verr)
        assert ll.t0 == 2.0
        assert ll.gp_kernel == gp_kernel

    def test_gploglikelihood_calculation(self, test_gp_data) -> None:
        """Test GP log-likelihood calculation with valid parameters."""
        time, vel, verr = test_gp_data
        gp_kernel = GPKernel("Quasiperiodic")
        ll = GPLogLikelihood(
            time=time, vel=vel, verr=verr, t0=2.0,
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel
        )

        params = {
            "P_b": 2.0, "K_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g": 0.0, "gd": 0.0, "gdd": 0.0, "jit": 2.0
        }
        hyperparams = {
            "gp_amp": 1.0,
            "gp_lambda_e": 50.0,
            "gp_lambda_p": 0.5,
            "gp_period": 10.0,
        }

        log_like = ll(params, hyperparams)
        assert np.isfinite(log_like)
        # JAX returns JAX Array types, so we need to check for those as well
        assert isinstance(log_like, (float, np.floating, jnp.ndarray))

    def test_gploglikelihood_invalid_planet(self, test_gp_data) -> None:
        """Test GP log-likelihood returns -inf for invalid planet parameters."""
        time, vel, verr = test_gp_data
        gp_kernel = GPKernel("Quasiperiodic")
        ll = GPLogLikelihood(
            time=time, vel=vel, verr=verr, t0=2.0,
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel
        )

        params = {
            "P_b": -1.0,  # Invalid negative period
            "K_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g": 0.0, "gd": 0.0, "gdd": 0.0, "jit": 1.0
        }
        hyperparams = {
            "gp_amp": 1.0,
            "gp_lambda_e": 50.0,
            "gp_lambda_p": 0.5,
            "gp_period": 10.0,
        }

        log_like = ll(params, hyperparams)
        assert log_like == -np.inf


class TestGPLogPosterior:
    """Tests for the GPLogPosterior class."""

    def test_gplogposterior_init(self, test_gp_data, test_gp_circular_params, test_gp_hyperparams,
                                   test_gp_priors, test_gp_hyperpriors) -> None:
        """Test GPLogPosterior initialization."""
        time, vel, verr = test_gp_data
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams
        priors = test_gp_priors
        hyperpriors = test_gp_hyperpriors
        gp_kernel = GPKernel("Quasiperiodic")

        # Extract fixed and free params/hyperparams
        fixed_params = {k: v.value for k, v in params.items() if v.fixed}
        free_params_names = [k for k, v in params.items() if not v.fixed]
        fixed_hyperparams = {k: v.value for k, v in hyperparams.items() if v.fixed}
        free_hyperparams_names = [k for k, v in hyperparams.items() if not v.fixed]

        lpost = GPLogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel,
            priors=priors,
            hyperpriors=hyperpriors,
            fixed_params=fixed_params,
            fixed_hyperparams=fixed_hyperparams,
            free_params_names=free_params_names,
            free_hyperparams_names=free_hyperparams_names,
            time=time, vel=vel, verr=verr, t0=2.0
        )

        assert lpost.planet_letters == ["b"]
        assert lpost.gp_kernel == gp_kernel

    def test_gplogposterior_valid_calculation(self, test_gp_data, test_gp_circular_params, test_gp_hyperparams,
                                               test_gp_priors, test_gp_hyperpriors) -> None:
        """Test GP log-posterior calculation with valid parameters."""
        time, vel, verr = test_gp_data
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams
        priors = test_gp_priors
        hyperpriors = test_gp_hyperpriors
        gp_kernel = GPKernel("Quasiperiodic")

        fixed_params = {k: v.value for k, v in params.items() if v.fixed}
        free_params_names = [k for k, v in params.items() if not v.fixed]
        fixed_hyperparams = {k: v.value for k, v in hyperparams.items() if v.fixed}
        free_hyperparams_names = [k for k, v in hyperparams.items() if not v.fixed]

        lpost = GPLogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel,
            priors=priors,
            hyperpriors=hyperpriors,
            fixed_params=fixed_params,
            fixed_hyperparams=fixed_hyperparams,
            free_params_names=free_params_names,
            free_hyperparams_names=free_hyperparams_names,
            time=time, vel=vel, verr=verr, t0=2.0
        )

        combined_dict = {
            "K_b": 5.0, "jit": 1.0,
            "gp_amp": 1.0, "gp_lambda_e": 50.0, "gp_lambda_p": 0.5, "gp_period": 10.0
        }
        log_post = lpost.log_probability(combined_dict)

        assert np.isfinite(log_post)
        # JAX returns JAX Array types, so we need to check for those as well
        assert isinstance(log_post, (float, np.floating, jnp.ndarray))

    def test_gplogposterior_invalid_prior(self, test_gp_data, test_gp_circular_params, test_gp_hyperparams,
                                           test_gp_priors, test_gp_hyperpriors) -> None:
        """Test GP log-posterior returns -inf when prior is invalid."""
        time, vel, verr = test_gp_data
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams
        priors = test_gp_priors
        hyperpriors = test_gp_hyperpriors
        gp_kernel = GPKernel("Quasiperiodic")

        fixed_params = {k: v.value for k, v in params.items() if v.fixed}
        free_params_names = [k for k, v in params.items() if not v.fixed]
        fixed_hyperparams = {k: v.value for k, v in hyperparams.items() if v.fixed}
        free_hyperparams_names = [k for k, v in hyperparams.items() if not v.fixed]

        lpost = GPLogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel,
            priors=priors,
            hyperpriors=hyperpriors,
            fixed_params=fixed_params,
            fixed_hyperparams=fixed_hyperparams,
            free_params_names=free_params_names,
            free_hyperparams_names=free_hyperparams_names,
            time=time, vel=vel, verr=verr, t0=2.0
        )

        combined_dict = {
            "K_b": -1.0, "jit": 1.0,  # Invalid K_b outside prior bounds
            "gp_amp": 1.0, "gp_lambda_e": 50.0, "gp_lambda_p": 0.5, "gp_period": 10.0
        }
        log_post = lpost.log_probability(combined_dict)

        assert log_post == -np.inf


class TestGPFitter:
    """Tests for the GPFitter class."""

    def test_gpfitter_init(self) -> None:
        """Test GPFitter initialization."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        assert fitter.planet_letters == ["b"]
        assert fitter.parameterisation.parameterisation == "P K e w Tc"
        assert fitter.gp_kernel == gp_kernel
        assert fitter.params == {}
        assert fitter.priors == {}
        assert fitter.hyperparams == {}
        assert fitter.hyperpriors == {}

    def test_add_data_valid(self, test_gp_data) -> None:
        """Test adding valid data to GPFitter."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        time, vel, verr = test_gp_data
        fitter.add_data(time, vel, verr, t0=2.0)

        np.testing.assert_array_equal(fitter.time, time)
        np.testing.assert_array_equal(fitter.vel, vel)
        np.testing.assert_array_equal(fitter.verr, verr)
        assert fitter.t0 == 2.0

    def test_add_data_mismatched_lengths(self) -> None:
        """Test error when data arrays have different lengths."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        time = np.array([0.0, 1.0])
        vel = np.array([5.0, -2.0, -5.0])  # Different length
        verr = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="Time, velocity, and uncertainty arrays must be the same length"):
            fitter.add_data(time, vel, verr, t0=2.0)

    def test_params_property_valid(self, test_gp_circular_params) -> None:
        """Test setting valid parameters via property."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        params = test_gp_circular_params
        fitter.params = params

        assert len(fitter.params) == 9  # 5 planetary + 3 trend params + jit
        assert "P_b" in fitter.params
        assert "jit" in fitter.params

    def test_hyperparams_property_valid(self, test_gp_circular_params, test_gp_hyperparams) -> None:
        """Test setting valid hyperparameters via property."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams

        fitter.params = params
        fitter.hyperparams = hyperparams

        assert len(fitter.hyperparams) == 4
        assert "gp_amp" in fitter.hyperparams
        assert "gp_period" in fitter.hyperparams

    def test_hyperparams_missing(self, test_gp_circular_params) -> None:
        """Test error when required hyperparameters are missing."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        params = test_gp_circular_params
        fitter.params = params

        # Missing gp_lambda_p and gp_period
        incomplete_hyperparams = {
            "gp_amp": Parameter(1.0, "m/s"),
            "gp_lambda_e": Parameter(50.0, "d"),
        }

        with pytest.raises(ValueError, match="Missing required hyperparameters"):
            fitter.hyperparams = incomplete_hyperparams

    def test_add_priors_valid(self, test_gp_circular_params, test_gp_priors) -> None:
        """Test adding valid priors."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        params = test_gp_circular_params
        priors = test_gp_priors

        fitter.params = params
        fitter.priors = priors

        assert len(fitter.priors) == 2
        assert "K_b" in fitter.priors
        assert "jit" in fitter.priors

    def test_add_hyperpriors_valid(self, test_gp_circular_params, test_gp_hyperparams, test_gp_priors, test_gp_hyperpriors) -> None:
        """Test adding valid hyperpriors."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams
        priors = test_gp_priors
        hyperpriors = test_gp_hyperpriors

        fitter.params = params
        fitter.hyperparams = hyperparams
        fitter.priors = priors
        fitter.hyperpriors = hyperpriors

        assert len(fitter.hyperpriors) == 4
        assert "gp_amp" in fitter.hyperpriors
        assert "gp_period" in fitter.hyperpriors

    def test_get_free_params(self, test_gp_circular_params) -> None:
        """Test getting free parameters."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        params = test_gp_circular_params
        fitter.params = params

        free_params = fitter.free_params_dict
        free_names = fitter.free_params_names
        free_vals = fitter.free_params_values

        assert len(free_params) == 2  # K_b and jit
        assert "K_b" in free_names
        assert "jit" in free_names
        assert len(free_vals) == 2
        assert 5.0 in free_vals  # K_b value
        assert 1.0 in free_vals  # jit value

    def test_get_free_hyperparams(self, test_gp_circular_params, test_gp_hyperparams) -> None:
        """Test getting free hyperparameters."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams

        fitter.params = params
        fitter.hyperparams = hyperparams

        free_hyperparams = fitter.free_hyperparams_dict
        free_hypernames = fitter.free_hyperparams_names
        free_hypervals = fitter.free_hyperparams_values

        assert len(free_hyperparams) == 4  # All 4 GP hyperparams
        assert "gp_amp" in free_hypernames
        assert "gp_period" in free_hypernames
        assert len(free_hypervals) == 4


class TestGPFitterIntegration:
    """Integration tests for complete GPFitter workflow."""

    def test_complete_setup(self, test_gp_data, test_gp_circular_params, test_gp_hyperparams,
                             test_gp_priors, test_gp_hyperpriors) -> None:
        """Test complete GPFitter setup without running MCMC."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)

        # Add data
        time, vel, verr = test_gp_data
        fitter.add_data(time, vel, verr, t0=2.0)

        # Add parameters and hyperparameters
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams
        fitter.params = params
        fitter.hyperparams = hyperparams

        # Add priors and hyperpriors
        priors = test_gp_priors
        hyperpriors = test_gp_hyperpriors
        fitter.priors = priors
        fitter.hyperpriors = hyperpriors

        # Verify everything is set up correctly
        assert len(fitter.params) == 9
        assert len(fitter.hyperparams) == 4
        assert len(fitter.priors) == 2
        assert len(fitter.hyperpriors) == 4
        assert len(fitter.free_params_names) == 2
        assert len(fitter.free_hyperparams_names) == 4
        assert fitter.ndim == 6  # 2 free params + 4 free hyperparams
