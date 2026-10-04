import logging
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import ravest.prior
from ravest.fit import (
    Fitter,
    GPFitter,
    GPLogLikelihood,
    GPLogPosterior,
    LogLikelihood,
    LogPosterior,
    LogPrior,
)
from ravest.gp import GPKernel
from ravest.param import Parameter, Parameterisation


@pytest.fixture
def test_data():
    """Simple synthetic RV data for testing (single instrument)."""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    vel = np.array([5.0, -2.0, -5.0, 2.0, 3.0])
    velerr = np.array([1.0, 1.1, 0.9, 0.85, 1.5])
    instrument = np.array(["HARPS", "HARPS", "HARPS", "HARPS", "HARPS"])
    return time, vel, velerr, instrument


@pytest.fixture
def test_data_multi_instrument():
    """Synthetic RV data with two instruments for testing."""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    vel = np.array([5.0, -2.0, -5.0, 102.0, 103.0, 98.0])  # HIRES has +100 offset
    velerr = np.array([1.0, 1.1, 0.9, 0.85, 1.5, 1.2])
    instrument = np.array(["HARPS", "HARPS", "HARPS", "HIRES", "HIRES", "HIRES"])
    return time, vel, velerr, instrument


@pytest.fixture
def test_circular_params():
    """Simple circular orbit parameters for testing (single instrument: HARPS)."""
    return {
        "P_b": Parameter(2.0, fixed=True),
        "K_b": Parameter(5.0, fixed=False),
        "e_b": Parameter(0.0, fixed=True),
        "w_b": Parameter(np.pi/2, fixed=True),
        "Tc_b": Parameter(0.0, fixed=True),
        "g_HARPS": Parameter(0.0, fixed=True),
        "gd": Parameter(0.0, fixed=True),
        "gdd": Parameter(0.0, fixed=True),
        "jit_HARPS": Parameter(1.0, fixed=False),
    }


@pytest.fixture
def test_circular_params_multi_instrument():
    """Circular orbit parameters for two instruments (HARPS and HIRES)."""
    return {
        "P_b": Parameter(2.0, fixed=True),
        "K_b": Parameter(5.0, fixed=False),
        "e_b": Parameter(0.0, fixed=True),
        "w_b": Parameter(np.pi/2, fixed=True),
        "Tc_b": Parameter(0.0, fixed=True),
        "g_HARPS": Parameter(0.0, fixed=False),
        "g_HIRES": Parameter(100.0, fixed=False),
        "gd": Parameter(0.0, fixed=True),
        "gdd": Parameter(0.0, fixed=True),
        "jit_HARPS": Parameter(1.0, fixed=False),
        "jit_HIRES": Parameter(2.0, fixed=False),
    }


@pytest.fixture
def test_simple_priors():
    """Simple priors for testing (single instrument: HARPS)."""
    return {
        "K_b": ravest.prior.Uniform(0, 20),
        "jit_HARPS": ravest.prior.Uniform(0, 5),
    }


@pytest.fixture
def test_simple_priors_multi_instrument():
    """Priors for two instruments (HARPS and HIRES)."""
    return {
        "K_b": ravest.prior.Uniform(0, 20),
        "g_HARPS": ravest.prior.Uniform(-10, 10),
        "g_HIRES": ravest.prior.Uniform(90, 110),
        "jit_HARPS": ravest.prior.Uniform(0, 5),
        "jit_HIRES": ravest.prior.Uniform(0, 5),
    }


class TestFitter:
    """Tests for the main Fitter class."""

    def test_fitter_init(self) -> None:
        """Test Fitter initialization."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        assert fitter.planet_letters == ["b"]
        assert fitter.parameterisation.parameterisation == "P K e w Tc"
        assert fitter.params == {}
        assert fitter.priors == {}

    def test_fitter_init_rejects_string_parameterisation(self) -> None:
        """Passing the parameterisation name as a string (not a Parameterisation) raises."""
        with pytest.raises(TypeError, match="parameterisation must be a Parameterisation object"):
            Fitter(["b"], "P K e w Tc")

    def test_add_data_valid(self, test_data) -> None:
        """Test adding valid data."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        np.testing.assert_array_equal(fitter.time, time)
        np.testing.assert_array_equal(fitter.vel, vel)
        np.testing.assert_array_equal(fitter.velerr, velerr)
        np.testing.assert_array_equal(fitter.instrument, instrument)
        assert fitter.t0 == 2.0
        assert fitter.unique_instruments == ["HARPS"]

    def test_add_data_multi_instrument(self, test_data_multi_instrument) -> None:
        """Test adding data with multiple instruments."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data_multi_instrument
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        np.testing.assert_array_equal(fitter.instrument, instrument)
        assert set(fitter.unique_instruments) == {"HARPS", "HIRES"}

    def test_add_data_mismatched_lengths(self) -> None:
        """Test error when data arrays have different lengths."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time = np.array([0.0, 1.0])
        vel = np.array([5.0, -2.0, -5.0])  # Different length
        velerr = np.array([1.0, 1.0])
        instrument = np.array(["HARPS", "HARPS"])

        with pytest.raises(ValueError, match="arrays must be the same length"):
            fitter.add_data(time, vel, velerr, instrument, t0=2.0)

    def test_params_property_valid(self, test_data, test_circular_params) -> None:
        """Test setting valid parameters via property."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params
        fitter.params = params

        assert len(fitter.params) == 9  # 5 planetary + 2 trend params + g_HARPS + jit_HARPS
        assert "P_b" in fitter.params
        assert "jit_HARPS" in fitter.params
        assert "g_HARPS" in fitter.params

    def test_add_params_wrong_count(self, test_data) -> None:
        """Test error when wrong number of parameters provided."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = {"P_b": Parameter(2.0, fixed=False)}  # Too few params

        with pytest.raises(ValueError, match="Missing required parameters.*Expected 9 parameters, got 1"):
            fitter.params = params

    def test_add_params_missing_planetary_param(self, test_data, test_circular_params) -> None:
        """Test error when planetary parameter is missing."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params.copy()
        del params["P_b"]  # Remove required parameter

        with pytest.raises(ValueError, match="Missing required parameters.*Expected 9 parameters, got 8"):
            fitter.params = params

    def test_add_params_unexpected_param(self, test_data, test_circular_params) -> None:
        """Test error when unexpected parameter is provided."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params.copy()
        params["invalid_param"] = Parameter(1.0, fixed=False)  # Add unexpected parameter

        # Should raise generic unexpected-parameter error, NOT the legacy g/jit hint
        with pytest.raises(ValueError, match="Unexpected parameters.*Expected 9 parameters, got 10"):
            fitter.params = params

    def test_add_params_legacy_only(self, test_data) -> None:
        """Test error when only legacy g and jit parameters are provided (nothing else)."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = {
            "g": Parameter(0.0, fixed=False),
            "jit": Parameter(1.0, fixed=False),
        }

        with pytest.raises(ValueError, match="Single-instrument 'g' and 'jit' parameters are no longer supported"):
            fitter.params = params

    def test_add_params_legacy_single_instrument(self, test_data, test_circular_params) -> None:
        """Test error when legacy g/jit are used instead of g_HARPS/jit_HARPS (single instrument).

        The error message should name the correct per-instrument parameter names.
        """
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params.copy()
        del params["g_HARPS"]
        del params["jit_HARPS"]
        params["g"] = Parameter(0.0, fixed=False)
        params["jit"] = Parameter(1.0, fixed=False)

        with pytest.raises(ValueError, match="Single-instrument 'g' and 'jit' parameters are no longer supported.*g_HARPS.*jit_HARPS"):
            fitter.params = params

    def test_add_params_legacy_partial_multi_instrument(self, test_data_multi_instrument, test_circular_params_multi_instrument) -> None:
        """Test error when legacy g/jit are used for one instrument in a multi-instrument setup.

        User provides g_HARPS/jit_HARPS correctly, but g/jit instead of g_HIRES/jit_HIRES.
        """
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data_multi_instrument
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params_multi_instrument.copy()
        del params["g_HIRES"]
        del params["jit_HIRES"]
        params["g"] = Parameter(100.0, fixed=False)
        params["jit"] = Parameter(2.0, fixed=False)

        with pytest.raises(ValueError, match="Single-instrument 'g' and 'jit' parameters are no longer supported"):
            fitter.params = params

    def test_add_params_legacy_alongside_correct_multi_instrument(self, test_data_multi_instrument, test_circular_params_multi_instrument) -> None:
        """Test error when legacy g/jit are provided alongside all correct per-instrument params.

        All 11 required parameters are present, but g and jit are also included.
        """
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data_multi_instrument
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params_multi_instrument.copy()
        params["g"] = Parameter(0.0, fixed=False)    # Legacy, on top of all correct params
        params["jit"] = Parameter(1.0, fixed=False)  # Legacy, on top of all correct params

        with pytest.raises(ValueError, match="Single-instrument 'g' and 'jit' parameters are no longer supported"):
            fitter.params = params

    def test_add_priors_valid(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test adding valid priors."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params
        priors = test_simple_priors

        fitter.params = params
        fitter.priors = priors

        assert len(fitter.priors) == 2
        assert "K_b" in fitter.priors
        assert "jit_HARPS" in fitter.priors

    def test_add_priors_missing_prior(self, test_data, test_circular_params) -> None:
        """Test error when prior is missing for free parameter."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params
        priors = {"K_b": ravest.prior.Uniform(0, 20)}  # Missing jit_HARPS prior

        fitter.params = params
        with pytest.raises(ValueError, match="Missing priors for parameters.*jit_HARPS"):
            fitter.priors = priors

    def test_add_priors_invalid_initial_value(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test error when initial parameter value is outside prior bounds."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params.copy()
        params["K_b"] = Parameter(25.0, fixed=False)  # Outside uniform prior [0, 20]
        priors = test_simple_priors

        fitter.params = params
        with pytest.raises(ValueError, match="Initial value 25.0 of parameter K_b is invalid"):
            fitter.priors = priors

    def test_add_priors_too_many_warning(self, test_data, test_circular_params) -> None:
        """Test warning when too many priors provided (for fixed params)."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params
        fitter.params = params

        # Add priors for both free AND fixed parameters
        priors = {
            "K_b": ravest.prior.Uniform(0, 20),
            "jit_HARPS": ravest.prior.Uniform(0, 5),
            "P_b": ravest.prior.Uniform(1, 5),  # This is fixed!
        }

        with pytest.raises(ValueError, match="Unexpected priors.*P_b"):
            fitter.priors = priors

    def test_get_free_params(self, test_data, test_circular_params) -> None:
        """Test getting free parameters."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params
        fitter.params = params

        free_params = fitter.free_params_dict
        free_names = fitter.free_params_names
        free_vals = fitter.free_params_values

        assert len(free_params) == 2  # K_b and jit_HARPS
        assert "K_b" in free_names
        assert "jit_HARPS" in free_names
        assert len(free_vals) == 2
        assert 5.0 in free_vals  # K_b value
        assert 1.0 in free_vals  # jit_HARPS value

    def test_get_fixed_params(self, test_data, test_circular_params) -> None:
        """Test getting fixed parameters."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = test_circular_params
        fitter.params = params

        fixed_params = fitter.fixed_params_dict
        fixed_names = fitter.fixed_params_names
        fixed_vals = fitter.fixed_params_values

        assert len(fixed_params) == 7  # All except K_b and jit_HARPS
        assert "P_b" in fixed_names
        assert "e_b" in fixed_names
        assert "g_HARPS" in fixed_names
        assert len(fixed_vals) == 7


class TestLogLikelihood:
    """Tests for the LogLikelihood class."""

    def test_loglikelihood_init(self, test_data) -> None:
        """Test LogLikelihood initialization."""
        time, vel, velerr, instrument = test_data
        unique_instruments = np.unique(instrument)
        ll = LogLikelihood(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        np.testing.assert_array_equal(ll.time, time)
        np.testing.assert_array_equal(ll.vel, vel)
        np.testing.assert_array_equal(ll.velerr, velerr)
        assert ll.t0 == 2.0

    def test_loglikelihood_calculation(self, test_data) -> None:
        """Test log-likelihood calculation with valid parameters."""
        time, vel, velerr, instrument = test_data
        unique_instruments = np.unique(instrument)
        ll = LogLikelihood(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        params = {
            "P_b": 2.0, "K_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g_HARPS": 0.0, "gd": 0.0, "gdd": 0.0, "jit_HARPS": 2.0
        }

        log_like = ll(params)
        assert np.isfinite(log_like)
        assert isinstance(log_like, float)

    def test_loglikelihood_invalid_planet(self, test_data) -> None:
        """Test log-likelihood returns -inf for invalid planet parameters."""
        time, vel, velerr, instrument = test_data
        unique_instruments = np.unique(instrument)
        ll = LogLikelihood(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        params = {
            "P_b": -1.0,  # Invalid negative period
            "K_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g_HARPS": 0.0, "gd": 0.0, "gdd": 0.0, "jit_HARPS": 1.0
        }

        log_like = ll(params)
        assert log_like == -np.inf

    def test_loglikelihood_perfect_fit(self) -> None:
        """Test log-likelihood when model perfectly fits data."""
        # Create synthetic data from known model
        time = np.array([0.0, 0.5, 1.0, 1.5])
        # Constant velocity (no planet signal)
        vel = np.array([2.0, 2.0, 2.0, 2.0])
        velerr = np.array([1.0, 1.0, 1.0, 1.0])
        instrument = np.array(["HARPS", "HARPS", "HARPS", "HARPS"])
        unique_instruments = np.array(["HARPS"])

        ll = LogLikelihood(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=1.0
        )

        params = {
            "P_b": 10.0, "K_b": 0.5, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g_HARPS": 2.0, "gd": 0.0, "gdd": 0.0, "jit_HARPS": 1.0
        }

        log_like = ll(params)
        # Should be finite for valid parameters
        assert np.isfinite(log_like)

    def test_loglikelihood_multi_instrument(self, test_data_multi_instrument) -> None:
        """Test log-likelihood calculation with multiple instruments."""
        time, vel, velerr, instrument = test_data_multi_instrument
        unique_instruments = np.unique(instrument)
        ll = LogLikelihood(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        params = {
            "P_b": 2.0, "K_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g_HARPS": 0.0, "g_HIRES": 100.0, "gd": 0.0, "gdd": 0.0,
            "jit_HARPS": 1.0, "jit_HIRES": 2.0
        }

        log_like = ll(params)
        assert np.isfinite(log_like)
        assert isinstance(log_like, float)

    def test_loglikelihood_jitter_affects_result(self, test_data) -> None:
        """Test that per-instrument jitter affects log-likelihood."""
        time, vel, velerr, instrument = test_data
        unique_instruments = np.unique(instrument)
        ll = LogLikelihood(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        params_low_jit = {
            "P_b": 2.0, "K_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g_HARPS": 0.0, "gd": 0.0, "gdd": 0.0, "jit_HARPS": 0.1
        }
        params_high_jit = {
            "P_b": 2.0, "K_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g_HARPS": 0.0, "gd": 0.0, "gdd": 0.0, "jit_HARPS": 10.0
        }

        ll_low = ll(params_low_jit)
        ll_high = ll(params_high_jit)

        # Different jitter should give different log-likelihood
        assert ll_low != ll_high


class TestLogPrior:
    """Tests for the LogPrior class."""

    def test_logprior_init(self, test_simple_priors) -> None:
        """Test LogPrior initialization."""
        priors = test_simple_priors
        lp = LogPrior(priors)
        assert lp.priors == priors

    def test_logprior_valid_params(self, test_simple_priors) -> None:
        """Test log-prior calculation with valid parameters."""
        priors = test_simple_priors
        lp = LogPrior(priors)

        params = {"K_b": 10.0, "jit_HARPS": 2.0}
        log_prior = lp(params)

        assert np.isfinite(log_prior)
        assert isinstance(log_prior, float)

    def test_logprior_invalid_params(self, test_simple_priors) -> None:
        """Test log-prior returns -inf for parameters outside bounds."""
        priors = test_simple_priors
        lp = LogPrior(priors)

        params = {"K_b": -5.0, "jit_HARPS": 2.0}  # K_b outside [0, 20]
        log_prior = lp(params)

        assert log_prior == -np.inf

    def test_logprior_multiple_params(self) -> None:
        """Test log-prior sums correctly across multiple parameters."""
        priors = {
            "K_b": ravest.prior.Uniform(0, 10),  # log_prior = -log(10)
            "jit_HARPS": ravest.prior.Uniform(0, 5),   # log_prior = -log(5)
        }
        lp = LogPrior(priors)

        params = {"K_b": 5.0, "jit_HARPS": 2.5}
        log_prior = lp(params)

        expected = -np.log(10) - np.log(5)
        assert np.isclose(log_prior, expected)


class TestLogPosterior:
    """Tests for the LogPosterior class (integration tests)."""

    def test_logposterior_init(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test LogPosterior initialization."""
        time, vel, velerr, instrument = test_data
        unique_instruments = np.unique(instrument)
        params = test_circular_params
        priors = test_simple_priors

        # Extract fixed params
        fixed_params = {k: v.value for k, v in params.items() if v.fixed}
        free_param_names = [k for k, v in params.items() if not v.fixed]

        lpost = LogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            priors=priors,
            fixed_params=fixed_params,
            free_params_names=free_param_names,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        assert lpost.planet_letters == ["b"]

    def test_logposterior_valid_calculation(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test log-posterior calculation with valid parameters."""
        time, vel, velerr, instrument = test_data
        unique_instruments = np.unique(instrument)
        params = test_circular_params
        priors = test_simple_priors

        fixed_params = {k: v.value for k, v in params.items() if v.fixed}
        free_param_names = [k for k, v in params.items() if not v.fixed]

        lpost = LogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            priors=priors,
            fixed_params=fixed_params,
            free_params_names=free_param_names,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        free_params_dict = {"K_b": 5.0, "jit_HARPS": 1.0}
        log_post = lpost.log_probability(free_params_dict)

        assert np.isfinite(log_post)
        assert isinstance(log_post, float)

    def test_logposterior_invalid_prior(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test log-posterior returns -inf when prior is invalid."""
        time, vel, velerr, instrument = test_data
        unique_instruments = np.unique(instrument)
        params = test_circular_params
        priors = test_simple_priors

        fixed_params = {k: v.value for k, v in params.items() if v.fixed}
        free_param_names = [k for k, v in params.items() if not v.fixed]

        lpost = LogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            priors=priors,
            fixed_params=fixed_params,
            free_params_names=free_param_names,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        free_params_dict = {"K_b": -1.0, "jit_HARPS": 1.0}  # Invalid K_b
        log_post = lpost.log_probability(free_params_dict)

        assert log_post == -np.inf

    def test_negative_log_probability_for_MAP(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test MAP interface that takes list instead of dict."""
        time, vel, velerr, instrument = test_data
        unique_instruments = np.unique(instrument)
        params = test_circular_params
        priors = test_simple_priors

        fixed_params = {k: v.value for k, v in params.items() if v.fixed}
        free_param_names = [k for k, v in params.items() if not v.fixed]

        lpost = LogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            priors=priors,
            fixed_params=fixed_params,
            free_params_names=free_param_names,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        free_params_vals = [5.0, 1.0]  # K_b, jit_HARPS
        neg_log_post = lpost._negative_log_probability_for_MAP(free_params_vals)

        assert np.isfinite(neg_log_post)
        assert isinstance(neg_log_post, float)

        # Should be negative of log_probability
        free_params_dict = {"K_b": 5.0, "jit_HARPS": 1.0}
        log_post = lpost.log_probability(free_params_dict)
        assert np.isclose(neg_log_post, -log_post)


class TestFitterIntegration:
    """Integration tests for complete Fitter workflow."""

    def test_complete_setup(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test complete Fitter setup without running MCMC."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))

        # Add data
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        # Add parameters
        params = test_circular_params
        fitter.params = params

        # Add priors
        priors = test_simple_priors
        fitter.priors = priors

        # Verify everything is set up correctly
        assert len(fitter.params) == 9
        assert len(fitter.priors) == 2
        assert len(fitter.free_params_names) == 2
        assert len(fitter.fixed_params_names) == 7

    def test_multi_planet_setup(self, test_data) -> None:
        """Test setup with multiple planets."""
        fitter = Fitter(["b", "c"], Parameterisation("P K e w Tc"))

        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        # Multi-planet parameters (single instrument: HARPS)
        params = {
            "P_b": Parameter(2.0, fixed=True),
            "K_b": Parameter(5.0, fixed=False),
            "e_b": Parameter(0.0, fixed=True),
            "w_b": Parameter(np.pi/2, fixed=True),
            "Tc_b": Parameter(0.0, fixed=True),

            "P_c": Parameter(4.0, fixed=True),
            "K_c": Parameter(3.0, fixed=False),
            "e_c": Parameter(0.0, fixed=True),
            "w_c": Parameter(np.pi/2, fixed=True),
            "Tc_c": Parameter(1.0, fixed=True),

            "g_HARPS": Parameter(0.0, fixed=True),
            "gd": Parameter(0.0, fixed=True),
            "gdd": Parameter(0.0, fixed=True),
            "jit_HARPS": Parameter(1.0, fixed=False),
        }

        priors = {
            "K_b": ravest.prior.Uniform(0, 20),
            "K_c": ravest.prior.Uniform(0, 20),
            "jit_HARPS": ravest.prior.Uniform(0, 5),
        }

        fitter.params = params
        fitter.priors = priors

        assert len(fitter.params) == 14  # 5*2 planets + 4 system (g_HARPS, gd, gdd, jit_HARPS)
        assert len(fitter.priors) == 3   # K_b, K_c, jit_HARPS
        assert len(fitter.free_params_names) == 3

    def test_multi_instrument_setup(self, test_data_multi_instrument, test_circular_params_multi_instrument, test_simple_priors_multi_instrument) -> None:
        """Test setup with multiple instruments."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))

        time, vel, velerr, instrument = test_data_multi_instrument
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        fitter.params = test_circular_params_multi_instrument
        fitter.priors = test_simple_priors_multi_instrument

        # 5 planetary + 2 trend (gd, gdd) + 2 gamma (g_HARPS, g_HIRES) + 2 jitter (jit_HARPS, jit_HIRES)
        assert len(fitter.params) == 11
        assert "g_HARPS" in fitter.params
        assert "g_HIRES" in fitter.params
        assert "jit_HARPS" in fitter.params
        assert "jit_HIRES" in fitter.params

    def test_params_all_fixed_warns(self, test_data, test_circular_params) -> None:
        """Test that setting all parameters as fixed issues a UserWarning."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = {k: Parameter(v.value, fixed=True) for k, v in test_circular_params.items()}

        with pytest.warns(UserWarning, match="All parameters are fixed"):
            fitter.params = params

    def test_find_map_estimate_all_fixed_raises(self, test_data, test_circular_params) -> None:
        """Test that find_map_estimate raises a clear error when all parameters are fixed.

        scipy.minimize cannot handle a zero-dimensional parameter space and produces
        a cryptic _MaxFuncCallError. We guard against this with an explicit ValueError.
        """
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        # Set all parameters as fixed - no priors needed as there are no free params
        params = {k: Parameter(v.value, fixed=True) for k, v in test_circular_params.items()}
        with pytest.warns(UserWarning):
            fitter.params = params

        with pytest.raises(ValueError, match="no free parameters to optimise"):
            fitter.find_map_estimate()

    def test_generate_walker_positions_random_all_fixed_raises(self, test_data, test_circular_params) -> None:
        """Test that generate_initial_walker_positions_random raises when all parameters are fixed."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = {k: Parameter(v.value, fixed=True) for k, v in test_circular_params.items()}
        with pytest.warns(UserWarning):
            fitter.params = params

        with pytest.raises(ValueError, match="no free parameters to sample"):
            fitter.generate_initial_walker_positions_random(nwalkers=10)

    def test_run_mcmc_all_fixed_raises(self, test_data, test_circular_params) -> None:
        """Test that run_mcmc raises a clear error when all parameters are fixed."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = {k: Parameter(v.value, fixed=True) for k, v in test_circular_params.items()}
        with pytest.warns(UserWarning):
            fitter.params = params

        dummy_positions = np.empty((10, 0))
        with pytest.raises(ValueError, match="no free parameters to sample"):
            fitter.run_mcmc(dummy_positions, nwalkers=10, max_steps=10, progress=False)


class TestAdaptiveConvergence:
    """Tests for adaptive convergence feature in run_mcmc."""

    @pytest.fixture
    def setup_fitter(self, test_data, test_circular_params, test_simple_priors):
        """Setup a basic fitter for MCMC tests."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = test_circular_params
        fitter.priors = test_simple_priors

        # Generate initial positions
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(
            map_result, nwalkers=10
        )
        return fitter, initial_positions

    def test_fixed_length_mode(self, setup_fitter):
        """Test that fixed-length mode (check_convergence=False) runs for exactly max_steps."""
        fitter, initial_positions = setup_fitter
        max_steps = 100

        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=max_steps,
            progress=False,
            check_convergence=False
        )

        # Check that sampler ran for exactly max_steps
        assert fitter.sampler is not None
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape[0] == max_steps  # Should be exactly max_steps

    def test_adaptive_mode_runs(self, setup_fitter):
        """Test that adaptive mode (check_convergence=True) runs without errors."""
        fitter, initial_positions = setup_fitter

        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=500,
            progress=False,
            check_convergence=True,
            convergence_check_interval=50,
            convergence_check_start=20
        )

        # Check that sampler exists and has run
        assert fitter.sampler is not None
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape[0] > 0  # Should have some samples
        assert chain.shape[0] <= 500  # Should not exceed max_steps

    def test_adaptive_mode_stops_early(self, setup_fitter):
        """Test that adaptive mode can stop before max_steps."""
        fitter, initial_positions = setup_fitter

        # Use a large max_steps but expect early stopping for this simple problem
        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=10000,
            progress=False,
            check_convergence=True,
            convergence_check_interval=100,
            convergence_check_start=50
        )

        # For a simple problem, we expect it might converge before max_steps
        # (though this isn't guaranteed, so we just check it ran successfully)
        assert fitter.sampler is not None
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape[0] <= 10000

    def test_backward_compatibility_positional_args(self, setup_fitter):
        """Test backward compatibility with positional arguments."""
        fitter, initial_positions = setup_fitter

        # Old style: run_mcmc(initial_positions, nwalkers, nsteps)
        # New style: max_steps replaces nsteps
        fitter.run_mcmc(initial_positions, 10, 100, False)

        assert fitter.sampler is not None
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape[0] == 100

    def test_convergence_check_interval_parameter(self, setup_fitter):
        """Test that convergence_check_interval parameter is respected."""
        fitter, initial_positions = setup_fitter

        # This should run without errors even with different intervals
        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=300,
            progress=False,
            check_convergence=True,
            convergence_check_interval=200,  # Check only once or twice
            convergence_check_start=20
        )

        assert fitter.sampler is not None

    def test_convergence_check_start_parameter(self, setup_fitter):
        """Test that convergence_check_start parameter affects convergence checking."""
        fitter, initial_positions = setup_fitter

        # Test with different convergence_check_start values
        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=200,
            progress=False,
            check_convergence=True,
            convergence_check_interval=50,
            convergence_check_start=100  # Don't check before iteration 100
        )

        assert fitter.sampler is not None
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape[0] <= 200

    def test_max_steps_smaller_than_interval_raises(self, setup_fitter):
        """check_convergence with max_steps below the first check interval raises."""
        fitter, initial_positions = setup_fitter
        with pytest.raises(ValueError, match="No convergence check would ever run"):
            fitter.run_mcmc(
                initial_positions,
                nwalkers=10,
                max_steps=50,
                progress=False,
                check_convergence=True,
                convergence_check_interval=100,
                convergence_check_start=0,
            )

    def test_convergence_check_start_beyond_max_steps_raises(self, setup_fitter):
        """check_convergence with convergence_check_start beyond max_steps raises."""
        fitter, initial_positions = setup_fitter
        with pytest.raises(ValueError, match="No convergence check would ever run"):
            fitter.run_mcmc(
                initial_positions,
                nwalkers=10,
                max_steps=100,
                progress=False,
                check_convergence=True,
                convergence_check_interval=50,
                convergence_check_start=200,
            )

    def test_plot_autocorr_without_convergence_check_raises(self, setup_fitter):
        """Test that plotting without convergence checking raises informative error."""
        fitter, initial_positions = setup_fitter

        # Run without convergence checking
        fitter.run_mcmc(initial_positions, nwalkers=10, max_steps=100, progress=False, check_convergence=False)

        # Should raise ValueError when trying to plot
        with pytest.raises(ValueError, match="No autocorrelation history available"):
            fitter.plot_autocorr_estimates()

    def test_plot_autocorr_stores_history(self, setup_fitter):
        """Test that autocorr history is stored when convergence checking enabled."""
        fitter, initial_positions = setup_fitter

        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=300,
            progress=False,
            check_convergence=True,
            convergence_check_interval=100,
            convergence_check_start=20
        )

        # Check that history was stored
        assert hasattr(fitter, 'autocorr_history')
        assert len(fitter.autocorr_history) > 0
        assert isinstance(fitter.autocorr_history, dict)

        # Check that keys are iteration numbers
        for key in fitter.autocorr_history.keys():
            assert isinstance(key, (int, np.integer))

        # Check that values are tau arrays
        for tau in fitter.autocorr_history.values():
            assert isinstance(tau, np.ndarray)
            assert tau.shape == (len(fitter.free_params_names),)

    def test_plot_autocorr_all_params(self, setup_fitter):
        """Test plotting all parameters (default behaviour)."""
        fitter, initial_positions = setup_fitter

        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=300,
            progress=False,
            check_convergence=True,
            convergence_check_interval=100,
            convergence_check_start=20
        )

        # Should not raise any errors
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        fitter.plot_autocorr_estimates()

    def test_plot_autocorr_specific_params(self, setup_fitter):
        """Test plotting specific parameters only."""
        fitter, initial_positions = setup_fitter

        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=300,
            progress=False,
            check_convergence=True,
            convergence_check_interval=100,
            convergence_check_start=20
        )

        # Should plot only specified parameter
        import matplotlib
        matplotlib.use('Agg')
        fitter.plot_autocorr_estimates(params=['K_b'])

    def test_plot_autocorr_mean(self, setup_fitter):
        """Test plotting mean tau."""
        fitter, initial_positions = setup_fitter

        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=300,
            progress=False,
            check_convergence=True,
            convergence_check_interval=100,
            convergence_check_start=20
        )

        # Should plot mean instead of individual params
        import matplotlib
        matplotlib.use('Agg')
        fitter.plot_autocorr_estimates(plot_mean=True)

    def test_plot_autocorr_no_legend(self, setup_fitter):
        """Test plotting without legend."""
        fitter, initial_positions = setup_fitter

        fitter.run_mcmc(
            initial_positions,
            nwalkers=10,
            max_steps=300,
            progress=False,
            check_convergence=True,
            convergence_check_interval=100,
            convergence_check_start=20
        )

        # Should plot without legend
        import matplotlib
        matplotlib.use('Agg')
        fitter.plot_autocorr_estimates(show_legend=False)


class TestRVCalculations:
    """Tests for RV calculation methods."""

    @pytest.fixture
    def setup_fitter_for_rv(self, test_data, test_circular_params, test_simple_priors):
        """Setup fitter with data and params for RV calculations."""
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = test_circular_params
        fitter.priors = test_simple_priors
        return fitter

    @pytest.fixture
    def setup_fitter_two_planets(self, test_data):
        """Two-planet fitter (b and c) for freeze_params planet-letter tests."""
        fitter = Fitter(["b", "c"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = {
            "P_b": Parameter(2.0, fixed=True), "K_b": Parameter(5.0, fixed=False),
            "e_b": Parameter(0.0, fixed=True), "w_b": Parameter(np.pi/2, fixed=True),
            "Tc_b": Parameter(0.0, fixed=True),
            "P_c": Parameter(8.0, fixed=True), "K_c": Parameter(3.0, fixed=False),
            "e_c": Parameter(0.0, fixed=True), "w_c": Parameter(np.pi/2, fixed=True),
            "Tc_c": Parameter(1.0, fixed=True),
            "g_HARPS": Parameter(0.0, fixed=True), "gd": Parameter(0.0, fixed=True),
            "gdd": Parameter(0.0, fixed=True), "jit_HARPS": Parameter(1.0, fixed=False),
        }
        fitter.priors = {
            "K_b": ravest.prior.Uniform(0, 20), "K_c": ravest.prior.Uniform(0, 20),
            "jit_HARPS": ravest.prior.Uniform(0, 5),
        }
        return fitter

    def test_calculate_rv_planet_custom(self, setup_fitter_for_rv):
        """Test custom planet RV against hand-calculated circular orbit values.

        With e=0, w=pi/2, P=2, K=5, Tc=0:
        RV = K * cos(2*pi*(t - Tc)/P + w)
           = 5 * cos(pi*t + pi/2)
           = -5 * sin(pi*t)
        """
        fitter = setup_fitter_for_rv
        times = np.array([0.25, 0.5, 0.75, 1.25])

        # Build params dict
        params = fitter.build_params_dict(fitter.free_params_values)

        # Calculate RV
        rv = fitter.calculate_rv_planet_custom('b', times, params)

        assert isinstance(rv, np.ndarray)
        assert len(rv) == len(times)
        assert np.all(np.isfinite(rv))

        # Verify against exact analytical solution for circular orbit
        expected = -5.0 * np.sin(np.pi * times)
        np.testing.assert_allclose(rv, expected, atol=1e-10)

    def test_calculate_rv_trend_custom(self, setup_fitter_for_rv):
        """Test custom trend RV calculation."""
        fitter = setup_fitter_for_rv
        times = np.array([0.0, 1.0, 2.0, 3.0])

        # Build params dict
        params = fitter.build_params_dict(fitter.free_params_values)

        # Calculate trend RV
        rv_trend = fitter.calculate_rv_trend_custom(times, params)

        assert isinstance(rv_trend, np.ndarray)
        assert len(rv_trend) == len(times)
        assert np.all(np.isfinite(rv_trend))

    def test_calculate_rv_trend_custom_with_nonzero_trend(self, test_data):
        """Test trend calculation with non-zero trend parameters.

        Note: In the new multi-instrument API, the trend only includes gd and gdd.
        The gamma offset is per-instrument and handled separately.
        """
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        # Set up params with non-zero trend (gd only - no global gamma)
        params = {
            "P_b": Parameter(2.0, fixed=True),
            "K_b": Parameter(5.0, fixed=False),
            "e_b": Parameter(0.0, fixed=True),
            "w_b": Parameter(np.pi/2, fixed=True),
            "Tc_b": Parameter(0.0, fixed=True),
            "g_HARPS": Parameter(10.0, fixed=True),  # Per-instrument gamma
            "gd": Parameter(0.5, fixed=True),  # Non-zero slope
            "gdd": Parameter(0.0, fixed=True),
            "jit_HARPS": Parameter(1.0, fixed=False),
        }
        fitter.params = params

        times = np.array([0.0, 1.0, 2.0, 3.0])
        params_dict = fitter.build_params_dict(fitter.free_params_values)

        rv_trend = fitter.calculate_rv_trend_custom(times, params_dict)

        # Trend only includes gd and gdd, NOT gamma offset
        # trend(t) = gd*(t - t0) + gdd*(t - t0)^2
        # With gd=0.5, gdd=0.0, t0=2.0:
        expected_trend = 0.5 * (times - 2.0)
        np.testing.assert_allclose(rv_trend, expected_trend)

    def test_calculate_rv_total_custom(self, setup_fitter_for_rv):
        """Test custom total RV calculation (planet + trend)."""
        fitter = setup_fitter_for_rv
        times = np.array([0.0, 1.0, 2.0, 3.0])

        # Build params dict
        params = fitter.build_params_dict(fitter.free_params_values)

        # Calculate total RV
        rv_total = fitter.calculate_rv_total_custom(times, params)

        # Also calculate components separately
        rv_planet = fitter.calculate_rv_planet_custom('b', times, params)
        rv_trend = fitter.calculate_rv_trend_custom(times, params)

        # Total should equal sum of components
        np.testing.assert_allclose(rv_total, rv_planet + rv_trend)

    def test_build_params_dict_from_array(self, setup_fitter_for_rv):
        """Test building params dict from array."""
        fitter = setup_fitter_for_rv

        # Build from array
        params = fitter.build_params_dict(fitter.free_params_values)

        assert isinstance(params, dict)
        assert len(params) == 9  # All params (free + fixed)
        assert "P_b" in params
        assert "K_b" in params
        assert "jit_HARPS" in params

    def test_build_params_dict_from_dict(self, setup_fitter_for_rv):
        """Test building params dict from dict."""
        fitter = setup_fitter_for_rv

        # Build from dict
        free_params_dict = fitter.free_params_dict
        free_params_values_dict = {k: v.value for k, v in free_params_dict.items()}
        params = fitter.build_params_dict(free_params_values_dict)

        assert isinstance(params, dict)
        assert len(params) == 9  # All params (free + fixed)

    def test_calculate_rv_planet_from_samples(self, setup_fitter_for_rv):
        """Test calculating planet RV from MCMC samples."""
        fitter = setup_fitter_for_rv

        # Run short MCMC
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(map_result, nwalkers=10)
        fitter.run_mcmc(initial_positions, nwalkers=10, max_steps=50, progress=False)

        times = np.array([0.0, 1.0, 2.0])

        # Calculate RV from samples
        rv_samples = fitter.calculate_rv_planet_from_samples('b', times, discard_start=10, thin=5)

        # Should have shape (n_samples, n_times)
        assert rv_samples.ndim == 2
        assert rv_samples.shape[1] == len(times)
        assert np.all(np.isfinite(rv_samples))

    def test_calculate_rv_trend_from_samples(self, setup_fitter_for_rv):
        """Test calculating trend RV from MCMC samples."""
        fitter = setup_fitter_for_rv

        # Run short MCMC
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(map_result, nwalkers=10)
        fitter.run_mcmc(initial_positions, nwalkers=10, max_steps=50, progress=False)

        times = np.array([0.0, 1.0, 2.0])

        # Calculate trend RV from samples
        trend_samples = fitter.calculate_rv_trend_from_samples(times, discard_start=10, thin=5)

        # Should have shape (n_samples, n_times)
        assert trend_samples.ndim == 2
        assert trend_samples.shape[1] == len(times)
        assert np.all(np.isfinite(trend_samples))

    def _run_short_mcmc(self, fitter):
        """Run a short MCMC on the fitter (helper for freeze_params tests)."""
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(map_result, nwalkers=10)
        fitter.run_mcmc(initial_positions, nwalkers=10, max_steps=50, progress=False)

    def test_resolve_freeze_params_none(self, setup_fitter_for_rv):
        """None passes straight through as None (no freezing)."""
        fitter = setup_fitter_for_rv
        assert fitter._resolve_freeze_params(None) is None

    def test_resolve_freeze_params_explicit_values(self, setup_fitter_for_rv):
        """Explicit float values are returned unchanged (as floats)."""
        fitter = setup_fitter_for_rv
        with pytest.warns(UserWarning, match="already fixed, not free"):
            resolved = fitter._resolve_freeze_params({"P_b": 2.0, "Tc_b": 0.5})
        assert resolved == {"P_b": 2.0, "Tc_b": 0.5}
        assert all(isinstance(v, float) for v in resolved.values())

    def test_resolve_freeze_params_none_value_uses_median(self, setup_fitter_for_rv):
        """A None value resolves to the parameter's posterior median."""
        fitter = setup_fitter_for_rv
        self._run_short_mcmc(fitter)

        # K_b is the only free planet parameter in this fixture
        resolved = fitter._resolve_freeze_params({"K_b": None}, discard_start=10, thin=5)
        samples_dict = fitter.get_samples_dict(discard_start=10, thin=5)
        expected = float(np.median(samples_dict["K_b"]))
        assert resolved["K_b"] == pytest.approx(expected)

    def test_resolve_freeze_params_unknown_key_raises(self, setup_fitter_for_rv):
        """An unrecognised key raises ValueError."""
        fitter = setup_fitter_for_rv
        with pytest.raises(ValueError, match="Unknown freeze_params key"):
            fitter._resolve_freeze_params({"P_c": None})

    def test_resolve_freeze_params_rejects_non_planet_params(self, setup_fitter_for_rv):
        """Trend and instrument parameters cannot be frozen (planet params only)."""
        fitter = setup_fitter_for_rv
        for key in ("jit_HARPS", "g_HARPS", "gd", "gdd"):
            with pytest.raises(ValueError, match="Unknown freeze_params key"):
                fitter._resolve_freeze_params({key: None})

    def test_resolve_freeze_params_wrong_planet_warns(self, setup_fitter_two_planets):
        """Freezing a parameter for a planet other than planet_letter warns."""
        fitter = setup_fitter_two_planets
        # Plotting 'b' but freezing K_c (planet c, and free -> no fixed warning)
        with pytest.warns(UserWarning, match="different planet"):
            resolved = fitter._resolve_freeze_params({"K_c": 3.0}, planet_letter="b")
        # Still applied (not banned)
        assert resolved == {"K_c": 3.0}

    def test_resolve_freeze_params_correct_planet_no_warn(self, setup_fitter_two_planets):
        """Freezing a free parameter of the target planet does not warn."""
        fitter = setup_fitter_two_planets
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            fitter._resolve_freeze_params({"K_b": 5.0}, planet_letter="b")

    def test_resolve_freeze_params_no_planet_letter_no_wrong_planet_warn(self, setup_fitter_two_planets):
        """Without planet_letter, no cross-planet warning is emitted."""
        fitter = setup_fitter_two_planets
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            fitter._resolve_freeze_params({"K_c": 3.0})  # K_c free, no planet_letter
        assert not [w for w in record if "different planet" in str(w.message)]

    def test_plot_posterior_phase_freeze_wrong_planet_warns_once(self, setup_fitter_two_planets):
        """Plotting one planet but freezing another's parameter warns exactly once."""
        import matplotlib
        matplotlib.use('Agg')

        fitter = setup_fitter_two_planets
        self._run_short_mcmc(fitter)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")  # worst case: no de-duplication
            # Plot planet 'b' but freeze K_c (planet c, free)
            fitter.plot_posterior_phase('b', discard_start=10, thin=5, freeze_params={"K_c": None})
        wrong_planet = [w for w in record if "different planet" in str(w.message)]
        assert len(wrong_planet) == 1

    def test_resolve_freeze_params_fixed_param_warns(self, setup_fitter_for_rv):
        """Freezing a parameter that is already fixed warns (but is allowed)."""
        fitter = setup_fitter_for_rv
        # P_b is fixed in this fixture; freezing it should warn.
        with pytest.warns(UserWarning, match="already fixed"):
            resolved = fitter._resolve_freeze_params({"P_b": 9.9})
        # ...and still resolve to the requested value (not banned).
        assert resolved == {"P_b": 9.9}

    def test_resolve_freeze_params_free_param_no_warn(self, setup_fitter_for_rv):
        """Freezing a free parameter does not warn."""
        fitter = setup_fitter_for_rv
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            fitter._resolve_freeze_params({"K_b": 5.0})

    def test_plot_posterior_phase_freeze_fixed_warns_once(self, setup_fitter_for_rv):
        """A fixed-parameter freeze warns exactly once across the whole plot."""
        import matplotlib
        matplotlib.use('Agg')

        fitter = setup_fitter_for_rv
        self._run_short_mcmc(fitter)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")  # worst case: no de-duplication
            fitter.plot_posterior_phase('b', discard_start=10, thin=5, freeze_params={"P_b": None, "Tc_b": None})
        fixed_warnings = [w for w in record if "already fixed" in str(w.message)]
        assert len(fixed_warnings) == 1

    def test_calculate_rv_planet_from_samples_freeze_constant(self, setup_fitter_for_rv):
        """Freezing all planet parameters makes every sample's RV identical."""
        fitter = setup_fitter_for_rv
        self._run_short_mcmc(fitter)

        times = np.array([0.0, 0.5, 1.0, 1.5])
        frozen = {"P_b": 2.0, "K_b": 5.0, "e_b": 0.0, "w_b": np.pi / 2, "Tc_b": 0.0}

        with pytest.warns(UserWarning, match="already fixed, not free"):
            rv_samples = fitter.calculate_rv_planet_from_samples('b', times, discard_start=10, thin=5, freeze_params=frozen)

        # With every planet parameter frozen, the planet RV no longer depends on
        # the sample, so all rows are identical and match a single custom calc.
        assert np.allclose(rv_samples, rv_samples[0:1], atol=1e-12)
        params = fitter.build_params_dict(fitter.free_params_values) | frozen
        expected = fitter.calculate_rv_planet_custom('b', times, params)
        np.testing.assert_allclose(rv_samples[0], expected, atol=1e-12)

    def test_calculate_rv_planet_from_samples_freeze_none_matches_median(self, setup_fitter_for_rv):
        """Freezing at None gives the same result as freezing at the median value."""
        fitter = setup_fitter_for_rv
        self._run_short_mcmc(fitter)

        times = np.array([0.0, 0.5, 1.0])
        # K_b is the only free planet parameter in this fixture
        samples_dict = fitter.get_samples_dict(discard_start=10, thin=5)
        k_med = float(np.median(samples_dict["K_b"]))

        rv_none = fitter.calculate_rv_planet_from_samples('b', times, discard_start=10, thin=5, freeze_params={"K_b": None})
        rv_val = fitter.calculate_rv_planet_from_samples('b', times, discard_start=10, thin=5, freeze_params={"K_b": k_med})
        np.testing.assert_allclose(rv_none, rv_val, atol=1e-12)

    def test_calculate_rv_planet_from_samples_no_freeze_unchanged(self, setup_fitter_for_rv):
        """Default (freeze_params=None) is unchanged from the original behaviour."""
        fitter = setup_fitter_for_rv
        self._run_short_mcmc(fitter)

        times = np.array([0.0, 1.0, 2.0])
        rv_default = fitter.calculate_rv_planet_from_samples('b', times, discard_start=10, thin=5)
        rv_none = fitter.calculate_rv_planet_from_samples('b', times, discard_start=10, thin=5, freeze_params=None)
        np.testing.assert_array_equal(rv_default, rv_none)

    def test_plot_posterior_phase_freeze_params(self, setup_fitter_for_rv):
        """plot_posterior_phase runs with freeze_params (median and explicit)."""
        import matplotlib
        matplotlib.use('Agg')

        fitter = setup_fitter_for_rv
        self._run_short_mcmc(fitter)

        # None -> median, and explicit float, both accepted
        with pytest.warns(UserWarning, match="already fixed, not free"):
            fitter.plot_posterior_phase('b', discard_start=10, thin=5, freeze_params={"P_b": None, "Tc_b": None})
        with pytest.warns(UserWarning, match="already fixed, not free"):
            fitter.plot_posterior_phase('b', discard_start=10, thin=5, freeze_params={"P_b": 2.0, "Tc_b": 0.0})

    def test_plot_posterior_phase_freeze_params_unknown_key_raises(self, setup_fitter_for_rv):
        """plot_posterior_phase rejects an unknown freeze_params key."""
        import matplotlib
        matplotlib.use('Agg')

        fitter = setup_fitter_for_rv
        self._run_short_mcmc(fitter)

        with pytest.raises(ValueError, match="Unknown freeze_params key"):
            fitter.plot_posterior_phase('b', discard_start=10, thin=5, freeze_params={"Xyz_b": None})

    def test_calculate_rv_total_from_samples(self, setup_fitter_for_rv):
        """Test calculating total RV from MCMC samples."""
        fitter = setup_fitter_for_rv

        # Run short MCMC
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(map_result, nwalkers=10)
        fitter.run_mcmc(initial_positions, nwalkers=10, max_steps=50, progress=False)

        times = np.array([0.0, 1.0, 2.0])

        # Calculate total RV from samples
        total_samples = fitter.calculate_rv_total_from_samples(times, discard_start=10, thin=5)

        # Should have shape (n_samples, n_times)
        assert total_samples.ndim == 2
        assert total_samples.shape[1] == len(times)
        assert np.all(np.isfinite(total_samples))


# --- GPFitter fixtures and tests ---


@pytest.fixture
def test_gp_data():
    """Simple synthetic RV data with GP-like correlation for testing (single instrument)."""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    vel = np.array([5.0, -2.0, -5.0, 2.0, 3.0, -1.0])
    velerr = np.array([1.0, 1.1, 0.9, 0.85, 1.5, 1.0])
    instrument = np.array(["HARPS", "HARPS", "HARPS", "HARPS", "HARPS", "HARPS"])
    return time, vel, velerr, instrument


@pytest.fixture
def test_gp_circular_params():
    """Simple circular orbit parameters for GP testing (single instrument: HARPS)."""
    return {
        "P_b": Parameter(2.0, fixed=True),
        "K_b": Parameter(5.0, fixed=False),
        "e_b": Parameter(0.0, fixed=True),
        "w_b": Parameter(np.pi/2, fixed=True),
        "Tc_b": Parameter(0.0, fixed=True),
        "g_HARPS": Parameter(0.0, fixed=True),
        "gd": Parameter(0.0, fixed=True),
        "gdd": Parameter(0.0, fixed=True),
        "jit_HARPS": Parameter(1.0, fixed=False),
    }


@pytest.fixture
def test_gp_hyperparams():
    """Simple GP hyperparameters for testing."""
    return {
        "gp_amp": Parameter(1.0, fixed=False),
        "gp_lambda_e": Parameter(50.0, fixed=False),
        "gp_lambda_p": Parameter(0.5, fixed=False),
        "gp_period": Parameter(10.0, fixed=False),
    }


@pytest.fixture
def test_gp_priors():
    """Simple priors for GP testing (single instrument: HARPS)."""
    return {
        "K_b": ravest.prior.Uniform(0, 20),
        "jit_HARPS": ravest.prior.Uniform(0, 5),
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


@pytest.fixture
def test_gp_all_params(test_gp_circular_params, test_gp_hyperparams):
    """Every GPFitter parameter: the circular orbit's, then the QP kernel's."""
    return test_gp_circular_params | test_gp_hyperparams


@pytest.fixture
def test_gp_all_priors(test_gp_priors, test_gp_hyperpriors):
    """A prior for every free parameter in test_gp_all_params, GP ones included."""
    return test_gp_priors | test_gp_hyperpriors


@pytest.fixture
def test_gp_data_multi_instrument():
    """Synthetic RV data with two instruments for GP testing."""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    vel = np.array([5.0, -2.0, -5.0, 102.0, 103.0, 98.0])  # HIRES has +100 offset
    velerr = np.array([1.0, 1.1, 0.9, 0.85, 1.5, 1.2])
    instrument = np.array(["HARPS", "HARPS", "HARPS", "HIRES", "HIRES", "HIRES"])
    return time, vel, velerr, instrument


@pytest.fixture
def test_gp_circular_params_multi_instrument():
    """Circular orbit parameters for two instruments (HARPS and HIRES) with GP."""
    return {
        "P_b": Parameter(2.0, fixed=True),
        "K_b": Parameter(5.0, fixed=False),
        "e_b": Parameter(0.0, fixed=True),
        "w_b": Parameter(np.pi/2, fixed=True),
        "Tc_b": Parameter(0.0, fixed=True),
        "g_HARPS": Parameter(0.0, fixed=False),
        "g_HIRES": Parameter(100.0, fixed=False),
        "gd": Parameter(0.0, fixed=True),
        "gdd": Parameter(0.0, fixed=True),
        "jit_HARPS": Parameter(1.0, fixed=False),
        "jit_HIRES": Parameter(2.0, fixed=False),
    }


class TestGPLogLikelihood:
    """Tests for the GPLogLikelihood class."""

    def test_gploglikelihood_init(self, test_gp_data) -> None:
        """Test GPLogLikelihood initialization."""
        time, vel, velerr, instrument = test_gp_data
        unique_instruments = np.unique(instrument)
        gp_kernel = GPKernel("Quasiperiodic")
        ll = GPLogLikelihood(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        np.testing.assert_array_equal(ll.time, time)
        np.testing.assert_array_equal(ll.vel, vel)
        np.testing.assert_array_equal(ll.velerr, velerr)
        assert ll.t0 == 2.0
        assert ll.gp_kernel == gp_kernel
        np.testing.assert_array_equal(ll.unique_instruments, ["HARPS"])

    def test_gploglikelihood_calculation(self, test_gp_data) -> None:
        """Test GP log-likelihood calculation with valid parameters."""
        time, vel, velerr, instrument = test_gp_data
        unique_instruments = np.unique(instrument)
        gp_kernel = GPKernel("Quasiperiodic")
        ll = GPLogLikelihood(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        params = {
            "P_b": 2.0, "K_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g_HARPS": 0.0, "gd": 0.0, "gdd": 0.0, "jit_HARPS": 2.0
        }
        hyperparams = {
            "gp_amp": 1.0,
            "gp_lambda_e": 50.0,
            "gp_lambda_p": 0.5,
            "gp_period": 10.0,
        }

        log_like = ll(params | hyperparams)
        assert np.isfinite(log_like)
        # JAX returns JAX Array types, so we need to check for those as well
        assert isinstance(log_like, (float, np.floating, jnp.ndarray))

    def test_gploglikelihood_invalid_planet(self, test_gp_data) -> None:
        """Test GP log-likelihood returns -inf for invalid planet parameters."""
        time, vel, velerr, instrument = test_gp_data
        unique_instruments = np.unique(instrument)
        gp_kernel = GPKernel("Quasiperiodic")
        ll = GPLogLikelihood(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        params = {
            "P_b": -1.0,  # Invalid negative period
            "K_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "Tc_b": 0.0,
            "g_HARPS": 0.0, "gd": 0.0, "gdd": 0.0, "jit_HARPS": 1.0
        }
        hyperparams = {
            "gp_amp": 1.0,
            "gp_lambda_e": 50.0,
            "gp_lambda_p": 0.5,
            "gp_period": 10.0,
        }

        log_like = ll(params | hyperparams)
        assert log_like == -np.inf


class TestGPLogPosterior:
    """Tests for the GPLogPosterior class."""

    def test_gplogposterior_init(self, test_gp_data, test_gp_circular_params, test_gp_hyperparams,
                                   test_gp_priors, test_gp_hyperpriors) -> None:
        """Test GPLogPosterior initialization."""
        time, vel, velerr, instrument = test_gp_data
        unique_instruments = np.unique(instrument)
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams
        priors = test_gp_priors
        hyperpriors = test_gp_hyperpriors
        gp_kernel = GPKernel("Quasiperiodic")

        # Extract fixed and free params (GP hyperparameters included)
        all_params = params | hyperparams
        fixed_params = {k: v.value for k, v in all_params.items() if v.fixed}
        free_params_names = [k for k, v in all_params.items() if not v.fixed]

        lpost = GPLogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel,
            priors=priors | hyperpriors,
            fixed_params=fixed_params,
            free_params_names=free_params_names,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        assert lpost.planet_letters == ["b"]
        assert lpost.gp_kernel == gp_kernel
        np.testing.assert_array_equal(lpost.unique_instruments, ["HARPS"])

    def test_gplogposterior_valid_calculation(self, test_gp_data, test_gp_circular_params, test_gp_hyperparams,
                                               test_gp_priors, test_gp_hyperpriors) -> None:
        """Test GP log-posterior calculation with valid parameters."""
        time, vel, velerr, instrument = test_gp_data
        unique_instruments = np.unique(instrument)
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams
        priors = test_gp_priors
        hyperpriors = test_gp_hyperpriors
        gp_kernel = GPKernel("Quasiperiodic")

        all_params = params | hyperparams
        fixed_params = {k: v.value for k, v in all_params.items() if v.fixed}
        free_params_names = [k for k, v in all_params.items() if not v.fixed]

        lpost = GPLogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel,
            priors=priors | hyperpriors,
            fixed_params=fixed_params,
            free_params_names=free_params_names,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        combined_dict = {
            "K_b": 5.0, "jit_HARPS": 1.0,
            "gp_amp": 1.0, "gp_lambda_e": 50.0, "gp_lambda_p": 0.5, "gp_period": 10.0
        }
        log_post = lpost.log_probability(combined_dict)

        assert np.isfinite(log_post)
        # JAX returns JAX Array types, so we need to check for those as well
        assert isinstance(log_post, (float, np.floating, jnp.ndarray))

    def test_gplogposterior_invalid_prior(self, test_gp_data, test_gp_circular_params, test_gp_hyperparams,
                                           test_gp_priors, test_gp_hyperpriors) -> None:
        """Test GP log-posterior returns -inf when prior is invalid."""
        time, vel, velerr, instrument = test_gp_data
        unique_instruments = np.unique(instrument)
        params = test_gp_circular_params
        hyperparams = test_gp_hyperparams
        priors = test_gp_priors
        hyperpriors = test_gp_hyperpriors
        gp_kernel = GPKernel("Quasiperiodic")

        all_params = params | hyperparams
        fixed_params = {k: v.value for k, v in all_params.items() if v.fixed}
        free_params_names = [k for k, v in all_params.items() if not v.fixed]

        lpost = GPLogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=gp_kernel,
            priors=priors | hyperpriors,
            fixed_params=fixed_params,
            free_params_names=free_params_names,
            time=time, vel=vel, velerr=velerr,
            instrument=instrument, unique_instruments=unique_instruments, t0=2.0
        )

        combined_dict = {
            "K_b": -1.0, "jit_HARPS": 1.0,  # Invalid K_b outside prior bounds
            "gp_amp": 1.0, "gp_lambda_e": 50.0, "gp_lambda_p": 0.5, "gp_period": 10.0
        }
        log_post = lpost.log_probability(combined_dict)

        assert log_post == -np.inf


class TestGPOneDictInternals:
    """GPLogPosterior and GPLogLikelihood take one dict, GP hyperparameters included.

    Reference values were computed with the earlier two-dict (params + hyperparams) API at the
    same points, so these also check that merging the dicts changes no number.
    """

    TIME = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    VEL = np.array([5.0, -2.0, -5.0, 2.0, 3.0, -1.0])
    VELERR = np.array([1.0, 1.1, 0.9, 0.85, 1.5, 1.0])
    INSTRUMENT = np.array(["HARPS"] * 6)

    @classmethod
    def _data(cls):
        return dict(time=cls.TIME, vel=cls.VEL, velerr=cls.VELERR,
                    instrument=cls.INSTRUMENT, unique_instruments=np.array(["HARPS"]), t0=2.0)

    def _posterior(self, case):
        """One-dict GPLogPosterior for case "A" or "B".

        A: P K e w Tc, every hyperparameter free. B: P K secosw sesinw Tc with priors on e_b,
        w_b (log-Jacobian correction applies), gp_lambda_p fixed.
        """
        U = ravest.prior.Uniform
        if case == "A":
            return GPLogPosterior(
                planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
                gp_kernel=GPKernel("Quasiperiodic"),
                priors={"K_b": U(0, 20), "jit_HARPS": U(0, 5), "gp_amp": U(0, 10),
                        "gp_lambda_e": U(1, 100), "gp_lambda_p": U(0.1, 2.0), "gp_period": U(1, 50)},
                fixed_params={"P_b": 2.0, "e_b": 0.0, "w_b": np.pi / 2, "Tc_b": 0.0,
                              "g_HARPS": 0.0, "gd": 0.0, "gdd": 0.0},
                free_params_names=["K_b", "jit_HARPS", "gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"],
                **self._data(),
            )
        return GPLogPosterior(
            planet_letters=["b"], parameterisation=Parameterisation("P K secosw sesinw Tc"),
            gp_kernel=GPKernel("Quasiperiodic"),
            priors={"K_b": U(0, 20), "e_b": U(0, 1), "w_b": U(-np.pi, np.pi), "jit_HARPS": U(0, 5),
                    "gp_amp": U(0, 10), "gp_lambda_e": U(1, 100), "gp_period": U(1, 50)},
            fixed_params={"P_b": 2.0, "Tc_b": 0.0, "g_HARPS": 0.0, "gd": 0.0, "gdd": 0.0,
                          "gp_lambda_p": 0.5},
            free_params_names=["K_b", "secosw_b", "sesinw_b", "jit_HARPS",
                               "gp_amp", "gp_lambda_e", "gp_period"],
            **self._data(),
        )

    POINTS = {
        "A": {"K_b": 5.0, "jit_HARPS": 1.0,
              "gp_amp": 1.0, "gp_lambda_e": 50.0, "gp_lambda_p": 0.5, "gp_period": 10.0},
        "B": {"K_b": 5.0, "secosw_b": 0.2, "sesinw_b": -0.1, "jit_HARPS": 1.0,
              "gp_amp": 1.0, "gp_lambda_e": 50.0, "gp_period": 10.0},
    }
    REFERENCE = {"A": -39.71988723240225, "B": -40.312193275974195}

    def test_likelihood_one_dict_matches_reference(self) -> None:
        """GPLogLikelihood(params) with the GP names in params gives the two-dict value."""
        ll = GPLogLikelihood(planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
                             gp_kernel=GPKernel("Quasiperiodic"), **self._data())
        params = {"P_b": 2.0, "K_b": 5.0, "e_b": 0.1, "w_b": 1.0, "Tc_b": 0.3,
                  "g_HARPS": 0.5, "gd": 0.1, "gdd": 0.01, "jit_HARPS": 2.0,
                  "gp_amp": 1.5, "gp_lambda_e": 30.0, "gp_lambda_p": 0.7, "gp_period": 8.0}

        assert float(ll(params)) == pytest.approx(-25.523497814852725, rel=1e-12)

    def test_likelihood_rejects_separate_hyperparams(self) -> None:
        """There is no separate hyperparams argument."""
        ll = GPLogLikelihood(planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
                             gp_kernel=GPKernel("Quasiperiodic"), **self._data())

        with pytest.raises(TypeError):
            ll({"P_b": 2.0}, {"gp_amp": 1.0})

    @pytest.mark.parametrize("case", ["A", "B"])
    def test_posterior_one_dict_matches_reference(self, case) -> None:
        """log_probability gives the two-dict value, incl. prior conversion to e, w (case B)."""
        lp = self._posterior(case)

        assert float(lp.log_probability(self.POINTS[case])) == pytest.approx(
            self.REFERENCE[case], rel=1e-12)

    def test_map_objective_takes_one_list(self) -> None:
        """The MAP objective reads one list in free_params_names order."""
        lp = self._posterior("A")
        values = [self.POINTS["A"][name] for name in lp.free_params_names]

        assert lp._negative_log_probability_for_MAP(values) == pytest.approx(
            -self.REFERENCE["A"], rel=1e-12)

    @pytest.mark.parametrize("name, value", [("gp_period", 60.0), ("gp_amp", -1.0)],
                             ids=["outside_prior", "unphysical"])
    def test_posterior_minus_inf_for_bad_gp_value(self, name, value) -> None:
        """A GP value outside its prior, or unphysical for the kernel, gives -inf."""
        lp = self._posterior("A")

        assert lp.log_probability(self.POINTS["A"] | {name: value}) == -np.inf

    @pytest.mark.parametrize("keyword", ["hyperpriors", "fixed_hyperparams", "free_hyperparams_names"])
    def test_posterior_has_no_hyper_keywords(self, keyword) -> None:
        """The separate hyperparameter arguments are gone."""
        lp = self._posterior("A")
        kwargs = dict(planet_letters=lp.planet_letters, parameterisation=lp.parameterisation,
                      gp_kernel=lp.gp_kernel, priors=lp.priors, fixed_params=lp.fixed_params,
                      free_params_names=lp.free_params_names, **self._data())
        kwargs[keyword] = {} if keyword != "free_hyperparams_names" else []

        with pytest.raises(TypeError):
            GPLogPosterior(**kwargs)

    def test_posterior_has_no_log_hyperprior(self) -> None:
        """One LogPrior covers the GP names too."""
        assert not hasattr(self._posterior("A"), "log_hyperprior")


class TestLogProbSignatures:
    """The posterior and likelihood classes take keyword-only arguments in one order.

    Model (planet_letters, parameterisation, gp_kernel), then fit setup (priors, fixed_params,
    free_params_names), then data (time, vel, velerr, instrument, unique_instruments, t0). Each
    class takes only what it needs.
    """

    MODEL = ["planet_letters", "parameterisation"]
    SETUP = ["priors", "fixed_params", "free_params_names"]
    DATA = ["time", "vel", "velerr", "instrument", "unique_instruments", "t0"]
    ORDER = {
        LogPosterior: MODEL + SETUP + DATA,
        GPLogPosterior: MODEL + ["gp_kernel"] + SETUP + DATA,
        LogLikelihood: MODEL + DATA,
        GPLogLikelihood: MODEL + ["gp_kernel"] + DATA,
    }
    CLASSES = list(ORDER)
    IDS = [cls.__name__ for cls in CLASSES]

    @classmethod
    def _kwargs(cls, klass):
        """Valid arguments for klass, in the agreed order."""
        values = dict(
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc"),
            gp_kernel=GPKernel("Quasiperiodic"),
            priors={"K_b": ravest.prior.Uniform(0, 20)},
            fixed_params={"P_b": 2.0, "e_b": 0.0, "w_b": np.pi / 2, "Tc_b": 0.0,
                          "g_HARPS": 0.0, "jit_HARPS": 0.0, "gd": 0.0, "gdd": 0.0},
            free_params_names=["K_b"],
            time=np.array([0.0, 1.0, 2.0]), vel=np.array([1.0, -1.0, 0.5]),
            velerr=np.array([1.0, 1.0, 1.0]), instrument=np.array(["HARPS"] * 3),
            unique_instruments=np.array(["HARPS"]), t0=1.0,
        )
        return {name: values[name] for name in cls.ORDER[klass]}

    @staticmethod
    def _parameters(klass):
        """inspect.Parameter objects of klass.__init__, without self."""
        import inspect
        return list(inspect.signature(klass.__init__).parameters.values())[1:]

    @pytest.mark.parametrize("klass", CLASSES, ids=IDS)
    def test_keyword_only_in_order(self, klass) -> None:
        """Every argument is keyword-only, in the agreed order."""
        import inspect
        parameters = self._parameters(klass)

        assert [p.name for p in parameters] == self.ORDER[klass]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters)

    @pytest.mark.parametrize("klass", CLASSES, ids=IDS)
    def test_positional_call_raises(self, klass) -> None:
        """Passing the arguments positionally is refused, even in the right order."""
        kwargs = self._kwargs(klass)

        with pytest.raises(TypeError, match="takes 1 positional argument"):
            klass(*kwargs.values())

    @pytest.mark.parametrize("klass", CLASSES, ids=IDS)
    def test_keyword_call_works(self, klass) -> None:
        """Passing every argument by keyword constructs the object."""
        klass(**self._kwargs(klass))

    @pytest.mark.parametrize("gp_class, partner", [(GPLogPosterior, LogPosterior),
                                                   (GPLogLikelihood, LogLikelihood)],
                             ids=["GPLogPosterior", "GPLogLikelihood"])
    def test_gp_signature_is_partner_plus_gp_kernel(self, gp_class, partner) -> None:
        """The GP class takes its partner's arguments (same kinds and annotations), plus gp_kernel."""
        gp_parameters = self._parameters(gp_class)
        partner_parameters = self._parameters(partner)

        assert [p.name for p in gp_parameters if p.name != "gp_kernel"] == [p.name for p in partner_parameters]
        assert [p for p in gp_parameters if p.name != "gp_kernel"] == partner_parameters


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

    def test_gpfitter_init_rejects_string_parameterisation(self) -> None:
        """Passing the parameterisation name as a string (not a Parameterisation) raises."""
        gp_kernel = GPKernel("Quasiperiodic")
        with pytest.raises(TypeError, match="parameterisation must be a Parameterisation object"):
            GPFitter(["b"], "P K e w Tc", gp_kernel)

    def test_add_data_valid(self, test_gp_data) -> None:
        """Test adding valid data to GPFitter."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        np.testing.assert_array_equal(fitter.time, time)
        np.testing.assert_array_equal(fitter.vel, vel)
        np.testing.assert_array_equal(fitter.velerr, velerr)
        np.testing.assert_array_equal(fitter.instrument, instrument)
        assert fitter.unique_instruments == ["HARPS"]
        assert fitter.t0 == 2.0

    def test_add_data_mismatched_lengths(self) -> None:
        """Test error when data arrays have different lengths."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        time = np.array([0.0, 1.0])
        vel = np.array([5.0, -2.0, -5.0])  # Different length
        velerr = np.array([1.0, 1.0])
        instrument = np.array(["HARPS", "HARPS"])

        with pytest.raises(ValueError, match="arrays must be the same length"):
            fitter.add_data(time, vel, velerr, instrument, t0=2.0)

    @staticmethod
    def _fitter(data):
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
        time, vel, velerr, instrument = data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        return fitter

    def test_params_property_valid(self, test_gp_data, test_gp_all_params) -> None:
        """`params` holds every parameter, the GP hyperparameters last in the kernel's order."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params

        assert len(fitter.params) == 13  # 5 planetary + g_HARPS + jit_HARPS + 2 trend + 4 GP
        assert list(fitter.params)[-4:] == ["gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]

    def test_params_missing_gp_names(self, test_gp_data, test_gp_circular_params) -> None:
        """The first params assignment must include the GP hyperparameters too."""
        fitter = self._fitter(test_gp_data)
        params = test_gp_circular_params | {
            "gp_amp": Parameter(1.0, fixed=False),
            "gp_lambda_e": Parameter(50.0, fixed=False),
        }

        with pytest.raises(ValueError, match="Missing required parameters.*gp_"):
            fitter.params = params

    def test_params_unexpected_gp_name(self, test_gp_data, test_gp_all_params) -> None:
        """A GP name the kernel does not have is rejected like any unexpected parameter."""
        fitter = self._fitter(test_gp_data)
        params = test_gp_all_params | {"gp_scale": Parameter(1.0, fixed=False)}

        with pytest.raises(ValueError, match="Unexpected parameters.*gp_scale"):
            fitter.params = params

    @pytest.mark.parametrize("name, value", [("gp_amp", 0.0), ("gp_lambda_e", -1.0), ("gp_period", np.inf)])
    def test_params_invalid_gp_value(self, test_gp_data, test_gp_all_params, name, value) -> None:
        """GP values are checked on assignment: finite and, for the QP kernel, positive."""
        fitter = self._fitter(test_gp_data)
        params = test_gp_all_params | {name: Parameter(value, fixed=False)}

        with pytest.raises(ValueError, match=name):
            fitter.params = params

    def test_params_partial_update_of_gp_value(self, test_gp_data, test_gp_all_params) -> None:
        """After the first full assignment, a GP value can be updated on its own."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params
        fitter.params = {"gp_period": Parameter(12.0, fixed=True)}

        assert fitter.params["gp_period"].value == 12.0
        assert "gp_period" in fitter.fixed_params_names

    def test_add_priors_valid(self, test_gp_data, test_gp_all_params, test_gp_all_priors) -> None:
        """`priors` holds a prior for every free parameter, the GP ones included."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params
        fitter.priors = test_gp_all_priors

        assert list(fitter.priors) == ["K_b", "jit_HARPS", "gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]

    def test_add_priors_missing_gp_prior(self, test_gp_data, test_gp_all_params, test_gp_all_priors) -> None:
        """A free GP hyperparameter without a prior is reported like any other."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params
        priors = dict(test_gp_all_priors)
        del priors["gp_period"]

        with pytest.raises(ValueError, match="Missing priors for parameters.*gp_period"):
            fitter.priors = priors

    def test_add_priors_for_fixed_gp_param_rejected(self, test_gp_data, test_gp_all_params,
                                                    test_gp_all_priors) -> None:
        """A prior on a fixed GP hyperparameter is unexpected, as for any fixed parameter."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params | {"gp_lambda_p": Parameter(0.5, fixed=True)}

        with pytest.raises(ValueError, match="Unexpected priors.*gp_lambda_p"):
            fitter.priors = test_gp_all_priors

    def test_add_priors_gp_initial_value_outside_prior(self, test_gp_data, test_gp_all_params,
                                                       test_gp_all_priors) -> None:
        """A GP starting value outside its prior is rejected when the priors are set."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params | {"gp_period": Parameter(60.0, fixed=False)}

        with pytest.raises(ValueError, match="Initial value 60.0 of parameter gp_period is invalid"):
            fitter.priors = test_gp_all_priors

    def test_get_free_params(self, test_gp_data, test_gp_all_params) -> None:
        """free_params_* include the free GP hyperparameters: they are the chain's columns."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params

        assert fitter.free_params_names == ["K_b", "jit_HARPS", "gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]
        assert fitter.free_params_values == [5.0, 1.0, 1.0, 50.0, 0.5, 10.0]
        assert list(fitter.free_params_dict) == fitter.free_params_names
        assert fitter.ndim == 6

    @pytest.mark.parametrize("name", [
        "hyperparams", "hyperpriors",
        "free_hyperparams_dict", "free_hyperparams_names", "free_hyperparams_values",
        "fixed_hyperparams_dict", "fixed_hyperparams_names", "fixed_hyperparams_values",
        "fixed_hyperparams_values_dict",
    ])
    def test_hyperparams_family_removed(self, test_gp_data, test_gp_all_params, name) -> None:
        """The separate hyperparameter accessors are gone."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params

        assert not hasattr(fitter, name)

    def test_add_data_multi_instrument(self, test_gp_data_multi_instrument) -> None:
        """Test adding data with multiple instruments."""
        fitter = self._fitter(test_gp_data_multi_instrument)

        np.testing.assert_array_equal(fitter.instrument, test_gp_data_multi_instrument[3])
        assert set(fitter.unique_instruments) == {"HARPS", "HIRES"}

    def test_add_params_wrong_count(self, test_gp_data) -> None:
        """Test error when wrong number of parameters provided."""
        fitter = self._fitter(test_gp_data)
        params = {"P_b": Parameter(2.0, fixed=False)}  # Too few params

        with pytest.raises(ValueError, match="Missing required parameters.*Expected 13 parameters, got 1"):
            fitter.params = params

    def test_add_params_missing_planetary_param(self, test_gp_data, test_gp_all_params) -> None:
        """Test error when planetary parameter is missing."""
        fitter = self._fitter(test_gp_data)
        params = dict(test_gp_all_params)
        del params["P_b"]

        with pytest.raises(ValueError, match="Missing required parameters.*Expected 13 parameters, got 12"):
            fitter.params = params

    def test_add_params_unexpected_param(self, test_gp_data, test_gp_all_params) -> None:
        """Test error when unexpected parameter is provided."""
        fitter = self._fitter(test_gp_data)
        params = test_gp_all_params | {"invalid_param": Parameter(1.0, fixed=False)}

        with pytest.raises(ValueError, match="Unexpected parameters.*Expected 13 parameters, got 14"):
            fitter.params = params

    def test_add_priors_missing_prior(self, test_gp_data, test_gp_all_params, test_gp_hyperpriors) -> None:
        """Test error when prior is missing for free parameter."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params
        priors = {"K_b": ravest.prior.Uniform(0, 20)} | test_gp_hyperpriors  # Missing jit_HARPS prior

        with pytest.raises(ValueError, match="Missing priors for parameters.*jit_HARPS"):
            fitter.priors = priors

    def test_add_priors_invalid_initial_value(self, test_gp_data, test_gp_all_params, test_gp_all_priors) -> None:
        """Test error when initial parameter value is outside prior bounds."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params | {"K_b": Parameter(25.0, fixed=False)}  # Outside uniform prior [0, 20]

        with pytest.raises(ValueError, match="Initial value 25.0 of parameter K_b is invalid"):
            fitter.priors = test_gp_all_priors

    def test_add_priors_too_many_warning(self, test_gp_data, test_gp_all_params, test_gp_all_priors) -> None:
        """Test error when priors provided for fixed params."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params
        priors = test_gp_all_priors | {"P_b": ravest.prior.Uniform(1, 5)}  # P_b is fixed!

        with pytest.raises(ValueError, match="Unexpected priors.*P_b"):
            fitter.priors = priors

    def test_get_fixed_params(self, test_gp_data, test_gp_all_params) -> None:
        """Test getting fixed parameters."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params

        assert fitter.fixed_params_names == ["P_b", "e_b", "w_b", "Tc_b", "g_HARPS", "gd", "gdd"]
        assert len(fitter.fixed_params_values) == 7

    def test_params_all_fixed_warns(self, test_gp_data, test_gp_all_params) -> None:
        """Setting every parameter, GP ones included, as fixed issues a UserWarning."""
        fitter = self._fitter(test_gp_data)
        params = {k: Parameter(v.value, fixed=True) for k, v in test_gp_all_params.items()}

        with pytest.warns(UserWarning, match="All parameters are fixed"):
            fitter.params = params

    def test_free_gp_param_alone_does_not_warn(self, test_gp_data, test_gp_all_params) -> None:
        """One free GP hyperparameter is enough to sample, so there is no all-fixed warning."""
        fitter = self._fitter(test_gp_data)
        params = {k: Parameter(v.value, fixed=k != "gp_amp") for k, v in test_gp_all_params.items()}

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitter.params = params

        assert fitter.free_params_names == ["gp_amp"]

    @staticmethod
    def _all_fixed_fitter(data, all_params):
        fitter = TestGPFitter._fitter(data)
        with pytest.warns(UserWarning):
            fitter.params = {k: Parameter(v.value, fixed=True) for k, v in all_params.items()}
        return fitter

    def test_find_map_estimate_all_fixed_raises(self, test_gp_data, test_gp_all_params) -> None:
        """Test that find_map_estimate raises a clear error when all parameters are fixed.

        scipy.minimize cannot handle a zero-dimensional parameter space and produces
        a cryptic _MaxFuncCallError. We guard against this with an explicit ValueError.
        """
        fitter = self._all_fixed_fitter(test_gp_data, test_gp_all_params)

        with pytest.raises(ValueError, match="no free parameters to optimise"):
            fitter.find_map_estimate()

    def test_generate_walker_positions_random_all_fixed_raises(self, test_gp_data, test_gp_all_params) -> None:
        """Test that generate_initial_walker_positions_random raises when all parameters are fixed."""
        fitter = self._all_fixed_fitter(test_gp_data, test_gp_all_params)

        with pytest.raises(ValueError, match="no free parameters to sample"):
            fitter.generate_initial_walker_positions_random(nwalkers=10)

    def test_run_mcmc_all_fixed_raises(self, test_gp_data, test_gp_all_params) -> None:
        """Test that run_mcmc raises a clear error when all parameters are fixed."""
        fitter = self._all_fixed_fitter(test_gp_data, test_gp_all_params)

        dummy_positions = np.empty((10, 0))
        with pytest.raises(ValueError, match="no free parameters to sample"):
            fitter.run_mcmc(dummy_positions, nwalkers=10, max_steps=10, progress=False)


class TestGPFitterMCMC:
    """Tests for GPFitter MCMC functionality."""

    @pytest.fixture
    def setup_gpfitter(self, test_gp_data, test_gp_all_params, test_gp_all_priors):
        """Setup a fully configured GPFitter for MCMC tests."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = test_gp_all_params
        fitter.priors = test_gp_all_priors

        nwalkers = 14  # >= 2 * ndim (ndim=6: 2 free params + 4 free GP hyperparameters)
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(
            map_result, nwalkers=nwalkers
        )
        return fitter, initial_positions, nwalkers

    def test_fixed_length_mode(self, setup_gpfitter):
        """Test that fixed-length mode runs for exactly max_steps."""
        fitter, initial_positions, nwalkers = setup_gpfitter
        max_steps = 50

        fitter.run_mcmc(
            initial_positions,
            nwalkers=nwalkers,
            max_steps=max_steps,
            progress=False,
            check_convergence=False
        )

        assert fitter.sampler is not None
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape[0] == max_steps

    def test_adaptive_mode_runs(self, setup_gpfitter):
        """Test that adaptive convergence mode runs without errors."""
        fitter, initial_positions, nwalkers = setup_gpfitter

        fitter.run_mcmc(
            initial_positions,
            nwalkers=nwalkers,
            max_steps=100,
            progress=False,
            check_convergence=True,
            convergence_check_interval=50,
            convergence_check_start=20
        )

        assert fitter.sampler is not None
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape[0] > 0
        assert chain.shape[0] <= 100

    def test_backward_compatibility_positional_args(self, setup_gpfitter):
        """Test backward compatibility with positional arguments."""
        fitter, initial_positions, nwalkers = setup_gpfitter

        # Old style: run_mcmc(initial_positions, nwalkers, nsteps, progress)
        fitter.run_mcmc(initial_positions, nwalkers, 50, False)

        assert fitter.sampler is not None
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape[0] == 50

    def test_sample_retrieval_np(self, setup_gpfitter):
        """Test get_samples_np returns correct shape."""
        fitter, initial_positions, nwalkers = setup_gpfitter
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=50, progress=False)

        # Flat=False: (nsteps, nwalkers, ndim)
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape == (50, nwalkers, fitter.ndim)

        # Flat=True: (nsteps*nwalkers, ndim)
        flat_chain = fitter.get_samples_np(flat=True)
        assert flat_chain.shape == (50 * nwalkers, fitter.ndim)

    def test_sample_retrieval_df(self, setup_gpfitter):
        """Test get_samples_df returns correct columns."""
        fitter, initial_positions, nwalkers = setup_gpfitter
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=50, progress=False)

        df = fitter.get_samples_df()
        assert list(df.columns) == fitter.free_params_names
        assert fitter.free_params_names == ["K_b", "jit_HARPS", "gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]
        assert len(df) == 50 * nwalkers

    def test_sample_retrieval_dict(self, setup_gpfitter):
        """Test get_samples_dict returns correct keys."""
        fitter, initial_positions, nwalkers = setup_gpfitter
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=50, progress=False)

        samples_dict = fitter.get_samples_dict()
        assert list(samples_dict) == fitter.free_params_names
        for v in samples_dict.values():
            assert len(v) == 50 * nwalkers

    def test_get_mcmc_posterior_dict(self, setup_gpfitter):
        """Test posterior dict includes fixed + free params, GP hyperparameters among them."""
        fitter, initial_positions, nwalkers = setup_gpfitter
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=50, progress=False)

        posterior = fitter.get_mcmc_posterior_dict()

        # Should contain every parameter
        assert set(posterior.keys()) == set(fitter.params)

        # Fixed params should be floats, free should be arrays
        for name in fitter.fixed_params_names:
            assert isinstance(posterior[name], (float, np.floating))
        for name in fitter.free_params_names:
            assert isinstance(posterior[name], np.ndarray)


class TestGPRVCalculations:
    """Tests for GPFitter RV calculation methods."""

    @pytest.fixture
    def setup_gpfitter_for_rv(self, test_gp_data, test_gp_all_params, test_gp_all_priors):
        """Setup GPFitter with data/params/priors for RV calculations."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = test_gp_all_params
        fitter.priors = test_gp_all_priors
        return fitter

    def test_calculate_rv_planet_custom(self, setup_gpfitter_for_rv):
        """Test custom planet RV against hand-calculated circular orbit values.

        With e=0, w=pi/2, P=2, K=5, Tc=0:
        RV = K * cos(2*pi*(t - Tc)/P + w)
           = 5 * cos(pi*t + pi/2)
           = -5 * sin(pi*t)
        """
        fitter = setup_gpfitter_for_rv
        times = np.array([0.25, 0.5, 0.75, 1.25])

        params = fitter.build_params_dict(fitter.free_params_values)
        rv = fitter.calculate_rv_planet_custom('b', times, params)

        expected = -5.0 * np.sin(np.pi * times)
        np.testing.assert_allclose(rv, expected, atol=1e-10)

    def test_calculate_rv_trend_custom_zero(self, setup_gpfitter_for_rv):
        """Test trend RV is zero when gd=0 and gdd=0."""
        fitter = setup_gpfitter_for_rv
        times = np.array([0.0, 1.0, 2.0, 3.0])

        params = fitter.build_params_dict(fitter.free_params_values)
        rv_trend = fitter.calculate_rv_trend_custom(times, params)

        np.testing.assert_allclose(rv_trend, 0.0, atol=1e-15)

    def test_calculate_rv_trend_custom_nonzero(self, test_gp_data, test_gp_hyperparams,
                                               test_gp_hyperpriors):
        """Test trend RV with nonzero gd."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = {
            "P_b": Parameter(2.0, fixed=True),
            "K_b": Parameter(5.0, fixed=False),
            "e_b": Parameter(0.0, fixed=True),
            "w_b": Parameter(np.pi/2, fixed=True),
            "Tc_b": Parameter(0.0, fixed=True),
            "g_HARPS": Parameter(0.0, fixed=True),
            "gd": Parameter(0.5, fixed=True),
            "gdd": Parameter(0.0, fixed=True),
            "jit_HARPS": Parameter(1.0, fixed=False),
        }
        fitter.params = params | test_gp_hyperparams
        fitter.priors = {"K_b": ravest.prior.Uniform(0, 20), "jit_HARPS": ravest.prior.Uniform(0, 5)} | test_gp_hyperpriors

        times = np.array([0.0, 1.0, 2.0, 3.0])
        params_dict = fitter.build_params_dict(fitter.free_params_values)
        rv_trend = fitter.calculate_rv_trend_custom(times, params_dict)

        # trend(t) = gd*(t - t0) + gdd*(t - t0)^2, with gd=0.5, gdd=0.0, t0=2.0
        expected_trend = 0.5 * (times - 2.0)
        np.testing.assert_allclose(rv_trend, expected_trend)

    def test_calculate_rv_total_custom(self, setup_gpfitter_for_rv):
        """Test total RV = planet + trend + GP."""
        fitter = setup_gpfitter_for_rv
        times = np.array([0.25, 0.5, 0.75, 1.25])

        params = fitter.build_params_dict(fitter.free_params_values)

        rv_total = fitter.calculate_rv_total_custom(times, params)
        rv_planet = fitter.calculate_rv_planet_custom('b', times, params)
        rv_trend = fitter.calculate_rv_trend_custom(times, params)
        rv_gp = fitter.calculate_rv_gp_custom(times, params)

        np.testing.assert_allclose(rv_total, rv_planet + rv_trend + rv_gp)

    def test_calculate_rv_gp_custom(self, setup_gpfitter_for_rv):
        """Test GP contribution is non-zero and finite."""
        fitter = setup_gpfitter_for_rv
        times = np.array([0.25, 0.5, 0.75, 1.25])

        params = fitter.build_params_dict(fitter.free_params_values)
        rv_gp = fitter.calculate_rv_gp_custom(times, params)

        assert isinstance(rv_gp, np.ndarray)
        assert len(rv_gp) == len(times)
        assert np.all(np.isfinite(rv_gp))

    def test_build_params_dict_from_array(self, setup_gpfitter_for_rv):
        """Test building params dict from array includes every parameter, GP ones too."""
        fitter = setup_gpfitter_for_rv

        params = fitter.build_params_dict(fitter.free_params_values)

        assert isinstance(params, dict)
        # 9 non-GP params + 4 GP hyperparameters
        assert len(params) == 13
        assert "P_b" in params
        assert "K_b" in params
        assert "jit_HARPS" in params
        assert "gp_amp" in params
        assert "gp_period" in params

    def test_build_params_dict_from_dict(self, setup_gpfitter_for_rv):
        """Test building params dict from dict input."""
        fitter = setup_gpfitter_for_rv

        free_values = {k: v.value for k, v in fitter.free_params_dict.items()}

        params = fitter.build_params_dict(free_values)

        assert isinstance(params, dict)
        assert len(params) == 13

    def test_calculate_rv_planet_from_samples(self, setup_gpfitter_for_rv):
        """Test calculating planet RV from MCMC samples."""
        fitter = setup_gpfitter_for_rv
        nwalkers = 14
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(
            map_result, nwalkers=nwalkers
        )
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=50, progress=False)

        times = np.array([0.0, 1.0, 2.0])
        rv_samples = fitter.calculate_rv_planet_from_samples(
            'b', times, discard_start=10, thin=5, progress=False
        )

        assert rv_samples.ndim == 2
        assert rv_samples.shape[1] == len(times)
        assert np.all(np.isfinite(rv_samples))

    def test_calculate_rv_total_from_samples(self, setup_gpfitter_for_rv):
        """Test calculating total RV from MCMC samples."""
        fitter = setup_gpfitter_for_rv
        nwalkers = 14
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(
            map_result, nwalkers=nwalkers
        )
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=50, progress=False)

        times = np.array([0.0, 1.0, 2.0])
        total_samples = fitter.calculate_rv_total_from_samples(
            times, discard_start=10, thin=5, progress=False
        )

        assert total_samples.ndim == 2
        assert total_samples.shape[1] == len(times)
        assert np.all(np.isfinite(total_samples))

    def test_calculate_rv_gp_from_samples(self, setup_gpfitter_for_rv):
        """Test calculating GP RV from MCMC samples."""
        fitter = setup_gpfitter_for_rv
        nwalkers = 14
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(
            map_result, nwalkers=nwalkers
        )
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=50, progress=False)

        times = np.array([0.0, 1.0, 2.0])
        gp_samples = fitter.calculate_rv_gp_from_samples(
            times, discard_start=10, thin=5, progress=False
        )

        assert gp_samples.ndim == 2
        assert gp_samples.shape[1] == len(times)
        assert np.all(np.isfinite(gp_samples))

    def _run_short_gp_mcmc(self, fitter):
        """Run a short MCMC on the GPFitter (helper for freeze_params tests)."""
        nwalkers = 14
        map_result = fitter.find_map_estimate()
        initial_positions = fitter.generate_initial_walker_positions_from_map(map_result, nwalkers=nwalkers)
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=50, progress=False)

    def test_resolve_freeze_params_none(self, setup_gpfitter_for_rv):
        """None passes straight through as None (no freezing)."""
        fitter = setup_gpfitter_for_rv
        assert fitter._resolve_freeze_params(None) is None

    def test_resolve_freeze_params_unknown_key_raises(self, setup_gpfitter_for_rv):
        """An unrecognised key raises ValueError."""
        fitter = setup_gpfitter_for_rv
        with pytest.raises(ValueError, match="Unknown freeze_params key"):
            fitter._resolve_freeze_params({"P_c": None})

    def test_resolve_freeze_params_rejects_non_planet_params(self, setup_gpfitter_for_rv):
        """Trend, instrument and GP hyperparameters cannot be frozen."""
        fitter = setup_gpfitter_for_rv
        for key in ("jit_HARPS", "g_HARPS", "gd", "gp_amp", "gp_period"):
            with pytest.raises(ValueError, match="Unknown freeze_params key"):
                fitter._resolve_freeze_params({key: None})

    def test_resolve_freeze_params_fixed_param_warns(self, setup_gpfitter_for_rv):
        """Freezing a parameter that is already fixed warns (but is allowed)."""
        fitter = setup_gpfitter_for_rv
        with pytest.warns(UserWarning, match="already fixed"):
            resolved = fitter._resolve_freeze_params({"P_b": 9.9})
        assert resolved == {"P_b": 9.9}

    def test_calculate_rv_planet_from_samples_freeze_constant(self, setup_gpfitter_for_rv):
        """Freezing all planet parameters makes every sample's RV identical."""
        fitter = setup_gpfitter_for_rv
        self._run_short_gp_mcmc(fitter)

        times = np.array([0.0, 0.5, 1.0, 1.5])
        frozen = {"P_b": 2.0, "K_b": 5.0, "e_b": 0.0, "w_b": np.pi / 2, "Tc_b": 0.0}

        with pytest.warns(UserWarning, match="already fixed, not free"):
            rv_samples = fitter.calculate_rv_planet_from_samples('b', times, discard_start=10, thin=5, progress=False, freeze_params=frozen)

        # With every planet parameter frozen, the planet RV no longer depends on
        # the sample, so all rows are identical and match a single custom calc.
        assert np.allclose(rv_samples, rv_samples[0:1], atol=1e-12)
        params = fitter.build_params_dict(fitter.free_params_values) | frozen
        expected = fitter.calculate_rv_planet_custom('b', times, params)
        np.testing.assert_allclose(rv_samples[0], expected, atol=1e-12)

    def test_plot_posterior_phase_freeze_params(self, setup_gpfitter_for_rv):
        """GPFitter.plot_posterior_phase runs with freeze_params (median and explicit)."""
        import matplotlib
        matplotlib.use('Agg')

        fitter = setup_gpfitter_for_rv
        self._run_short_gp_mcmc(fitter)

        # None -> median, and explicit float, both accepted
        with pytest.warns(UserWarning, match="already fixed, not free"):
            fitter.plot_posterior_phase('b', discard_start=10, thin=5, n_smooth=50, freeze_params={"P_b": None, "Tc_b": None})
        with pytest.warns(UserWarning, match="already fixed, not free"):
            fitter.plot_posterior_phase('b', discard_start=10, thin=5, n_smooth=50, freeze_params={"P_b": 2.0, "Tc_b": 0.0})


class TestGPFitterNdim:
    """GPFitter.ndim is the number of free parameters, GP hyperparameters included."""

    @staticmethod
    def _fitter(test_gp_data):
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        return fitter

    def test_ndim_counts_free_gp_params(self, test_gp_data, test_gp_all_params) -> None:
        """Free GP hyperparameters count towards ndim; fixed ones don't."""
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params
        assert fitter.ndim == len(fitter.free_params_names) == 6

        fitter.params = {"gp_lambda_p": Parameter(0.5, fixed=True)}
        assert fitter.ndim == 5

    def test_params_reassigned(self, test_gp_data, test_gp_all_params, test_gp_all_priors) -> None:
        """Re-assigning params keeps ndim, and so BIC and AICc, unchanged.

        calculate_bic and calculate_aicc use ndim as the number of free parameters,
        so a stale ndim would change them silently for the same model and point.
        """
        fitter = self._fitter(test_gp_data)
        fitter.params = test_gp_all_params
        fitter.priors = test_gp_all_priors
        point = {name: p.value for name, p in fitter.params.items()}
        bic_before = fitter.calculate_bic(point)
        aicc_before = fitter.calculate_aicc(point)

        fitter.params = test_gp_all_params

        assert fitter.ndim == 6
        assert fitter.calculate_bic(point) == bic_before
        assert fitter.calculate_aicc(point) == aicc_before


class TestGPFitterIntegration:
    """Integration tests for complete GPFitter workflow."""

    def test_complete_setup(self, test_gp_data, test_gp_all_params, test_gp_all_priors) -> None:
        """Test complete GPFitter setup without running MCMC."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)

        # Add data
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        # Add parameters and priors, GP hyperparameters included
        fitter.params = test_gp_all_params
        fitter.priors = test_gp_all_priors

        # Verify everything is set up correctly
        assert len(fitter.params) == 13
        assert len(fitter.priors) == 6
        assert len(fitter.free_params_names) == 6
        assert fitter.ndim == 6  # 2 free params + 4 free GP hyperparameters
        assert list(fitter.unique_instruments) == ["HARPS"]

    def test_complete_workflow(self, test_gp_data, test_gp_all_params, test_gp_all_priors) -> None:
        """Test complete workflow: setup -> MAP -> walkers -> MCMC -> sample retrieval."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)

        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = test_gp_all_params
        fitter.priors = test_gp_all_priors

        # MAP
        map_result = fitter.find_map_estimate()
        assert map_result.success or np.isfinite(map_result.fun)

        # Walker init
        nwalkers = 14  # >= 2 * ndim (ndim=6)
        initial_positions = fitter.generate_initial_walker_positions_from_map(
            map_result, nwalkers=nwalkers
        )
        assert initial_positions.shape == (nwalkers, fitter.ndim)

        # MCMC
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=50, progress=False)
        assert fitter.sampler is not None

        # Sample retrieval: the chain's columns are free_params_names, in order
        chain = fitter.get_samples_np(flat=False)
        assert chain.shape == (50, nwalkers, fitter.ndim)

        df = fitter.get_samples_df()
        assert len(df) == 50 * nwalkers
        assert list(df.columns) == fitter.free_params_names

    def test_map_prints_one_dict_in_order(self, test_gp_data, test_gp_all_params, test_gp_all_priors,
                                          capsys) -> None:
        """find_map_estimate prints one results dict, like Fitter, in free_params_names order."""
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
        fitter.add_data(*test_gp_data, t0=2.0)
        fitter.params = test_gp_all_params
        fitter.priors = test_gp_all_priors

        fitter.find_map_estimate()
        out = capsys.readouterr().out

        assert out.count("MAP parameter results:") == 1
        assert "hyperparameter" not in out
        positions = [out.index(f"'{name}'") for name in fitter.free_params_names]
        assert positions == sorted(positions)

    def test_multi_planet_setup(self, test_gp_data, test_gp_hyperparams) -> None:
        """Test setup with two planets and GP kernel."""
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b", "c"], Parameterisation("P K e w Tc"), gp_kernel)

        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)

        params = {
            "P_b": Parameter(2.0, fixed=True),
            "K_b": Parameter(5.0, fixed=False),
            "e_b": Parameter(0.0, fixed=True),
            "w_b": Parameter(np.pi/2, fixed=True),
            "Tc_b": Parameter(0.0, fixed=True),
            "P_c": Parameter(4.0, fixed=True),
            "K_c": Parameter(3.0, fixed=False),
            "e_c": Parameter(0.0, fixed=True),
            "w_c": Parameter(np.pi/2, fixed=True),
            "Tc_c": Parameter(1.0, fixed=True),
            "g_HARPS": Parameter(0.0, fixed=True),
            "gd": Parameter(0.0, fixed=True),
            "gdd": Parameter(0.0, fixed=True),
            "jit_HARPS": Parameter(1.0, fixed=False),
        }
        fitter.params = params | test_gp_hyperparams

        assert len(fitter.params) == 18  # 5*2 planets + 4 system + 4 GP
        assert fitter.free_params_names == ["K_b", "K_c", "jit_HARPS",
                                            "gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]
        assert fitter.ndim == 7


class TestGPFitterKeywords:
    """GPFitter's methods take the same keywords as Fitter's; GP hyperparameters are just params."""

    @pytest.mark.parametrize("method", [
        "calculate_log_likelihood", "calculate_chi2", "calculate_aicc", "calculate_bic",
        "build_params_dict", "plot_custom_rv", "plot_custom_phase", "plot_autocorr_estimates",
    ])
    def test_signature_matches_fitter(self, method) -> None:
        """Same parameter names, in the same order, as the Fitter method."""
        import inspect
        fitter_names = list(inspect.signature(getattr(Fitter, method)).parameters)
        gpfitter_names = list(inspect.signature(getattr(GPFitter, method)).parameters)

        assert gpfitter_names == fitter_names

    @pytest.fixture
    def fitter(self, test_gp_data, test_gp_all_params, test_gp_all_priors):
        """A GPFitter with every parameter and prior set, GP ones included."""
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
        fitter.add_data(*test_gp_data, t0=2.0)
        fitter.params = test_gp_all_params
        fitter.priors = test_gp_all_priors
        return fitter

    def test_statistics_take_params_dict(self, fitter) -> None:
        """calculate_* take params_dict=, one dict with the GP hyperparameters in it."""
        point = fitter.build_params_dict(free_params=fitter.free_params_values)

        assert point == {name: p.value for name, p in fitter.params.items()}
        for method in ("calculate_log_likelihood", "calculate_chi2", "calculate_aicc", "calculate_bic"):
            assert np.isfinite(getattr(fitter, method)(params_dict=point))

    def test_custom_plots_take_params(self, fitter) -> None:
        """plot_custom_rv and plot_custom_phase take params=, GP hyperparameters included."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        point = fitter.build_params_dict(fitter.free_params_values)

        fitter.plot_custom_rv(params=point, n_smooth=50)
        fitter.plot_custom_phase("b", params=point)
        plt.close("all")

    def test_plot_autocorr_estimates_takes_gp_names_in_params(self, fitter) -> None:
        """plot_autocorr_estimates(params=...) accepts GP names; there is no hyperparams= keyword."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nwalkers = 2 * fitter.ndim
        rng = np.random.default_rng(0)
        centre = np.array(fitter.free_params_values)
        positions = centre * (1 + 0.01 * rng.standard_normal((nwalkers, fitter.ndim)))
        fitter.run_mcmc(positions, nwalkers=nwalkers, max_steps=300, progress=False,
                        check_convergence=True, convergence_check_interval=100,
                        convergence_check_start=20)

        fitter.plot_autocorr_estimates(params=["K_b", "gp_amp"])
        with pytest.raises(TypeError):
            fitter.plot_autocorr_estimates(hyperparams=["gp_amp"])
        plt.close("all")


class TestWalkerInitialisationWidths:
    """Tests for how generate_initial_walker_positions_random draws each prior type."""

    NWALKERS = 400
    SEED = 20260828
    # A sample standard deviation from 400 draws has a ~3.5% standard error, so this
    # band sits ~4 sigma from either boundary and ~25 sigma from a 2-sigma draw.
    LO, HI = 0.85, 1.15

    @staticmethod
    def _make_fitter(test_data, params, priors, parameterisation="P K e w Tc"):
        fitter = Fitter(["b"], Parameterisation(parameterisation))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = params
        if priors is not None:
            fitter.priors = priors
        return fitter

    @staticmethod
    def _unbounded_prior_params():
        """Circular params with K_b far from the K > 0 validity edge."""
        return {
            "P_b": Parameter(2.0, fixed=True),
            "K_b": Parameter(50.0, fixed=False),
            "e_b": Parameter(0.0, fixed=True),
            "w_b": Parameter(np.pi / 2, fixed=True),
            "Tc_b": Parameter(0.0, fixed=True),
            "g_HARPS": Parameter(0.0, fixed=True),
            "gd": Parameter(0.0, fixed=True),
            "gdd": Parameter(0.0, fixed=True),
            "jit_HARPS": Parameter(1.0, fixed=False),
        }

    def test_normal_prior_parameter_drawn_at_one_sigma(self, test_data) -> None:
        """A Normal prior on a parameter draws at scale=std, not 2*std.

        K_b sits 10 sigma clear of the K > 0 validity check, so no draw is rejected
        and the sample standard deviation is unbiased.
        """
        prior_std = 5.0
        fitter = self._make_fitter(
            test_data,
            self._unbounded_prior_params(),
            {"K_b": ravest.prior.Normal(50.0, prior_std),
             "jit_HARPS": ravest.prior.Uniform(0, 5)},
        )

        np.random.seed(self.SEED)
        positions = fitter.generate_initial_walker_positions_random(self.NWALKERS)

        k_column = positions[:, fitter.free_params_names.index("K_b")]
        ratio = np.std(k_column, ddof=1) / prior_std
        assert self.LO <= ratio <= self.HI, f"drawn at {ratio:.2f} sigma, expected 1"

    def test_halfnormal_prior_parameter_drawn_at_one_sigma(self, test_data) -> None:
        """A HalfNormal prior on a parameter draws at scale=std, not 2*std.

        Draws are abs(...) so none is ever rejected; the mean of |N(0, s)| is
        s*sqrt(2/pi), which separates 1 sigma from 2 sigma just as cleanly.
        """
        prior_std = 2.0
        fitter = self._make_fitter(
            test_data,
            self._unbounded_prior_params(),
            {"K_b": ravest.prior.Uniform(0, 100),
             "jit_HARPS": ravest.prior.HalfNormal(prior_std)},
        )

        np.random.seed(self.SEED)
        positions = fitter.generate_initial_walker_positions_random(self.NWALKERS)

        jit_column = positions[:, fitter.free_params_names.index("jit_HARPS")]
        expected_mean = prior_std * np.sqrt(2 / np.pi)
        ratio = np.mean(jit_column) / expected_mean
        assert self.LO <= ratio <= self.HI, f"drawn at {ratio:.2f} sigma, expected 1"

    def test_params_and_gp_params_share_one_width(
        self, test_gp_data, test_gp_hyperparams, test_gp_hyperpriors
    ) -> None:
        """Planet and GP parameters are drawn at the same width in one GP fit.

        This is the invariant the fix establishes: before it, gp_amp started within
        1 sigma of its prior while K_b started within 2 sigma of its prior.
        """
        param_std, hyper_std = 5.0, 1.0
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = (self._unbounded_prior_params() | test_gp_hyperparams
                         | {"gp_amp": Parameter(5.0, fixed=False)})
        fitter.priors = (test_gp_hyperpriors
                         | {"K_b": ravest.prior.Normal(50.0, param_std),
                            "jit_HARPS": ravest.prior.Uniform(0, 5),
                            "gp_amp": ravest.prior.Normal(5.0, hyper_std)})

        np.random.seed(self.SEED)
        positions = fitter.generate_initial_walker_positions_random(self.NWALKERS)

        columns = fitter.free_params_names
        param_ratio = np.std(positions[:, columns.index("K_b")], ddof=1) / param_std
        hyper_ratio = np.std(positions[:, columns.index("gp_amp")], ddof=1) / hyper_std

        assert self.LO <= param_ratio <= self.HI
        assert self.LO <= hyper_ratio <= self.HI
        assert abs(param_ratio - hyper_ratio) < 0.25

    def test_beta_gp_prior_draws_inside_its_support(
        self, test_gp_data, test_gp_circular_params, test_gp_hyperparams,
        test_gp_priors, test_gp_hyperpriors
    ) -> None:
        """A Beta prior on a GP hyperparameter draws from [0, 1], not from its shape parameters.

        Beta.a and Beta.b are shape parameters, not bounds. Drawing uniform(a, b)
        put every walker outside the support, so every draw scored -inf and walker
        generation exhausted max_attempts and raised.
        """
        gp_kernel = GPKernel("Quasiperiodic")
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), gp_kernel)
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = test_gp_circular_params | test_gp_hyperparams
        fitter.priors = test_gp_priors | test_gp_hyperpriors | {"gp_lambda_p": ravest.prior.Beta(2.0, 5.0)}

        positions = fitter.generate_initial_walker_positions_random(nwalkers=50)

        columns = fitter.free_params_names
        lambda_p = positions[:, columns.index("gp_lambda_p")]
        assert np.all((lambda_p >= 0.0) & (lambda_p <= 1.0))

    @staticmethod
    def _transformed_params():
        return {
            "P_b": Parameter(2.0, fixed=True),
            "K_b": Parameter(5.0, fixed=False),
            "secosw_b": Parameter(0.0177, fixed=False),
            "sesinw_b": Parameter(0.0767, fixed=False),
            "Tc_b": Parameter(0.0, fixed=True),
            "g_HARPS": Parameter(0.0, fixed=True),
            "gd": Parameter(0.0, fixed=True),
            "gdd": Parameter(0.0, fixed=True),
            "jit_HARPS": Parameter(1.0, fixed=False),
        }

    @staticmethod
    def _transformed_priors():
        return {
            "K_b": ravest.prior.Uniform(0, 20),
            "jit_HARPS": ravest.prior.Uniform(0, 5),
            "e_b": ravest.prior.Uniform(0, 0.9),
            "w_b": ravest.prior.Uniform(0, 2 * np.pi),
        }

    def test_transformed_parameterisation_does_not_warn(self, test_data, caplog) -> None:
        """Priors given on e/w while fitting secosw/sesinw is expected, not warned about.

        This is the normal path for every eccentric fit, so it must neither raise nor
        warn. It is logged at DEBUG level only.
        """
        fitter = self._make_fitter(
            test_data, self._transformed_params(), self._transformed_priors(),
            parameterisation="P K secosw sesinw Tc",
        )

        with caplog.at_level(logging.DEBUG):
            fitter.generate_initial_walker_positions_random(nwalkers=8)

        warnings_logged = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings_logged == []
        debug_messages = " ".join(r.message for r in caplog.records)
        assert "secosw_b" in debug_messages and "e_b" in debug_messages

    def test_transformed_parameterisation_still_draws_the_ball(self, test_data) -> None:
        """Making the secosw/sesinw case explicit did not change its numbers.

        Drawing secosw/sesinw from the e/w priors instead was tested and made sampling
        worse, so the ball is deliberate and must stay put.
        """
        fitter = self._make_fitter(
            test_data, self._transformed_params(), self._transformed_priors(),
            parameterisation="P K secosw sesinw Tc",
        )

        np.random.seed(self.SEED)
        positions = fitter.generate_initial_walker_positions_random(self.NWALKERS)

        for name in ("secosw_b", "sesinw_b"):
            centre = fitter.params[name].value
            column = positions[:, fitter.free_params_names.index(name)]
            expected_spread = abs(centre) * 0.1 + 0.01
            assert abs(np.mean(column) - centre) < 0.5 * expected_spread
            assert self.LO <= np.std(column, ddof=1) / expected_spread <= self.HI

    def test_parameter_with_no_prior_anywhere_raises(self, test_data) -> None:
        """Free parameters with no prior in either parameterisation raise, naming them.

        Route: priors never set at all.
        """
        fitter = self._make_fitter(test_data, self._unbounded_prior_params(), None)

        with pytest.raises(ValueError, match="No prior for free parameter") as excinfo:
            fitter.generate_initial_walker_positions_random(nwalkers=8)

        assert "K_b" in str(excinfo.value)
        assert "jit_HARPS" in str(excinfo.value)

    def test_parameter_freed_after_priors_set_raises(self, test_data) -> None:
        """A parameter freed after priors were assigned raises, naming only it.

        Route: priors set and validated for the free parameters at the time, then
        params re-assigned with another parameter free. The params setter does not
        re-check priors, so this must be caught before fitting.
        """
        params = self._unbounded_prior_params()
        params["jit_HARPS"] = Parameter(1.0, fixed=True)
        fitter = self._make_fitter(
            test_data, params, {"K_b": ravest.prior.Uniform(0, 100)},
        )
        fitter.params = self._unbounded_prior_params()  # jit_HARPS now free

        with pytest.raises(ValueError, match="No prior for free parameter") as excinfo:
            fitter.generate_initial_walker_positions_random(nwalkers=8)

        assert "jit_HARPS" in str(excinfo.value)
        assert "K_b" not in str(excinfo.value)

    def _gp_transformed_fitter(self, test_gp_data, test_gp_hyperparams,
                               test_gp_hyperpriors, priors):
        """GPFitter fitting secosw/sesinw, for the GPFitter copy of the no-prior cases.

        GPFitter is not a subclass of Fitter, so its initialiser is a separate copy
        and needs its own coverage.
        """
        fitter = GPFitter(["b"], Parameterisation("P K secosw sesinw Tc"),
                          GPKernel("Quasiperiodic"))
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = self._transformed_params() | test_gp_hyperparams
        fitter.priors = priors | test_gp_hyperpriors
        return fitter

    def test_gpfitter_transformed_parameterisation_does_not_warn(
        self, test_gp_data, test_gp_hyperparams, test_gp_hyperpriors, caplog
    ) -> None:
        """GPFitter treats priors on e/w while fitting secosw/sesinw as expected too."""
        fitter = self._gp_transformed_fitter(
            test_gp_data, test_gp_hyperparams, test_gp_hyperpriors,
            self._transformed_priors(),
        )

        with caplog.at_level(logging.DEBUG):
            np.random.seed(self.SEED)
            positions = fitter.generate_initial_walker_positions_random(nwalkers=8)

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
        debug_messages = " ".join(r.message for r in caplog.records)
        assert "secosw_b" in debug_messages and "e_b" in debug_messages

        centre = fitter.params["secosw_b"].value
        column = positions[:, fitter.free_params_names.index("secosw_b")]
        assert np.all(np.abs(column - centre) < 5 * (abs(centre) * 0.1 + 0.01))

    def test_gpfitter_parameter_with_no_prior_anywhere_raises(
        self, test_gp_data, test_gp_circular_params, test_gp_hyperparams,
        test_gp_hyperpriors
    ) -> None:
        """GPFitter raises for free parameters with no prior, naming them.

        Route: priors never set at all.
        """
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"),
                          GPKernel("Quasiperiodic"))
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = test_gp_circular_params | test_gp_hyperparams

        with pytest.raises(ValueError, match="No prior for free parameter") as excinfo:
            fitter.generate_initial_walker_positions_random(nwalkers=8)

        assert "K_b" in str(excinfo.value)
        assert "jit_HARPS" in str(excinfo.value)

    def test_gpfitter_parameter_freed_after_priors_set_raises(
        self, test_gp_data, test_gp_hyperparams, test_gp_hyperpriors
    ) -> None:
        """GPFitter raises for a parameter freed after priors were assigned.

        Route: priors set for the free parameters at the time, then params
        re-assigned with another parameter free.
        """
        params = self._unbounded_prior_params()
        params["jit_HARPS"] = Parameter(1.0, fixed=True)
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"),
                          GPKernel("Quasiperiodic"))
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        fitter.params = params | test_gp_hyperparams
        fitter.priors = {"K_b": ravest.prior.Uniform(0, 100)} | test_gp_hyperpriors
        fitter.params = self._unbounded_prior_params()  # jit_HARPS now free

        with pytest.raises(ValueError, match="No prior for free parameter") as excinfo:
            fitter.generate_initial_walker_positions_random(nwalkers=8)

        assert "jit_HARPS" in str(excinfo.value)
        assert "K_b" not in str(excinfo.value)


class TestPriorPresenceValidation:
    """Every entry point that samples or optimises refuses free parameters without a prior.

    Without the check, a free parameter with no prior contributes nothing to the
    log-prior, so it is sampled or optimised completely unconstrained with no error.
    Two routes reach that state: priors never set, and a parameter freed by
    re-assigning params after priors were set (the params setter does not re-check).
    """

    @staticmethod
    def _params(jit_fixed=False):
        return {
            "P_b": Parameter(2.0, fixed=True),
            "K_b": Parameter(5.0, fixed=False),
            "e_b": Parameter(0.0, fixed=True),
            "w_b": Parameter(np.pi / 2, fixed=True),
            "Tc_b": Parameter(0.0, fixed=True),
            "g_HARPS": Parameter(0.0, fixed=True),
            "gd": Parameter(0.0, fixed=True),
            "gdd": Parameter(0.0, fixed=True),
            "jit_HARPS": Parameter(1.0, fixed=jit_fixed),
        }

    def _fitter(self, test_data, route):
        fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
        time, vel, velerr, instrument = test_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        if route == "never_set":
            fitter.params = self._params()
        else:  # "freed_after"
            fitter.params = self._params(jit_fixed=True)
            fitter.priors = {"K_b": ravest.prior.Uniform(0, 20)}
            fitter.params = self._params()  # jit_HARPS now free, with no prior
        return fitter

    def _gpfitter(self, test_gp_data, test_gp_hyperparams, test_gp_hyperpriors, route):
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
        time, vel, velerr, instrument = test_gp_data
        fitter.add_data(time, vel, velerr, instrument, t0=2.0)
        if route == "never_set":
            fitter.params = self._params() | test_gp_hyperparams
        else:  # "freed_after"
            fitter.params = self._params(jit_fixed=True) | test_gp_hyperparams
            fitter.priors = {"K_b": ravest.prior.Uniform(0, 20)} | test_gp_hyperpriors
            fitter.params = self._params()  # jit_HARPS now free, with no prior
        return fitter

    @staticmethod
    def _call(fitter, entry_point, ndim):
        if entry_point == "find_map_estimate":
            fitter.find_map_estimate()
        elif entry_point == "around_point":
            fitter.generate_initial_walker_positions_around_point(
                centre=np.ones(ndim), nwalkers=2 * ndim
            )
        elif entry_point == "run_mcmc":
            positions = 1 + 0.01 * np.random.default_rng(0).standard_normal((2 * ndim, ndim))
            fitter.run_mcmc(positions, nwalkers=2 * ndim, max_steps=2, progress=False)

    @pytest.mark.parametrize("route", ["never_set", "freed_after"])
    @pytest.mark.parametrize("entry_point", ["find_map_estimate", "around_point", "run_mcmc"])
    def test_fitter_raises_for_free_param_without_prior(self, test_data, entry_point, route) -> None:
        """Fitter refuses a free parameter without a prior at every entry point, by either route."""
        fitter = self._fitter(test_data, route)

        with pytest.raises(ValueError, match="No prior for free parameter") as excinfo:
            self._call(fitter, entry_point, len(fitter.free_params_names))

        assert "jit_HARPS" in str(excinfo.value)

    @pytest.mark.parametrize("route", ["never_set", "freed_after"])
    @pytest.mark.parametrize("entry_point", ["find_map_estimate", "around_point", "run_mcmc"])
    def test_gpfitter_raises_for_free_param_without_prior(
        self, test_gp_data, test_gp_hyperparams, test_gp_hyperpriors, entry_point, route
    ) -> None:
        """GPFitter refuses a free parameter without a prior at every entry point, by either route."""
        fitter = self._gpfitter(test_gp_data, test_gp_hyperparams, test_gp_hyperpriors, route)

        with pytest.raises(ValueError, match="No prior for free parameter") as excinfo:
            self._call(fitter, entry_point, fitter.ndim)

        assert "jit_HARPS" in str(excinfo.value)

    @pytest.mark.parametrize(
        "entry_point", ["random", "find_map_estimate", "around_point", "run_mcmc"]
    )
    def test_gpfitter_raises_for_free_gp_param_without_prior(
        self, test_gp_data, test_gp_all_params, test_gp_all_priors, entry_point
    ) -> None:
        """A GP hyperparameter freed after priors were set is refused like any other parameter.

        gp_amp is the first GP hyperparameter, so a check that only dropped the last
        name, or mis-paired names and values, would not pass.
        """
        fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
        fitter.add_data(*test_gp_data, t0=2.0)
        fitter.params = test_gp_all_params | {"gp_amp": Parameter(1.0, fixed=True)}
        fitter.priors = {k: v for k, v in test_gp_all_priors.items() if k != "gp_amp"}
        fitter.params = {"gp_amp": Parameter(1.0, fixed=False)}  # now free, with no prior

        with pytest.raises(ValueError, match="No prior for free parameter") as excinfo:
            if entry_point == "random":
                fitter.generate_initial_walker_positions_random(nwalkers=2 * fitter.ndim)
            else:
                self._call(fitter, entry_point, fitter.ndim)

        assert "gp_amp" in str(excinfo.value)
        assert "K_b" not in str(excinfo.value)

    @pytest.mark.parametrize("kind, name", [("Fitter", "jit_HARPS"), ("GPFitter", "gp_amp")])
    def test_walker_loop_refuses_param_without_prior_if_check_bypassed(
        self, test_data, test_gp_data, test_gp_all_params, test_gp_all_priors, monkeypatch,
        kind, name
    ) -> None:
        """The random walker loop names a free parameter with no prior, even past the up-front check.

        Unreachable through the public API while the up-front check runs; without the guard,
        such a parameter would silently start in a ball around its current value, as if its
        prior had been given on default-parameterisation equivalents.
        """
        if kind == "Fitter":
            fitter = self._fitter(test_data, "freed_after")
        else:
            fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
            fitter.add_data(*test_gp_data, t0=2.0)
            fitter.params = test_gp_all_params | {"gp_amp": Parameter(1.0, fixed=True)}
            fitter.priors = {k: v for k, v in test_gp_all_priors.items() if k != "gp_amp"}
            fitter.params = {"gp_amp": Parameter(1.0, fixed=False)}  # now free, with no prior
        monkeypatch.setattr(fitter, "_validate_before_fit", lambda: {})

        with pytest.raises(ValueError, match=f"No prior for free parameter {name}"):
            fitter.generate_initial_walker_positions_random(nwalkers=2 * fitter.ndim)


class TestMinimumWalkers:
    """run_mcmc refuses fewer than 2 * ndim walkers and never changes the caller's nwalkers.

    emcee's stretch move needs at least 2 * ndim walkers. Raising up front names the
    minimum, instead of quietly raising self.nwalkers above the number of starting
    positions supplied.
    """

    def _fitter(self, kind, test_data, test_circular_params, test_simple_priors,
                test_gp_data, test_gp_hyperparams, test_gp_hyperpriors):
        """Fitter (ndim 2: K_b, jit_HARPS) or GPFitter (ndim 6: plus the four QP hyperparameters)."""
        if kind == "Fitter":
            fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
            fitter.add_data(*test_data, t0=2.0)
            fitter.params = test_circular_params
            fitter.priors = test_simple_priors
            centre = np.array([5.0, 1.0])
        else:
            fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
            fitter.add_data(*test_gp_data, t0=2.0)
            fitter.params = test_circular_params | test_gp_hyperparams
            fitter.priors = test_simple_priors | test_gp_hyperpriors
            centre = np.array([5.0, 1.0, 1.0, 50.0, 0.5, 10.0])
        assert fitter.ndim == len(centre)
        return fitter, centre

    @staticmethod
    def _positions(centre, nwalkers):
        rng = np.random.default_rng(0)
        return centre * (1 + 0.01 * rng.standard_normal((nwalkers, len(centre))))

    @pytest.fixture(params=["Fitter", "GPFitter"])
    def fitter_and_centre(self, request, test_data, test_circular_params, test_simple_priors,
                          test_gp_data, test_gp_hyperparams, test_gp_hyperpriors):
        """Each fitter class with its walker centre, in free-parameter order."""
        return self._fitter(request.param, test_data, test_circular_params, test_simple_priors,
                            test_gp_data, test_gp_hyperparams, test_gp_hyperpriors)

    def test_too_few_walkers_raises(self, fitter_and_centre) -> None:
        """One walker short of 2 * ndim raises, naming the minimum, ndim and the value given."""
        fitter, centre = fitter_and_centre
        ndim = fitter.ndim
        nwalkers = 2 * ndim - 1

        expected = (
            f"nwalkers must be at least 2 * ndim = {2 * ndim} "
            f"({ndim} free parameters), got {nwalkers}."
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            fitter.run_mcmc(self._positions(centre, nwalkers), nwalkers=nwalkers,
                            max_steps=2, progress=False)

    def test_too_few_walkers_checked_before_positions_shape(self, fitter_and_centre) -> None:
        """The walker minimum is reported even when initial_positions has the wrong shape too."""
        fitter, centre = fitter_and_centre
        nwalkers = 2 * fitter.ndim - 1

        with pytest.raises(ValueError, match="nwalkers must be at least"):
            fitter.run_mcmc(self._positions(centre, 2 * fitter.ndim), nwalkers=nwalkers,
                            max_steps=2, progress=False)

    def test_too_few_walkers_leaves_nwalkers_unchanged(self, fitter_and_centre) -> None:
        """A refused run does not touch self.nwalkers."""
        fitter, centre = fitter_and_centre
        before = getattr(fitter, "nwalkers", None)
        nwalkers = 2 * fitter.ndim - 1

        with pytest.raises(ValueError):
            fitter.run_mcmc(self._positions(centre, nwalkers), nwalkers=nwalkers,
                            max_steps=2, progress=False)

        assert getattr(fitter, "nwalkers", None) == before

    @pytest.mark.parametrize("extra", [0, 1, 6], ids=["exactly_2ndim", "odd", "even_above"])
    def test_enough_walkers_runs(self, fitter_and_centre, extra) -> None:
        """2 * ndim walkers or more (odd counts included) run, and self.nwalkers is the argument."""
        fitter, centre = fitter_and_centre
        nwalkers = 2 * fitter.ndim + extra

        fitter.run_mcmc(self._positions(centre, nwalkers), nwalkers=nwalkers,
                        max_steps=2, progress=False)

        assert fitter.nwalkers == nwalkers
        assert fitter.sampler.nwalkers == nwalkers
        assert fitter.get_samples_np(flat=False).shape == (2, nwalkers, fitter.ndim)


class TestParamOrder:
    """Parameters, priors and chain columns follow one fixed order, whatever the dict order.

    Planets by letter, each in the parameterisation's order; then g_ and jit_ per instrument,
    instruments sorted case-insensitively; then gd, gdd; then (GPFitter) the kernel's
    hyperparameters. Fixed parameters are skipped in the free names, so those are the chain's
    columns.
    """

    INSTRUMENTS = ["HIRES", "apf", "harps"]

    EXPECTED_ORDER = [
        "P_b", "K_b", "secosw_b", "sesinw_b", "Tc_b",
        "P_c", "K_c", "secosw_c", "sesinw_c", "Tc_c",
        "g_apf", "jit_apf", "g_harps", "jit_harps", "g_HIRES", "jit_HIRES",
        "gd", "gdd",
    ]

    QP_ORDER = ["gp_amp", "gp_lambda_e", "gp_lambda_p", "gp_period"]

    FIXED = {"Tc_c", "jit_harps", "gd", "gdd", "gp_lambda_p"}

    EXPECTED_PRIORS = [
        "P_b", "K_b", "e_b", "w_b", "Tp_b",
        "P_c", "K_c", "secosw_c", "sesinw_c",
        "g_apf", "jit_apf", "g_harps", "g_HIRES", "jit_HIRES",
    ]

    def _expected(self, make):
        """The full parameter order for the fitter that make builds."""
        return self.EXPECTED_ORDER + (self.QP_ORDER if make == "_gpfitter" else [])

    def _priors(self, make):
        """Scrambled priors for the fitter that make builds (GPFitter's include the free GP names)."""
        gp = {"gp_period": ravest.prior.Uniform(1, 50), "gp_amp": ravest.prior.Uniform(0, 10),
              "gp_lambda_e": ravest.prior.Uniform(1, 100)}
        return (gp if make == "_gpfitter" else {}) | self._scrambled_priors()

    @classmethod
    def _add_data(cls, fitter):
        time = np.linspace(0.0, 20.0, 9)
        vel = np.zeros(9)
        velerr = np.ones(9)
        instrument = np.array(cls.INSTRUMENTS * 3)
        fitter.add_data(time, vel, velerr, instrument, t0=10.0)

    @classmethod
    def _scrambled_params(cls):
        """Every parameter, in reverse of the expected order."""
        values = {"P": 5.0, "K": 3.0, "secosw": 0.1, "sesinw": 0.1, "Tc": 0.0}
        params = {}
        for name in cls.EXPECTED_ORDER:
            base, _, suffix = name.partition("_")
            if base in values:
                value = values[base] + (7.0 if base == "P" and suffix == "c" else 0.0)
            elif base == "jit":
                value = 1.0
            else:
                value = 0.0
            params[name] = Parameter(value, fixed=name in cls.FIXED)
        return dict(reversed(params.items()))

    @staticmethod
    def _scrambled_priors():
        """Priors in no particular order.

        Planet b's are on e, w and Tp (default-parameterisation equivalents), planet c's on its
        own parameters.
        """
        U = ravest.prior.Uniform
        return {
            "jit_HIRES": U(0, 10),
            "Tp_b": U(-20, 20),
            "sesinw_c": U(-1, 1),
            "g_apf": U(-10, 10),
            "w_b": U(-np.pi, np.pi),
            "K_c": U(0, 10),
            "e_b": U(0, 1),
            "g_harps": U(-10, 10),
            "P_c": U(10, 14),
            "secosw_c": U(-1, 1),
            "jit_apf": U(0, 10),
            "K_b": U(0, 10),
            "g_HIRES": U(-10, 10),
            "P_b": U(4, 6),
        }

    def _fitter(self):
        fitter = Fitter(["c", "b"], Parameterisation("P K secosw sesinw Tc"))
        self._add_data(fitter)
        fitter.params = self._scrambled_params()
        return fitter

    def _gpfitter(self):
        fitter = GPFitter(["c", "b"], Parameterisation("P K secosw sesinw Tc"),
                          GPKernel("Quasiperiodic"))
        self._add_data(fitter)
        fitter.params = {
            "gp_period": Parameter(10.0, fixed=False),
            "gp_lambda_p": Parameter(0.5, fixed=True),
            "gp_amp": Parameter(1.0, fixed=False),
            "gp_lambda_e": Parameter(50.0, fixed=False),
        } | self._scrambled_params()
        return fitter

    @pytest.mark.parametrize("cls", [Fitter, GPFitter])
    def test_unique_instruments_sorted_case_insensitively(self, cls) -> None:
        """Case-insensitive alphabetical, with upper case first when two names differ only in case."""
        args = (["b"], Parameterisation("P K e w Tp"))
        fitter = cls(*args) if cls is Fitter else cls(*args, GPKernel("Quasiperiodic"))
        instrument = np.array(["HIRES", "apf", "harps", "HARPS"])
        fitter.add_data(np.arange(4.0), np.zeros(4), np.ones(4), instrument, t0=0.0)

        assert list(fitter.unique_instruments) == ["apf", "HARPS", "harps", "HIRES"]

    def test_param_order_fitter(self) -> None:
        """Planets sorted by letter, instruments case-insensitively, trend last."""
        assert self._fitter()._param_order() == self.EXPECTED_ORDER

    def test_param_order_gpfitter(self) -> None:
        """GPFitter appends the kernel's hyperparameters, in the kernel's own order."""
        assert self._gpfitter()._param_order() == self.EXPECTED_ORDER + self.QP_ORDER

    @pytest.mark.parametrize("make", ["_fitter", "_gpfitter"])
    def test_params_stored_in_order(self, make) -> None:
        """`params` assigned in reverse order come back in the fixed order."""
        fitter = getattr(self, make)()

        assert list(fitter.params) == self._expected(make)

    @pytest.mark.parametrize("make", ["_fitter", "_gpfitter"])
    def test_partial_update_keeps_order(self, make) -> None:
        """A partial params update changes values, not positions."""
        fitter = getattr(self, make)()
        fitter.params = {"gd": Parameter(0.0, fixed=True), "K_b": Parameter(4.0, fixed=False)}

        assert list(fitter.params) == self._expected(make)
        assert fitter.params["K_b"].value == 4.0

    @pytest.mark.parametrize("make", ["_fitter", "_gpfitter"])
    def test_free_and_fixed_names_in_order(self, make) -> None:
        """free_params_names skips fixed parameters; both lists keep the fixed order."""
        fitter = getattr(self, make)()

        assert fitter.free_params_names == [n for n in self._expected(make) if n not in self.FIXED]
        assert fitter.fixed_params_names == [n for n in self._expected(make) if n in self.FIXED]

    @pytest.mark.parametrize("make", ["_fitter", "_gpfitter"])
    def test_priors_stored_in_order(self, make) -> None:
        """Priors follow the free parameters' order.

        A default-parameterisation equivalent takes the slot of the parameter it constrains
        (e_b, w_b for secosw_b, sesinw_b; Tp_b for Tc_b).
        """
        fitter = getattr(self, make)()
        fitter.priors = self._priors(make)

        gp = ["gp_amp", "gp_lambda_e", "gp_period"] if make == "_gpfitter" else []
        assert list(fitter.priors) == self.EXPECTED_PRIORS + gp

    @pytest.mark.parametrize("make", ["_fitter", "_gpfitter"])
    def test_partial_priors_update_keeps_order(self, make) -> None:
        """A partial priors update replaces the function, not its position."""
        fitter = getattr(self, make)()
        fitter.priors = self._priors(make)
        before = list(fitter.priors)
        new_prior = ravest.prior.Uniform(0, 20)

        fitter.priors = {"K_c": new_prior}

        assert list(fitter.priors) == before
        assert fitter.priors["K_c"] is new_prior

    @pytest.mark.parametrize("make", ["_fitter", "_gpfitter"])
    def test_ndim_is_read_only(self, make) -> None:
        """`ndim` is derived, so it cannot be assigned."""
        fitter = getattr(self, make)()

        with pytest.raises(AttributeError):
            fitter.ndim = 3

    def test_fitter_ndim_is_number_of_free_params(self) -> None:
        """Fitter.ndim is len(free_params_names), after re-assignment too."""
        fitter = self._fitter()
        assert fitter.ndim == len(fitter.free_params_names) == 14

        fitter.params = {"K_c": Parameter(3.0, fixed=True)}
        assert fitter.ndim == 13

    @pytest.mark.parametrize("make", ["_fitter", "_gpfitter"])
    def test_ndim_follows_in_place_edit(self, make) -> None:
        """Replacing a parameter in place (bypassing the setter) cannot leave ndim stale."""
        fitter = getattr(self, make)()
        before = fitter.ndim

        fitter.params["K_c"] = Parameter(3.0, fixed=True)

        assert fitter.ndim == before - 1

    def test_gpfitter_ndim_counts_free_gp_params(self) -> None:
        """GPFitter.ndim includes the free GP hyperparameters (three of the four here)."""
        fitter = self._gpfitter()

        assert fitter.ndim == len(fitter.free_params_names) == 14 + 3


def _set_value(name, value):
    def edit(fitter):
        fitter.params[name].value = value
    return edit


def _set_fixed(name, fixed):
    def edit(fitter):
        fitter.params[name].fixed = fixed
    return edit


def _replace_param(name, param):
    def edit(fitter):
        fitter.params[name] = param
    return edit


def _delete_param(name):
    def edit(fitter):
        del fitter.params[name]
    return edit


def _replace_prior(name, prior):
    def edit(fitter):
        fitter.priors[name] = prior
    return edit


def _delete_prior(name):
    def edit(fitter):
        del fitter.priors[name]
    return edit


# (id, edit, exception, match): in-place edits that skip the params/priors setters
IN_PLACE_EDITS = [
    ("value_outside_prior", _set_value("K_b", 25.0), ValueError,
     "Initial value 25.0 of parameter K_b is invalid"),
    ("value_unphysical", _set_value("jit_HARPS", -1.0), ValueError, "Invalid jitter jit_HARPS"),
    ("fixed_not_bool", _set_fixed("K_b", np.False_), TypeError, "fixed"),
    ("param_replaced_as_fixed", _replace_param("K_b", Parameter(5.0, fixed=True)), ValueError,
     "Unexpected priors.*K_b"),
    ("param_deleted", _delete_param("gd"), ValueError, "Missing required parameters.*gd"),
    ("param_added", _replace_param("foo", Parameter(1.0, fixed=True)), ValueError,
     "Unexpected parameters.*foo"),
    ("prior_excludes_value", _replace_prior("K_b", ravest.prior.Uniform(10, 20)), ValueError,
     "Initial value 5.0 of parameter K_b is invalid"),
    ("prior_deleted", _delete_prior("K_b"), ValueError, "(No prior|Missing priors).*K_b"),
    ("prior_on_fixed_param", _replace_prior("P_b", ravest.prior.Uniform(1, 5)), ValueError,
     "Unexpected priors.*P_b"),
]

GP_IN_PLACE_EDITS = [
    ("gp_value_unphysical", _set_value("gp_amp", -1.0), ValueError, "gp_amp must be positive"),
    ("gp_value_outside_prior", _set_value("gp_period", 60.0), ValueError,
     "Initial value 60.0 of parameter gp_period is invalid"),
]

ENTRY_POINTS = ["find_map_estimate", "random", "around_point", "from_map", "run_mcmc"]


class TestPointOfUseValidation:
    """Every method that fits re-checks params and priors, so in-place edits cannot skip validation.

    fitter.params and fitter.priors return the live dicts, so editing them in place bypasses the
    setters' checks. find_map_estimate, run_mcmc and the three walker initialisers run the full
    params and priors checks first, before anything else (including their own shape checks).
    """

    @staticmethod
    def _fitter(kind, test_data, test_circular_params, test_simple_priors,
                test_gp_data, test_gp_all_params, test_gp_all_priors):
        """Fitter (free K_b, jit_HARPS) or GPFitter (plus the four free QP hyperparameters)."""
        if kind == "Fitter":
            fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
            fitter.add_data(*test_data, t0=2.0)
            fitter.params = test_circular_params
            fitter.priors = test_simple_priors
        else:
            fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
            fitter.add_data(*test_gp_data, t0=2.0)
            fitter.params = test_gp_all_params
            fitter.priors = test_gp_all_priors
        return fitter

    @staticmethod
    def _call(fitter, entry_point, centre):
        """Call one entry point, with inputs sized for the fitter as it was before any edit."""
        ndim = len(centre)
        nwalkers = 2 * ndim
        if entry_point == "find_map_estimate":
            fitter.find_map_estimate()
        elif entry_point == "random":
            fitter.generate_initial_walker_positions_random(nwalkers=nwalkers)
        elif entry_point == "around_point":
            fitter.generate_initial_walker_positions_around_point(centre=centre, nwalkers=nwalkers)
        elif entry_point == "from_map":
            import types
            fitter.generate_initial_walker_positions_from_map(types.SimpleNamespace(x=centre),
                                                              nwalkers=nwalkers)
        elif entry_point == "run_mcmc":
            rng = np.random.default_rng(0)
            positions = centre * (1 + 0.01 * rng.standard_normal((nwalkers, ndim)))
            fitter.run_mcmc(positions, nwalkers=nwalkers, max_steps=2, progress=False)

    @pytest.fixture(params=["Fitter", "GPFitter"])
    def fitter(self, request, test_data, test_circular_params, test_simple_priors,
               test_gp_data, test_gp_all_params, test_gp_all_priors):
        """Each fitter class, fully set up through the setters."""
        return self._fitter(request.param, test_data, test_circular_params, test_simple_priors,
                            test_gp_data, test_gp_all_params, test_gp_all_priors)

    @pytest.mark.parametrize("entry_point", ENTRY_POINTS)
    @pytest.mark.parametrize("edit_id, edit, exc, match", IN_PLACE_EDITS,
                             ids=[e[0] for e in IN_PLACE_EDITS])
    def test_in_place_edit_refused(self, fitter, entry_point, edit_id, edit, exc, match) -> None:
        """Each in-place edit is caught at each entry point, with the setters' own message."""
        centre = np.array(fitter.free_params_values)
        edit(fitter)

        with pytest.raises(exc, match=match):
            self._call(fitter, entry_point, centre)

    @pytest.mark.parametrize("entry_point", ENTRY_POINTS)
    @pytest.mark.parametrize("edit_id, edit, exc, match", GP_IN_PLACE_EDITS,
                             ids=[e[0] for e in GP_IN_PLACE_EDITS])
    def test_gp_in_place_edit_refused(self, test_data, test_circular_params, test_simple_priors,
                                      test_gp_data, test_gp_all_params, test_gp_all_priors,
                                      entry_point, edit_id, edit, exc, match) -> None:
        """In-place edits to GP hyperparameters are caught the same way."""
        fitter = self._fitter("GPFitter", test_data, test_circular_params, test_simple_priors,
                              test_gp_data, test_gp_all_params, test_gp_all_priors)
        centre = np.array(fitter.free_params_values)
        edit(fitter)

        with pytest.raises(exc, match=match):
            self._call(fitter, entry_point, centre)

    @pytest.mark.parametrize("entry_point", ENTRY_POINTS)
    def test_valid_in_place_edit_accepted(self, fitter, entry_point) -> None:
        """A valid in-place edit passes the check, and the new value is the one used."""
        fitter.params["K_b"].value = 6.0
        centre = np.array(fitter.free_params_values)

        self._call(fitter, entry_point, centre)

        assert fitter.free_params_values[0] == 6.0

    def test_check_does_not_rewrite_priors(self, fitter) -> None:
        """Checking at fit time leaves the priors dict as it was: same object, same contents."""
        priors = fitter.priors
        before = dict(priors)

        fitter.generate_initial_walker_positions_random(nwalkers=2 * fitter.ndim)

        assert fitter.priors is priors
        assert fitter.priors == before


class TestLogProbFactories:
    """Fitters build their posteriors and likelihoods through two factory methods.

    _build_log_posterior() and _build_log_likelihood() build from the fitter as it is now; on a
    GPFitter they build the GP classes. Every entry point that needs one calls the factory.
    """

    POSTERIOR = {"Fitter": LogPosterior, "GPFitter": GPLogPosterior}
    LIKELIHOOD = {"Fitter": LogLikelihood, "GPFitter": GPLogLikelihood}
    DATA = ["time", "vel", "velerr", "instrument", "unique_instruments", "t0"]
    FACTORY_USERS = [
        ("find_map_estimate", "_build_log_posterior"),
        ("random", "_build_log_posterior"),
        ("around_point", "_build_log_posterior"),
        ("run_mcmc", "_build_log_posterior"),
        ("calculate_log_likelihood", "_build_log_likelihood"),
        ("calculate_chi2", "_build_log_likelihood"),
    ]

    @pytest.fixture(params=["Fitter", "GPFitter"])
    def fitter(self, request, test_data, test_circular_params, test_simple_priors,
               test_gp_data, test_gp_all_params, test_gp_all_priors):
        """Each fitter class, fully set up through the setters."""
        return TestPointOfUseValidation._fitter(request.param, test_data, test_circular_params,
                                                test_simple_priors, test_gp_data, test_gp_all_params,
                                                test_gp_all_priors)

    def test_build_log_posterior_type(self, fitter) -> None:
        """Exactly LogPosterior on a Fitter, GPLogPosterior on a GPFitter."""
        assert type(fitter._build_log_posterior()) is self.POSTERIOR[type(fitter).__name__]

    def test_build_log_likelihood_type(self, fitter) -> None:
        """Exactly LogLikelihood on a Fitter, GPLogLikelihood on a GPFitter."""
        assert type(fitter._build_log_likelihood()) is self.LIKELIHOOD[type(fitter).__name__]

    def test_build_log_posterior_reads_current_state(self, fitter) -> None:
        """Built from the fitter as it is now: an in-place edit and new priors show up."""
        fitter.params["P_b"].value = 2.5
        fitter.priors = {"K_b": ravest.prior.Uniform(0, 30)}

        lp = fitter._build_log_posterior()

        assert lp.fixed_params == fitter.fixed_params_values_dict
        assert lp.fixed_params["P_b"] == 2.5
        assert lp.priors is fitter.priors
        assert lp.free_params_names == fitter.free_params_names
        assert lp.planet_letters == fitter.planet_letters
        assert lp.parameterisation is fitter.parameterisation
        for name in self.DATA:
            assert getattr(lp, name) is getattr(fitter, name)
        if isinstance(fitter, GPFitter):
            assert lp.gp_kernel is fitter.gp_kernel

    def test_build_log_likelihood_reads_current_state(self, fitter) -> None:
        """Built from the fitter's model and data."""
        ll = fitter._build_log_likelihood()

        assert ll.planet_letters == fitter.planet_letters
        assert ll.parameterisation is fitter.parameterisation
        for name in self.DATA:
            assert getattr(ll, name) is getattr(fitter, name)
        if isinstance(fitter, GPFitter):
            assert ll.gp_kernel is fitter.gp_kernel

    @pytest.mark.parametrize("entry_point, factory", FACTORY_USERS, ids=[e for e, _ in FACTORY_USERS])
    def test_entry_point_builds_through_factory(self, fitter, monkeypatch, entry_point, factory) -> None:
        """Each entry point gets its posterior or likelihood from the factory."""
        original = getattr(type(fitter), factory)
        built = []

        def spy():
            obj = original(fitter)
            built.append(obj)
            return obj

        monkeypatch.setattr(fitter, factory, spy)
        if entry_point in ("calculate_log_likelihood", "calculate_chi2"):
            point = fitter.build_params_dict(free_params=fitter.free_params_values)
            getattr(fitter, entry_point)(params_dict=point)
        else:
            TestPointOfUseValidation._call(fitter, entry_point, np.array(fitter.free_params_values))

        assert built


class TestGPLogPosteriorSubclass:
    """GPLogPosterior is a LogPosterior that builds a GP likelihood and checks kernel values first.

    The prior conversion, log-probability corrections and MAP objective are inherited. The
    recorded values in TestGPOneDictInternals check that no number changes.
    """

    def test_is_subclass(self) -> None:
        """GPLogPosterior inherits from LogPosterior."""
        assert issubclass(GPLogPosterior, LogPosterior)

    def test_defines_only_gp_parts(self) -> None:
        """GPLogPosterior defines only its constructor, its likelihood and the kernel check."""
        import inspect
        defined = {name for name, value in vars(GPLogPosterior).items() if inspect.isfunction(value)}

        assert defined == {"__init__", "_build_log_likelihood", "log_probability"}

    @pytest.mark.parametrize("klass, likelihood", [(LogPosterior, LogLikelihood),
                                                   (GPLogPosterior, GPLogLikelihood)],
                             ids=["LogPosterior", "GPLogPosterior"])
    def test_log_likelihood_attribute(self, klass, likelihood) -> None:
        """Both hold their likelihood as log_likelihood, built by _build_log_likelihood()."""
        lp = klass(**TestLogProbSignatures._kwargs(klass))

        assert type(lp.log_likelihood) is likelihood
        assert type(lp._build_log_likelihood()) is likelihood
        assert not hasattr(lp, "gp_log_likelihood")
        if klass is GPLogPosterior:
            assert lp.log_likelihood.gp_kernel is lp.gp_kernel


class TestGPFitterSubclass:
    """GPFitter is a Fitter with a GP: it defines only what the GP changes.

    Everything else is inherited, which also fixes the places where GPFitter's copies of
    Fitter's methods had drifted apart.
    """

    GP_DEFINED = {
        # Fitter's version plus the GP part, via super()
        "__init__", "_param_order", "_validate_astrophysical_validity",
        "_get_default_parameterisation_equivalent_free_param_name",
        "calculate_rv_total_from_samples", "calculate_rv_total_custom",
        # The GP posterior and likelihood
        "_build_log_posterior", "_build_log_likelihood",
        # GP statistics, RVs and plots
        "calculate_chi2", "_compute_gp_chi2", "calculate_rv_gp_from_samples", "calculate_rv_gp_custom",
        "_plot_rv", "_plot_phase", "plot_posterior_rv", "plot_posterior_phase",
        # progress=True by default
        "calculate_rv_planet_from_samples", "_calculate_rv_planet_from_samples",
        "calculate_rv_trend_from_samples",
        # Default titles differ from Fitter's
        "plot_MAP_rv", "plot_MAP_phase", "plot_custom_rv", "plot_custom_phase",
        "plot_best_sample_rv", "plot_best_sample_phase",
    }

    @pytest.fixture
    def fitted(self, test_data, test_circular_params, test_simple_priors,
               test_gp_data, test_gp_all_params, test_gp_all_priors):
        """A GPFitter after a short MCMC run."""
        fitter = TestPointOfUseValidation._fitter("GPFitter", test_data, test_circular_params,
                                                  test_simple_priors, test_gp_data, test_gp_all_params,
                                                  test_gp_all_priors)
        nwalkers = 2 * fitter.ndim
        rng = np.random.default_rng(0)
        centre = np.array(fitter.free_params_values)
        positions = centre * (1 + 0.01 * rng.standard_normal((nwalkers, fitter.ndim)))
        fitter.run_mcmc(positions, nwalkers=nwalkers, max_steps=20, progress=False)
        return fitter

    def test_is_subclass(self) -> None:
        """GPFitter inherits from Fitter."""
        assert issubclass(GPFitter, Fitter)

    def test_defines_only_gp_parts(self) -> None:
        """GPFitter defines exactly these; anything else comes from Fitter."""
        defined = {name for name in vars(GPFitter)
                   if name == "__init__" or not (name.startswith("__") and name.endswith("__"))}

        assert defined == self.GP_DEFINED

    def test_gp_kernel_must_be_gpkernel(self) -> None:
        """A kernel name passed as a plain string is refused."""
        with pytest.raises(TypeError, match="gp_kernel must be a GPKernel"):
            GPFitter(["b"], Parameterisation("P K e w Tc"), "Quasiperiodic")

    @pytest.mark.parametrize("kind", ["Fitter", "GPFitter"])
    def test_params_before_add_data_raises(self, kind, test_circular_params, test_gp_all_params) -> None:
        """Setting params before add_data() names the missing step on both classes."""
        if kind == "Fitter":
            fitter = Fitter(["b"], Parameterisation("P K e w Tc"))
            params = test_circular_params
        else:
            fitter = GPFitter(["b"], Parameterisation("P K e w Tc"), GPKernel("Quasiperiodic"))
            params = test_gp_all_params

        with pytest.raises(RuntimeError, match=r"add_data\(\) must be called"):
            fitter.params = params

    @pytest.mark.parametrize("title, expected", [("My title", "My title"), (None, ""), ("", "")],
                             ids=["custom", "none", "empty"])
    def test_plot_corner_title(self, fitted, title, expected) -> None:
        """plot_corner draws the title given, or none for None or ""."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close("all")

        fitted.plot_corner(title=title)

        assert plt.gcf().get_suptitle() == expected
        plt.close("all")

    def test_rv_total_from_samples_progress_false(self, fitted, capsys) -> None:
        """progress=False reaches the trend, planet and GP calculations: no bars at all."""
        capsys.readouterr()

        fitted.calculate_rv_total_from_samples(np.linspace(0, 5, 7), discard_start=10, progress=False)

        assert "from samples" not in capsys.readouterr().err

    def test_rv_total_from_samples_progress_true(self, fitted, capsys) -> None:
        """progress=True shows a bar for each of the trend, planet and GP calculations."""
        capsys.readouterr()

        fitted.calculate_rv_total_from_samples(np.linspace(0, 5, 7), discard_start=10, progress=True)

        err = capsys.readouterr().err
        for label in ("Calculating trend RV", "Calculating planet b RV", "Calculating GP"):
            assert label in err
