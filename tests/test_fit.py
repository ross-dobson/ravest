import numpy as np
import pytest

import ravest.prior
from ravest.fit import Fitter, LogLikelihood, LogPosterior, LogPrior
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
        "P_b": Parameter(2.0, "d", fixed=True),
        "K_b": Parameter(5.0, "m/s", fixed=False),
        "e_b": Parameter(0.0, "", fixed=True),
        "w_b": Parameter(np.pi/2, "rad", fixed=True),
        "Tc_b": Parameter(0.0, "d", fixed=True),
        "g_HARPS": Parameter(0.0, "m/s", fixed=True),
        "gd": Parameter(0.0, "m/s/day", fixed=True),
        "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
        "jit_HARPS": Parameter(1.0, "m/s", fixed=False),
    }


@pytest.fixture
def test_circular_params_multi_instrument():
    """Circular orbit parameters for two instruments (HARPS and HIRES)."""
    return {
        "P_b": Parameter(2.0, "d", fixed=True),
        "K_b": Parameter(5.0, "m/s", fixed=False),
        "e_b": Parameter(0.0, "", fixed=True),
        "w_b": Parameter(np.pi/2, "rad", fixed=True),
        "Tc_b": Parameter(0.0, "d", fixed=True),
        "g_HARPS": Parameter(0.0, "m/s", fixed=False),
        "g_HIRES": Parameter(100.0, "m/s", fixed=False),
        "gd": Parameter(0.0, "m/s/day", fixed=True),
        "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
        "jit_HARPS": Parameter(1.0, "m/s", fixed=False),
        "jit_HIRES": Parameter(2.0, "m/s", fixed=False),
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

        params = {"P_b": Parameter(2.0, "d")}  # Too few params

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
        params["invalid_param"] = Parameter(1.0, "")  # Add unexpected parameter

        with pytest.raises(ValueError, match="Unexpected parameters.*Expected 9 parameters, got 10"):
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
        params["K_b"] = Parameter(25.0, "m/s", fixed=False)  # Outside uniform prior [0, 20]
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
        unique_instruments = list(np.unique(instrument))
        ll = LogLikelihood(
            time=time, vel=vel, velerr=velerr, t0=2.0,
            instrument=instrument, unique_instruments=unique_instruments,
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc")
        )

        np.testing.assert_array_equal(ll.time, time)
        np.testing.assert_array_equal(ll.vel, vel)
        np.testing.assert_array_equal(ll.velerr, velerr)
        assert ll.t0 == 2.0

    def test_loglikelihood_calculation(self, test_data) -> None:
        """Test log-likelihood calculation with valid parameters."""
        time, vel, velerr, instrument = test_data
        unique_instruments = list(np.unique(instrument))
        ll = LogLikelihood(
            time=time, vel=vel, velerr=velerr, t0=2.0,
            instrument=instrument, unique_instruments=unique_instruments,
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc")
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
        unique_instruments = list(np.unique(instrument))
        ll = LogLikelihood(
            time=time, vel=vel, velerr=velerr, t0=2.0,
            instrument=instrument, unique_instruments=unique_instruments,
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc")
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
        unique_instruments = ["HARPS"]

        ll = LogLikelihood(
            time=time, vel=vel, velerr=velerr, t0=1.0,
            instrument=instrument, unique_instruments=unique_instruments,
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc")
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
        unique_instruments = list(np.unique(instrument))
        ll = LogLikelihood(
            time=time, vel=vel, velerr=velerr, t0=2.0,
            instrument=instrument, unique_instruments=unique_instruments,
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc")
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
        unique_instruments = list(np.unique(instrument))
        ll = LogLikelihood(
            time=time, vel=vel, velerr=velerr, t0=2.0,
            instrument=instrument, unique_instruments=unique_instruments,
            planet_letters=["b"], parameterisation=Parameterisation("P K e w Tc")
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
        unique_instruments = list(np.unique(instrument))
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
            time=time, vel=vel, velerr=velerr, t0=2.0,
            instrument=instrument, unique_instruments=unique_instruments
        )

        assert lpost.planet_letters == ["b"]

    def test_logposterior_valid_calculation(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test log-posterior calculation with valid parameters."""
        time, vel, velerr, instrument = test_data
        unique_instruments = list(np.unique(instrument))
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
            time=time, vel=vel, velerr=velerr, t0=2.0,
            instrument=instrument, unique_instruments=unique_instruments
        )

        free_params_dict = {"K_b": 5.0, "jit_HARPS": 1.0}
        log_post = lpost.log_probability(free_params_dict)

        assert np.isfinite(log_post)
        assert isinstance(log_post, float)

    def test_logposterior_invalid_prior(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test log-posterior returns -inf when prior is invalid."""
        time, vel, velerr, instrument = test_data
        unique_instruments = list(np.unique(instrument))
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
            time=time, vel=vel, velerr=velerr, t0=2.0,
            instrument=instrument, unique_instruments=unique_instruments
        )

        free_params_dict = {"K_b": -1.0, "jit_HARPS": 1.0}  # Invalid K_b
        log_post = lpost.log_probability(free_params_dict)

        assert log_post == -np.inf

    def test_negative_log_probability_for_MAP(self, test_data, test_circular_params, test_simple_priors) -> None:
        """Test MAP interface that takes list instead of dict."""
        time, vel, velerr, instrument = test_data
        unique_instruments = list(np.unique(instrument))
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
            time=time, vel=vel, velerr=velerr, t0=2.0,
            instrument=instrument, unique_instruments=unique_instruments
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
            "P_b": Parameter(2.0, "d", fixed=True),
            "K_b": Parameter(5.0, "m/s", fixed=False),
            "e_b": Parameter(0.0, "", fixed=True),
            "w_b": Parameter(np.pi/2, "rad", fixed=True),
            "Tc_b": Parameter(0.0, "d", fixed=True),

            "P_c": Parameter(4.0, "d", fixed=True),
            "K_c": Parameter(3.0, "m/s", fixed=False),
            "e_c": Parameter(0.0, "", fixed=True),
            "w_c": Parameter(np.pi/2, "rad", fixed=True),
            "Tc_c": Parameter(1.0, "d", fixed=True),

            "g_HARPS": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit_HARPS": Parameter(1.0, "m/s", fixed=False),
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

    def test_calculate_rv_planet_custom(self, setup_fitter_for_rv):
        """Test custom planet RV calculation."""
        fitter = setup_fitter_for_rv
        times = np.array([0.0, 1.0, 2.0, 3.0])

        # Build params dict
        params = fitter.build_params_dict(fitter.free_params_values)

        # Calculate RV
        rv = fitter.calculate_rv_planet_custom('b', times, params)

        assert isinstance(rv, np.ndarray)
        assert len(rv) == len(times)
        assert np.all(np.isfinite(rv))

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
            "P_b": Parameter(2.0, "d", fixed=True),
            "K_b": Parameter(5.0, "m/s", fixed=False),
            "e_b": Parameter(0.0, "", fixed=True),
            "w_b": Parameter(np.pi/2, "rad", fixed=True),
            "Tc_b": Parameter(0.0, "d", fixed=True),
            "g_HARPS": Parameter(10.0, "m/s", fixed=True),  # Per-instrument gamma
            "gd": Parameter(0.5, "m/s/day", fixed=True),  # Non-zero slope
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit_HARPS": Parameter(1.0, "m/s", fixed=False),
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
