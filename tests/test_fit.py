import numpy as np
import pytest

import ravest.prior
from ravest.fit import Fitter, LogLikelihood, LogPosterior, LogPrior
from ravest.param import Parameter, Parameterisation


@pytest.fixture
def test_data():
    """Simple synthetic RV data for testing"""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    vel = np.array([5.0, -2.0, -5.0, 2.0, 3.0])
    verr = np.array([1.0, 1.1, 0.9, 0.85, 1.5])
    return time, vel, verr


@pytest.fixture
def test_circular_params():
    """Simple circular orbit parameters for testing"""
    return {
        "per_b": Parameter(2.0, "d", fixed=True),
        "k_b": Parameter(5.0, "m/s", fixed=False),
        "e_b": Parameter(0.0, "", fixed=True),
        "w_b": Parameter(np.pi/2, "rad", fixed=True),
        "tc_b": Parameter(0.0, "d", fixed=True),
        "g": Parameter(0.0, "m/s", fixed=True),
        "gd": Parameter(0.0, "m/s/day", fixed=True),
        "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
        "jit": Parameter(1.0, "m/s", fixed=False),
    }


@pytest.fixture
def test_simple_priors():
    """Simple priors for testing"""
    return {
        "k_b": ravest.prior.Uniform(0, 20),
        "jit": ravest.prior.Uniform(0, 5),
    }


class TestFitter:
    """Tests for the main Fitter class"""

    def test_fitter_init(self):
        """Test Fitter initialization"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        assert fitter.planet_letters == ["b"]
        assert fitter.parameterisation.parameterisation == "per k e w tc"
        assert fitter.params == {}
        assert fitter.priors == {}

    def test_add_data_valid(self, test_data):
        """Test adding valid data"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        time, vel, verr = test_data
        fitter.add_data(time, vel, verr, t0=2.0)

        np.testing.assert_array_equal(fitter.time, time)
        np.testing.assert_array_equal(fitter.vel, vel)
        np.testing.assert_array_equal(fitter.verr, verr)
        assert fitter.t0 == 2.0

    def test_add_data_mismatched_lengths(self):
        """Test error when data arrays have different lengths"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        time = np.array([0.0, 1.0])
        vel = np.array([5.0, -2.0, -5.0])  # Different length
        verr = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="Time, velocity, and uncertainty arrays must be the same length"):
            fitter.add_data(time, vel, verr, t0=2.0)

    def test_params_property_valid(self, test_circular_params):
        """Test setting valid parameters via property"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = test_circular_params
        fitter.params = params

        assert len(fitter.params) == 9  # 5 planetary + 3 trend params + jit
        assert "per_b" in fitter.params
        assert "jit" in fitter.params

    def test_add_params_wrong_count(self):
        """Test error when wrong number of parameters provided"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = {"per_b": Parameter(2.0, "d")}  # Too few params, only 1 out of 9 provided

        with pytest.raises(ValueError, match="Missing required parameters.*Expected 9 parameters, got 1"):
            fitter.params = params

    def test_add_params_missing_planetary_param(self, test_circular_params):
        """Test error when planetary parameter is missing"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = test_circular_params.copy()
        del params["per_b"]  # Remove required parameter

        with pytest.raises(ValueError, match="Missing required parameters.*Expected 9 parameters, got 8"):
            fitter.params = params

    def test_add_params_unexpected_param(self, test_circular_params):
        """Test error when unexpected parameter is provided"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = test_circular_params.copy()
        params["invalid_param"] = Parameter(1.0, "")  # Add unexpected parameter

        with pytest.raises(ValueError, match="Unexpected parameters.*Expected 9 parameters, got 10"):
            fitter.params = params

    def test_add_priors_valid(self, test_circular_params, test_simple_priors):
        """Test adding valid priors"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = test_circular_params
        priors = test_simple_priors

        fitter.params = params
        fitter.priors = priors

        assert len(fitter.priors) == 2
        assert "k_b" in fitter.priors
        assert "jit" in fitter.priors

    def test_add_priors_missing_prior(self, test_circular_params):
        """Test error when prior is missing for free parameter"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = test_circular_params
        priors = {"k_b": ravest.prior.Uniform(0, 20)}  # Missing jit prior

        fitter.params = params
        with pytest.raises(ValueError, match="Missing priors for"):
            fitter.priors = priors

    def test_add_priors_invalid_initial_value(self, test_circular_params, test_simple_priors):
        """Test error when initial parameter value is outside prior bounds"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = test_circular_params.copy()
        params["k_b"].value = 25.0  # Outside uniform prior [0, 20]
        priors = test_simple_priors

        fitter.params = params
        with pytest.raises(ValueError, match="Initial value 25.0 of parameter k_b is invalid"):
            fitter.priors = priors

    def test_add_priors_too_many_warning(self, test_circular_params):
        """Test warning when too many priors provided (for fixed params)"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = test_circular_params
        fitter.params = params

        # Add priors for both free AND fixed parameters
        priors = {
            "k_b": ravest.prior.Uniform(0, 20),
            "jit": ravest.prior.Uniform(0, 5),
            "per_b": ravest.prior.Uniform(1, 5),  # This is fixed!
        }

        with pytest.raises(ValueError, match="Unexpected priors.*Expected 2 priors, got 3"):
            fitter.priors = priors

    def test_get_free_params(self, test_circular_params):
        """Test getting free parameters"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = test_circular_params
        fitter.params = params

        free_params = fitter.free_params_dict
        free_names = fitter.free_params_names
        free_vals = fitter.free_params_values

        assert len(free_params) == 2  # k_b and jit
        assert "k_b" in free_names
        assert "jit" in free_names
        assert len(free_vals) == 2
        assert 5.0 in free_vals  # k_b value
        assert 1.0 in free_vals  # jit value

    def test_get_fixed_params(self, test_circular_params):
        """Test getting fixed parameters"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))
        params = test_circular_params
        fitter.params = params

        fixed_params = fitter.fixed_params_dict
        fixed_names = fitter.fixed_params_names
        fixed_vals = fitter.fixed_params_values

        assert len(fixed_params) == 7  # All except k_b and jit
        assert "per_b" in fixed_names
        assert "e_b" in fixed_names
        assert len(fixed_vals) == 7


class TestLogLikelihood:
    """Tests for the LogLikelihood class"""

    def test_loglikelihood_init(self, test_data):
        """Test LogLikelihood initialization"""
        time, vel, verr = test_data
        ll = LogLikelihood(
            time=time, vel=vel, verr=verr, t0=2.0,
            planet_letters=["b"], parameterisation=Parameterisation("per k e w tc")
        )

        np.testing.assert_array_equal(ll.time, time)
        np.testing.assert_array_equal(ll.vel, vel)
        np.testing.assert_array_equal(ll.verr, verr)
        assert ll.t0 == 2.0
        assert len(ll.expected_params) == 9

    def test_loglikelihood_calculation(self, test_data):
        """Test log-likelihood calculation with valid parameters"""
        time, vel, verr = test_data
        ll = LogLikelihood(
            time=time, vel=vel, verr=verr, t0=2.0,
            planet_letters=["b"], parameterisation=Parameterisation("per k e w tc")
        )

        params = {
            "per_b": 2.0, "k_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "tc_b": 0.0,
            "g": 0.0, "gd": 0.0, "gdd": 0.0, "jit": 2.0
        }

        log_like = ll(params)
        assert np.isfinite(log_like)
        assert isinstance(log_like, float)

    def test_loglikelihood_invalid_planet(self, test_data):
        """Test log-likelihood returns -inf for invalid planet parameters"""
        time, vel, verr = test_data
        ll = LogLikelihood(
            time=time, vel=vel, verr=verr, t0=2.0,
            planet_letters=["b"], parameterisation=Parameterisation("per k e w tc")
        )

        params = {
            "per_b": -1.0,  # Invalid negative period
            "k_b": 5.0, "e_b": 0.0, "w_b": np.pi/2, "tc_b": 0.0,
            "g": 0.0, "gd": 0.0, "gdd": 0.0, "jit": 1.0
        }

        log_like = ll(params)
        assert log_like == -np.inf

    def test_loglikelihood_perfect_fit(self):
        """Test log-likelihood when model perfectly fits data"""
        # Create synthetic data from known model
        time = np.array([0.0, 0.5, 1.0, 1.5])
        # Constant velocity (no planet signal)
        vel = np.array([2.0, 2.0, 2.0, 2.0])
        verr = np.array([1.0, 1.0, 1.0, 1.0])

        ll = LogLikelihood(
            time=time, vel=vel, verr=verr, t0=1.0,
            planet_letters=["b"], parameterisation=Parameterisation("per k e w tc")
        )

        params = {
            "per_b": 10.0, "k_b": 0.5, "e_b": 0.0, "w_b": np.pi/2, "tc_b": 0.0,
            "g": 2.0, "gd": 0.0, "gdd": 0.0, "jit": 1.0
        }

        log_like = ll(params)
        # Should be finite for valid parameters
        assert np.isfinite(log_like)


class TestLogPrior:
    """Tests for the LogPrior class"""

    def test_logprior_init(self, test_simple_priors):
        """Test LogPrior initialization"""
        priors = test_simple_priors
        lp = LogPrior(priors)
        assert lp.priors == priors

    def test_logprior_valid_params(self, test_simple_priors):
        """Test log-prior calculation with valid parameters"""
        priors = test_simple_priors
        lp = LogPrior(priors)

        params = {"k_b": 10.0, "jit": 2.0}
        log_prior = lp(params)

        assert np.isfinite(log_prior)
        assert isinstance(log_prior, float)

    def test_logprior_invalid_params(self, test_simple_priors):
        """Test log-prior returns -inf for parameters outside bounds"""
        priors = test_simple_priors
        lp = LogPrior(priors)

        params = {"k_b": -5.0, "jit": 2.0}  # k_b outside [0, 20]
        log_prior = lp(params)

        assert log_prior == -np.inf

    def test_logprior_multiple_params(self):
        """Test log-prior sums correctly across multiple parameters"""
        priors = {
            "k_b": ravest.prior.Uniform(0, 10),  # log_prior = -log(10)
            "jit": ravest.prior.Uniform(0, 5),   # log_prior = -log(5)
        }
        lp = LogPrior(priors)

        params = {"k_b": 5.0, "jit": 2.5}
        log_prior = lp(params)

        expected = -np.log(10) - np.log(5)
        assert np.isclose(log_prior, expected)


class TestLogPosterior:
    """Tests for the LogPosterior class (integration tests)"""

    def test_logposterior_init(self, test_data, test_circular_params, test_simple_priors):
        """Test LogPosterior initialization"""
        time, vel, verr = test_data
        params = test_circular_params
        priors = test_simple_priors

        # Extract fixed params
        fixed_params = {k: v for k, v in params.items() if v.fixed}
        free_param_names = [k for k, v in params.items() if not v.fixed]

        lpost = LogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("per k e w tc"),
            priors=priors,
            fixed_params=fixed_params,
            free_params_names=free_param_names,
            time=time, vel=vel, verr=verr, t0=2.0
        )

        assert lpost.planet_letters == ["b"]
        assert len(lpost.expected_params) == 9

    def test_logposterior_valid_calculation(self, test_data, test_circular_params, test_simple_priors):
        """Test log-posterior calculation with valid parameters"""
        time, vel, verr = test_data
        params = test_circular_params
        priors = test_simple_priors

        fixed_params = {k: v for k, v in params.items() if v.fixed}
        free_param_names = [k for k, v in params.items() if not v.fixed]

        lpost = LogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("per k e w tc"),
            priors=priors,
            fixed_params=fixed_params,
            free_params_names=free_param_names,
            time=time, vel=vel, verr=verr, t0=2.0
        )

        free_params_dict = {"k_b": 5.0, "jit": 1.0}
        log_post = lpost.log_probability(free_params_dict)

        assert np.isfinite(log_post)
        assert isinstance(log_post, float)

    def test_logposterior_invalid_prior(self, test_data, test_circular_params, test_simple_priors):
        """Test log-posterior returns -inf when prior is invalid"""
        time, vel, verr = test_data
        params = test_circular_params
        priors = test_simple_priors

        fixed_params = {k: v for k, v in params.items() if v.fixed}
        free_param_names = [k for k, v in params.items() if not v.fixed]

        lpost = LogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("per k e w tc"),
            priors=priors,
            fixed_params=fixed_params,
            free_params_names=free_param_names,
            time=time, vel=vel, verr=verr, t0=2.0
        )

        free_params_dict = {"k_b": -1.0, "jit": 1.0}  # Invalid k_b
        log_post = lpost.log_probability(free_params_dict)

        assert log_post == -np.inf

    def test_negative_log_probability_for_MAP(self, test_data, test_circular_params, test_simple_priors):
        """Test MAP interface that takes list instead of dict"""
        time, vel, verr = test_data
        params = test_circular_params
        priors = test_simple_priors

        fixed_params = {k: v for k, v in params.items() if v.fixed}
        free_param_names = [k for k, v in params.items() if not v.fixed]

        lpost = LogPosterior(
            planet_letters=["b"],
            parameterisation=Parameterisation("per k e w tc"),
            priors=priors,
            fixed_params=fixed_params,
            free_params_names=free_param_names,
            time=time, vel=vel, verr=verr, t0=2.0
        )

        free_params_vals = [5.0, 1.0]  # k_b, jit
        neg_log_post = lpost._negative_log_probability_for_MAP(free_params_vals)

        assert np.isfinite(neg_log_post)
        assert isinstance(neg_log_post, float)

        # Should be negative of log_probability
        free_params_dict = {"k_b": 5.0, "jit": 1.0}
        log_post = lpost.log_probability(free_params_dict)
        assert np.isclose(neg_log_post, -log_post)


class TestFitterIntegration:
    """Integration tests for complete Fitter workflow"""

    def test_complete_setup(self, test_data, test_circular_params, test_simple_priors):
        """Test complete Fitter setup without running MCMC"""
        fitter = Fitter(["b"], Parameterisation("per k e w tc"))

        # Add data
        time, vel, verr = test_data
        fitter.add_data(time, vel, verr, t0=2.0)

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

    def test_multi_planet_setup(self, test_data):
        """Test setup with multiple planets"""
        fitter = Fitter(["b", "c"], Parameterisation("per k e w tc"))

        # Multi-planet parameters
        params = {
            "per_b": Parameter(2.0, "d", fixed=True),
            "k_b": Parameter(5.0, "m/s", fixed=False),
            "e_b": Parameter(0.0, "", fixed=True),
            "w_b": Parameter(np.pi/2, "rad", fixed=True),
            "tc_b": Parameter(0.0, "d", fixed=True),

            "per_c": Parameter(4.0, "d", fixed=True),
            "k_c": Parameter(3.0, "m/s", fixed=False),
            "e_c": Parameter(0.0, "", fixed=True),
            "w_c": Parameter(np.pi/2, "rad", fixed=True),
            "tc_c": Parameter(1.0, "d", fixed=True),

            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False),
        }

        priors = {
            "k_b": ravest.prior.Uniform(0, 20),
            "k_c": ravest.prior.Uniform(0, 20),
            "jit": ravest.prior.Uniform(0, 5),
        }

        time, vel, verr = test_data
        fitter.add_data(time, vel, verr, t0=2.0)
        fitter.params = params
        fitter.priors = priors

        assert len(fitter.params) == 14  # 5*2 planets + 4 system
        assert len(fitter.priors) == 3   # k_b, k_c, jit
        assert len(fitter.free_params_names) == 3
