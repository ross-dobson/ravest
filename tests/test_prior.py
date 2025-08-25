import json
from pathlib import Path

import numpy as np
import pytest

import ravest.prior


class TestUniform:
    """Tests for the Uniform prior class"""

    def test_uniform_init(self):
        """Test Uniform prior initialization"""
        prior = ravest.prior.Uniform(0, 10)
        assert prior.lower == 0
        assert prior.upper == 10

    def test_uniform_valid_values(self):
        """Test log probability for valid values"""
        prior = ravest.prior.Uniform(0, 10)

        # Inside bounds should return -log(width)
        expected = -np.log(10)
        assert np.isclose(prior(5.0), expected)
        assert np.isclose(prior(0.0), expected)  # At lower bound
        assert np.isclose(prior(10.0), expected)  # At upper bound

    def test_uniform_invalid_values(self):
        """Test log probability for values outside bounds"""
        prior = ravest.prior.Uniform(0, 10)

        assert prior(-1.0) == -np.inf  # Below lower bound
        assert prior(11.0) == -np.inf  # Above upper bound

    def test_uniform_repr(self):
        """Test string representation"""
        prior = ravest.prior.Uniform(2.5, 7.5)
        assert repr(prior) == "Uniform(2.5, 7.5)"


class TestGaussian:
    """Tests for the Gaussian prior class"""

    def test_gaussian_init(self):
        """Test Gaussian prior initialization"""
        prior = ravest.prior.Gaussian(0, 1)
        assert prior.mean == 0
        assert prior.std == 1

    def test_gaussian_at_mean(self):
        """Test log probability at the mean"""
        prior = ravest.prior.Gaussian(5.0, 2.0)

        # At mean, log_prob = -0.5 * ln(2*pi*sigma**2)
        expected = -0.5 * np.log(2 * np.pi * 4.0)  # sigma**2 = 4
        assert np.isclose(prior(5.0), expected)

    def test_gaussian_symmetric(self):
        """Test that Gaussian is symmetric around mean"""
        prior = ravest.prior.Gaussian(0, 1)

        # Should be symmetric around mean
        assert np.isclose(prior(1.0), prior(-1.0))
        assert np.isclose(prior(2.0), prior(-2.0))

    def test_gaussian_decreases_with_distance(self):
        """Test that probability decreases as we move away from mean"""
        prior = ravest.prior.Gaussian(0, 1)

        prob_at_mean = prior(0.0)
        prob_at_1std = prior(1.0)
        prob_at_2std = prior(2.0)

        assert prob_at_mean > prob_at_1std > prob_at_2std

    def test_gaussian_repr(self):
        """Test string representation"""
        prior = ravest.prior.Gaussian(1.5, 0.5)
        assert repr(prior) == "Gaussian(1.5, 0.5)"


class TestEccentricityPrior:
    """Tests for the EccentricityPrior class"""

    def test_eccentricity_prior_init_valid(self):
        """Test valid initialization"""
        prior = ravest.prior.EccentricityPrior(0.5)
        assert prior.upper == 0.5

    def test_eccentricity_prior_init_invalid(self):
        """Test invalid initialization parameters"""
        # Upper bound >= 1 should raise error
        with pytest.raises(ValueError, match="Upper bound of eccentricity must be less than 1"):
            ravest.prior.EccentricityPrior(1.0)

        with pytest.raises(ValueError, match="Upper bound of eccentricity must be less than 1"):
            ravest.prior.EccentricityPrior(1.5)

        # Upper bound <= 0 should raise error
        with pytest.raises(ValueError, match="Upper bound of eccentricity must be greater than 0"):
            ravest.prior.EccentricityPrior(0.0)

        with pytest.raises(ValueError, match="Upper bound of eccentricity must be greater than 0"):
            ravest.prior.EccentricityPrior(-0.1)

    def test_eccentricity_prior_valid_values(self):
        """Test log probability for valid eccentricity values"""
        prior = ravest.prior.EccentricityPrior(0.8)

        # Inside bounds should return -log(upper)
        expected = -np.log(0.8)
        assert np.isclose(prior(0.0), expected)  # At zero (inclusive)
        assert np.isclose(prior(0.4), expected)  # Middle value
        assert np.isclose(prior(0.8), expected)  # At upper bound

    def test_eccentricity_prior_invalid_values(self):
        """Test log probability for invalid eccentricity values"""
        prior = ravest.prior.EccentricityPrior(0.8)

        assert prior(-0.1) == -np.inf  # Below zero
        assert prior(0.9) == -np.inf   # Above upper bound
        assert prior(1.0) == -np.inf   # At 1.0

    def test_eccentricity_prior_repr(self):
        """Test string representation"""
        prior = ravest.prior.EccentricityPrior(0.7)
        assert repr(prior) == "EccentricityPrior(0.7)"


class TestTruncatedGaussian:
    """Tests for the TruncatedGaussian prior class"""

    def test_truncated_gaussian_init(self):
        """Test TruncatedGaussian prior initialization"""
        prior = ravest.prior.TruncatedGaussian(5.0, 1.0, 0.0, 10.0)
        assert prior.mean == 5.0
        assert prior.std == 1.0
        assert prior.lower == 0.0
        assert prior.upper == 10.0

    def test_truncated_gaussian_init_invalid_std(self):
        """Test invalid standard deviation"""
        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            ravest.prior.TruncatedGaussian(5.0, 0.0, 0.0, 10.0)

        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            ravest.prior.TruncatedGaussian(5.0, -1.0, 0.0, 10.0)

    def test_truncated_gaussian_init_invalid_bounds(self):
        """Test invalid bounds"""
        with pytest.raises(ValueError, match="Lower bound must be less than upper bound"):
            ravest.prior.TruncatedGaussian(5.0, 1.0, 10.0, 0.0)

        with pytest.raises(ValueError, match="Lower bound must be less than upper bound"):
            ravest.prior.TruncatedGaussian(5.0, 1.0, 5.0, 5.0)

    def test_truncated_gaussian_within_bounds(self):
        """Test log probability for values within bounds"""
        prior = ravest.prior.TruncatedGaussian(5.0, 1.0, 0.0, 10.0)

        # Should return finite values within bounds
        assert np.isfinite(prior(5.0))  # At mean
        assert np.isfinite(prior(6.0))  # Near mean
        assert np.isfinite(prior(1.0))  # Lower side
        assert np.isfinite(prior(9.0))  # Upper side

    def test_truncated_gaussian_outside_bounds(self):
        """Test log probability for values outside bounds"""
        prior = ravest.prior.TruncatedGaussian(5.0, 1.0, 0.0, 10.0)

        assert prior(-1.0) == -np.inf  # Below lower bound
        assert prior(11.0) == -np.inf  # Above upper bound

    def test_truncated_gaussian_at_bounds(self):
        """Test log probability at the boundary values"""
        prior = ravest.prior.TruncatedGaussian(5.0, 1.0, 0.0, 10.0)

        # At boundaries, should return finite values
        assert np.isfinite(prior(0.0))
        assert np.isfinite(prior(10.0))

    def test_truncated_gaussian_normalization(self):
        """Test that truncated Gaussian gives higher probabilities than unbounded (due to normalization)"""
        # Compare truncated vs regular Gaussian - truncated should have higher density due to renormalization

        mean, std = 0.0, 1.0
        lower, upper = -1.0, 1.0  # Narrow truncation

        truncated = ravest.prior.TruncatedGaussian(mean, std, lower, upper)
        regular_gaussian = ravest.prior.Gaussian(mean, std)

        # At the mean, truncated should give higher log probability due to renormalization
        x = 0.0
        trunc_logprob = truncated(x)
        regular_logprob = regular_gaussian(x)

        assert trunc_logprob > regular_logprob, "Truncated Gaussian should have higher density due to renormalization"

    def test_truncated_gaussian_repr(self):
        """Test string representation"""
        prior = ravest.prior.TruncatedGaussian(1.0, 0.5, -2.0, 4.0)
        assert repr(prior) == "TruncatedGaussian(1.0, 0.5, -2.0, 4.0)"


class TestBeta:
    """Tests for the Beta prior class"""

    def test_beta_init(self):
        """Test Beta prior initialization"""
        prior = ravest.prior.Beta(2.0, 5.0)
        assert prior.a == 2.0
        assert prior.b == 5.0

    def test_beta_init_invalid_a(self):
        """Test invalid a parameter"""
        with pytest.raises(ValueError, match="Value of a > 0 required"):
            ravest.prior.Beta(0.0, 5.0)

        with pytest.raises(ValueError, match="Value of a > 0 required"):
            ravest.prior.Beta(-1.0, 5.0)

    def test_beta_init_invalid_b(self):
        """Test invalid b parameter"""
        with pytest.raises(ValueError, match="Value of b > 0 required"):
            ravest.prior.Beta(2.0, 0.0)

        with pytest.raises(ValueError, match="Value of b > 0 required"):
            ravest.prior.Beta(2.0, -1.0)

    def test_beta_valid_values(self):
        """Test log probability for valid values"""
        prior = ravest.prior.Beta(2.0, 5.0)

        # Should return finite values for values in (0,1)
        assert np.isfinite(prior(0.1))
        assert np.isfinite(prior(0.5))
        assert np.isfinite(prior(0.9))

    def test_beta_out_of_bounds(self):
        """Test behavior for values outside [0,1]"""
        prior = ravest.prior.Beta(2.0, 3.0)

        # Below lower bound
        assert prior(-0.5) == -np.inf
        assert prior(-0.001) == -np.inf

        # Above upper bound
        assert prior(1.001) == -np.inf
        assert prior(1.5) == -np.inf

    def test_beta_uniform_case(self):
        """Test Beta(1,1) which should be uniform on [0,1]"""
        uniform_beta = ravest.prior.Beta(1.0, 1.0)

        # Should return 0.0 (log of 1.0) for all values in [0,1]
        test_values = [0.1, 0.5, 0.9]
        for x in test_values:
            result = uniform_beta(x)
            assert np.isclose(result, 0.0), f"Beta(1,1) should return 0.0 for x={x}, got {result}"

    def test_beta_repr(self):
        """Test string representation"""
        prior = ravest.prior.Beta(1.58, 4.4)
        assert repr(prior) == "Beta(1.58, 4.4)"

    @pytest.fixture(scope="class")
    def beta_reference_data(self):
        """Load reference test data generated from external Log Beta prior"""
        data_file = Path(__file__).parent / "data" / "beta_reference.json"
        with open(data_file, 'r') as f:
            return json.load(f)

    def test_beta_against_external_reference(self, beta_reference_data):
        """Test Beta implementation against external reference values for numerical accuracy"""

        for case in beta_reference_data:
            alpha = case['alpha']
            beta = case['beta']
            description = case['description']

            # Create our Beta prior instance
            prior = ravest.prior.Beta(alpha, beta)

            for x, expected_result in case['test_results']:
                if expected_result == '-inf':
                    expected = -np.inf
                elif expected_result == '+inf':
                    expected = +np.inf
                else:
                    expected = expected_result

                actual = prior(x)

                # Use appropriate comparison for infinite vs finite values
                if np.isinf(expected):
                    assert np.isinf(actual), (
                        f"Beta({alpha}, {beta}) [{description}] at x={x}: "
                        f"expected {expected}, got {actual}"
                    )
                    # Check sign of infinity matches
                    if np.isposinf(expected):
                        assert np.isposinf(actual), (
                            f"Beta({alpha}, {beta}) [{description}] at x={x}: "
                            f"expected +inf, got {actual}"
                        )
                    elif np.isneginf(expected):
                        assert np.isneginf(actual), (
                            f"Beta({alpha}, {beta}) [{description}] at x={x}: "
                            f"expected -inf, got {actual}"
                        )
                else:
                    assert np.isclose(actual, expected), (
                        f"Beta({alpha}, {beta}) [{description}] at x={x}: "
                        f"expected {expected:.15e}, got {actual:.15e}, "
                        f"diff={abs(actual - expected):.2e}"
                    )
