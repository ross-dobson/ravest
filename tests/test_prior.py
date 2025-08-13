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


class TestBoundedGaussian:
    """Tests for the BoundedGaussian prior class"""

    def test_bounded_gaussian_init(self):
        """Test BoundedGaussian prior initialization"""
        prior = ravest.prior.BoundedGaussian(5.0, 1.0, 0.0, 10.0)
        assert prior.mean == 5.0
        assert prior.std == 1.0
        assert prior.lower == 0.0
        assert prior.upper == 10.0

    def test_bounded_gaussian_within_bounds(self):
        """Test log probability for values within bounds"""
        prior = ravest.prior.BoundedGaussian(5.0, 1.0, 0.0, 10.0)

        # At mean, should match regular Gaussian
        expected_at_mean = -0.5 * np.log(2 * np.pi * 1.0)
        assert np.isclose(prior(5.0), expected_at_mean)

        # At other points within bounds
        expected_at_6 = -0.5 * ((6.0 - 5.0) / 1.0)**2 - 0.5 * np.log(2 * np.pi * 1.0)
        assert np.isclose(prior(6.0), expected_at_6)

    def test_bounded_gaussian_outside_bounds(self):
        """Test log probability for values outside bounds"""
        prior = ravest.prior.BoundedGaussian(5.0, 1.0, 0.0, 10.0)

        assert prior(-1.0) == -np.inf  # Below lower bound
        assert prior(11.0) == -np.inf  # Above upper bound

    def test_bounded_gaussian_at_bounds(self):
        """Test log probability at the boundary values"""
        prior = ravest.prior.BoundedGaussian(5.0, 1.0, 0.0, 10.0)

        # At boundaries, should behave like regular Gaussian
        expected_at_0 = -0.5 * ((0.0 - 5.0) / 1.0)**2 - 0.5 * np.log(2 * np.pi * 1.0)
        expected_at_10 = -0.5 * ((10.0 - 5.0) / 1.0)**2 - 0.5 * np.log(2 * np.pi * 1.0)

        assert np.isclose(prior(0.0), expected_at_0)
        assert np.isclose(prior(10.0), expected_at_10)

    def test_bounded_gaussian_repr(self):
        """Test string representation"""
        prior = ravest.prior.BoundedGaussian(1.0, 0.5, -2.0, 4.0)
        assert repr(prior) == "BoundedGaussian(1.0, 0.5, -2.0, 4.0)"
