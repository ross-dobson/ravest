#!/usr/bin/env python3
"""Test parameterisation flexibility - comprehensive test suite.

Tests both prior validation/setup and MCMC integration.
"""

import numpy as np
import pytest

from ravest.fit import Fitter
from ravest.param import Parameter, Parameterisation
from ravest.prior import Uniform


class TestParameterisationFlexibility:
    """Test parameterisation flexibility scenarios."""

    @pytest.fixture
    def mcmc_test_data(self):
        """Generate fake data for MCMC testing."""
        np.random.seed(42)
        time = np.linspace(0, 50, 100)
        # Simple sinusoidal RV curve with noise
        true_rv = 5 * np.sin(2 * np.pi * time / 5.0) + np.random.normal(0, 1, len(time))
        verr = np.ones_like(time) * 1.0
        return time, true_rv, verr

    def test_default_to_default_priors(self) -> None:
        """Test default parameterisation with default priors."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "e_b": Parameter(0.3, "", fixed=False),
            "w_b": Parameter(np.radians(47), "rad", fixed=False),
            "Tp_b": Parameter(68.7, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "e_b": Uniform(0, 1),
            "w_b": Uniform(-np.pi, np.pi),
            "Tp_b": Uniform(64, 70),
            "jit": Uniform(0, 5)
        }
        fitter = Fitter(["b"], Parameterisation("P K e w Tp"))
        fitter.params = params
        fitter.priors = priors
        # Should not raise exception

    def test_transformed_to_transformed_priors_secosw_sesinw(self) -> None:
        """Test secosw/sesinw parameterisation with secosw/sesinw priors."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "secosw_b": Parameter(0.3735, "", fixed=False),
            "sesinw_b": Parameter(0.4006, "", fixed=False),
            "Tc_b": Parameter(69.0, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "secosw_b": Uniform(-1, 1),
            "sesinw_b": Uniform(-1, 1),
            "Tc_b": Uniform(64, 70),
            "jit": Uniform(0, 5)
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.params = params
        fitter.priors = priors
        # Should not raise exception

    def test_transformed_to_transformed_priors_ecosw_esinw(self) -> None:
        """Test ecosw/esinw parameterisation with ecosw/esinw priors."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "ecosw_b": Parameter(0.1, "", fixed=False),
            "esinw_b": Parameter(0.2, "", fixed=False),
            "Tc_b": Parameter(69.0, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "ecosw_b": Uniform(-1, 1),
            "esinw_b": Uniform(-1, 1),
            "Tc_b": Uniform(64, 70),
            "jit": Uniform(0, 5)
        }
        fitter = Fitter(["b"], Parameterisation("P K ecosw esinw Tc"))
        fitter.params = params
        fitter.priors = priors
        # Should not raise exception

    def test_case3_secosw_sesinw_with_default_priors(self) -> None:
        """Test secosw/sesinw parameterisation with default priors (Case 3)."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "secosw_b": Parameter(0.3735, "", fixed=False),
            "sesinw_b": Parameter(0.4006, "", fixed=False),
            "Tc_b": Parameter(69.0, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "e_b": Uniform(0, 1),           # Default equivalent to secosw/sesinw
            "w_b": Uniform(-np.pi, np.pi),  # Default equivalent to secosw/sesinw
            "Tp_b": Uniform(64, 70),        # Default equivalent to tc
            "jit": Uniform(0, 5)
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.params = params
        fitter.priors = priors
        # Should not raise exception

    def test_case3_ecosw_esinw_with_default_priors(self) -> None:
        """Test ecosw/esinw parameterisation with default priors (Case 3)."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "ecosw_b": Parameter(0.1, "", fixed=False),
            "esinw_b": Parameter(0.2, "", fixed=False),
            "Tc_b": Parameter(69.0, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "e_b": Uniform(0, 1),           # Default equivalent to ecosw/esinw
            "w_b": Uniform(-np.pi, np.pi),  # Default equivalent to ecosw/esinw
            "Tp_b": Uniform(64, 70),        # Default equivalent to tc
            "jit": Uniform(0, 5)
        }
        fitter = Fitter(["b"], Parameterisation("P K ecosw esinw Tc"))
        fitter.params = params
        fitter.priors = priors
        # Should not raise exception

    def test_mixed_priors_per_parameter_flexibility(self) -> None:
        """Test mixed priors (per-parameter flexibility)."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "secosw_b": Parameter(0.3735, "", fixed=False),
            "sesinw_b": Parameter(0.4006, "", fixed=False),
            "Tc_b": Parameter(69.0, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),        # Current parameterisation
            "K_b": Uniform(0, 10),          # Current parameterisation
            "e_b": Uniform(0, 1),           # Default equivalent to secosw/sesinw
            "w_b": Uniform(-np.pi, np.pi),  # Default equivalent to secosw/sesinw
            "Tc_b": Uniform(64, 70),        # Current parameterisation
            "jit": Uniform(0, 5)            # Current parameterisation
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.params = params
        fitter.priors = priors
        # Should not raise exception

    def test_common_parameters_transformed_parameterisation(self) -> None:
        """Test common parameters only with transformed parameterisation."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "secosw_b": Parameter(0.3735, "", fixed=True),    # Fixed
            "sesinw_b": Parameter(0.4006, "", fixed=True),    # Fixed
            "Tc_b": Parameter(69.0, "days", fixed=True),      # Fixed
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "jit": Uniform(0, 5)
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.params = params
        fitter.priors = priors
        # Should not raise exception

    def test_common_parameters_default_parameterisation(self) -> None:
        """Test common parameters only with default parameterisation."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "e_b": Parameter(0.3, "", fixed=True),                # Fixed
            "w_b": Parameter(np.radians(47), "rad", fixed=True),  # Fixed
            "Tp_b": Parameter(68.7, "days", fixed=True),          # Fixed
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "jit": Uniform(0, 5)
        }
        fitter = Fitter(["b"], Parameterisation("P K e w Tp"))
        fitter.params = params
        fitter.priors = priors
        # Should not raise exception

    def test_mixed_secosw_sesinw_coupling_should_fail(self) -> None:
        """Test that mixed secosw/sesinw coupling is rejected."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "secosw_b": Parameter(0.3735, "", fixed=True),     # Fixed
            "sesinw_b": Parameter(0.4006, "", fixed=False),    # Free - violation!
            "Tc_b": Parameter(69.0, "days", fixed=True),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        with pytest.raises(ValueError, match="secosw_b and sesinw_b must both be fixed or both be free"):
            fitter.params = params

    def test_mixed_ecosw_esinw_coupling_should_fail(self) -> None:
        """Test that mixed ecosw/esinw coupling is rejected."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "ecosw_b": Parameter(0.1, "", fixed=True),      # Fixed
            "esinw_b": Parameter(0.2, "", fixed=False),     # Free - violation!
            "Tc_b": Parameter(69.0, "days", fixed=True),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        fitter = Fitter(["b"], Parameterisation("P K ecosw esinw Tc"))
        with pytest.raises(ValueError, match="ecosw_b and esinw_b must both be fixed or both be free"):
            fitter.params = params

    def test_missing_priors_should_fail(self) -> None:
        """Test that missing priors are rejected."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "e_b": Parameter(0.3, "", fixed=False),
            "w_b": Parameter(np.radians(47), "rad", fixed=False),
            "Tp_b": Parameter(68.7, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            # Missing e_b, w_b, tp_b, jit priors
        }
        fitter = Fitter(["b"], Parameterisation("P K e w Tp"))
        fitter.params = params
        with pytest.raises(ValueError, match="Missing priors for parameters"):
            fitter.priors = priors

    def test_invalid_parameter_values_should_fail(self) -> None:
        """Test that invalid parameter values are rejected."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(25.0, "m/s", fixed=False),     # Outside prior bounds [0, 20]
            "e_b": Parameter(0.3, "", fixed=True),
            "w_b": Parameter(np.radians(47), "rad", fixed=True),
            "Tp_b": Parameter(68.7, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 20),          # k_b = 25.0 is outside this range
            "Tp_b": Uniform(64, 70),
            "jit": Uniform(0, 5)
        }
        fitter = Fitter(["b"], Parameterisation("P K e w Tp"))
        fitter.params = params
        with pytest.raises(ValueError, match="Initial value 25.0 of parameter K_b is invalid for prior"):
            fitter.priors = priors

    def test_conflicting_priors_should_fail(self) -> None:
        """Test that providing both current and default parameterisation priors raises an error."""
        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "secosw_b": Parameter(0.1, "", fixed=False),
            "sesinw_b": Parameter(0.0, "", fixed=False),
            "Tc_b": Parameter(25.0, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "secosw_b": Uniform(-0.5, 0.5),   # Current parameterisation
            "sesinw_b": Uniform(-0.5, 0.5),   # Current parameterisation
            "e_b": Uniform(0, 0.5),           # CONFLICT with secosw_b!
            "w_b": Uniform(-np.pi, np.pi),    # CONFLICT with sesinw_b!
            "Tc_b": Uniform(20, 30),          # Current parameterisation
            "Tp_b": Uniform(20, 30),          # CONFLICT with tc_b!
            "jit": Uniform(0, 5)
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.params = params
        with pytest.raises(ValueError, match="Conflicting priors provided for both current and default parameterisations"):
            fitter.priors = priors

    # ==================== MCMC Integration Tests ====================

    def test_mcmc_default_parameterisation_default_priors(self, mcmc_test_data) -> None:
        """Test MCMC: Default parameterisation with default priors."""
        time, true_rv, verr = mcmc_test_data

        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "e_b": Parameter(0.1, "", fixed=False),
            "w_b": Parameter(0.0, "rad", fixed=False),
            "Tp_b": Parameter(25.0, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=False),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(3, 7),
            "K_b": Uniform(0, 10),
            "e_b": Uniform(0, 0.5),
            "w_b": Uniform(-np.pi, np.pi),
            "Tp_b": Uniform(20, 30),
            "g": Uniform(-10, 10),
            "jit": Uniform(0, 5)
        }

        # Setup fitter
        fitter = Fitter(["b"], Parameterisation("P K e w Tp"))
        fitter.add_data(time=time, vel=true_rv, verr=verr, t0=np.mean(time))
        fitter.params = params
        fitter.priors = priors

        # Verify setup
        assert set(fitter.free_params_names) == set(priors.keys())

        # Run short MCMC
        nwalkers = 2 * len(fitter.free_params_names)  # Minimum 2 walkers per free parameter
        initial_positions = fitter.generate_initial_walker_positions(nwalkers)
        fitter.run_mcmc(initial_positions=initial_positions, nwalkers=nwalkers, nsteps=50, progress=False)

        # Verify results
        samples = fitter.get_samples_np(flat=True)
        assert samples.shape == (nwalkers * 50, 7)  # nwalkers * 50 steps

        # Check that we get finite log probabilities
        lnprob = fitter.get_sampler_lnprob(flat=True)
        finite_count = np.sum(np.isfinite(lnprob))
        assert finite_count > 0  # Should have some finite values

    def test_mcmc_transformed_parameterisation_transformed_priors(self, mcmc_test_data) -> None:
        """Test MCMC: Transformed parameterisation with transformed priors."""
        time, true_rv, verr = mcmc_test_data

        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "secosw_b": Parameter(0.1, "", fixed=False),
            "sesinw_b": Parameter(0.0, "", fixed=False),
            "Tc_b": Parameter(25.0, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=False),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(3, 7),
            "K_b": Uniform(0, 10),
            "secosw_b": Uniform(-0.5, 0.5),
            "sesinw_b": Uniform(-0.5, 0.5),
            "Tc_b": Uniform(20, 30),
            "g": Uniform(-10, 10),
            "jit": Uniform(0, 5)
        }

        # Setup fitter
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.add_data(time=time, vel=true_rv, verr=verr, t0=np.mean(time))
        fitter.params = params
        fitter.priors = priors

        # Verify setup
        assert set(fitter.free_params_names) == set(priors.keys())

        # Run short MCMC
        nwalkers = 2 * len(fitter.free_params_names)  # Minimum 2 walkers per free parameter
        initial_positions = fitter.generate_initial_walker_positions(nwalkers)
        fitter.run_mcmc(initial_positions=initial_positions, nwalkers=nwalkers, nsteps=50, progress=False)

        # Verify results
        samples = fitter.get_samples_np(flat=True)
        assert samples.shape == (nwalkers * 50, 7)  # nwalkers * 50 steps

        # Check that we get finite log probabilities
        lnprob = fitter.get_sampler_lnprob(flat=True)
        finite_count = np.sum(np.isfinite(lnprob))
        assert finite_count > 0

    def test_mcmc_transformed_parameterisation_default_priors(self, mcmc_test_data) -> None:
        """Test MCMC: Transformed parameterisation with default priors (Case 3)."""
        time, true_rv, verr = mcmc_test_data

        params = {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "secosw_b": Parameter(0.1, "", fixed=False),
            "sesinw_b": Parameter(0.0, "", fixed=False),
            "Tc_b": Parameter(25.0, "days", fixed=False),
            "g": Parameter(0.0, "m/s", fixed=False),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit": Parameter(1.0, "m/s", fixed=False)
        }
        priors = {
            "P_b": Uniform(3, 7),
            "K_b": Uniform(0, 10),
            "e_b": Uniform(0, 0.5),           # Default equivalent to secosw/sesinw
            "w_b": Uniform(-np.pi, np.pi),    # Default equivalent to secosw/sesinw
            "Tp_b": Uniform(20, 30),          # Default equivalent to tc
            "g": Uniform(-10, 10),
            "jit": Uniform(0, 5)
        }

        # Setup fitter
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.add_data(time=time, vel=true_rv, verr=verr, t0=np.mean(time))
        fitter.params = params
        fitter.priors = priors

        # Verify setup - different from previous tests
        assert set(fitter.free_params_names) != set(priors.keys())  # Should be different!
        assert "secosw_b" in fitter.free_params_names
        assert "e_b" in fitter.priors

        # Run short MCMC
        nwalkers = 2 * len(fitter.free_params_names)  # Minimum 2 walkers per free parameter
        initial_positions = fitter.generate_initial_walker_positions(nwalkers)
        fitter.run_mcmc(initial_positions=initial_positions, nwalkers=nwalkers, nsteps=50, progress=False)

        # Verify results
        samples = fitter.get_samples_np(flat=True)
        assert samples.shape == (nwalkers * 50, 7)  # nwalkers * 50 steps

        # Check that we get finite log probabilities
        lnprob = fitter.get_sampler_lnprob(flat=True)
        finite_count = np.sum(np.isfinite(lnprob))
        assert finite_count > 0
