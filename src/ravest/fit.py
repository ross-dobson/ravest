"""Radial velocity fitting using MCMC and MAP estimation.

This module provides the main Fitter class for fitting radial velocity data
to planetary models using various parameterisations.
"""
# fit.py
import logging
import multiprocessing as mp
import os
import warnings
from typing import Callable, Dict

# Many builds of NumPy are linked against OpenBLAS or MKL, which can use multiple threads
# This can cause problems with multiprocessing (that we use to speed up emcee)
# So we set these to only use one thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import corner
import emcee
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import AutoLocator, AutoMinorLocator, MultipleLocator
from scipy.optimize import minimize
from tinygp import GaussianProcess, kernels
from tqdm import tqdm

import ravest.model
from ravest.gp import GPKernel
from ravest.param import Parameter, Parameterisation

# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO)


class Fitter:
    """Main class for fitting radial velocity data to planetary models.

    Supports MCMC sampling, MAP estimation, and various parameterisations.
    Handles multiple planets, trends, and jitter parameters.
    """

    def __init__(self, planet_letters: list[str], parameterisation: Parameterisation) -> None:
        """Initialize the Fitter object.

        Parameters
        ----------
        planet_letters : list[str]
            List of single-character planet identifiers (e.g., ['b', 'c', 'd']).
            Used to distinguish parameters for different planets in the system.
        parameterisation : Parameterisation
            The orbital parameterisation to use for fitting. Defines which orbital
            elements are used as free/fixed parameters (e.g., 'Default', 'EccentricityWind').
        """
        self.planet_letters = planet_letters
        self.parameterisation = parameterisation

        # Initialize parameter storage
        self._params: Dict[str, Parameter] = {}
        self._priors: Dict[str, Callable[[float], float]] = {}

    def add_data(self, time: np.ndarray, vel: np.ndarray, verr: np.ndarray, t0: float) -> None:
        """Add the data to the Fitter object.

        Parameters
        ----------
        time : array-like
            Time of each observation [days]
        vel : array-like
            Radial velocity at each time [m/s]
        verr : array-like
            Uncertainty on the radial velocity at each time [m/s]
        t0 : float
            Reference time for the trend [days].
            Recommended to set this as mean or median of input `time` array.
        """
        if len(time) != len(vel) or len(time) != len(verr):
            raise ValueError("Time, velocity, and uncertainty arrays must be the same length.")

        self.time = np.ascontiguousarray(time)
        self.vel = np.ascontiguousarray(vel)
        self.verr = np.ascontiguousarray(verr)
        self.t0 = t0

    @property
    def params(self) -> Dict[str, Parameter]:
        """Parameters dictionary. Set via: fitter.params = param_dict."""
        return self._params

    @params.setter
    def params(self, new_params: Dict[str, Parameter]) -> None:
        """Set parameters with a dict, checking all required params are present.

        You can update all or some of the parameters at once, example:
        >>> fitter.params = {"g": Parameter(1.0, "m/s"), "gd": Parameter(0.1, "m/s/d")}  # only update trend parameters
        >>> fitter.params = {"P_c": Parameter(5.0, "d"), "K_c": Parameter(3.5, "m/s")}  # only update some of planet C parameters

        Parameters
        ----------
        new_params : dict
            Dictionary of new parameter values to set.

            The keys of this dictionary should match the parameter names expected
            by the Fitter object: all required parameters for the
            chosen parameterisation, with planet letters (not required for
            Trend or jitter parameters.)

        Raises
        ------
        ValueError
            If any of the required parameters are missing or invalid.
        """
        # Update the current _params dict with the new entries
        merged_params = dict(self._params)
        merged_params.update(new_params)

        # Validate the complete parameter set
        self._validate_complete_params(merged_params)

        # If validation passes, update the actual params
        self._params.update(new_params)

        # Update ndim based on new free parameters
        self.ndim = len(self.free_params_values)

    @property
    def priors(self) -> dict:
        """Priors dictionary. Set via: fitter.priors = prior_dict."""
        return self._priors

    @priors.setter
    def priors(self, new_priors: dict[str, Callable[[float], float]]) -> None:
        """Set prior functions using a dict, checking all required priors are present.

        Priors must be provided for all free parameters. You can set all priors
        at once or update individual priors.

        Parameters
        ----------
        new_priors : dict
            Dictionary of prior functions to set. Keys should be parameter names
            that match free parameters, values should be callable prior functions.

        Examples
        --------
        >>> from ravest.prior import Uniform
        >>> fitter.priors = {"K_b": Uniform(0, 100), "P_b": Uniform(1, 30)}

        Raises
        ------
        ValueError
            If any required priors are missing, unexpected priors are provided,
            or initial parameter values are outside prior bounds.
        """
        self._set_priors_with_validation(new_priors)

    def _validate_complete_params(self, params: Dict[str, Parameter]) -> None:
        """Validate that params dict has required parameters, astrophysically valid values."""
        # Build complete set of expected parameters
        expected_params = set()

        # Add planetary parameters
        for planet_letter in self.planet_letters:
            for par_name in self.parameterisation.pars:
                expected_params.add(f"{par_name}_{planet_letter}")

        # Add trend parameters
        expected_params.update(["g", "gd", "gdd"])

        # Add jitter parameter
        expected_params.add("jit")

        # Convert to sets for easy comparison
        provided_params = set(params.keys())

        # Check for unexpected parameters
        unexpected_params = provided_params - expected_params
        if unexpected_params:
            raise ValueError(
                f"Unexpected parameters: {unexpected_params}. "
                f"Expected {len(expected_params)} parameters, got {len(provided_params)}"
            )

        # Check for missing parameters
        missing_params = expected_params - provided_params
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {missing_params}. "
                f"Expected {len(expected_params)} parameters, got {len(provided_params)}"
            )

        # Validate astrophysical validity of all parameters
        params_values = {name: param.value for name, param in params.items()}
        self._validate_astrophysical_validity(params_values)

        # Validate parameter coupling constraints
        # i.e. if two parameters both need to be fixed or free together
        self._validate_parameter_coupling(params)

    def _validate_astrophysical_validity(self, params_values: Dict[str, float]) -> None:
        """Validate that all parameter values are astrophysically valid."""
        # First, check that ALL parameters are finite (not NaN or infinite)
        invalid_params = { name: value for name, value in params_values.items() if not np.isfinite(value) }
        if invalid_params:
            raise ValueError( "Invalid parameters detected: " + ", ".join(f"{k}={v}" for k, v in invalid_params.items()) )

        # Validate planetary parameters for each planet
        for planet_letter in self.planet_letters:
            planet_params = {}
            for par_name in self.parameterisation.pars:
                key = f"{par_name}_{planet_letter}"
                planet_params[par_name] = params_values[key]

            # Validate this planet's parameters in current parameterisation
            self.parameterisation.validate_planetary_params(planet_params)

        # Validate trend parameters are finite real numbers (already checked above, but kept for clarity)
        for trend_param in ["g", "gd", "gdd"]:
            trend_value = params_values[trend_param]
            if not np.isfinite(trend_value):
                raise ValueError(f"Invalid trend parameter {trend_param}: {trend_value} is not a finite real number")

        # Validate jitter parameter
        jit_value = params_values["jit"]
        if jit_value < 0:
            raise ValueError(f"Invalid jitter: {jit_value} < 0")

    def _validate_parameter_coupling(self, params: Dict[str, Parameter]) -> None:
        """Validate parameter coupling constraints (e.g., secosw/sesinw must both be free or both fixed)."""
        for planet_letter in self.planet_letters:

            # Check secosw/sesinw coupling
            secosw_key = f"secosw_{planet_letter}"
            sesinw_key = f"sesinw_{planet_letter}"
            if secosw_key in params and sesinw_key in params:
                secosw_fixed = params[secosw_key].fixed
                sesinw_fixed = params[sesinw_key].fixed
                if secosw_fixed != sesinw_fixed:
                    raise ValueError(f"Parameters {secosw_key} and {sesinw_key} must both be fixed or both be free")

            # Check ecosw/esinw coupling
            ecosw_key = f"ecosw_{planet_letter}"
            esinw_key = f"esinw_{planet_letter}"
            if ecosw_key in params and esinw_key in params:
                ecosw_fixed = params[ecosw_key].fixed
                esinw_fixed = params[esinw_key].fixed
                if ecosw_fixed != esinw_fixed:
                    raise ValueError(f"Parameters {ecosw_key} and {esinw_key} must both be fixed or both be free")

    def _set_priors_with_validation(self, new_priors: dict[str, Callable[[float], float]]) -> None:
        """Set priors with validation. Supports partial updates. Can be current or default parameterisation."""
        # Create merged priors dict (in case user is only updating some priors, not all)
        merged_priors_dict = dict(self._priors)  # get existing priors
        merged_priors_dict.update(new_priors)  # overwrite with newer functions, if supplied
        provided_prior_param_names = set(merged_priors_dict.keys())

        # There are two possibilities for priors:
        # 1. The prior has been given for the parameter, in the current parameterisation
        #    (this can also include if the user is fitting in the default parameterisation)
        # 2. The prior has been given for the Default parameterisation's equivalent parameter instead
        #    (e.g. e & w instead of secosw & sesinw, or Tp instead of Tc)
        # If not, then prior isn't given for either the Current or Default parameterisation, raise an Exception
        validated_priors = {}
        missing_priors = []
        conflicts = []

        # in the current parameterisation, which (free) parameters do we expect priors for?
        current_parameterisation_free_param_names = set(self.free_params_names)
        for free_param_name in current_parameterisation_free_param_names:
            if free_param_name in provided_prior_param_names:
                # Prior was provided for the param in the current parameterisation
                validated_priors[free_param_name] = merged_priors_dict[free_param_name]

                # Check if user ALSO provided equivalent default priors (conflict!)
                default_parameterisation_equivalent_free_param_names = self._get_default_parameterisation_equivalent_free_param_name(free_param_name)
                if default_parameterisation_equivalent_free_param_names:
                    for equiv_param in default_parameterisation_equivalent_free_param_names:
                        if equiv_param in provided_prior_param_names:
                            conflicts.append((free_param_name, equiv_param))
            else:
                # We haven't been provided the prior for the free parameter in the current parameterisation
                # So let's check if we were given the prior for the equivalent parameter in the default parameterisation instead
                default_parameterisation_equivalent_free_param_names = self._get_default_parameterisation_equivalent_free_param_name(free_param_name)

                # remember that one parameter in current parameterisation (e.g. secosw) might map to more than one equivalent in default parameterisation (e.g. both e & w)
                if default_parameterisation_equivalent_free_param_names and all(eq in provided_prior_param_names for eq in default_parameterisation_equivalent_free_param_names):
                    # Found all required default equivalents
                    for equiv in default_parameterisation_equivalent_free_param_names:
                        validated_priors[equiv] = merged_priors_dict[equiv]
                else:
                    # Missing prior for a free parameter in both the current parameterisation, and its equivalent in the default parameterisation
                    if default_parameterisation_equivalent_free_param_names:
                        missing_priors.append(f"{free_param_name} (or equivalent {default_parameterisation_equivalent_free_param_names})")
                    else:
                        missing_priors.append(free_param_name)

        # Check for conflicts after processing all parameters
        if conflicts:
            conflict_strs = [f"{current} vs {default}" for current, default in conflicts]
            raise ValueError(f"Conflicting priors provided for both current and default parameterisations: {', '.join(conflict_strs)}. Please provide priors for either the current parameterisation OR the equivalent default parameterisation, but not both.")

        if missing_priors:
            raise ValueError(f"Missing priors for parameters: {missing_priors}")

        # Check for unexpected priors - only allow priors that were validated above
        expected_prior_param_names = set(validated_priors.keys())
        unexpected_prior_param_names = provided_prior_param_names - expected_prior_param_names
        if unexpected_prior_param_names:
            raise ValueError(
                f"Unexpected priors supplied for parameters: {unexpected_prior_param_names}. "
                f"Priors expected only for parameters: {expected_prior_param_names}"
            )

        # Check parameter values work with priors
        self._check_params_values_against_priors(validated_priors, current_parameterisation_free_param_names)

        # Update the priors with the new values
        self._priors.update(new_priors)

    def _get_default_parameterisation_equivalent_free_param_name(self, free_param: str) -> str:
        """Get the names of the default parameterisation equivalent parameter(s), for a single free parameter from the current parameterisation.

        Note this can be more than one: e.g. if you have secosw, this affects both e & w in the default parameterisation
        Whereas Tc just maps to Tp alone
        """
        # Extract planet letter if this is a planetary parameter
        if '_' in free_param:
            base_param, planet_letter = free_param.rsplit('_', 1)
            if planet_letter in self.planet_letters:
                # This is a planetary parameter

                if base_param in ['secosw', 'sesinw']:
                    # Both secosw and sesinw map to e,w equivalents
                    partner_param = 'sesinw' if base_param == 'secosw' else 'secosw'
                    partner_key = f"{partner_param}_{planet_letter}"
                    if partner_key in self.free_params_names:
                        return [f"e_{planet_letter}", f"w_{planet_letter}"]

                elif base_param in ['ecosw', 'esinw']:
                    # Both ecosw and esinw map to e,w equivalents
                    partner_param = 'esinw' if base_param == 'ecosw' else 'ecosw'
                    partner_key = f"{partner_param}_{planet_letter}"
                    if partner_key in self.free_params_names:
                        return [f"e_{planet_letter}", f"w_{planet_letter}"]

                elif base_param == 'Tc':
                    # Tc can use Tp equivalent
                    return [f"Tp_{planet_letter}"]

                elif base_param in ['P', 'K', 'e', 'w', 'Tp']:
                    # These are default parameterisation parameters anyway
                    # So there are no alternative priors to look for (therefore the prior is missing)
                    return None
            else:
                raise Exception(f"Parameter {free_param} has invalid planet letter {planet_letter}")
        else:
            # Non-planetary parameter (g, gd, gdd, jit)
            if free_param in ['g', 'gd', 'gdd', 'jit']:
                # These are the same in all parameterisations
                # So there are no alternative priors to look for (therefore the prior is missing)
                return None

    def _check_params_values_against_priors(self, validated_priors: dict[str, Callable[[float], float]], current_free_param_names: list[str]) -> None:
        """Check parameter values against priors (including if Prior is for the Default parameterisation equivalent parameter)."""
        for prior_param_name, prior_function in validated_priors.items():
            if prior_param_name in current_free_param_names:
                # This prior is in current parameterisation - check directly
                param_value = self.params[prior_param_name].value
                log_prior_probability = prior_function(param_value)
                if not np.isfinite(log_prior_probability):
                    raise ValueError(f"Initial value {param_value} of parameter {prior_param_name} is invalid for prior {prior_function}.")
            else:
                # This prior is in default parameterisation - need to convert parameter value
                # Get the current parameter value and convert to default
                default_param_value = self._convert_single_param_to_default(prior_param_name)
                log_prior_probability = prior_function(default_param_value)
                if not np.isfinite(log_prior_probability):
                    raise ValueError(f"Initial value {default_param_value} of parameter {prior_param_name} (in default parameterisation) is invalid for prior {prior_function}.")

    def _convert_single_param_to_default(self, default_param_name: str) -> float:
        """Convert a single parameter from current to default parameterisation."""
        # Extract planet letter if this is a planetary parameter
        if '_' in default_param_name:
            base_param, planet_letter = default_param_name.rsplit('_', 1)
            if planet_letter in self.planet_letters:
                # Get all current parameters for this planet (we need all five parameters to do a conversion)
                planet_params_dict = {}
                for par_name in self.parameterisation.pars:
                    param_key = f"{par_name}_{planet_letter}"
                    planet_params_dict[par_name] = self.params[param_key].value

                # Convert all the planetary parameters to the default parameterisation
                default_planet_params = self.parameterisation.convert_pars_to_default_parameterisation(planet_params_dict)

                # Return just the requested parameter in the default parameterisation
                return default_planet_params[base_param]

        # For non-planetary parameters (g, gd, gdd, jit), they're the same in all parameterisations
        if default_param_name in self.params:
            return self.params[default_param_name].value

        raise ValueError(f"Cannot convert parameter {default_param_name} to default parameterisation")

    @property
    def free_params_dict(self) -> Dict[str, Parameter]:
        """Free parameters as dict."""
        free_pars = {}
        for par in self.params:
            if self.params[par].fixed is False:
                free_pars[par] = self.params[par]
        return free_pars

    @property
    def free_params_values(self) -> list[float]:
        """Values of free parameters as list."""
        return [param.value for param in self.free_params_dict.values()]

    @property
    def free_params_names(self) -> list[str]:
        """Names of free parameters as list."""
        return list(self.free_params_dict.keys())

    @property
    def fixed_params_dict(self) -> Dict[str, Parameter]:
        """Fixed parameters as dict, mapping names to Parameter objects."""
        fixed_pars = {}
        for par in self.params:
            if self.params[par].fixed is True:
                fixed_pars[par] = self.params[par]
        return fixed_pars

    @property
    def fixed_params_values(self) -> list[float]:
        """Values of fixed parameters, as list."""
        return [param.value for param in self.fixed_params_dict.values()]

    @property
    def fixed_params_names(self) -> list[str]:
        """Names of fixed parameters, as list."""
        return list(self.fixed_params_dict.keys())

    @property
    def fixed_params_values_dict(self) -> Dict[str, float]:
        """Fixed parameters as dict mapping names to just the values."""
        return dict(zip(self.fixed_params_names, self.fixed_params_values))

    def find_map_estimate(self, method: str = "Powell") -> scipy.optimize.OptimizeResult:
        """Find Maximum A Posteriori (MAP) estimate of parameters.

        Parameters
        ----------
        method : str, optional
            Optimization method to use (default: "Powell")

        Returns
        -------
        scipy.optimize.OptimizeResult
            The optimization result containing the MAP estimate

        Raises
        ------
        Warning
            If MAP optimization fails to converge
        """
        # Initialize log-posterior object
        lp = LogPosterior(
            self.planet_letters,
            self.parameterisation,
            self.priors,
            self.fixed_params_values_dict,
            self.free_params_names,
            self.time,
            self.vel,
            self.verr,
            self.t0,
        )

        initial_guess = self.free_params_values

        # Perform MAP optimization
        def negative_log_posterior(*args: float) -> float:
            return lp._negative_log_probability_for_MAP(*args)

        map_results = minimize(negative_log_posterior, initial_guess, method=method)

        if map_results.success is False:
            print(map_results)
            warnings.warn("MAP did not succeed. Check the initial values of the parameters, and the prior functions.")

        # Print results as dictionary (to show param names too)
        map_results_dict = dict(zip(self.free_params_names, map_results.x))
        print("MAP parameter results:", map_results_dict)

        # Return the scipy OptimizeResult object so that user can inspect fully if needed
        return map_results

    def generate_initial_walker_positions_random(self, nwalkers: int, verbose: bool = False, max_attempts: int = 1000) -> np.ndarray:
        """Generate random initial walker positions that satisfy priors and are astrophysically valid.

        Creates random starting positions for MCMC walkers by sampling from
        appropriate distributions based on each parameter's prior type. Ensures
        that parameter combinations are astrophysically valid (e.g., eccentricity < 1).

        Parameters
        ----------
        nwalkers : int
            Number of MCMC walkers to generate positions for
        verbose : bool, default False
            If True, print walker positions during generation
        max_attempts : int, default 1000
            Maximum attempts to generate a valid walker position

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, ndim) where ndim is the number of free parameters.
            Each row represents the starting position for one walker in the order of
            free_params_names.

        Raises
        ------
        ValueError
            If a prior type is not supported for walker generation or if unable
            to generate valid positions after max_attempts

        Examples
        --------
        >>> # Generate positions for 40 walkers
        >>> nwalkers = 10 * len(fitter.free_params_names)
        >>> initial_positions = fitter.generate_initial_walker_positions_random(nwalkers)
        >>> fitter.run_mcmc(initial_positions, nwalkers, max_steps=2000)
        """
        if verbose:
            print("Free parameters:", self.free_params_names)

        mcmc_init = []

        for walker_idx in range(nwalkers):
            attempts = 0
            while attempts < max_attempts:
                walker_position = []
                for param_name in self.free_params_names:
                    # Check if we have a direct prior for this parameter
                    # (because user may be fitting in a transformed parameterisation, but gave priors in the default parameterisation instead)
                    if param_name in self.priors:
                        prior = self.priors[param_name]

                        if isinstance(prior, ravest.prior.Normal):
                            walker_position.append(np.random.normal(loc=prior.mean, scale=2*prior.std))

                        elif isinstance(prior, ravest.prior.HalfNormal):
                            walker_position.append(np.abs(np.random.normal(loc=0, scale=2*prior.std)))

                        elif isinstance(prior, ravest.prior.Uniform):
                            walker_position.append(np.random.uniform(low=prior.lower, high=prior.upper))

                        elif isinstance(prior, ravest.prior.TruncatedNormal):
                            walker_position.append(np.random.uniform(low=prior.lower, high=prior.upper))

                        elif isinstance(prior, ravest.prior.Beta):
                            walker_position.append(np.random.uniform(low=0, high=1))

                        elif isinstance(prior, ravest.prior.EccentricityUniform):
                            walker_position.append(np.random.uniform(low=0, high=prior.upper))

                        else:
                            raise ValueError(f"Unsupported prior type for walker generation: {type(prior)}")

                    else:
                        # No direct prior for this parameter (this happens if fitting in a transformed parameterisation, but prior is in Default)
                        # Instead use current value + small perturbation
                        centre_val = self.params[param_name].value
                        # Add small random perturbation (10% of current value + small fixed amount for near-zero values)
                        perturbation = np.random.normal(0, abs(centre_val) * 0.1 + 0.01)
                        walker_position.append(centre_val + perturbation)

                # Check astrophysical validity and prior compliance
                try:
                    # Convert walker position to full parameter dict (free + fixed)
                    free_params_dict = dict(zip(self.free_params_names, walker_position))
                    all_params_dict = self.fixed_params_values_dict | free_params_dict

                    # Check astrophysical validity
                    self._validate_astrophysical_validity(all_params_dict)

                    # Check prior compliance using LogPosterior, rather than calling priors direct
                    # (because it handles Transformed->Default parameter transformations already, if needed)
                    lp = LogPosterior(
                        self.planet_letters,
                        self.parameterisation,
                        self.priors,
                        self.fixed_params_values_dict,
                        self.free_params_names,
                        self.time,
                        self.vel,
                        self.verr,
                        self.t0,
                    )
                    # Check the log-prior probability is finite (i.e. proposed initial values are within prior bounds)
                    params_for_prior = lp._convert_params_for_prior_evaluation(free_params_dict)
                    log_prior = lp.log_prior(params_for_prior)
                    if not np.isfinite(log_prior):
                        raise ValueError(f"Outside prior bounds (log_prior = {log_prior})")

                    # If both astrophysical and priors validations pass, we have a valid walker position
                    break
                except ValueError:
                    # Validation failed. Generate a new set of values and try again.
                    attempts += 1
                    continue

            if attempts >= max_attempts:
                raise ValueError(f"Could not generate astrophysically valid walker {walker_idx} after {max_attempts} attempts. "
                               f"Consider relaxing priors or checking parameter constraints.")

            if verbose:
                print(f"Walker {walker_idx} position: {walker_position} (valid after {attempts + 1} attempts)")
            mcmc_init.append(walker_position)

        mcmc_init = np.array(mcmc_init)
        if verbose:
            print(f"Generated MCMC initial positions with shape: {mcmc_init.shape}")

        return mcmc_init

    def generate_initial_walker_positions_around_point(
        self,
        centre: np.ndarray | list,
        nwalkers: int,
        scale: float = 1e-4,
        relative: bool = True,
        verbose: bool = False,
        max_attempts: int = 1000
    ) -> np.ndarray:
        """Generate initial walker positions in a ball around a supplied centre point.

        Creates starting positions for MCMC walkers clustered around a centre point
        (e.g., MAP estimate). Each walker is generated by adding small random perturbations
        to the centre values. Validates that both the centre point and all generated
        walker positions satisfy priors and are astrophysically valid.

        Parameters
        ----------
        centre : np.ndarray or list
            Centre point for walker positions. Must have length equal to the number
            of free parameters and be in the order of free_params_names.
        nwalkers : int
            Number of MCMC walkers to generate positions for
        scale : float, default 1e-4
            Scale of perturbations around centre point
        relative : bool, default True
            If True, perturbations scale with parameter values (scale * centre * random).
            If False, perturbations are absolute (scale * random).
        verbose : bool, default False
            If True, print walker positions during generation
        max_attempts : int, default 1000
            Maximum attempts to generate a valid walker position

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, ndim) where ndim is the number of free parameters.
            Each row represents the starting position for one walker in the order of
            free_params_names.

        Raises
        ------
        ValueError
            If centre has wrong length, if centre point is invalid, or if unable
            to generate valid positions after max_attempts

        Examples
        --------
        >>> # Generate walkers around MAP estimate
        >>> map_result = fitter.find_map_estimate()
        >>> initial_positions = fitter.generate_initial_walker_positions_around_point(
        ...     centre=map_result.x, nwalkers=40, scale=1e-4
        ... )
        >>> fitter.run_mcmc(initial_positions, nwalkers=40, max_steps=2000)
        """
        centre = np.asarray(centre)

        if len(centre) != len(self.free_params_names):
            raise ValueError(
                f"Centre must have length {len(self.free_params_names)} "
                f"(number of free parameters), got {len(centre)}"
            )

        if verbose:
            print("Free parameters:", self.free_params_names)
            print(f"Centre values: {centre}")

        # Validate centre point first
        try:
            free_params_dict = dict(zip(self.free_params_names, centre))
            all_params_dict = self.fixed_params_values_dict | free_params_dict

            # Check astrophysical validity
            self._validate_astrophysical_validity(all_params_dict)

            # Check prior compliance
            lp = LogPosterior(
                self.planet_letters,
                self.parameterisation,
                self.priors,
                self.fixed_params_values_dict,
                self.free_params_names,
                self.time,
                self.vel,
                self.verr,
                self.t0,
            )
            params_for_prior = lp._convert_params_for_prior_evaluation(free_params_dict)
            log_prior = lp.log_prior(params_for_prior)
            if not np.isfinite(log_prior):
                raise ValueError(f"Centre point outside prior bounds (log_prior = {log_prior})")

            if verbose:
                print(f"Centre point validated (log_prior = {log_prior})")

        except ValueError as e:
            raise ValueError(f"Supplied centre point is not valid: {e}")

        # Generate walker positions around centre
        mcmc_init = []

        for walker_idx in range(nwalkers):
            attempts = 0
            while attempts < max_attempts:
                # Generate perturbation
                if relative:
                    # Relative perturbation: scales with parameter values
                    perturbation = scale * np.random.randn(len(centre)) * np.abs(centre)
                else:
                    # Absolute perturbation: same scale for all parameters
                    perturbation = scale * np.random.randn(len(centre))

                walker_position = centre + perturbation

                # Validate this walker position
                try:
                    free_params_dict = dict(zip(self.free_params_names, walker_position))
                    all_params_dict = self.fixed_params_values_dict | free_params_dict

                    # Check astrophysical validity
                    self._validate_astrophysical_validity(all_params_dict)

                    # Check prior compliance
                    params_for_prior = lp._convert_params_for_prior_evaluation(free_params_dict)
                    log_prior = lp.log_prior(params_for_prior)
                    if not np.isfinite(log_prior):
                        raise ValueError(f"Outside prior bounds (log_prior = {log_prior})")

                    # If validation passes, we have a valid walker position
                    break
                except ValueError:
                    # Validation failed, try again
                    attempts += 1
                    continue

            if attempts >= max_attempts:
                raise ValueError(
                    f"Could not generate astrophysically valid walker {walker_idx} after {max_attempts} attempts. "
                    f"Consider using a larger scale parameter or checking that the centre point is not too close to prior/physical boundaries."
                )

            if verbose:
                print(f"Walker {walker_idx} position: {walker_position} (valid after {attempts + 1} attempts)")
            mcmc_init.append(walker_position)

        mcmc_init = np.array(mcmc_init)
        if verbose:
            print(f"Generated MCMC initial positions with shape: {mcmc_init.shape}")

        return mcmc_init

    def generate_initial_walker_positions_from_map(
        self,
        map_result: scipy.optimize.OptimizeResult,
        nwalkers: int,
        scale: float = 1e-4,
        relative: bool = True,
        verbose: bool = False,
        max_attempts: int = 1000
    ) -> np.ndarray:
        """Generate initial walker positions around MAP estimate.

        Convenience function that generates walker positions clustered around
        MAP parameter estimates from a pre-computed MAP result.

        Parameters
        ----------
        map_result : scipy.optimize.OptimizeResult
            Result from find_map_estimate()
        nwalkers : int
            Number of MCMC walkers to generate positions for
        scale : float, default 1e-4
            Scale of perturbations around MAP values
        relative : bool, default True
            If True, perturbations scale with parameter values.
            If False, perturbations are absolute.
        verbose : bool, default False
            If True, print walker positions during generation
        max_attempts : int, default 1000
            Maximum attempts to generate a valid walker position

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, ndim) where ndim is the number of free parameters.
            Each row represents the starting position for one walker.

        Raises
        ------
        ValueError
            If unable to generate valid positions

        Examples
        --------
        >>> # Find MAP then generate walkers around it
        >>> map_result = fitter.find_map_estimate()
        >>> initial_positions = fitter.generate_initial_walker_positions_from_map(
        ...     map_result=map_result, nwalkers=40
        ... )
        >>> fitter.run_mcmc(initial_positions, nwalkers=40, max_steps=2000)
        """
        return self.generate_initial_walker_positions_around_point(
            centre=map_result.x,
            nwalkers=nwalkers,
            scale=scale,
            relative=relative,
            verbose=verbose,
            max_attempts=max_attempts
        )

    def run_mcmc(self, initial_positions : np.ndarray, nwalkers: int, max_steps: int = 5000, progress: bool = True, multiprocessing: bool = False, check_convergence: bool = False, convergence_check_interval: int = 1000, convergence_check_start: int = 0) -> None:
        """Run MCMC sampling from given initial walker positions.

        Parameters
        ----------
        initial_positions  : np.ndarray
            Starting positions for all MCMC walkers. Shape must be (nwalkers, ndim)
            where ndim is the number of free parameters. Each row represents the
            starting position for one walker in the order of free_params_names.
        nwalkers : int
            Number of MCMC walkers (must match first dimension of initial_positions )
        max_steps : int, optional
            Maximum number of MCMC steps to run. If check_convergence=False, runs for
            exactly this many steps. If check_convergence=True, runs up to this many
            steps, stopping early when convergence criteria are met (default: 5000)
        progress : bool, optional
            Whether to show progress bar during MCMC (default: True)
        multiprocessing : bool, optional
            Whether to use multiprocessing for MCMC (default: False)
        check_convergence : bool, optional
            If True, check for convergence and stop early when criteria met.
            Convergence requires: chain length > 50 times max autocorrelation time,
            and autocorrelation time estimate stable to 1 percent.
            If False, run for exactly max_steps (default: False)
        convergence_check_interval : int, optional
            Steps between convergence checks (only used if check_convergence=True) (default: 1000)
        convergence_check_start : int, optional
            Minimum iteration before starting convergence checks. Set this sensibly
            (e.g. 2x burn-in) to avoid inaccurate tau estimates on short chains (default: 0)
        """
        # Initialize log-posterior object for MCMC sampling
        lp = LogPosterior(
            self.planet_letters,
            self.parameterisation,
            self.priors,
            self.fixed_params_values_dict,
            self.free_params_names,
            self.time,
            self.vel,
            self.verr,
            self.t0,
        )

        # Enforce minimum number of walkers (though users ideally should have many more than this)
        if nwalkers < 2 * self.ndim:
            logging.warning(f"nwalkers should be at least 2 * ndim. You have {nwalkers} walkers and {self.ndim} dimensions. Setting nwalkers to {2 * self.ndim}.")
            self.nwalkers = 2 * self.ndim
        else:
            self.nwalkers = nwalkers

        # Validate walker positions shape
        if initial_positions .shape != (nwalkers, self.ndim):
            raise ValueError(f"initial_positions  must have shape ({nwalkers}, {self.ndim}), got {initial_positions .shape}")

        # Validate every walker position for astrophysical validity and prior compliance
        # (we don't want to start any chains in invalid parameter space)
        for i, walker_position in enumerate(initial_positions ):
            walker_params_dict = dict(zip(self.free_params_names, walker_position))
            all_params_dict = self.fixed_params_values_dict | walker_params_dict

            # Check astrophysical validity
            try:
                self._validate_astrophysical_validity(all_params_dict)
            except ValueError as e:
                raise ValueError(f"Walker {i} has invalid astrophysical parameters: {e}") from e

            # Check prior compliance
            params_for_prior = lp._convert_params_for_prior_evaluation(walker_params_dict)
            log_prior = lp.log_prior(params_for_prior)
            if not np.isfinite(log_prior):
                raise ValueError(f"Walker {i} is outside prior bounds (log_prior = {log_prior})")

        # TODO: parameter_names argument does slightly impact performance - but not sure if it can be avoided, we do need the names
        # and I'm not sure constructing the dictionary later ourselves manually is any quicker than passing parameter_names argument

        # Create sampler
        if multiprocessing:
            pool = mp.get_context("spawn").Pool()  # Use 'spawn' instead of 'fork' to avoid issues on some Linux platforms
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability,
                                            parameter_names=self.free_params_names,
                                            pool=pool)
        else:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability,
                                            parameter_names=self.free_params_names)

        # Warn if convergence arguments provided but convergence checking disabled
        if not check_convergence:
            if convergence_check_interval != 1000 or convergence_check_start != 0:
                logging.warning(
                    "Convergence checking arguments provided but check_convergence=False. "
                    "These arguments will be ignored. Did you forget to set check_convergence=True?"
                )

        # Run MCMC with or without convergence checking
        if not check_convergence:
            # Fixed-length mode - run for exactly max_steps
            logging.info(f"Starting MCMC for {max_steps} steps...")
            sampler.run_mcmc(initial_state=initial_positions, nsteps=max_steps, progress=progress)
            logging.info("...MCMC done.")
        else:
            # Convergence checking - run up to max_steps, stopping early if converged
            logging.info(f"Starting MCMC with convergence checks. (Maximum {max_steps} steps, checking convergence every {convergence_check_interval} steps after iteration {convergence_check_start})...")

            # Initialize autocorrelation history storage
            self.autocorr_history = {}

            old_tau = np.inf

            for sample in sampler.sample(initial_state=initial_positions, iterations=max_steps, progress=progress):
                # Only check at specified intervals
                if sampler.iteration % convergence_check_interval != 0:
                    continue

                # Don't check before we have reached convergence_check_start
                if sampler.iteration < convergence_check_start:
                    continue

                # Get autocorrelation time estimate
                tau = sampler.get_autocorr_time(tol=0)

                # Store autocorrelation history for plotting/diagnostics later
                self.autocorr_history[sampler.iteration] = tau.copy()

                # Log progress
                logging.info(f"Convergence check: Step {sampler.iteration}: mean(tau)={np.mean(tau):.1f}, max(tau)={np.max(tau):.1f}")

                # Check convergence criteria
                check_chain_length = np.all(sampler.iteration > 50 * tau)  # Chain length > 50 * tau
                check_stable_tau = np.all(np.abs(old_tau - tau) / tau < 0.01)  # Tau stable to 1 percent
                converged = check_chain_length and check_stable_tau

                if converged:
                    logging.info(f"Converged at iteration {sampler.iteration}")
                    break
                else:
                    logging.info(f"Not yet converged (N/50>tau check: {check_chain_length}, tau stability check: {check_stable_tau})")

                # Warn if approaching max steps without convergence
                if sampler.iteration > 0.8 * max_steps:
                    logging.warning(f"Approaching max iterations ({max_steps}) without convergence! (max tau={np.max(tau):.1f}, tau stability change={np.abs(old_tau - tau) / tau})")

                # Update old tau for next check
                old_tau = tau

            # Final log
            final_steps = sampler.iteration
            logging.info(f"MCMC complete: {final_steps} steps total")

        # Close multiprocessing pool if used
        if multiprocessing:
            pool.close()
            pool.join()

        self.sampler = sampler

    def get_samples_np(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:
        """Return a contiguous numpy array of MCMC samples.

        Samples can be discarded from the start and/or the end of the array. You can
        also thin (take only every n-th sample), and you can flatten the array
        so that each walker's chain is merged into one chain.

        This is the foundational method for accessing MCMC samples. All the other
        get_samples methods build on this.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        flat : bool, optional
            Whether to flatten each walker's chain into one chain. (default: False)
            If True, return flattened array with shape (nsteps_after_discard_thin * nwalkers, ndim)
            If False, return unflattened array with shape (nsteps_after_discard_thin, nwalkers, ndim)

        Returns
        -------
        np.ndarray
            Contiguous array of MCMC samples. Shape depends on `flat` parameter:
            - flat=False: (nsteps_after_discard_thin, nwalkers, ndim)
            - flat=True: (nsteps_after_discard_thin * nwalkers, ndim)

        Notes
        -----
        We enforce np.ascontiguousarray() on the return, because np.reshape() does
        not guarantee a contiguous array in memory.
        """
        # Get the full chain from emcee without any processing
        full_samples = self.sampler.get_chain(discard=0, thin=1, flat=False)

        # Match emcee's slicing logic: [discard + thin - 1 : end : thin]
        # But adapted - we also allow for discarding from the end
        start_idx = discard_start + thin - 1
        if discard_end == 0:
            end_idx = full_samples.shape[0]
        else:
            end_idx = full_samples.shape[0] - discard_end

        # Check the start and end points are valid
        if start_idx >= end_idx:
            raise ValueError(f"Invalid parameters: start_idx ({start_idx}) >= end_idx ({end_idx}). "
                            f"Try reducing discard_start ({discard_start}), discard_end ({discard_end}), or thin ({thin}).")

        # Apply the slicing
        samples = full_samples[start_idx:end_idx:thin]

        # Flatten if requested (after discarding) - flatten steps and walkers into single dimension
        if flat:
            # (steps, walkers, ndim) -> (steps*walkers, ndim)
            nsteps, nwalkers, ndim = samples.shape
            samples = samples.reshape(nsteps * nwalkers, ndim)

        return np.ascontiguousarray(samples)

    def get_samples_df(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1) -> pd.DataFrame:
        """Return a pandas DataFrame of flattened MCMC samples.

        Each row represents one sample, each column represents one parameter.
        Built on get_samples_np().

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)

        Returns
        -------
        pd.DataFrame
            DataFrame with shape (nsteps_after_discard_thin * nwalkers, ndim).
            Columns are parameter names.
        """
        flat_samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        return pd.DataFrame(flat_samples, columns=self.free_params_names)

    def get_samples_dict(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1) -> Dict[str, np.ndarray]:
        """Return a dict of flattened MCMC samples.

        Each parameter gets a 1D (flattened) contiguous array of all its samples.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)

        Returns
        -------
        dict
            Dictionary mapping parameter names to 1D arrays of samples.
            Each array has shape (nsteps_after_discard_thin * nwalkers,)

        Examples
        --------
        >>> samples_dict = fitter.get_samples_dict(discard_start=1000)
        >>> K_b_samples = samples_dict['K_b']  # All samples for parameter K for planet b
        """
        flat_samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        param_names = self.free_params_names

        # Direct numpy slicing - much faster than pandas operations
        return {name: flat_samples[:, i] for i, name in enumerate(param_names)}

    def get_sampler_lnprob(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:
        """Returns the log probability at each step of the sampler.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        flat : bool, optional
            If True, return flattened array shape (nsteps_after_discard_thin * nwalkers)
            If False, return unflattened array shape (nsteps_after_discard_thin, nwalkers) (default: False)

        Returns
        -------
        np.ndarray
            Array of log probabilities of the function at each sample.
        """
        # Get the full log prob chain from emcee without any processing
        full_lnprob = self.sampler.get_log_prob(discard=0, thin=1, flat=False)

        # Match emcee's slicing logic: [discard + thin - 1 : end : thin]
        # But adapted - we also allow for discarding from the end
        start_idx = discard_start + thin - 1
        if discard_end == 0:
            end_idx = full_lnprob.shape[0]
        else:
            end_idx = full_lnprob.shape[0] - discard_end

        # Check the start and end points are valid
        if start_idx >= end_idx:
            raise ValueError(f"Invalid parameters: start_idx ({start_idx}) >= end_idx ({end_idx}). "
                            f"Try reducing discard_start ({discard_start}), discard_end ({discard_end}), or thin ({thin}).")

        # Apply the slicing
        lnprob = full_lnprob[start_idx:end_idx:thin]

        # Flatten if requested (after discarding) - flatten steps and walkers into single dimension
        if flat:
            # (steps, walkers) -> (steps*walkers,)
            nsteps, nwalkers = lnprob.shape
            lnprob = lnprob.reshape(nsteps * nwalkers)

        return np.ascontiguousarray(lnprob)

    def get_mcmc_posterior_dict(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1) -> dict:
        """Return dict combining MCMC samples for free params, and the fixed values for the fixed params.

        This method creates a unified dictionary containing all model parameters:
        fixed parameters as single float values, and free parameters as arrays
        of MCMC samples. This format is ideal for functions like calculate_mpsini
        that need all parameters (whether free or fixed), and that should propagate
        uncertainties from the free parameters samples.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)

        Returns
        -------
        dict
            Dictionary of all parameters:
            - Fixed parameters: single float values
            - Free parameters: 1D arrays of MCMC samples with shape (nsteps_after_discard_thin * nwalkers,)
        """
        fixed_params_dict = self.fixed_params_values_dict
        free_samples_dict = self.get_samples_dict(discard_start=discard_start, discard_end=discard_end, thin=thin)
        return fixed_params_dict | free_samples_dict

    def calculate_log_likelihood(self, params_dict: Dict[str, float]) -> float:
        """Calculate log-likelihood for given parameter values.

        Note this does not include (log-)prior probabilities, this is just the
        (log-) *likelihood* primarily for use in AIC & BIC calculation.

        Parameters
        ----------
        params_dict : dict
            Dictionary of all parameter values (both fixed and free parameters)

        Returns
        -------
        float
            The log-likelihood value
        """
        # Create LogLikelihood object (same as in find_map_estimate and run_mcmc)
        log_likelihood = LogLikelihood(
            time=self.time,
            vel=self.vel,
            verr=self.verr,
            t0=self.t0,
            planet_letters=self.planet_letters,
            parameterisation=self.parameterisation,
        )
        return log_likelihood(params_dict)

    def build_params_dict(self, free_params: np.ndarray | list | Dict[str, float]) -> Dict[str, float]:
        """Build a params dict by providing free param vals, combine with fixed param vals.

        Takes free parameter float values (which can be from any source e.g. MAP results, MCMC posteriors,
        or any custom values) and combines them with the fixed parameter values to create
        a complete parameter dictionary. This dict is ideal for calculating chi2, log-likelihood,
        AIC, and BIC.

        This is designed for a single value per parameter. For combining the MCMC posterior
        chains for free parameters and the fixed values for fixed parameters, use
        `get_mcmc_posterior_dict` method.

        Parameters
        ----------
        free_params : list, np.ndarray, or dict
            Free parameter values from any source:
            - list/array: values in order of self.free_params_names
            - dict: mapping of free param names to values

        Returns
        -------
        Dict[str, float]
            Complete parameters dict with both free and fixed parameter values

        Examples
        --------
        >>> # From MAP optimization result
        >>> map_result = fitter.find_map_estimate()
        >>> params = fitter.build_params_dict(map_result.x)
        >>> aic = fitter.calculate_aic(params)
        >>>
        >>> # From best MCMC sample
        >>> best_sample = fitter.get_sample_with_best_lnprob(discard_start=1000)
        >>> params = fitter.build_params_dict(best_sample)
        >>> bic = fitter.calculate_bic(params)
        >>>
        >>> # From custom array of values (in order of free_params_names)
        >>> custom_values = [5.0, 50.0, 0.1, 0.0, 2450000.0]  # example values
        >>> params = fitter.build_params_dict(custom_values)
        >>> log_like = fitter.calculate_log_likelihood(params)
        """
        if isinstance(free_params, dict):
            # Validate that all expected free parameters are present
            expected_names = set(self.free_params_names)
            provided_names = set(free_params.keys())

            missing = expected_names - provided_names
            if missing:
                raise ValueError(f"Missing required free parameters: {missing}")

            extra = provided_names - expected_names
            if extra:
                raise ValueError(f"Unexpected parameters provided: {extra}")

            return self.fixed_params_values_dict | free_params
        else:
            # Validate that array/list has correct length
            if len(free_params) != len(self.free_params_names):
                raise ValueError(
                    f"Expected {len(self.free_params_names)} free parameter values "
                    f"but got {len(free_params)} "
                    f"(expecting {len(self.free_params_names)} values for {self.free_params_names})"
                )

            free_dict = dict(zip(self.free_params_names, free_params))
            return self.fixed_params_values_dict | free_dict

    def calculate_chi2(self, params_dict: Dict[str, float]) -> float:
        """Calculate chi-squared for given parameter values.

        Uses LogLikelihood to avoid code duplication. Works backwards from
        log-likelihood: ll = -0.5 * (chi2 + penalty_term)

        Parameters
        ----------
        params_dict : dict
            Dictionary of all parameter values (both fixed and free parameters)

        Returns
        -------
        float
            Chi-squared value: sum((data - model)^2 / (error^2 + jitter^2))
        """
        # Create LogLikelihood instance to reuse RV model calculation
        ll = LogLikelihood(
            self.time, self.vel, self.verr, self.t0,
            self.planet_letters, self.parameterisation
        )

        # Get log-likelihood
        log_like = ll(params_dict)

        # Work backwards to get chi2
        # ll = -0.5 * (chi2 + penalty_term)
        # chi2 = -2 * ll - penalty_term
        verr_jitter_squared = self.verr**2 + params_dict["jit"]**2
        penalty_term = np.sum(np.log(2 * np.pi * verr_jitter_squared))
        chi2 = -2 * log_like - penalty_term

        return chi2

    def calculate_aic(self, params_dict: Dict[str, float]) -> float:
        """Calculate Akaike Information Criterion (AIC) for given parameters.

        AIC = 2*k - 2*ln(L), where k is the number of free parameters
        and L is the likelihood.

        Parameters
        ----------
        params_dict : dict
            Dictionary of all parameter values (both fixed and free parameters)

        Returns
        -------
        float
            AIC value
        """
        log_like = self.calculate_log_likelihood(params_dict)
        return 2 * self.ndim - 2 * log_like

    def calculate_bic(self, params_dict: Dict[str, float]) -> float:
        """Calculate Bayesian Information Criterion (BIC) for given parameters.

        BIC = k*ln(n) - 2*ln(L), where k is the number of free parameters,
        n is the number of observations, and L is the likelihood.

        Parameters
        ----------
        params_dict : dict
            Dictionary of all parameter values (both fixed and free parameters)

        Returns
        -------
        float
            BIC value
        """
        log_like = self.calculate_log_likelihood(params_dict)
        return self.ndim * np.log(len(self.time)) - 2 * log_like

    def get_sample_with_best_lnprob(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1) -> Dict[str, float]:
        """Get parameter values from the MCMC sample with the highest log probability.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)

        Returns
        -------
        Dict[str, float]
            Dictionary of parameter names to values from the best sample
        """
        # Get samples and log probabilities
        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        lnprob = self.get_sampler_lnprob(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)

        # Find index of maximum log probability
        best_idx = np.argmax(lnprob)
        best_lnprob = lnprob[best_idx]

        print(f"Best sample found with log probability {best_lnprob:.6f} at index {best_idx} of samples (with discard_start={discard_start}, discard_end={discard_end}, thin={thin})")

        # Get parameter values at that index
        best_values = samples[best_idx]

        # Return as dictionary
        return dict(zip(self.free_params_names, best_values))

    def plot_autocorr_estimates(
        self,
        params: list[str] | None = None,
        plot_mean: bool = False,
        show_legend: bool = True,
        title: str | None = "Autocorrelation Time Estimates",
        xlabel: str | None = "Step number",
        ylabel: str | None = r"Autocorrelation time $\tau$",
        save: bool = False,
        fname: str = "autocorr_plot.png",
        dpi: int = 100
    ) -> None:
        r"""Plot autocorrelation time estimates from adaptive MCMC run.

        Shows how autocorrelation time evolved during the MCMC run and
        the convergence threshold line (N / 50).

        Only available if run_mcmc was called with check_convergence=True.

        Parameters
        ----------
        params : list[str] or None, optional
            List of parameter names to plot. If None, plots all free parameters (default: None)
        plot_mean : bool, optional
            If True, plot mean tau instead of individual parameter taus.
            Overrides params argument (default: False)
        show_legend : bool, optional
            Whether to show legend (default: True)
        title : str or None, optional
            Plot title (default: "Autocorrelation Time Estimates"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label (default: "Step number"). Set to None or "" to skip.
        ylabel : str or None, optional
            Y-axis label (default: r"Autocorrelation time $\tau$"). Set to None or "" to skip.
        save : bool, optional
            Save the plot to path `fname` (default: False)
        fname : str, optional
            The path to save the plot to (default: "autocorr_plot.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)

        Raises
        ------
        ValueError
            If no autocorrelation history is available (run_mcmc was not called
            with check_convergence=True, or has not been called yet)
        """
        # Check if data available
        if not hasattr(self, 'autocorr_history') or len(self.autocorr_history) == 0:
            raise ValueError(
                "No autocorrelation history available. "
                "Please run run_mcmc() with check_convergence=True first."
            )

        iterations = np.array(list(self.autocorr_history.keys()))
        max_iteration = np.max(iterations)
        tau_history = np.array(list(self.autocorr_history.values()))  # Shape: (n_checks, n_params)

        # Create plot
        fig, ax = plt.subplots(1, figsize=(10, 6))
        if title:
            fig.suptitle(title)

        # Plot convergence threshold (N/50)
        ax.plot([0, max_iteration], [0, max_iteration / 50], "--k", linewidth=2,
                label="N/50 convergence threshold")

        if plot_mean:
            # Plot mean tau
            mean_tau = np.mean(tau_history, axis=1)
            ax.plot(iterations, mean_tau, linewidth=2, label="Mean τ")
        else:
            # Determine which parameters to plot
            if params is None:
                params_to_plot = self.free_params_names
                indices_to_plot = range(len(self.free_params_names))
            else:
                params_to_plot = []
                indices_to_plot = []
                for param in params:
                    if param in self.free_params_names:
                        idx = self.free_params_names.index(param)
                        params_to_plot.append(param)
                        indices_to_plot.append(idx)
                    else:
                        logging.warning(f"Parameter '{param}' not found in free parameters, skipping")

            # Plot individual parameter taus
            for i, param_name in zip(indices_to_plot, params_to_plot):
                ax.plot(iterations, tau_history[:, i], alpha=0.7, label=param_name)

        ax.set_xlim(0, iterations.max())
        ax.set_ylim(0, tau_history.max() * 1.1)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if show_legend:
            ax.legend(loc='upper left')

        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def plot_chains(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, truths: list = None, title: str | None = "Chains plot", xlabel: str | None = "Step number", save: bool = False, fname: str = "chains_plot.png", dpi: int = 100) -> None:
        """Plot MCMC chains for all free parameters.

        Displays the evolution of each free parameter across MCMC steps for all walkers.
        Useful for diagnosing convergence, burn-in, and mixing of the MCMC chains.
        Each parameter gets its own subplot showing all walker traces.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        truths : list, optional
            List of true parameter values to overplot as horizontal lines.
            Must match the number of free parameters. Use None for parameters
            without known truth values (default: None)
        title : str or None, optional
            Plot title (default: "Chains plot"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label (default: "Step number"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "chains_plot.png")
        dpi : int, optional
            Resolution for saving (default: 100)
        """
        # Scale figure height to maintain consistent subplot size
        subplot_height_inches = 1.25
        fig, axes = plt.subplots(self.ndim, figsize=(10, self.ndim * subplot_height_inches),
                                sharex=True, constrained_layout=True)
        if title:
            fig.suptitle(title)

        if self.ndim == 1:
            axes = [axes]

        if truths is not None:
            if not len(truths) == self.ndim:
                raise ValueError(f"Length of truths ({len(truths)}) must match number of free parameters ({self.ndim})")

        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=False)
        for i in range(self.ndim):
            ax = axes[i]
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.free_params_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

            to_plot = samples[:, :, i]
            ax.plot(to_plot, "k", alpha=0.3)
            if truths is not None and truths[i] is not None:
                ax.axhline(truths[i], color="tab:blue")

        if xlabel:
            axes[-1].set_xlabel(xlabel)
        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def plot_lnprob(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, title: str | None = "Log Probability Traces", xlabel: str | None = "Step number", ylabel: str | None = "Log probability", save: bool = False, fname: str = "lnprob_plot.png", dpi: int = 100) -> None:
        """Plot log probability traces for all walkers.

        Useful for diagnosing MCMC convergence and identifying problematic
        walkers/parameters. You can use `discard_start` and `discard_end` to
        focus in on specific steps in the chains.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        title : str or None, optional
            Plot title (default: "Log Probability Traces"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label (default: "Step number"). Set to None or "" to skip.
        ylabel : str or None, optional
            Y-axis label (default: "Log probability"). Set to None or "" to skip.
        save : bool, optional
            Save the plot to path `fname` (default: False)
        fname : str, optional
            The path to save the plot to (default: "lnprob_plot.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)
        """
        fig, ax = plt.subplots(1, figsize=(10, 6))
        if title:
            fig.suptitle(title)

        lnprobs = self.get_sampler_lnprob(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=False)

        nsteps, nwalkers = lnprobs.shape
        for i in range(nwalkers):
            to_plot = lnprobs[:, i]
            ax.plot(to_plot, "k", alpha=0.3)

        ax.set_xlim(0, nsteps)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def plot_corner(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, plot_datapoints: bool = False, truths: list[float] = None, title: str | None = "Corner plots", save: bool = False, fname: str = "corner_plot.png", dpi: int = 100) -> None:
        """Create a corner plot of MCMC samples.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        plot_datapoints : bool, optional
            Show individual data points in addition to contours (default: False)
        truths : list of float, optional
            True parameter values to overplot as vertical/horizontal lines (default: None).
            Must match the order of free parameters if provided.
        title : str or None, optional
            Plot title (default: "Corner plots"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "corner_plot.png")
        dpi : int, optional
            Resolution for saving (default: 100)
        """
        flat_samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        param_names = self.free_params_names
        fig = corner.corner(
            flat_samples, labels=param_names, show_titles=True,
            plot_datapoints=plot_datapoints, quantiles=[0.1585, 0.5, 0.8415],
            truths=truths,
        )
        if title:
            fig.suptitle(title)
        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def _plot_rv(self, params: Dict[str, float], title: str = "RV Model", ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Time [days]", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "rv_plot.png", dpi: int = 100) -> None:
        """Helper function to plot RV model with given parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameter values (both free and fixed)
        title : str, optional
            Plot title (default: "RV Model"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main RV plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Time [days]"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "rv_plot.png")
        dpi : int, optional
            Resolution for saving (default: 100)

        Returns
        -------
        np.ndarray
            Time array used for evaluation
        np.ndarray
            RV values at evaluation times
        """
        # Create smooth time curve for plotting
        _tmin, _tmax = self.time.min(), self.time.max()
        _trange = _tmax - _tmin
        tsmooth = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # Initialize arrays for planetary contributions
        rv_all_planets_smooth = np.zeros(len(tsmooth))
        rv_all_planets_obs = np.zeros(len(self.time))

        # Add planetary contributions
        for letter in self.planet_letters:
            planet_params = {}
            for par_name in self.parameterisation.pars:
                key = f"{par_name}_{letter}"
                planet_params[par_name] = params[key]

            planet = ravest.model.Planet(letter, self.parameterisation, planet_params)

            # Calculate for both time arrays in single loop
            rv_all_planets_smooth += planet.radial_velocity(tsmooth)
            rv_all_planets_obs += planet.radial_velocity(self.time)

        # Add trend contribution
        trend_params = {key: params[key] for key in ["g", "gd", "gdd"]}
        trend = ravest.model.Trend(params=trend_params, t0=self.t0)
        rv_total_smooth = rv_all_planets_smooth + trend.radial_velocity(tsmooth)
        rv_total_obs = rv_all_planets_obs + trend.radial_velocity(self.time)

        # Get jitter value for error bars
        jit_value = params["jit"]
        verr_with_jit = np.sqrt(self.verr**2 + jit_value**2)

        # Calculate residuals
        residuals = self.vel - rv_total_obs

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5),
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})

        # Main RV plot
        ax1.errorbar(self.time, self.vel, yerr=self.verr, marker=".", color="tab:blue",
                    ecolor="tab:blue", linestyle="None", markersize=8, zorder=4, label="Data")
        ax1.errorbar(self.time, self.vel, yerr=verr_with_jit, marker="None",
                    ecolor="tab:blue", linestyle="None", alpha=0.5, zorder=3, label="Jitter")

        ax1.plot(tsmooth, rv_total_smooth, label="Model", color="black", zorder=2)
        ax1.set_xlim(tsmooth[0], tsmooth[-1])
        if ylabel_main:
            ax1.set_ylabel(ylabel_main)
        if title:
            ax1.set_title(title)
        ax1.legend(loc="upper right")
        ax1.tick_params(axis='x', labelbottom=False, bottom=True, top=False, direction='in')  # Remove x-axis labels from top plot
        ax1.tick_params(axis='y', direction='in')

        # Set y-axis ticks automatically based on data range
        ax1.yaxis.set_major_locator(AutoLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='y', which='minor', direction='in', length=3)

        # Residuals plot
        ax2.errorbar(self.time, residuals, yerr=self.verr, marker=".", color="tab:blue",
                    ecolor="tab:blue", linestyle="None", markersize=8, zorder=4)
        ax2.errorbar(self.time, residuals, yerr=verr_with_jit, marker="None",
                    ecolor="tab:blue", linestyle="None", alpha=0.5, zorder=3)
        ax2.axhline(0, color="k", linestyle="--", zorder=2)
        ax2.set_xlim(tsmooth[0], tsmooth[-1])

        # Set symmetric y-limits for residuals
        max_abs_residual = np.max(np.abs(residuals + verr_with_jit))
        ax2.set_ylim(-max_abs_residual * 1.1, max_abs_residual * 1.1)

        if xlabel:
            ax2.set_xlabel(xlabel)
        if ylabel_residuals:
            ax2.set_ylabel(ylabel_residuals)
        ax2.tick_params(axis='x', direction='in')
        ax2.tick_params(axis='y', direction='in')
        ax2.tick_params(axis='x', top=True, labeltop=False)  # Add ticks on shared border

        # Set y-axis ticks automatically based on residuals range
        ax2.yaxis.set_major_locator(AutoLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(axis='y', which='minor', direction='in', length=3)

        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def _plot_phase(self, planet_letter: str, params: Dict[str, float], title: str = None, ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Orbital phase", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "phase_plot.png", dpi: int = 100) -> None:
        """Helper function to plot phase-folded RV model for a single planet with given parameters.

        Parameters
        ----------
        planet_letter : str
            Letter identifying the planet to plot (e.g., 'b', 'c', 'd')
        params : dict
            Dictionary of parameter values (both free and fixed)
        title : str, optional
            Plot title (default: f"Planet {planet_letter} Phase Plot"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main phase plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Orbital phase"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "phase_plot.png")
        dpi : int, optional
            Resolution for saving (default: 100)
        """
        if title is None:
            title = f"Planet {planet_letter} Phase Plot"

        # get smooth linear time curve for plotting
        _tmin, _tmax = self.time.min(), self.time.max()
        _trange = _tmax - _tmin
        tsmooth = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # Get jitter value for error bars
        jit_value = params["jit"]
        verr_with_jit = np.sqrt(self.verr**2 + jit_value**2)

        # Get period and time of conjunction for this planet
        P = params[f"P_{planet_letter}"]

        # Convert to tc if needed
        if "Tc" in self.parameterisation.pars:
            Tc = params[f"Tc_{planet_letter}"]
        elif "e" in self.parameterisation.pars and "w" in self.parameterisation.pars:
            _e = params[f"e_{planet_letter}"]
            _w = params[f"w_{planet_letter}"]
            _Tp = params[f"Tp_{planet_letter}"]
            Tc = self.parameterisation.convert_tp_to_tc(_Tp, P, _e, _w)
        else:
            # Fall back to default parameterisation conversion
            planet_params = {par: params[f"{par}_{planet_letter}"] for par in self.parameterisation.pars}
            default_params = self.parameterisation.convert_pars_to_default_parameterisation(planet_params)
            Tc = self.parameterisation.convert_tp_to_tc(default_params["Tp"], P, default_params["e"], default_params["w"])

        # Calculate phase-folded time arrays (in units of orbital phase)
        t_fold_sorted, inds = ravest.model.fold_time_series(self.time, P, Tc)
        tsmooth_fold_sorted, smooth_inds = ravest.model.fold_time_series(tsmooth, P, Tc)

        # Calculate RV contribution from this planet only
        planet_params = {par: params[f"{par}_{planet_letter}"] for par in self.parameterisation.pars}
        planet = ravest.model.Planet(planet_letter, self.parameterisation, planet_params)
        planet_rv_tsmooth = planet.radial_velocity(tsmooth)
        planet_rv_data = planet.radial_velocity(self.time)
        planet_rv_sorted = planet_rv_tsmooth[smooth_inds]

        # Calculate all other contributions (other planets + trend) at data times
        other_rv = np.zeros(len(self.time))
        for other_letter in self.planet_letters:
            if other_letter != planet_letter:
                other_params = {par: params[f"{par}_{other_letter}"] for par in self.parameterisation.pars}
                other_planet = ravest.model.Planet(other_letter, self.parameterisation, other_params)
                other_rv += other_planet.radial_velocity(self.time)

        # Add trend
        trend_params = {key: params[key] for key in ["g", "gd", "gdd"]}
        trend = ravest.model.Trend(params=trend_params, t0=self.t0)
        other_rv += trend.radial_velocity(self.time)

        # Calculate data with other components subtracted
        data_minus_others = self.vel - other_rv

        # Sort the data according to phase folding
        data_minus_others_sorted = data_minus_others[inds]
        verr_sorted = self.verr[inds]
        verr_with_jit_sorted = verr_with_jit[inds]

        # Calculate residuals (data - model for this planet)
        residuals = data_minus_others - planet_rv_data
        residuals_sorted = residuals[inds]

        # Create figure with subplots (main plot + residuals)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5),
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})

        # Main phase plot
        ax1.errorbar(t_fold_sorted, data_minus_others_sorted, yerr=verr_sorted, marker=".",
                    linestyle="None", color="tab:blue", markersize=8, zorder=4, label="Data")
        ax1.errorbar(t_fold_sorted, data_minus_others_sorted, yerr=verr_with_jit_sorted, marker="None",
                    linestyle="None", color="tab:blue", alpha=0.5, zorder=3, label="Jitter")

        # Plot phase-folded model for this planet
        ax1.plot(tsmooth_fold_sorted, planet_rv_sorted, label="Model", color="black", zorder=2)
        ax1.set_xlim(-0.5, 0.5)
        ax1.xaxis.set_major_locator(MultipleLocator(0.25))  # Set x-ticks every 0.25
        if ylabel_main:
            ax1.set_ylabel(ylabel_main)
        ax1.legend(loc="upper right")
        if title:
            ax1.set_title(title)
        ax1.tick_params(axis='x', labelbottom=False, bottom=True, top=False, direction='in')  # Remove x-axis labels from top plot
        ax1.tick_params(axis='y', direction='in')

        # Set y-axis ticks automatically based on phase data range
        ax1.yaxis.set_major_locator(AutoLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='y', which='minor', direction='in', length=3)

        # Annotate with planet info
        K_value = params[f"K_{planet_letter}"]
        s = f"Planet {planet_letter}\nP={P:.2f} d\nK={K_value:.2f} m/s"
        ax1.annotate(s, xy=(0, 1), xycoords="axes fraction",
                    xytext=(+0.5, -0.5), textcoords="offset fontsize", va="top")

        # Residuals plot (phase-folded)
        ax2.errorbar(t_fold_sorted, residuals_sorted, yerr=verr_sorted, marker=".",
                    linestyle="None", color="tab:blue", markersize=8, zorder=4)
        ax2.errorbar(t_fold_sorted, residuals_sorted, yerr=verr_with_jit_sorted, marker="None",
                    linestyle="None", color="tab:blue", alpha=0.5, zorder=3)
        ax2.axhline(0, color="k", linestyle="--", zorder=2)
        ax2.set_xlim(-0.5, 0.5)
        ax2.xaxis.set_major_locator(MultipleLocator(0.25))  # Set x-ticks every 0.25

        # Set symmetric y-limits for residuals
        max_abs_residual = np.max(np.abs(residuals_sorted + verr_with_jit_sorted))
        ax2.set_ylim(-max_abs_residual * 1.1, max_abs_residual * 1.1)

        if xlabel:
            ax2.set_xlabel(xlabel)
        if ylabel_residuals:
            ax2.set_ylabel(ylabel_residuals)
        ax2.tick_params(axis='x', direction='in')
        ax2.tick_params(axis='y', direction='in')
        ax2.tick_params(axis='x', top=True, labeltop=False)  # Add ticks on shared border

        # Set y-axis ticks automatically based on residuals range
        ax2.yaxis.set_major_locator(AutoLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(axis='y', which='minor', direction='in', length=3)

        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def plot_posterior_rv(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, show_CI: bool = True, title: str | None = "Posterior RV", ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Time [days]", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "posterior_rv.png", dpi: int = 100) -> None:
        """Plot the posterior RV model with uncertainty bands from MCMC samples.

        Calculates RV model predictions for each MCMC sample, then plots the median
        with optional 68% CI (16th-84th percentile) uncertainty bands. Shows both
        the full model and residuals vs data.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        show_CI : bool, optional
            Show 68.3% credible interval band (default: True)
        title : str or None, optional
            Title for the main RV plot (default: "Posterior RV"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main RV plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Time [days]"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot to path `fname` (default: False)
        fname : str, optional
            The path to save the plot to (default: "posterior_rv.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)
        """
        # Create smooth time curve for plotting (same as _plot_rv helper)
        _tmin, _tmax = self.time.min(), self.time.max()
        _trange = _tmax - _tmin
        tsmooth = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # Calculate posterior RV predictions
        rv_all_planets_trend_matrix_smooth = self.calculate_rv_total_from_samples(times=tsmooth, discard_start=discard_start, discard_end=discard_end, thin=thin)
        rv_all_planets_trend_matrix_obs = self.calculate_rv_total_from_samples(times=self.time, discard_start=discard_start, discard_end=discard_end, thin=thin)

        # Calculate percentiles
        rv_percentiles_smooth = np.percentile(rv_all_planets_trend_matrix_smooth, [15.85, 50, 84.15], axis=0)
        rv_percentiles_obs = np.percentile(rv_all_planets_trend_matrix_obs, [15.85, 50, 84.15], axis=0)

        # Calculate residuals using median model at data times
        residuals = self.vel - rv_percentiles_obs[1]

        # Get jitter samples for error bars
        samples_dict = self.get_samples_dict(discard_start=discard_start, discard_end=discard_end, thin=thin)
        if 'jit' in samples_dict:
            jit_median = np.median(samples_dict['jit'])
        else:
            jit_median = self.fixed_params_values_dict['jit']
        verr_with_jit = np.sqrt(self.verr**2 + jit_median**2)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})

        # Main RV plot
        ax1.errorbar(self.time, self.vel, yerr=self.verr, marker=".", color="tab:blue", ecolor="tab:blue", linestyle="None", markersize=8, zorder=4, label="Data")
        ax1.errorbar(self.time, self.vel, yerr=verr_with_jit, marker="None", ecolor="tab:blue", linestyle="None", alpha=0.5, zorder=3, label="Jitter")

        # Plot median model and uncertainty
        ax1.plot(tsmooth, rv_percentiles_smooth[1], label="Model", color="black", zorder=2)
        if show_CI:
            ax1.fill_between(tsmooth, rv_percentiles_smooth[0], rv_percentiles_smooth[2], color="tab:gray", alpha=0.3, edgecolor="none", label="68.3% CI")

        ax1.set_xlim(tsmooth[0], tsmooth[-1])
        if ylabel_main:
            ax1.set_ylabel(ylabel_main)
        if title:
            ax1.set_title(title)
        ax1.legend(loc="upper right")
        ax1.tick_params(axis='x', labelbottom=False, bottom=True, top=False, direction='in')
        ax1.tick_params(axis='y', direction='in')

        # Set y-axis ticks automatically based on data range
        ax1.yaxis.set_major_locator(AutoLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='y', which='minor', direction='in', length=3)

        # Residuals plot
        ax2.errorbar(self.time, residuals, yerr=self.verr, marker=".", color="tab:blue", ecolor="tab:blue", linestyle="None", markersize=8, zorder=4)
        ax2.errorbar(self.time, residuals, yerr=verr_with_jit, marker="None", ecolor="tab:blue", linestyle="None", alpha=0.5, zorder=3)
        ax2.axhline(0, color="k", linestyle="--", zorder=2)
        ax2.set_xlim(tsmooth[0], tsmooth[-1])

        # Set symmetric y-limits for residuals plot, so 0 is in centre
        max_abs_residual = np.max(np.abs(residuals + verr_with_jit))
        ax2.set_ylim(-max_abs_residual * 1.1, max_abs_residual * 1.1)

        if xlabel:
            ax2.set_xlabel(xlabel)
        if ylabel_residuals:
            ax2.set_ylabel(ylabel_residuals)
        ax2.tick_params(axis='x', direction='in')
        ax2.tick_params(axis='y', direction='in')
        ax2.tick_params(axis='x', top=True, labeltop=False)

        # Set y-axis ticks automatically based on residuals range
        ax2.yaxis.set_major_locator(AutoLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(axis='y', which='minor', direction='in', length=3)

        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def plot_posterior_phase(self, planet_letter: str, discard_start: int = 0, discard_end: int = 0, thin: int = 1, show_CI: bool = True, title: str | None = None, ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Orbital phase", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "posterior_phase.png", dpi: int = 100) -> None:
        """Plot phase-folded RV model with uncertainty bands from MCMC samples.

        Shows the phase-folded planetary signal with uncertainty bands calculated
        from MCMC samples. Removes contributions from trends and other planets.

        Parameters
        ----------
        planet_letter : str
            Letter identifying the planet to plot (e.g., 'b', 'c', 'd')
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        show_CI : bool, optional
            Show 68.3% credible interval band (default: True)
        title : str or None, optional
            Title for the main phase plot (default: "Posterior Phase Plot - Planet {planet_letter}"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main phase plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Orbital phase"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot to path `fname` (default: False)
        fname : str, optional
            The path to save the plot to (default: "posterior_phase.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)
        """
        # Get period (handle both free and fixed cases)
        samples_dict = self.get_samples_dict(discard_start=discard_start, discard_end=discard_end, thin=thin)

        # Combine with fixed parameters and fixed hyperparameters
        params = samples_dict | self.fixed_params_values_dict

        # Create smooth time array (same approach as _plot_rv)
        _tmin, _tmax = self.time.min(), self.time.max()
        _trange = _tmax - _tmin
        tsmooth = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # Get jitter value for error bars
        jit_med = np.median(params["jit"])
        verr_with_jit = np.sqrt(self.verr**2 + jit_med**2)

        # Get period value
        _P = params[f'P_{planet_letter}']

        # Get (or calculate) Tc for this planet for folding around
        if "Tc" in self.parameterisation.pars:
            _Tc = params[f"Tc_{planet_letter}"]
        else:
            # Fall back to default parameterisation conversion (as it has P, e, w and Tp, so we can definitely get Tc)
            planet_params = {par: params[f"{par}_{planet_letter}"] for par in self.parameterisation.pars}
            default_params = self.parameterisation.convert_pars_to_default_parameterisation(planet_params)
            _Tc = self.parameterisation.convert_tp_to_tc(default_params["Tp"], _P, default_params["e"], default_params["w"])

        # just for the folding, take the median value of the P and Tc samples
        Tc_med = np.median(_Tc)
        P_med = np.median(_P)

        # Phase fold both data and smooth times
        t_fold, inds = ravest.model.fold_time_series(self.time, P_med, Tc_med)
        tsmooth_fold_sorted, smooth_inds = ravest.model.fold_time_series(tsmooth, P_med, Tc_med)


        # Calculate RV components from MCMC samples (matrix of n_samples x n_obs, or n_samples x n_smooth)
        rv_planet_data = self.calculate_rv_planet_from_samples(planet_letter, self.time, discard_start, discard_end, thin)
        rv_planet_smooth = self.calculate_rv_planet_from_samples(planet_letter, tsmooth, discard_start, discard_end, thin)
        rv_trend_data = self.calculate_rv_trend_from_samples(self.time, discard_start, discard_end, thin)

        # Calculate RV contributions from all OTHER planets (not the target planet)
        rv_other_planets_data = np.zeros_like(rv_trend_data)
        for other_letter in self.planet_letters:
            if other_letter != planet_letter:
                rv_other_planet = self.calculate_rv_planet_from_samples(other_letter, self.time, discard_start, discard_end, thin)
                rv_other_planets_data += rv_other_planet

        # Combine all non-target contributions (other planets + trend)
        rv_others_total_data = rv_trend_data + rv_other_planets_data

        # Calculate percentiles across these matrices of RVs
        rv_planet_data_percs = np.percentile(rv_planet_data, [15.85, 50, 84.15], axis=0)
        rv_planet_smooth_percs = np.percentile(rv_planet_smooth, [15.85, 50, 84.15], axis=0)
        rv_others_total_percs = np.percentile(rv_others_total_data, [15.85, 50, 84.15], axis=0)

        # Remove all other contributions from observed data (using median of combined other contributions)
        data_minus_others = self.vel - rv_others_total_percs[1]

        # That gives us the component of the data just due to this planet. So now we can see how our modelled planet compares
        residuals = data_minus_others - rv_planet_data_percs[1]


        # Create figure with subplots (main plot + residuals)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})

        # Main phase plot - plot data with other contributions removed, sorted by phase
        ax1.errorbar(t_fold, data_minus_others[inds], yerr=self.verr[inds], marker=".",
                    linestyle="None", color="tab:blue", markersize=8, zorder=4, label="Data")
        ax1.errorbar(t_fold, data_minus_others[inds], yerr=verr_with_jit[inds], marker="None",
                    linestyle="None", color="tab:blue", alpha=0.5, zorder=3, label="Jitter")

        # Plot planet model with uncertainty, sorted by phase
        ax1.plot(tsmooth_fold_sorted, rv_planet_smooth_percs[1][smooth_inds],
                linestyle="-", color="black", zorder=3, label="Model")
        if show_CI:
            ax1.fill_between(tsmooth_fold_sorted, rv_planet_smooth_percs[0][smooth_inds],
                            rv_planet_smooth_percs[2][smooth_inds], color="tab:gray", alpha=0.3, edgecolor="none", label="68.3% CI")

        ax1.set_xlim(-0.5, 0.5)
        ax1.xaxis.set_major_locator(MultipleLocator(0.25))  # Set x-ticks every 0.25
        if ylabel_main:
            ax1.set_ylabel(ylabel_main)
        if title is None:
            ax1.set_title(f"Posterior Phase Plot - Planet {planet_letter}")
        elif title:
            ax1.set_title(title)
        ax1.legend(loc="upper right")
        ax1.tick_params(axis='x', labelbottom=False, bottom=True, top=False, direction='in')
        ax1.tick_params(axis='y', direction='in')

        # Set y-axis ticks automatically based on phase data range
        ax1.yaxis.set_major_locator(AutoLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='y', which='minor', direction='in', length=3)

        # Residuals plot (phase-folded)
        ax2.errorbar(t_fold, residuals[inds], yerr=self.verr[inds], marker=".",
                    linestyle="None", color="tab:blue", markersize=8, zorder=4)
        ax2.errorbar(t_fold, residuals[inds], yerr=verr_with_jit[inds], marker="None",
                    linestyle="None", color="tab:blue", alpha=0.5, zorder=3)
        ax2.axhline(0, color="k", linestyle="--", zorder=2)
        ax2.set_xlim(-0.5, 0.5)
        ax2.xaxis.set_major_locator(MultipleLocator(0.25))  # Set x-ticks every 0.25

        # Set symmetric y-limits for residuals
        max_abs_residual = np.max(np.abs(residuals[inds] + verr_with_jit[inds]))
        ax2.set_ylim(-max_abs_residual * 1.1, max_abs_residual * 1.1)

        if xlabel:
            ax2.set_xlabel(xlabel)
        if ylabel_residuals:
            ax2.set_ylabel(ylabel_residuals)
        ax2.tick_params(axis='x', direction='in')
        ax2.tick_params(axis='y', direction='in')
        ax2.tick_params(axis='x', top=True, labeltop=False)

        # Set y-axis ticks automatically based on residuals range
        ax2.yaxis.set_major_locator(AutoLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(axis='y', which='minor', direction='in', length=3)

        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def calculate_rv_planet_from_samples(self, planet_letter: str, times: np.ndarray, discard_start: int = 0, discard_end: int = 0, thin: int = 1, progress: bool = False) -> np.ndarray:
        """Calculate planetary RV for each MCMC sample.

        This calculates RV(params_i) for each MCMC sample i, preserving
        parameter correlations. This differs from using median parameters
        which may not represent actual samples from the posterior.

        Parameters
        ----------
        planet_letter : str
            Planet letter (e.g., 'b', 'c')
        times : np.ndarray
            Time points to calculate RV at
        discard_start : int, optional
            Discard first N steps (default: 0)
        discard_end : int, optional
            Discard last N steps (default: 0)
        thin : int, optional
            Use every Nth sample (default: 1)
        progress : bool, optional
            Show progress bar (default: False)

        Returns
        -------
        np.ndarray
            Shape (n_samples, len(times)) - RV for each sample
        """
        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        planet_rvs = np.zeros((len(samples), len(times)))

        iterator = tqdm(enumerate(samples), total=len(samples), disable=not progress, desc=f"Calculating planet {planet_letter} RV from samples")
        for i, row in iterator:
            # Build complete params dict for this sample
            params = self.build_params_dict(row)

            # Use custom method
            planet_rvs[i, :] = self.calculate_rv_planet_custom(planet_letter, times, params)

        return planet_rvs

    def calculate_rv_trend_from_samples(self, times: np.ndarray, discard_start: int = 0, discard_end: int = 0, thin: int = 1, progress: bool = False) -> np.ndarray:
        """Calculate trend RV for each MCMC sample.

        This calculates RV_trend(params_i) for each MCMC sample i, preserving
        parameter correlations. This differs from using median parameters
        which may not represent actual samples from the posterior.

        Parameters
        ----------
        times : np.ndarray
            Time points to calculate RV at
        discard_start : int, optional
            Discard first N steps (default: 0)
        discard_end : int, optional
            Discard last N steps (default: 0)
        thin : int, optional
            Use every Nth sample (default: 1)
        progress : bool, optional
            Show progress bar (default: False)

        Returns
        -------
        np.ndarray
            Shape (n_samples, len(times)) - Trend RV for each sample
        """
        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        trend_rvs = np.zeros((len(samples), len(times)))

        iterator = tqdm(enumerate(samples), total=len(samples), disable=not progress, desc="Calculating trend RV from samples")
        for i, row in iterator:
            # Build complete params dict for this sample
            params = self.build_params_dict(row)

            # Use custom method
            trend_rvs[i, :] = self.calculate_rv_trend_custom(times, params)

        return trend_rvs

    def calculate_rv_total_from_samples(self, times: np.ndarray, discard_start: int = 0, discard_end: int = 0, thin: int = 1, progress: bool = False) -> np.ndarray:
        """Calculate total RV (planets + trend) for each MCMC sample.

        This calculates RV_total(params_i) for each MCMC sample i, preserving
        parameter correlations. This differs from using median parameters
        which may not represent actual samples from the posterior.

        Parameters
        ----------
        times : np.ndarray
            Time points to calculate RV at
        discard_start : int, optional
            Discard first N steps (default: 0)
        discard_end : int, optional
            Discard last N steps (default: 0)
        thin : int, optional
            Use every Nth sample (default: 1)
        progress : bool, optional
            Show progress bar (default: False)

        Returns
        -------
        np.ndarray
            Shape (n_samples, len(times)) - Total RV for each sample
        """
        # Get trend RV for all samples
        total_rvs = self.calculate_rv_trend_from_samples(times, discard_start, discard_end, thin, progress)

        # Add each planet's RV
        for planet_letter in self.planet_letters:
            planet_rvs = self.calculate_rv_planet_from_samples(planet_letter, times, discard_start, discard_end, thin, progress)
            total_rvs += planet_rvs

        return total_rvs

    def calculate_rv_planet_custom(self, planet_letter: str, times: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """Calculate planetary RV for a single set of custom parameters.

        Useful for calculating RV with specific parameter values (e.g., best lnprob
        sample, median parameters, or experimental values).

        Parameters
        ----------
        planet_letter : str
            Planet letter (e.g., 'b', 'c')
        times : np.ndarray
            Time points to calculate RV at
        params : dict[str, float]
            Complete parameter dictionary (both free and fixed parameters).
            Can be created using build_params_dict() or manually constructed.

        Returns
        -------
        np.ndarray
            RV values at the requested times

        Examples
        --------
        >>> # Using MAP result
        >>> map_result = fitter.find_map_estimate()
        >>> params = fitter.build_params_dict(map_result.x)
        >>> rv = fitter.calculate_rv_planet_custom('b', times, params)
        >>>
        >>> # Using best lnprob sample
        >>> best_params = fitter.get_sample_with_best_lnprob(discard_start=1000)
        >>> params = fitter.build_params_dict(best_params)
        >>> rv = fitter.calculate_rv_planet_custom('b', times, params)
        """
        # Extract planet parameters
        planet_params = {}
        for par in self.parameterisation.pars:
            key = f"{par}_{planet_letter}"
            planet_params[par] = params[key]

        # Calculate planet RV
        planet = ravest.model.Planet(planet_letter, self.parameterisation, planet_params)
        return planet.radial_velocity(times)

    def calculate_rv_trend_custom(self, times: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """Calculate trend RV for a single set of custom parameters.

        Useful for calculating RV with specific parameter values (e.g., best lnprob
        sample, median parameters, or experimental values).

        Parameters
        ----------
        times : np.ndarray
            Time points to calculate RV at
        params : dict[str, float]
            Complete parameter dictionary (both free and fixed parameters).
            Can be created using build_params_dict() or manually constructed.

        Returns
        -------
        np.ndarray
            Trend RV values at the requested times

        Examples
        --------
        >>> # Using MAP result
        >>> map_result = fitter.find_map_estimate()
        >>> params = fitter.build_params_dict(map_result.x)
        >>> trend_rv = fitter.calculate_rv_trend_custom(times, params)
        """
        # Calculate trend RV
        trend = ravest.model.Trend(params={"g": params["g"], "gd": params["gd"], "gdd": params["gdd"]}, t0=self.t0)
        return trend.radial_velocity(times)

    def calculate_rv_total_custom(self, times: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """Calculate total RV (trend + all planets) for a single set of custom parameters.

        Useful for calculating RV with specific parameter values (e.g., best lnprob
        sample, median parameters, or experimental values).

        Parameters
        ----------
        times : np.ndarray
            Time points to calculate RV at
        params : dict[str, float]
            Complete parameter dictionary (both free and fixed parameters).
            Can be created using build_params_dict() or manually constructed.

        Returns
        -------
        np.ndarray
            Total RV values (trend + all planets) at the requested times

        Examples
        --------
        >>> # Using MAP result
        >>> map_result = fitter.find_map_estimate()
        >>> params = fitter.build_params_dict(map_result.x)
        >>> total_rv = fitter.calculate_rv_total_custom(times, params)
        >>>
        >>> # Using median parameters
        >>> samples_df = fitter.get_samples_df(discard_start=1000)
        >>> median_values = samples_df.median().to_dict()
        >>> params = fitter.build_params_dict(median_values)
        >>> total_rv = fitter.calculate_rv_total_custom(times, params)
        """
        # Calculate trend
        total_rv = self.calculate_rv_trend_custom(times, params)

        # Add each planet
        for planet_letter in self.planet_letters:
            planet_rv = self.calculate_rv_planet_custom(planet_letter, times, params)
            total_rv += planet_rv

        return total_rv

    def plot_MAP_rv(self, map_result: scipy.optimize.OptimizeResult, title: str | None = "MAP RV", ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Time [days]", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "MAP_rv.png", dpi: int = 100) -> None:
        """Plot radial velocity data and model using MAP parameter estimates.

        Parameters
        ----------
        map_result : scipy.optimize.OptimizeResult
            Result from find_map_estimate() containing the MAP parameters
        title : str or None, optional
            Plot title (default: "MAP RV"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main RV plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Time [days]"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "MAP_rv.png")
        dpi : int, optional
            Resolution for saving (default: 100)
        """
        # Get MAP parameter values from the optimization result
        map_params = dict(zip(self.free_params_names, map_result.x))

        # Combine with fixed parameters
        all_params = self.fixed_params_values_dict | map_params

        # Use helper function to create the plot
        self._plot_rv(all_params, title=title, ylabel_main=ylabel_main, xlabel=xlabel, ylabel_residuals=ylabel_residuals, save=save, fname=fname, dpi=dpi)

    def plot_MAP_phase(self, planet_letter: str, map_result: scipy.optimize.OptimizeResult, title: str | None = None, ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Orbital phase", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "MAP_phase.png", dpi: int = 100) -> None:
        """Plot phase-folded radial velocity data and model using MAP parameter estimates.

        Parameters
        ----------
        planet_letter : str
            Letter identifying the planet to plot (e.g., 'b', 'c', 'd')
        map_result : scipy.optimize.OptimizeResult
            Result from find_map_estimate() containing the MAP parameters
        title : str or None, optional
            Plot title (default: f"MAP Phase Plot - Planet {planet_letter}"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main phase plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Orbital phase"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "MAP_phase.png")
        dpi : int, optional
            Resolution for saving (default: 100)
        """
        # Get MAP parameter values from the optimization result
        map_params = dict(zip(self.free_params_names, map_result.x))

        # Combine with fixed parameters
        all_params = self.fixed_params_values_dict | map_params

        # Set default title if not provided
        if title is None:
            title = f"MAP Phase Plot - Planet {planet_letter}"

        # Use helper function to create the plot
        self._plot_phase(planet_letter, all_params, title=title, ylabel_main=ylabel_main, xlabel=xlabel, ylabel_residuals=ylabel_residuals, save=save, fname=fname, dpi=dpi)

    def plot_custom_rv(self, params: dict, title: str | None = "Custom RV Plot", ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Time [days]", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "custom_rv.png", dpi: int = 100) -> None:
        """Plot radial velocity data and model using custom parameter values.

        Allows plotting with arbitrary parameter values for exploring parameter space
        or comparing theoretical models.

        Parameters
        ----------
        params : dict
            Dictionary of parameter values to use for plotting. Keys should match
            parameter names, values should be floats. Must include all required
            parameters for the current parameterisation.
        title : str or None, optional
            Plot title (default: "Custom RV Plot"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main RV plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Time [days]"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "custom_rv.png")
        dpi : int, optional
            Resolution for saving (default: 100)

        Examples
        --------
        >>> # Plot with custom values (must include all required parameters)
        >>> fitter.plot_custom_rv({"P_b": 4.25, "K_b": 55.0, "e_b": 0.1,
        ...                        "w_b": 1.57, "Tc_b": 2456325.5,
        ...                        "g": -10.2, "gd": 0.0, "gdd": 0.0, "jit": 2.0})
        """
        # Validate that all required parameters are present
        expected_params = set(self.free_params_names + list(self.fixed_params_names))
        provided_params = set(params.keys())
        missing_params = expected_params - provided_params
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Use helper function to create the plot
        self._plot_rv(params, title=title, ylabel_main=ylabel_main, xlabel=xlabel, ylabel_residuals=ylabel_residuals, save=save, fname=fname, dpi=dpi)

    def plot_custom_phase(self, planet_letter: str, params: dict, title: str | None = None, ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Orbital phase", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "custom_phase.png", dpi: int = 100) -> None:
        """Plot phase-folded radial velocity data and model using custom parameter values.

        Allows plotting phase-folded data with arbitrary parameter values for exploring
        parameter space or comparing theoretical models.

        Parameters
        ----------
        planet_letter : str
            Letter identifying the planet to plot (e.g., 'b', 'c', 'd')
        params : dict
            Dictionary of parameter values to use for plotting. Keys should match
            parameter names, values should be floats. Must include all required
            parameters for the current parameterisation.
        title : str or None, optional
            Plot title (default: f"Custom Phase Plot - Planet {planet_letter}"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main phase plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Orbital phase"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "custom_phase.png")
        dpi : int, optional
            Resolution for saving (default: 100)

        Examples
        --------
        >>> # Plot phase curve with custom values
        >>> fitter.plot_custom_phase("b", {"P_b": 4.25, "K_b": 55.0, "e_b": 0.1,
        ...                               "w_b": 1.57, "Tc_b": 2456325.5,
        ...                               "g": -10.2, "gd": 0.0, "gdd": 0.0, "jit": 2.0})
        """
        # Validate that all required parameters are present
        expected_params = set(self.free_params_names + list(self.fixed_params_names))
        provided_params = set(params.keys())
        missing_params = expected_params - provided_params
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Set default title if not provided
        if title is None:
            title = f"Custom Phase Plot - Planet {planet_letter}"

        # Use helper function to create the plot
        self._plot_phase(planet_letter, params, title=title, ylabel_main=ylabel_main, xlabel=xlabel, ylabel_residuals=ylabel_residuals, save=save, fname=fname, dpi=dpi)

    def plot_best_sample_rv(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, title: str | None = "Best Sample RV Plot", ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Time [days]", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "best_sample_rv.png", dpi: int = 100) -> None:
        """Plot radial velocity data and model using parameter values from the MCMC sample with highest log probability.

        This is useful for comparing with plot_MAP_rv() to diagnose potential issues with
        MAP convergence or MCMC mixing. The two plots should be very similar if both
        MAP and MCMC are working correctly.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        title : str or None, optional
            Title for the main RV plot (default: "Best Sample RV Plot"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main RV plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Time [days]"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "best_sample_rv.png")
        dpi : int, optional
            Resolution for saving (default: 100)
        """
        # Get parameter values from best sample
        best_sample_params = self.get_sample_with_best_lnprob(discard_start=discard_start, discard_end=discard_end, thin=thin)

        # Combine with fixed parameters
        all_params = self.fixed_params_values_dict | best_sample_params

        # Use helper function to create the plot
        self._plot_rv(all_params, title=title, ylabel_main=ylabel_main, xlabel=xlabel, ylabel_residuals=ylabel_residuals, save=save, fname=fname, dpi=dpi)

    def plot_best_sample_phase(self, planet_letter: str, discard_start: int = 0, discard_end: int = 0, thin: int = 1, title: str | None = None, ylabel_main: str | None = "Radial velocity [m/s]", xlabel: str | None = "Orbital phase", ylabel_residuals: str | None = "Residuals [m/s]", save: bool = False, fname: str = "best_sample_phase.png", dpi: int = 100) -> None:
        """Plot phase-folded radial velocity data and model using parameter values from the MCMC sample with highest log probability.

        This is useful for comparing with plot_MAP_phase() to diagnose potential issues with
        MAP convergence or MCMC mixing. The two plots should be very similar if both
        MAP and MCMC are working correctly.

        Parameters
        ----------
        planet_letter : str
            Letter identifying the planet to plot (e.g., 'b', 'c', 'd')
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        title : str or None, optional
            Title for the main phase plot (default: "Best Sample Phase Plot - Planet {planet_letter}"). Set to None or "" to skip.
        ylabel_main : str or None, optional
            Y-axis label for main phase plot (default: "Radial velocity [m/s]"). Set to None or "" to skip.
        xlabel : str or None, optional
            X-axis label for residuals plot (default: "Orbital phase"). Set to None or "" to skip.
        ylabel_residuals : str or None, optional
            Y-axis label for residuals plot (default: "Residuals [m/s]"). Set to None or "" to skip.
        save : bool, optional
            Save the plot (default: False)
        fname : str, optional
            Filename to save (default: "best_sample_phase.png")
        dpi : int, optional
            Resolution for saving (default: 100)
        """
        # Get parameter values from best sample
        best_sample_params = self.get_sample_with_best_lnprob(discard_start=discard_start, discard_end=discard_end, thin=thin)

        # Combine with fixed parameters
        all_params = self.fixed_params_values_dict | best_sample_params

        # Set default title if not provided
        if title is None:
            title = f"Best Sample Phase Plot - Planet {planet_letter}"

        # Use helper function to create the plot
        self._plot_phase(planet_letter, all_params, title=title, ylabel_main=ylabel_main, xlabel=xlabel, ylabel_residuals=ylabel_residuals,
                        save=save, fname=fname, dpi=dpi)

class LogPosterior:
    """Log posterior probability for MCMC sampling.

    Combines log likelihood and log prior for Bayesian parameter estimation.
    """

    def __init__(
        self,
        planet_letters: list[str],
        parameterisation: Parameterisation,
        priors: dict[str, Callable[[float], float]],
        fixed_params: dict[str, float],
        free_params_names: list[str],
        time: np.ndarray,
        vel: np.ndarray,
        verr: np.ndarray,
        t0: float,
    ) -> None:
        """Initialize the LogPosterior object.

        Parameters
        ----------
        planet_letters : list[str]
            List of single-character planet identifiers.
        parameterisation : Parameterisation
            The orbital parameterisation to use.
        priors : dict[str, Callable[[float], float]]
            Dictionary mapping parameter names to their prior probability functions.
        fixed_params : dict[str, float]
            Dictionary of fixed parameter values.
        free_params_names : list[str]
            List of free parameter names to sample.
        time : np.ndarray
            Time of each observation [days].
        vel : np.ndarray
            Radial velocity at each time [m/s].
        verr : np.ndarray
            Uncertainty on the radial velocity at each time [m/s].
        t0 : float
            Reference time for the trend [days].
        """
        self.planet_letters = planet_letters
        self.parameterisation = parameterisation
        self.priors = priors
        self.fixed_params = fixed_params
        self.free_params_names = free_params_names
        self.time = time
        self.vel = vel
        self.verr = verr
        self.t0 = t0

        # Create log-likelihood and log-prior objects for later
        self.log_likelihood = LogLikelihood(time=self.time,
                                            vel=self.vel,
                                            verr=self.verr,
                                            t0=self.t0,
                                            planet_letters=self.planet_letters,
                                            parameterisation=self.parameterisation,
                                            )
        self.log_prior = LogPrior(self.priors)

    def _convert_params_for_prior_evaluation(self, free_params_dict: dict[str, float]) -> Dict[str, float]:
        """Convert free parameters for prior evaluation if needed.

        Parameters
        ----------
        free_params_dict : dict
            Free parameters in current parameterisation

        Returns
        -------
        dict
            Parameters with names/values converted for prior evaluation
        """
        # Three cases:
        # Case 1: User is fitting in transformed parameterisation, but priors are in same transformed parameterisation
        # Case 2: User is fitting in default parameterisation, and priors are also in default parameterisation
        # Case 3: User is fitting in transformed parameterisation, but priors are in default parameterisation

        # Simple detection: do prior keys match our current free parameter names?
        prior_keys = set(self.priors.keys())
        free_param_keys = set(self.free_params_names)
        if prior_keys == free_param_keys:
            # No conversion needed (Cases 1 & 2)
            return free_params_dict
        else:
            # Conversion needed (Case 3) - convert to default parameterisation equivalents
            # Start with just the non-planetary parameters that match
            params_for_prior = {key: value for key, value in free_params_dict.items()
                              if key in prior_keys}

            all_params = self.fixed_params | free_params_dict

            # Convert each planet's parameters
            for planet_letter in self.planet_letters:
                # Get current planet parameters
                planet_params = {par: all_params[f"{par}_{planet_letter}"]
                               for par in self.parameterisation.pars}

                # Convert to default parameterisation
                default_params = self.parameterisation.convert_pars_to_default_parameterisation(planet_params)

                # Add the converted parameter values for priors that need them
                for default_par, value in default_params.items():
                    default_param_key = f"{default_par}_{planet_letter}"
                    if default_param_key in prior_keys:  # Only add if we have a prior for it
                        params_for_prior[default_param_key] = value

            return params_for_prior

    def log_probability(self, free_params_dict: Dict[str, float]) -> float:
        """Calculate log posterior probability for given free parameters.

        Parameters
        ----------
        free_params_dict : Dict[str, float]
            Dictionary of free parameter values

        Returns
        -------
        float
            Log posterior probability (log likelihood + log prior)
        """
        # Fast fail for invalid jitter (before expensive prior/likelihood calculations)
        # We have to check jitter specifically because all other params will ultimately
        # get checked/raise Exceptions when they are used to calculate an RV.
        # Jitter doesn't directly contribute to calculated RV, so needs to be checked manually.
        _all_params_for_ll = self.fixed_params | free_params_dict
        if _all_params_for_ll["jit"] < 0:
            return -np.inf

        # Evaluate priors on the free parameters. If any parameters are outside priors
        # (i.e. priors are infinite), then fail fast by returning -inf early (before expensive likelihood calc).
        # We attempt to convert free parameters (if needed) for prior evaluation
        # This is for if the user is fitting in transformed parameterisation,
        # but defining their priors in the default parameterisation
        try:
            params_for_prior = self._convert_params_for_prior_evaluation(free_params_dict)
            lp = self.log_prior(params_for_prior)
        except ValueError:
            # Invalid parameter conversion (e.g., unphysical eccentricity)
            return -np.inf
        if not np.isfinite(lp):
            return -np.inf

        # Calculate log-likelihood with all parameters
        ll = self.log_likelihood(_all_params_for_ll)

        # Return combined log-posterior (log-likelihood + log-prior)
        logprob = ll + lp
        return logprob

    def _negative_log_probability_for_MAP(self, free_params_vals: list[float]) -> float:
        """For MAP: run __call__ only passing in a list, not dict, of params.

        Because scipy.optimize.minimise only takes list of values, not a dict,
        we need to assign the values back to their corresponding keys, and pass
        that to __call__().

        This does not check that the values are in the correct order, it is
        assumed. As we're dealing with dicts, this hopefully is the case.

        Parameters
        ----------
        free_params_vals : list
            float values of the free parameters
        """
        # Create dicts from the names and values
        # (Assumes the order of names matches the order of values)
        free_params_dict = dict(zip(self.free_params_names, free_params_vals))

        # Calculate *negative* log_probability (MAP is backwards from MCMC)
        logprob = self.log_probability(free_params_dict)
        neg_logprob = -logprob

        # Handle -inf log_probability to prevent scipy RuntimeWarnings during optimisation
        # scipy's optimizer can't handle -inf values in arithmetic operations
        # (This does mean there is a non-zero chance we could end up returning a solution that doesn't satisfy the prior functions)
        if not np.isfinite(neg_logprob):
            return 1e30  # Very large finite number instead of +inf

        return neg_logprob


class LogLikelihood:
    """Log likelihood calculation for radial velocity data.

    Calculates log likelihood given RV model parameters and data.
    """

    def __init__(
        self,
        time: np.ndarray,
        vel: np.ndarray,
        verr: np.ndarray,
        t0: float,
        planet_letters: list[str],
        parameterisation: Parameterisation,
    ) -> None:
        """Initialize the LogLikelihood object.

        Parameters
        ----------
        time : np.ndarray
            Time of each observation [days].
        vel : np.ndarray
            Radial velocity at each time [m/s].
        verr : np.ndarray
            Uncertainty on the radial velocity at each time [m/s].
        t0 : float
            Reference time for the trend [days].
        planet_letters : list[str]
            List of single-character planet identifiers.
        parameterisation : Parameterisation
            The orbital parameterisation to use.
        """
        self.time = time
        self.vel = vel
        self.verr = verr
        self.t0 = t0

        self.planet_letters = planet_letters
        self.parameterisation = parameterisation

    def __call__(self, params: Dict[str, float]) -> float:
        """Calculate log likelihood for given parameters.

        Parameters
        ----------
        params : Dict[str, float]
            Dictionary of all parameter values

        Returns
        -------
        float
            Log likelihood value
        """
        rv_total = np.zeros(len(self.time))

        # Step 1: Calculate RV contributions from each planet
        for letter in self.planet_letters:
            # get just the parameters for this planet (and strip the _letter suffix from the keys)
            _this_planet_params = {
                par: params[f"{par}_{letter}"]
                for par in self.parameterisation.pars
            }
            try:
                _this_planet = ravest.model.Planet(letter, self.parameterisation, _this_planet_params)
                _this_planet_rv = _this_planet.radial_velocity(self.time)
            except ValueError:
                # Planet.__init__ validates parameters and raises ValueError for invalid params
                return -np.inf  # fail-fast: return -inf log-likelihood

            # add this planet's RV contribution to the total
            rv_total += _this_planet_rv

        # Step 2: Calculate and add the RV from the system Trend
        _trend_keys = ["g", "gd", "gdd"]
        _trend_params = {key: params[key] for key in _trend_keys}
        _this_trend = ravest.model.Trend(params=_trend_params, t0=self.t0)
        _rv_trend = _this_trend.radial_velocity(self.time)
        rv_total += _rv_trend

        # Step 3: Calculate log-likelihood including jitter term
        verr_jitter_squared = self.verr**2 + params["jit"]**2
        penalty_term = np.log(2 * np.pi * verr_jitter_squared)
        residuals = rv_total - self.vel
        chi2 = residuals**2 / verr_jitter_squared
        ll = -0.5 * np.sum(chi2 + penalty_term)
        return ll


class LogPrior:
    """Log prior probability calculation.

    Evaluates log prior probabilities for model parameters.
    """

    def __init__(self, priors: dict[str, Callable[[float], float]]) -> None:
        self.priors = priors

    def __call__(self, params: Dict[str, float]) -> float:
        """Calculate log prior probability for given parameters.

        Parameters
        ----------
        params : Dict[str, float]
            Dictionary of parameter values

        Returns
        -------
        float
            Log prior probability
        """
        log_prior_probability = 0
        for param in params:
            # go into the `self.priors dict``, get the Prior object for this `param`
            # and call it with the value of said param, to get the prior probability
            log_prior_probability += self.priors[param](params[param])

        return log_prior_probability

class GPLogPosterior:
    """Log posterior probability for GP MCMC sampling.

    Combines GP log likelihood and log priors for both parameters and hyperparameters.
    """

    def __init__(
        self,
        planet_letters: list[str],
        parameterisation: Parameterisation,
        gp_kernel: GPKernel,
        priors: dict[str, Callable[[float], float]],
        hyperpriors: dict[str, Callable[[float], float]],
        fixed_params: dict[str, float],
        fixed_hyperparams: dict[str, float],
        free_params_names: list[str],
        free_hyperparams_names: list[str],
        time: np.ndarray,
        vel: np.ndarray,
        verr: np.ndarray,
        t0: float,
    ) -> None:
        """Initialize the GPLogPosterior object.

        Parameters
        ----------
        planet_letters : list[str]
            List of single-character planet identifiers.
        parameterisation : Parameterisation
            The orbital parameterisation to use.
        gp_kernel : GPKernel
            The Gaussian Process kernel to use.
        priors : dict[str, Callable[[float], float]]
            Dictionary mapping parameter names to their prior probability functions.
        hyperpriors : dict[str, Callable[[float], float]]
            Dictionary mapping hyperparameter names to their prior probability functions.
        fixed_params : dict[str, float]
            Dictionary of fixed parameter values.
        fixed_hyperparams : dict[str, float]
            Dictionary of fixed hyperparameter values.
        free_params_names : list[str]
            List of free parameter names to sample.
        free_hyperparams_names : list[str]
            List of free hyperparameter names to sample.
        time : np.ndarray
            Time of each observation [days].
        vel : np.ndarray
            Radial velocity at each time [m/s].
        verr : np.ndarray
            Uncertainty on the radial velocity at each time [m/s].
        t0 : float
            Reference time for the trend [days].
        """
        self.planet_letters = planet_letters
        self.parameterisation = parameterisation
        self.gp_kernel = gp_kernel
        self.priors = priors
        self.hyperpriors = hyperpriors
        self.fixed_params = fixed_params
        self.fixed_hyperparams = fixed_hyperparams
        self.free_params_names = free_params_names
        self.free_hyperparams_names = free_hyperparams_names
        self.time = time
        self.vel = vel
        self.verr = verr
        self.t0 = t0

        # Create GP log-likelihood and GP log-prior objects for later
        self.gp_log_likelihood = GPLogLikelihood(
            time=self.time,
            vel=self.vel,
            verr=self.verr,
            t0=self.t0,
            planet_letters=self.planet_letters,
            parameterisation=self.parameterisation,
            gp_kernel=self.gp_kernel,
        )

        # Create LogPrior objects for parameters and hyperparameters
        self.log_prior = LogPrior(self.priors)
        self.log_hyperprior = LogPrior(self.hyperpriors)

    def _convert_params_for_prior_evaluation(self, free_params_dict: dict[str, float]) -> Dict[str, float]:
        """Convert free parameters for prior evaluation if needed.

        Parameters
        ----------
        free_params_dict : dict
            Free parameters in current parameterisation

        Returns
        -------
        dict
            Parameters with names/values converted for prior evaluation
        """
        # Three cases:
        # Case 1: User is fitting in transformed parameterisation, but priors are in same transformed parameterisation
        # Case 2: User is fitting in default parameterisation, and priors are also in default parameterisation
        # Case 3: User is fitting in transformed parameterisation, but priors are in default parameterisation

        # Simple detection: do prior keys match our current free parameter names?
        prior_keys = set(self.priors.keys())
        free_param_keys = set(self.free_params_names)
        if prior_keys == free_param_keys:
            # No conversion needed (Cases 1 & 2)
            return free_params_dict
        else:
            # Conversion needed (Case 3) - convert to default parameterisation equivalents
            # Start with just the non-planetary parameters that match
            params_for_prior = {key: value for key, value in free_params_dict.items()
                              if key in prior_keys}

            all_params = self.fixed_params | free_params_dict

            # Convert each planet's parameters
            for planet_letter in self.planet_letters:
                # Get current planet parameters
                planet_params = {par: all_params[f"{par}_{planet_letter}"]
                               for par in self.parameterisation.pars}

                # Convert to default parameterisation
                default_params = self.parameterisation.convert_pars_to_default_parameterisation(planet_params)

                # Add the converted parameter values for priors that need them
                for default_par, value in default_params.items():
                    default_param_key = f"{default_par}_{planet_letter}"
                    if default_param_key in prior_keys:  # Only add if we have a prior for it
                        params_for_prior[default_param_key] = value

            return params_for_prior

    def log_probability(self, combined_params_hyperparams: Dict[str, float]) -> float:
        """Calculate log posterior probability for given free parameters and hyperparameters.

        Parameters
        ----------
        combined_params_hyperparams : Dict[str, float]
            Combined dictionary of free parameters and hyperparameters

        Returns
        -------
        float
            Log posterior probability (log likelihood + log prior + log hyperprior)
        """
        # Split the combined dictionary into parameters and hyperparameters
        free_params_dict = {name: combined_params_hyperparams[name] for name in self.free_params_names}
        free_hyperparams_dict = {name: combined_params_hyperparams[name] for name in self.free_hyperparams_names}
        # Fast fail for invalid jitter (before expensive prior/likelihood calculations)
        # We have to check jitter specifically because all other params will ultimately
        # get checked/raise Exceptions when they are used to calculate an RV.
        # Jitter doesn't directly contribute to calculated RV, so needs to be checked manually.
        _all_params_for_ll = self.fixed_params | free_params_dict
        if _all_params_for_ll["jit"] < 0:
            return -np.inf

        # Fast fail for invalid GP hyperparameters
        # This is a check for unphysical values, not for if they are within the hyperpriors or not
        try:
            all_hyperparams_values = self.fixed_hyperparams | free_hyperparams_dict
            self.gp_kernel._validate_hyperparams_values(all_hyperparams_values)
        except ValueError:
            return -np.inf

        # Evaluate priors on the free parameters. If any parameters are outside priors
        # (i.e. priors are infinite), then fail fast by returning -inf early (before expensive likelihood calc).
        # We attempt to convert free parameters (if needed) for prior evaluation
        # This is for if the user is fitting in transformed parameterisation,
        # but defining their priors in the default parameterisation
        try:
            params_for_prior = self._convert_params_for_prior_evaluation(free_params_dict)
            lp = self.log_prior(params_for_prior)
        except ValueError:
            # Invalid parameter conversion (e.g., unphysical eccentricity)
            return -np.inf
        if not np.isfinite(lp):
            return -np.inf

        # Evaluate hyperpriors on the free hyperparameters - fail fast if any hyperparameters are outside priors
        lhp = self.log_hyperprior(free_hyperparams_dict)
        if not np.isfinite(lhp):
            return -np.inf

        # Calculate GP log-likelihood with all parameters and hyperparameters
        all_params = self.fixed_params | free_params_dict
        all_hyperparams = self.fixed_hyperparams | free_hyperparams_dict
        ll = self.gp_log_likelihood(params=all_params, hyperparams=all_hyperparams)

        # Return combined log-posterior (log-likelihood + log-prior + log-hyperprior)
        logprob = ll + lp + lhp
        return logprob

    def _negative_log_probability_for_MAP(self, combined_free_params_hyperparams_vals: list[float]) -> float:
        """For MAP: run __call__ only passing in a list, not dict, of params.

        Because scipy.optimize.minimise only takes list of values, not a dict,
        we need to assign the values back to their corresponding keys, and pass
        that to __call__().

        This does not check that the values are in the correct order, it is
        assumed. As we're dealing with dicts, this hopefully is the case.

        Parameters
        ----------
        combined_free_params_hyperparams_vals : list
            Combined list of free parameter and free hyperparameter values
        """
        # Split the list back into params values and hyperparams values
        n_params = len(self.free_params_names)
        params_values = combined_free_params_hyperparams_vals[:n_params]
        hyperparams_values = combined_free_params_hyperparams_vals[n_params:]

        # Create combined dict from the names and values
        # (Assumes the order of names matches the order of values)
        params_dict = dict(zip(self.free_params_names, params_values))
        hyperparams_dict = dict(zip(self.free_hyperparams_names, hyperparams_values))
        combined_dict = params_dict | hyperparams_dict

        # Calculate *negative* log_probability (MAP is backwards from MCMC)
        logprob = self.log_probability(combined_dict)
        neg_logprob = -logprob

        # Handle -inf log_probability to prevent scipy RuntimeWarnings during optimisation
        # scipy's optimizer can't handle -inf values in arithmetic operations
        # (This does mean there is a non-zero chance we could end up returning a solution that doesn't satisfy the prior functions)
        if not np.isfinite(neg_logprob):
            return 1e30  # Very large finite number instead of +inf

        return neg_logprob
class GPLogLikelihood:
    """GP version of Log likelihood calculation for radial velocity data.

    Calculates log likelihood given RV model parameters and data, and GP hyperparameters.
    """

    def __init__(
        self,
        time: np.ndarray,
        vel: np.ndarray,
        verr: np.ndarray,
        t0: float,
        planet_letters: list[str],
        parameterisation: Parameterisation,
        gp_kernel: GPKernel,
    ) -> None:
        self.time = time
        self.vel = vel
        self.verr = verr
        self.t0 = t0
        self.planet_letters = planet_letters
        self.parameterisation = parameterisation
        self.gp_kernel = gp_kernel

        # Convert data to JAX array for tinygp
        self.jax_time = jnp.array(self.time)
        self.jax_vel = jnp.array(self.vel)
        self.jax_verr = jnp.array(self.verr)

    def _calculate_mean_model(self, params: Dict[str, float]) -> jnp.ndarray:
        """Calculate the Keplerian RV model (the mean function for the GP).

        Takes planetary parameters and trend parameters.

        Parameters
        ----------
        params : Dict[str, float]
            Dictionary of all parameter values

        Returns
        -------
        jnp.ndarray
            Mean model RV values at observation times
        """
        rv_total = jnp.zeros(len(self.time))

        # Step 1: Calculate RV contributions from each planet
        for letter in self.planet_letters:
            # get just the parameters for this planet (and strip the _letter suffix from the keys)
            _this_planet_params = {
                par: params[f"{par}_{letter}"]
                for par in self.parameterisation.pars
            }

            try:
                _this_planet = ravest.model.Planet(letter, self.parameterisation, _this_planet_params)
                _this_planet_rv = _this_planet.radial_velocity(self.time)
            except ValueError:
                # Planet.__init__ validates parameters and raises ValueError for invalid params
                return -np.inf  # fail-fast: return -inf log-likelihood

            # add this planet's RV contribution to the total
            rv_total += _this_planet_rv

        # Step 2: Calculate and add the RV from the system Trend
        _trend_keys = ["g", "gd", "gdd"]
        _trend_params = {key: params[key] for key in _trend_keys}
        _this_trend = ravest.model.Trend(params=_trend_params, t0=self.t0)
        _rv_trend = _this_trend.radial_velocity(self.time)
        rv_total += jnp.array(_rv_trend)

        return rv_total

    @staticmethod
    @jax.jit
    def _compute_gp_log_likelihood(
        kernel: kernels.Kernel,
        time_array: jnp.ndarray,
        vel_array: jnp.ndarray,
        verr_squared_array: jnp.ndarray,
        mean_model: jnp.ndarray
    ) -> float:
        """JIT-compiled GP log likelihood computation.

        This is the expensive numerical part that benefits from JIT compilation.
        """
        gp = GaussianProcess(kernel=kernel, X=time_array, diag=verr_squared_array)
        residuals = vel_array - mean_model
        return gp.log_probability(y=residuals)

    def __call__(self, params: Dict[str, float], hyperparams: Dict[str, float]) -> float:
        """Calculate GP log likelihood for given parameters and hyperparameters.

        Parameters
        ----------
        params : Dict[str, float]
            Dictionary of all parameter values
        hyperparams : Dict[str, float]
            Dictionary of all hyperparameter values

        Returns
        -------
        float
            Log likelihood value
        """
        # Calculate mean model (RV signal from planets + system trend)
        mean_model = self._calculate_mean_model(params)

        # Check if mean model calculation failed
        # (no point doing expensive GP calculation if we don't need to)
        if not jnp.isfinite(mean_model).all():
            return -np.inf

        # Build GP kernel with hyperparameters
        kernel = self.gp_kernel.build_kernel(hyperparams)

        # Add jitter to observational uncertainties
        jit_value = params["jit"]
        jit2_verr2 = self.jax_verr**2 + jit_value**2
        # N.B. we don't sqrt here - tinygp diag wants variance, not stddev

        # Use JIT-compiled helper for the expensive GP computation
        return self._compute_gp_log_likelihood(
            kernel=kernel,
            time_array=self.jax_time,
            vel_array=self.jax_vel,
            verr_squared_array=jit2_verr2,
            mean_model=mean_model
        )
