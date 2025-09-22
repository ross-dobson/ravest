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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from scipy.optimize import minimize

import ravest.model
from ravest.param import Parameter, Parameterisation

logging.basicConfig(level=logging.INFO)

def calculate_aic(log_likelihood: float, num_params: int) -> float:
    """Calculate Akaike Information Criterion (AIC).

    Parameters
    ----------
    log_likelihood : float
        The log-likelihood of the model
    num_params : int
        Number of free parameters in the model

    Returns
    -------
    float
        AIC = 2*k - 2*ln(L), where k is the number of parameters and L is the likelihood
    """
    return 2 * num_params - 2 * log_likelihood


def calculate_bic(log_likelihood: float, num_params: int, num_observations: int) -> float:
    """Calculate Bayesian Information Criterion (BIC).

    Parameters
    ----------
    log_likelihood : float
        The log-likelihood of the model
    num_params : int
        Number of free parameters in the model
    num_observations : int
        Number of data points used to fit the model

    Returns
    -------
    float
        BIC = k*ln(n) - 2*ln(L), where n is the number of observations,
        k is the number of parameters, and L is the likelihood
    """
    return num_params * np.log(num_observations) - 2 * log_likelihood


class Fitter:
    """Main class for fitting radial velocity data to planetary models.

    Supports MCMC sampling, MAP estimation, and various parameterisations.
    Handles multiple planets, trends, and jitter parameters.
    """

    def __init__(self, planet_letters: list[str], parameterisation: Parameterisation) -> None:
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
        # TODO rename this function?
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
                # TODO: this is a bit inefficient as we loop through all free params again
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
        self.ndim = len(self.free_params_values)  # TODO: would this ever change (here)? I think this may only change if user changes self.params, not self.priors

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
        # TODO: where and why is this used, rather than fixed_params_dict?
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

    def generate_initial_walker_positions(self, nwalkers: int, verbose: bool = False, max_attempts: int = 1000) -> np.ndarray:
        """Generate initial walker positions that satisfy priors and are astrophysically valid.

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
        >>> initial_positions = fitter.generate_initial_walker_positions(nwalkers)
        >>> fitter.run_mcmc(initial_positions, nwalkers, nsteps=2000)
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
                            walker_position.append(np.random.normal(loc=prior.mean, scale=prior.std))

                        elif isinstance(prior, ravest.prior.HalfNormal):
                            walker_position.append(np.abs(np.random.normal(loc=0, scale=prior.std)))

                        elif isinstance(prior, ravest.prior.Uniform):
                            walker_position.append(np.random.uniform(low=prior.lower, high=prior.upper))

                        elif isinstance(prior, ravest.prior.TruncatedNormal):
                            walker_position.append(np.random.uniform(low=prior.lower, high=prior.upper))

                        elif isinstance(prior, ravest.prior.Beta):
                            walker_position.append(np.random.uniform(low=prior.a, high=prior.b))

                        elif isinstance(prior, ravest.prior.EccentricityUniform):
                            walker_position.append(np.random.uniform(low=0, high=prior.upper))

                        else:
                            raise ValueError(f"Unsupported prior type for walker generation: {type(prior)}")

                    else:
                        # No direct prior for this parameter - use current value + small perturbation
                        center_val = self.params[param_name].value
                        # Add small random perturbation (10% of current value + small fixed amount for near-zero values)
                        perturbation = np.random.normal(0, abs(center_val) * 0.1 + 0.01)
                        walker_position.append(center_val + perturbation)

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
                    # Validation failed, try again
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

    def run_mcmc(self, initial_positions : np.ndarray, nwalkers: int, nsteps: int = 5000, progress: bool = True, multiprocessing: bool = False) -> None:
        """Run MCMC sampling from given initial walker positions.

        Parameters
        ----------
        initial_positions  : np.ndarray
            Starting positions for all MCMC walkers. Shape must be (nwalkers, ndim)
            where ndim is the number of free parameters. Each row represents the
            starting position for one walker in the order of free_params_names.
        nwalkers : int
            Number of MCMC walkers (must match first dimension of initial_positions )
        nsteps : int, optional
            Number of MCMC steps to run (default: 5000)
        progress : bool, optional
            Whether to show progress bar during MCMC (default: True)
        multiprocessing : bool, optional
            Whether to use multiprocessing for MCMC (default: False)
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
            # Use LogPosterior's parameter conversion for consistency
            params_for_prior = lp._convert_params_for_prior_evaluation(walker_params_dict)
            log_prior = lp.log_prior(params_for_prior)
            if not np.isfinite(log_prior):
                raise ValueError(f"Walker {i} is outside prior bounds (log_prior = {log_prior})")

        # Store number of walkers
        self.nwalkers = nwalkers

        # Use provided walker positions directly
        mcmc_init = initial_positions

        # TODO: parameter_names argument does slightly impact performance - but not sure if it can be avoided, we do need the names
        # and I'm not sure constructing the dictionary later ourselves manually is any quicker than passing parameter_names argument

        if multiprocessing:
            logging.info("Starting MCMC (with multiprocessing)...")
            with mp.get_context("spawn").Pool() as pool:  # Use 'spawn' instead of 'fork' to avoid issues on some Linux platforms
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability,
                                                parameter_names=self.free_params_names,
                                                pool=pool)
                sampler.run_mcmc(initial_state=mcmc_init, nsteps=nsteps, progress=True)
        else:
            logging.info("Starting MCMC...")
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability,
                                                parameter_names=self.free_params_names)
            sampler.run_mcmc(initial_state=mcmc_init, nsteps=nsteps, progress=progress)

        self.sampler = sampler
        logging.info("...MCMC done.")

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

    def get_posterior_params_dict(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1) -> dict:
        """Return dict combining fixed parameter values, and MCMC samples for the free ones.

        This method creates a unified dictionary containing all model parameters:
        fixed parameters as single float values, and free parameters as arrays
        of MCMC samples. This format is ideal for functions like calculate_mpsini
        that need all parameters (whether free or fixed), and that should propagate uncertainties from
        the free parameters samples.

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


    def plot_chains(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, save: bool = False, fname: str = "chains_plot.png", dpi: int = 100) -> None:
        """Plot MCMC chains for all free parameters."""
        # Scale figure height to maintain consistent subplot size
        subplot_height_inches = 1.25
        fig, axes = plt.subplots(self.ndim, figsize=(10, self.ndim * subplot_height_inches),
                                sharex=True, constrained_layout=True)
        fig.suptitle("Chains plot")

        if self.ndim == 1:
            axes = [axes]

        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=False)
        for i in range(self.ndim):
            ax = axes[i]
            to_plot = samples[:, :, i]

            ax.plot(to_plot, "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.free_params_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("Step number")
        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def plot_lnprob(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, save: bool = False, fname: str = "lnprob_plot.png", dpi: int = 100) -> None:
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
        save : bool, optional
            Save the plot to path `fname` (default: False)
        fname : str, optional
            The path to save the plot to (default: "lnprob_plot.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)
        """
        fig, ax = plt.subplots(1, figsize=(10, 6))
        fig.suptitle("Log Probability Traces")

        lnprobs = self.get_sampler_lnprob(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=False)

        nsteps, nwalkers = lnprobs.shape
        for i in range(nwalkers):
            to_plot = lnprobs[:, i]
            ax.plot(to_plot, "k", alpha=0.3)

        ax.set_xlim(0, nsteps)
        ax.set_xlabel("Step number")
        ax.set_ylabel("Log probability")

        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def plot_corner(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, save: bool = False, fname: str = "corner_plot.png", dpi: int = 100) -> None:
        """Create a corner plot of MCMC samples.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
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
        plot_datapoints=False, quantiles=[0.16, 0.5, 0.84],
        )
        fig.suptitle("Corner plots")
        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def _posterior_rv(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each step in the MCMC chain, calculate the RV.

        The RVs are calculated at the times in `tlin`, which is a smooth time
        array generated for plotting purposes. This array consists of 1000
        uniformly spaced points spanning from the minimum to maximum of the
        observed `time` array, with a 1% buffer added to both ends.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain. (default: 1)

        Returns
        -------
        np.ndarray
            Array of RVs for each sample in the chain, at the times in `tlin`.
        np.ndarray
            The time array `tlin` at which the RVs are calculated.
        """
        # Get posterior parameter samples
        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)

        # Create smooth time curve for plotting
        _tmin, _tmax = self.time.min(), self.time.max()
        _trange = _tmax - _tmin
        tlin = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # store the rv for each sample here
        rv_array = np.zeros((len(samples), len(tlin)))

        # get the free parameter names and fixed parameter values
        # we don't need to call this repeatedly for each sample
        free_params_names = self.free_params_names
        fixed_params_dict = self.fixed_params_values_dict

        # TODO: can we replace this loop with something faster?
        for i, row in enumerate(samples):
            # Combine fixed and free parameters, so we have all values needed to evaluate the model
            free_params = dict(zip(free_params_names, row))
            params = fixed_params_dict | free_params
            this_row_rv = np.zeros(len(tlin))

            for letter in self.planet_letters:
                this_planet_params = {}
                for par in self.parameterisation.pars:
                    key = f"{par}_{letter}"
                    this_planet_params[par] = params[key]
                this_planet = ravest.model.Planet(letter, self.parameterisation, this_planet_params)
                this_planet_rv = this_planet.radial_velocity(tlin)
                this_row_rv += this_planet_rv
            # now we're outside the planet loop, do the trend
            this_trend = ravest.model.Trend(params={"g": params["g"], "gd": params["gd"], "gdd": params["gdd"]}, t0=self.t0)
            this_trend_rv = this_trend.radial_velocity(tlin)
            this_row_rv += this_trend_rv
            rv_array[i, :] = this_row_rv

        return rv_array, tlin

    def _plot_rv(self, params: Dict[str, float], title: str = "RV Model", save: bool = False, fname: str = "rv_plot.png", dpi: int = 100) -> None:
        """Helper function to plot RV model with given parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameter values (both free and fixed)
        title : str, optional
            Plot title (default: "RV Model")
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
        tlin = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # Calculate RV model for all planets and trend
        rv_total = np.zeros(len(tlin))

        # Add planetary contributions
        for letter in self.planet_letters:
            planet_params = {}
            for par_name in self.parameterisation.pars:
                key = f"{par_name}_{letter}"
                planet_params[par_name] = params[key]

            planet = ravest.model.Planet(letter, self.parameterisation, planet_params)
            rv_total += planet.radial_velocity(tlin)

        # Add trend contribution
        trend_params = {key: params[key] for key in ["g", "gd", "gdd"]}
        trend = ravest.model.Trend(params=trend_params, t0=self.t0)
        rv_total += trend.radial_velocity(tlin)

        # Get jitter value for error bars
        jit_value = params["jit"]
        verr_with_jit = np.sqrt(self.verr**2 + jit_value**2)

        # Calculate model at observed times for residuals
        rv_model_at_data = np.zeros(len(self.time))

        # Add planetary contributions at data times
        for letter in self.planet_letters:
            planet_params = {}
            for par_name in self.parameterisation.pars:
                key = f"{par_name}_{letter}"
                planet_params[par_name] = params[key]

            planet = ravest.model.Planet(letter, self.parameterisation, planet_params)
            rv_model_at_data += planet.radial_velocity(self.time)

        # Add trend contribution at data times
        trend_params = {key: params[key] for key in ["g", "gd", "gdd"]}
        trend = ravest.model.Trend(params=trend_params, t0=self.t0)
        rv_model_at_data += trend.radial_velocity(self.time)

        # Calculate residuals
        residuals = self.vel - rv_model_at_data

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5),
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})

        # Main RV plot
        ax1.errorbar(self.time, self.vel, yerr=self.verr, marker=".", color="tab:blue",
                    ecolor="tab:blue", linestyle="None", markersize=8, zorder=4, label="Data")
        ax1.errorbar(self.time, self.vel, yerr=verr_with_jit, marker="None",
                    ecolor="tab:blue", linestyle="None", alpha=0.5, zorder=3, label="Jitter")

        ax1.plot(tlin, rv_total, label="Model", color="black", zorder=2)
        ax1.set_xlim(tlin[0], tlin[-1])
        ax1.set_ylabel("Radial velocity [m/s]")
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
        ax2.set_xlim(tlin[0], tlin[-1])

        # Set symmetric y-limits for residuals
        max_abs_residual = np.max(np.abs(residuals + verr_with_jit))
        ax2.set_ylim(-max_abs_residual * 1.1, max_abs_residual * 1.1)

        ax2.set_xlabel("Time [days]")
        ax2.set_ylabel("Residuals [m/s]")
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

    def _plot_phase(self, planet_letter: str, params: Dict[str, float], title: str = None, save: bool = False, fname: str = "phase_plot.png", dpi: int = 100) -> None:
        """Helper function to plot phase-folded RV model for a single planet with given parameters.

        Parameters
        ----------
        planet_letter : str
            Letter identifying the planet to plot (e.g., 'b', 'c', 'd')
        params : dict
            Dictionary of parameter values (both free and fixed)
        title : str, optional
            Plot title (default: f"Planet {planet_letter} Phase Plot")
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
        tlin = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # Get jitter value for error bars
        jit_value = params["jit"]
        verr_with_jit = np.sqrt(self.verr**2 + jit_value**2)

        # Get period and time of conjunction for this planet
        p = params[f"P_{planet_letter}"]

        # Convert to tc if needed
        if "Tc" in self.parameterisation.pars:
            tc = params[f"Tc_{planet_letter}"]
        elif "e" in self.parameterisation.pars and "w" in self.parameterisation.pars:
            _e = params[f"e_{planet_letter}"]
            _w = params[f"w_{planet_letter}"]
            _tp = params[f"Tp_{planet_letter}"]
            tc = self.parameterisation.convert_tp_to_tc(_tp, p, _e, _w)
        else:
            # Fall back to default parameterisation conversion
            planet_params = {par: params[f"{par}_{planet_letter}"] for par in self.parameterisation.pars}
            default_params = self.parameterisation.convert_pars_to_default_parameterisation(planet_params)
            tc = self.parameterisation.convert_tp_to_tc(default_params["Tp"], p, default_params["e"], default_params["w"])

        # Calculate phase-folded time arrays (in units of orbital phase)
        t_fold = ((self.time - tc + 0.5*p) % p - 0.5*p) / p
        tlin_fold = ((tlin - tc + 0.5*p) % p - 0.5*p) / p

        # Sort the tlin_fold array for proper plotting
        lin_inds = np.argsort(tlin_fold)
        tlin_fold_sorted = tlin_fold[lin_inds]

        # Calculate RV contribution from this planet only
        planet_params = {par: params[f"{par}_{planet_letter}"] for par in self.parameterisation.pars}
        planet = ravest.model.Planet(planet_letter, self.parameterisation, planet_params)
        planet_rv_tlin = planet.radial_velocity(tlin)
        planet_rv_data = planet.radial_velocity(self.time)
        planet_rv_sorted = planet_rv_tlin[lin_inds]

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

        # Calculate residuals (data - model for this planet)
        residuals = data_minus_others - planet_rv_data

        # Create figure with subplots (main plot + residuals)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5),
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})

        # Main phase plot
        ax1.errorbar(t_fold, data_minus_others, yerr=self.verr, marker=".",
                    linestyle="None", color="tab:blue", markersize=8, zorder=4, label="Data")
        ax1.errorbar(t_fold, data_minus_others, yerr=verr_with_jit, marker="None",
                    linestyle="None", color="tab:blue", alpha=0.5, zorder=3, label="Jitter")

        # Plot phase-folded model for this planet
        ax1.plot(tlin_fold_sorted, planet_rv_sorted, label="Model", color="black", zorder=2)
        ax1.set_xlim(-0.5, 0.5)
        ax1.set_ylabel("Radial velocity [m/s]")
        ax1.legend(loc="upper right")
        ax1.set_title(title)
        ax1.tick_params(axis='x', labelbottom=False, bottom=True, top=False, direction='in')  # Remove x-axis labels from top plot
        ax1.tick_params(axis='y', direction='in')

        # Set y-axis ticks automatically based on phase data range
        ax1.yaxis.set_major_locator(AutoLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='y', which='minor', direction='in', length=3)

        # Annotate with planet info
        k_value = params[f"K_{planet_letter}"]
        s = f"Planet {planet_letter}\nP={p:.2f} d\nK={k_value:.2f} m/s"
        ax1.annotate(s, xy=(0, 1), xycoords="axes fraction",
                    xytext=(+0.5, -0.5), textcoords="offset fontsize", va="top")

        # Residuals plot (phase-folded)
        ax2.errorbar(t_fold, residuals, yerr=self.verr, marker=".",
                    linestyle="None", color="tab:blue", markersize=8, zorder=4)
        ax2.errorbar(t_fold, residuals, yerr=verr_with_jit, marker="None",
                    linestyle="None", color="tab:blue", alpha=0.5, zorder=3)
        ax2.axhline(0, color="k", linestyle="--", zorder=2)
        ax2.set_xlim(-0.5, 0.5)

        # Set symmetric y-limits for residuals
        max_abs_residual = np.max(np.abs(residuals + verr_with_jit))
        ax2.set_ylim(-max_abs_residual * 1.1, max_abs_residual * 1.1)

        ax2.set_xlabel("Orbital phase")
        ax2.set_ylabel("Residuals [m/s]")
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

    def plot_posterior_rv(self, discard_start: int = 0, discard_end: int = 0, thin: int = 1, save: bool = False, fname: str = "posterior_rv.png", dpi: int = 100) -> None:
        """Plot the posterior RV model using median parameter values.

        Uses the median values of each parameter from the MCMC chain to calculate
        and plot a single RV model curve, along with residuals showing the
        difference between the data and the model.

        Parameters
        ----------
        discard_start : int, optional
            Discard the first `discard_start` steps from the start of the chain (default: 0)
        discard_end : int, optional
            Discard the last `discard_end` steps from the end of the chain (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        save : bool, optional
            Save the plot to path `fname` (default: False)
        fname : str, optional
            The path to save the plot to (default: "posterior_rv.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)

        Returns
        -------
        tuple
            Time array and RV values from the helper function
        """
        # Get median parameter values from MCMC chain
        samples_dict = self.get_samples_dict(discard_start=discard_start, discard_end=discard_end, thin=thin)
        median_params = {name: np.median(samples) for name, samples in samples_dict.items()}

        # Combine with fixed parameters
        all_params = self.fixed_params_values_dict | median_params

        # Use helper function to create the plot
        self._plot_rv(all_params, title="Posterior RV", save=save, fname=fname, dpi=dpi)

    def _posterior_rv_planet(self, planet_letter: str, times: np.ndarray, discard_start: int = 0, discard_end: int = 0, thin: int = 1) -> np.ndarray:
        """Calculate the posterior rv for a planet, using all samples in the chain."""
        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        this_planet_rvs = np.zeros((len(samples), len(times))) # type: ignore
        fixed_params_dict = self.fixed_params_values_dict

        for i, row in enumerate(samples):  # type: ignore
            # Combine fixed and free parameters
            free_params = dict(zip(self.free_params_names, row))
            params = fixed_params_dict | free_params

            # get this planet's params
            this_planet_params = {}
            for par in self.parameterisation.pars:
                key = f"{par}_{planet_letter}"
                this_planet_params[par] = params[key]
            # calculate this planet's RV for each entry in the chain
            this_planet = ravest.model.Planet(planet_letter, self.parameterisation, this_planet_params)
            this_planet_rv = this_planet.radial_velocity(times)
            this_planet_rvs[i, :] = this_planet_rv

        return this_planet_rvs

    def _posterior_rv_trend(self, times: np.ndarray, discard_start: int = 0, discard_end: int = 0, thin: int = 1) -> np.ndarray:
        """Calculate the posterior rv for the trend, using all samples in the chain."""
        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        this_trend_rvs = np.zeros((len(samples), len(times))) # type: ignore
        fixed_params_dict = self.fixed_params_values_dict

        # for each sample in the chain, calculate the RV for the trend
        for i, row in enumerate(samples):  # type: ignore
            # Combine fixed and free parameters
            free_params = dict(zip(self.free_params_names, row))
            params = fixed_params_dict | free_params

            this_trend = ravest.model.Trend(params={"g": params["g"], "gd": params["gd"], "gdd": params["gdd"]}, t0=self.t0)
            this_trend_rv = this_trend.radial_velocity(times)
            this_trend_rvs[i, :] = this_trend_rv

        return this_trend_rvs

    def plot_posterior_phase(self, planet_letter: str, discard_start: int = 0, discard_end: int = 0, thin: int = 1, save: bool = False, fname: str = "posterior_phase.png", dpi: int = 100) -> None:
        """Plot the posterior phase-folded RV model for a single planet using median parameter values.

        Uses the median values of each parameter from the MCMC chain to calculate
        and plot a single RV model curve for the specified planet, phase-folded around
        that planet's period and time of conjunction. Includes residuals panel.

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
        save : bool, optional
            Save the plot to path `fname` (default: False)
        fname : str, optional
            The path to save the plot to (default: "posterior_phase.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)
        """
        # Get median parameter values from MCMC chain
        samples_dict = self.get_samples_dict(discard_start=discard_start, discard_end=discard_end, thin=thin)
        median_params = {name: np.median(samples) for name, samples in samples_dict.items()}

        # Combine with fixed parameters
        all_params = self.fixed_params_values_dict | median_params

        # Use helper function to create the plot
        self._plot_phase(planet_letter, all_params, title=f"Posterior Phase Plot - Planet {planet_letter}",
                        save=save, fname=fname, dpi=dpi)

    def plot_MAP_rv(self, map_result: scipy.optimize.OptimizeResult, save: bool = False, fname: str = "MAP_rv.png", dpi: int = 100) -> None:
        """Plot radial velocity data and model using MAP parameter estimates.

        Parameters
        ----------
        map_result : scipy.optimize.OptimizeResult
            Result from find_map_estimate() containing the MAP parameters
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
        self._plot_rv(all_params, title="MAP RV", save=save, fname=fname, dpi=dpi)

    def plot_MAP_phase(self, planet_letter: str, map_result: scipy.optimize.OptimizeResult, save: bool = False, fname: str = "MAP_phase.png", dpi: int = 100) -> None:
        """Plot phase-folded radial velocity data and model using MAP parameter estimates.

        Parameters
        ----------
        planet_letter : str
            Letter identifying the planet to plot (e.g., 'b', 'c', 'd')
        map_result : scipy.optimize.OptimizeResult
            Result from find_map_estimate() containing the MAP parameters
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

        # Use helper function to create the plot
        self._plot_phase(planet_letter, all_params, title=f"MAP Phase Plot - Planet {planet_letter}",
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
