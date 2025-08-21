# fit.py
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import ravest.model
from ravest.param import Parameterisation


class Fitter:

    def __init__(self, planet_letters: list[str], parameterisation: Parameterisation):
        self.planet_letters = planet_letters
        self.parameterisation = parameterisation
        self._params = {}
        self._priors = {}

    def add_data(self, time, vel, verr, t0):
        """Add the data to the Fitter object

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
    def params(self):
        """Parameters dictionary. Set via: fitter.params = param_dict"""
        return self._params

    @params.setter
    def params(self, new_params):
        """Set parameters with a dict, checking all required params are present.

        You can update all or some of the parameters at once, example:
        >>> fitter.params = {"g": 1.0, "gd": 0.1}  # only update trend parameters
        >>> fitter.params = {"per_c" : 5.0, "k_c": 3.5}  # only update some of planet C parameters

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
    def priors(self):
        """Priors dictionary. Set via: fitter.priors = prior_dict"""
        return self._priors

    @priors.setter
    def priors(self, new_priors):
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
        >>> fitter.priors = {"k_b": Uniform(0, 100), "per_b": Uniform(1, 30)}

        Raises
        ------
        ValueError
            If any required priors are missing, unexpected priors are provided,
            or initial parameter values are outside prior bounds.
        """
        self._set_priors_with_validation(new_priors)

    def _validate_complete_params(self, params: dict):
        """Validate that params dict contains exactly the required parameters."""
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

    def _set_priors_with_validation(self, new_priors: dict):
        """Set priors with validation. Supports partial updates."""
        # Create merged priors dict for validation
        merged_priors = dict(self._priors)
        merged_priors.update(new_priors)

        # Get expected priors (only for free parameters)
        expected_priors = set(self.free_params_names)
        provided_priors = set(merged_priors.keys())

        # Check for unexpected priors
        unexpected_priors = provided_priors - expected_priors
        if unexpected_priors:
            raise ValueError(
                f"Unexpected priors: {unexpected_priors}. "
                f"Expected {len(expected_priors)} priors, got {len(provided_priors)}"
            )

        # Check that all free parameters have priors
        missing_priors = expected_priors - provided_priors
        if missing_priors:
            raise ValueError(
                f"Missing priors for: {missing_priors}. "
                f"Expected {len(expected_priors)} priors, got {len(provided_priors)}"
            )

        # Validate that initial parameter values are within prior bounds (only for new priors)
        for par in new_priors:
            if par in self.free_params_names:  # Only validate if it's a free parameter
                prior_fn = new_priors[par]
                log_prior_prob = prior_fn(self.params[par].value)
                if not np.isfinite(log_prior_prob):
                    raise ValueError(f"Initial value {self.params[par].value} of parameter {par} is invalid for prior {new_priors[par]}.")

        # Update the priors
        self._priors.update(new_priors)
        self.ndim = len(self.free_params_values)

    @property
    def free_params_dict(self):
        """Free parameters as dict."""
        free_pars = {}
        for par in self.params:
            if self.params[par].fixed is False:
                free_pars[par] = self.params[par]
        return free_pars

    @property
    def free_params_values(self):
        """Values of free parameters as list."""
        return [param.value for param in self.free_params_dict.values()]

    @property
    def free_params_names(self):
        """Names of free parameters as list."""
        return list(self.free_params_dict.keys())

    @property
    def fixed_params_dict(self):
        """Fixed parameters as dict."""
        fixed_pars = {}
        for par in self.params:
            if self.params[par].fixed is True:
                fixed_pars[par] = self.params[par]
        return fixed_pars

    @property
    def fixed_params_values(self):
        """Values of fixed parameters as list."""
        return [param.value for param in self.fixed_params_dict.values()]

    @property
    def fixed_params_names(self):
        """Names of fixed parameters as list."""
        return list(self.fixed_params_dict.keys())

    @property
    def fixed_params_values_dict(self):
        """Fixed parameters as dict mapping names to values."""
        return dict(zip(self.fixed_params_names, self.fixed_params_values))

    def find_map_estimate(self, method="Powell"):
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
            self.fixed_params_dict,
            self.free_params_names,
            self.time,
            self.vel,
            self.verr,
            self.t0,
        )

        initial_guess = self.free_params_values

        # Perform MAP optimization
        def negative_log_posterior(*args):
            return lp._negative_log_probability_for_MAP(*args)

        map_results = minimize(negative_log_posterior, initial_guess, method=method)

        if map_results.success is False:
            print(map_results)
            raise Warning("MAP did not succeed. Check the initial values of the parameters, and the priors functions.")

        # Return results as dictionary for easy access
        map_results_dict = dict(zip(self.free_params_names, map_results.x))
        print("MAP results:", map_results_dict)

        return map_results


    def run_mcmc(self, initial_values, nwalkers, nsteps=5000, progress=True):
        """Run MCMC sampling from given initial parameter values.

        Parameters
        ----------
        initial_values : array-like
            Starting parameter values for MCMC. Should match the order of
            free parameters from get_free_params_names()
        nwalkers : int
            Number of MCMC walkers
        nsteps : int, optional
            Number of MCMC steps to run (default: 5000)
        progress : bool, optional
            Whether to show progress bar during MCMC (default: True)
        """
        # Initialize log-posterior object for MCMC sampling
        lp = LogPosterior(
            self.planet_letters,
            self.parameterisation,
            self.priors,
            self.fixed_params_dict,
            self.free_params_names,
            self.time,
            self.vel,
            self.verr,
            self.t0,
        )

        print("Starting MCMC...")
        if nwalkers < 2 * self.ndim:
            print(f"Warning: nwalkers should be at least 2 * ndim. You have {nwalkers} walkers and {self.ndim} dimensions. Setting nwalkers to {2 * self.ndim}.")
            self.nwalkers = 2 * self.ndim
        else:
            self.nwalkers = nwalkers

        mcmc_init = initial_values + 1e-5 * np.random.randn(self.nwalkers, self.ndim)
        # TODO: benchmark if parameter_names argument impacts MCMC performance
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability,
                                            parameter_names=self.free_params_names)
        sampler.run_mcmc(initial_state=mcmc_init, nsteps=nsteps, progress=progress)

        # TODO: multiprocessing disabled for now as it's causing slowdown
        # I suspect something might be being pickled that shouldn't be, but
        # this requires further investigation.
        #
        # with Pool() as pool:
        #     sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability,
        #                                     parameter_names=self.free_params_names,
        #                                     pool=pool)
        #     state = sampler.run_mcmc(mcmc_init, 10000, progress=True)
        print("...MCMC done.")
        self.sampler = sampler

    def get_samples_np(self, discard_start=0, discard_end=0, thin=1, flat=False):
        """Returns a contiguous numpy array of MCMC samples.

        Samples can be discarded from the start and/or the end of the array. You can
        also thin (take only every n-th sample), and you can flatten the array
        so that each walker's chain is merged into one chain.

        This is the foundational method for accessing MCMC samples. All the other
        sample methods build on this.

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

        # Sanity check the start and end points
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

    def get_samples_df(self, discard_start=0, discard_end=0, thin=1):
        """Returns a pandas DataFrame of flattened MCMC samples.

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

    def get_samples_dict(self, discard_start=0, discard_end=0, thin=1):
        """Returns a dict of flattened MCMC samples.

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
        >>> k_b_samples = samples_dict['k_b']  # All samples for parameter k for planet b
        """
        flat_samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        param_names = self.free_params_names

        # Direct numpy slicing - much faster than pandas operations
        return {name: flat_samples[:, i] for i, name in enumerate(param_names)}

    def get_sampler_lnprob(self, discard_start=0, discard_end=0, thin=1, flat=False):
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

        # Sanity check the start and end points
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

    def get_posterior_params_dict(self, discard_start=0, discard_end=0, thin=1):
        """Returns dict combining fixed parameters and MCMC samples.

        This method creates a unified dictionary containing all model parameters:
        fixed parameters as single float values, and free parameters as arrays
        of MCMC samples. This format is ideal for functions like calculate_mpsini
        that need all parameters, and that should propagate uncertainties from
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

    def plot_chains(self, discard_start=0, discard_end=0, thin=1, save=False, fname="chains_plot.png", dpi=100):
        fig, axes = plt.subplots(self.ndim, figsize=(10,1+(self.ndim*2/3)), sharex=True)
        # TODO: dynamically scale figure height based on number of parameters
        fig.suptitle("Chains plot")

        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=False)
        for i in range(self.ndim):
            ax = axes[i]
            to_plot = samples[:, :, i]  # type: ignore

            ax.plot(to_plot, "k", alpha=0.3)
            ax.set_xlim(0, len(samples))  # type: ignore
            ax.set_ylabel(self.free_params_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("Step number")
        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def plot_lnprob(self, discard_start=0, discard_end=0, thin=1, save=False, fname="lnprob_plot.png", dpi=100):
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

    def plot_corner(self, discard_start=0, discard_end=0, thin=1, save=False, fname="corner_plot.png", dpi=100):
        flat_samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)
        fig = corner.corner(
        flat_samples, labels=self.free_params_names, show_titles=True,
        plot_datapoints=False, quantiles=[0.16, 0.5, 0.84],
        )
        fig.suptitle("Corner plots")
        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def _posterior_rv(self, discard_start=0, discard_end=0, thin=1):
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
        """
        samples = self.get_samples_np(discard_start=discard_start, discard_end=discard_end, thin=thin, flat=True)

        # get smooth time curve for plotting
        _tmin, _tmax = self.time.min(), self.time.max()
        _trange = _tmax - _tmin
        tlin = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # store the rv for each sample here
        rv_array = np.zeros((len(samples), len(tlin)))

        # get the free parameter names and fixed parameter values
        # we don't need to call this repeatedly for each sample
        free_param_names = self.free_params_names
        fixed_params_dict = self.fixed_params_values_dict

        for i, row in enumerate(samples):  # type: ignore
            # Combine fixed and free parameters
            free_params = dict(zip(free_param_names, row))
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

    def plot_posterior_rv(self, discard_start=0, discard_end=0, thin=1, save=False, fname="posterior_rv.png", dpi=100):
        """Plot the posterior RV model (median & 16-84 percentiles).

        For each step in the MCMC chain, calculate the RV. Plot the median RV,
        and the 16th and 84th percentiles as shaded region. The RVs are
        calculated at the times in `tlin`, which is a smooth time array
        generated for plotting purposes. This array consists of 1000 uniformly
        spaced points spanning from the minimum to maximum of the observed
        `time` array, with a 1% buffer added to both ends.

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
        """
        # TODO: could combine with plot_posterior_phase() to avoid recalculating posterior RVs
        # This would improve performance for large MCMC chains

        # Get the posterior RVs, evaluated at each sample in the chains
        rv_array, tlin = self._posterior_rv(discard_start=discard_start, discard_end=discard_end, thin=thin)
        rv_percentiles = np.percentile(rv_array, [16, 50, 84], axis=0)

        # Get the new errorbars to include jit
        if "jit" in self.fixed_params_names:
            jit_median = self.fixed_params_dict["jit"].value
        else:
            jit_median = np.median(self.get_samples_df(discard_start=discard_start, discard_end=discard_end, thin=thin)["jit"])
        verr_with_jit = np.sqrt(self.verr**2 + jit_median**2)

        plt.figure(figsize=(8,3.5))
        # TODO - multi-instrument support. Need a way to match the color of the +jit errorbars to the data points
        plt.errorbar(self.time, self.vel, yerr=self.verr, marker=".", color="black", ecolor="black", linestyle="None", zorder=4)
        plt.errorbar(self.time, self.vel, yerr=verr_with_jit, marker="None", ecolor="black", linestyle="None", alpha=0.5, zorder=3)

        plt.plot(tlin, rv_percentiles[1], label="median", color="tab:blue", zorder=2)
        plt.xlim(tlin[0], tlin[-1])
        art = plt.fill_between(tlin, rv_percentiles[0], rv_percentiles[2], color="tab:blue", alpha=0.3, zorder=1)
        art.set_edgecolor("none")
        plt.xlabel("Time [days]")
        plt.ylabel("Radial velocity [m/s]")
        plt.title("Posterior RV")
        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def _posterior_rv_planet(self, planet_letter, times, discard_start=0, discard_end=0, thin=1):
        """calculate the posterior rv for a planet, using all samples in the chain"""
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

    def _posterior_rv_trend(self, times, discard_start=0, discard_end=0, thin=1):
        """calculate the posterior rv for the trend, using all samples in the chain"""
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

    def plot_posterior_phase(self, discard_start=0, discard_end=0, thin=1, save=False, fname="posterior_phase.png", dpi=100):
        """Plot the posterior RV model (median & 16-84 percentiles) in phase space.

        For each sample of parameters in the MCMC chain, calculate the RV. Fold the
        times around the median values of P and t_c for each planet. Plot the median RV
        and the 16th and 84th percentiles as shaded region. The RVs are
        calculated at the times in `tlin`, which is a smooth time array
        generated for plotting purposes. This array consists of 1000 uniformly
        spaced points spanning from the minimum to maximum of the observed
        `time` array, with a 1% buffer added to both ends.


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
            The path to save the plot to (default: "posterior_phase.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)
        """
        # TODO: could combine with plot_posterior_phase() to avoid recalculating posterior RVs
        # This would improve performance for large MCMC chains

        # get smooth linear time curve for plotting
        _tmin, _tmax = self.time.min(), self.time.max()
        _trange = _tmax - _tmin
        tlin = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # Each parameter is either fixed, or has a (flattened) chain of samples.
        # Construct a dict `params' to store whichever one we have.
        # Both work for taking a median, (the median of a fixed value is just
        # the fixed value) and the RV functions will propagate either a fixed
        # value or an array of values through the calculations.
        fixed_params_dict = self.fixed_params_values_dict
        samples_df = self.get_samples_df(discard_start=discard_start, discard_end=discard_end, thin=thin)
        samples_dict = samples_df.to_dict("list")
        params = fixed_params_dict | samples_dict

        # For each planet we need to:
        # 1) calculate the median values of p and tc
        # 2) fold the times around median p and tc
        # 2) calculate posterior rvs for each sample, at the data x times
        # 3) calculate posterior rvs for each sample, at the smooth tlin times

        # Dicts to store all of these, labelled with the planet letter
        tc_medians = {}
        p_medians = {}
        posterior_rvs = {}
        posterior_rvs_tlin = {}
        t_folds = {}
        tlin_folds = {}
        inds = {}
        lin_inds = {}

        for letter in self.planet_letters:
            ### 1) Calculate median values of p and tc
            # we (currently) don't ever reparameterise p, so it must be an entry of params dict
            p = params[f"per_{letter}"]

            # However, we might have tp, not tc, so we may need to convert it.
            # first check - does planet params contain tc? Otherwise, we need to convert tp to tc
            if "tc" in self.parameterisation.pars:
                tc = params[f"tc_{letter}"]
            # second check - do we have e and w, for converting tp to tc?
            elif ("e" in self.parameterisation.pars and "w" in self.parameterisation.pars):
                _e =  params[f"e_{letter}"]
                _w =  params[f"w_{letter}"]
                _tp = params[f"tp_{letter}"]
                tc = self.parameterisation.convert_tp_to_tc(_tp, p, _e, _w)
            # else, convert to default parameterisation, giving us e and w, then get tc
            else:
                # get the parameterisation for this planet. Combine with planet letter
                _keys = [f"{par}_{letter}" for par in self.parameterisation.pars]
                _inpars = {key: samples_df[key] for key in _keys}
                _default_parameterisation = self.parameterisation.convert_pars_to_default_parameterisation(_inpars)  # dict of converted pars
                tc = _default_parameterisation["tc"]

            # get the median of the p and tc
            p_medians[letter] = np.median(p)
            tc_medians[letter] = np.median(tc)

            ### 2) fold the times using those medians
            t_fold = (self.time - tc_medians[letter] + 0.5 * p_medians[letter]) % p_medians[letter] - 0.5 * p_medians[letter]
            tlin_fold = (tlin - tc_medians[letter] + 0.5 * p_medians[letter]) % p_medians[letter] - 0.5 * p_medians[letter]
            # rather than x axis being [-p, +p] days, we want to scale to [-0.5, 0.5] phase
            t_fold /= p_medians[letter]
            tlin_fold /= p_medians[letter]

            # store in the dicts
            t_folds[letter] = t_fold
            tlin_folds[letter] = tlin_fold
            inds[letter] = np.argsort(t_fold)
            lin_inds[letter] = np.argsort(tlin_fold)

            # Step 3: Calculate posterior RV matrix for current planet at data times
            # Generates matrix with shape (times, samples) for this planet's contribution
            posterior_rvs[letter] = self._posterior_rv_planet(letter, times=self.time, discard_start=discard_start, discard_end=discard_end, thin=thin)

            # Step 4: Calculate posterior RV matrix for current planet at smooth times
            # Used for plotting smooth model curve with uncertainties
            posterior_rvs_tlin[letter] = self._posterior_rv_planet(letter, times=tlin, discard_start=discard_start, discard_end=discard_end, thin=thin)

            # TODO: I'm not convinced that storing this all in dicts is ideal. It's a lot of writing and retrieving.
            # There might be a better way to do this without looping through planets twice.

        # Phase plot procedure:
        # 1. Calculate overall system trend RV
        # 2. For each planet, calculate posterior RVs for all other planets at data times
        # 3. Add system trend posterior RV matrix at data times
        # 4. Take median to create composite model excluding target planet
        # 5. Subtract composite model from data to isolate target planet signal
        # 6. Add jitter error bars to account for additional scatter
        # 7. Calculate target planet's posterior RVs at smooth time grid (tlin)
        # 8. Compute 16th, 50th, 84th percentiles of posterior RV matrix
        # 9. Plot median curve with uncertainty shaded region

        fig, axs = plt.subplots(len(self.planet_letters), figsize=(8, len(self.planet_letters)*10/3), sharex=True)
        fig.subplots_adjust(hspace=0)

        # Ensure axs is always an array for consistent indexing
        if len(self.planet_letters) == 1:
            axs = [axs]

        # 1) we need the system trend RV
        trend_rv = self._posterior_rv_trend(times=self.time, discard_start=discard_start, discard_end=discard_end, thin=thin)

        jit_median = np.median(params["jit"])
        verr_with_jit = np.sqrt(self.verr**2 + jit_median**2)

        for i, letter in enumerate(self.planet_letters):

            # 2) sum all of the posterior_rv matrices for the OTHER planets
            all_other_rvs = np.zeros((len(samples_df), len(self.time)))
            other_letters = [pl for pl in self.planet_letters if pl != letter]
            for ol in other_letters:
                all_other_rvs += posterior_rvs[ol]

            # 3) add on the posterior RV matrix for the system trend
            all_other_rvs += trend_rv

            # 4) take the median of this
            median_all_other_rvs = np.median(all_other_rvs, axis=0)

            # 5) subtract this median from the data, to get the residual RV for this planet
            # plot the t_fold, (data - median), observed errorbars
            axs[i].errorbar(t_folds[letter], self.vel-median_all_other_rvs, yerr=self.verr, marker=".", linestyle="None", color="black", zorder=4)

            # 6) overplot the jitter errorbars
            axs[i].errorbar(t_folds[letter], self.vel-median_all_other_rvs, yerr=verr_with_jit, marker="None", linestyle="None", color="black", alpha=0.5, zorder=3)

            # 7) get this planet's posterior RV matrix at smooth tlin times
            this_planet_rv = posterior_rvs_tlin[letter]

            # use the lin_inds to fold the RV. We do this so it's already in the
            # "correct" order to plot folded. Otherwise, matplotlib might
            # connect the first and last point with a horizontal line, and the
            # fill_between doesn't always work correctly.
            this_planet_inds = lin_inds[letter]
            this_planet_rv_folded = this_planet_rv[:,this_planet_inds] # get every sample's posterior RV, but in the order of the inds for folding (not in time order)

            # 8) calculate the 16,50,84 percentiles of the posterior RVs matrix
            rv_percentiles = np.percentile(this_planet_rv_folded, [16, 50, 84], axis=0)

            # 9) plot the median RV and the 16,50,84 percentiles as a shaded region
            axs[i].plot(tlin_folds[letter][lin_inds[letter]], rv_percentiles[1], label="median", color="tab:blue", zorder=2)
            axs[i].fill_between(tlin_folds[letter][lin_inds[letter]], rv_percentiles[0], rv_percentiles[2], color="tab:blue", alpha=0.3, zorder=1)
            axs[i].set_xlim(-0.5, 0.5)
            axs[i].set_ylabel("Radial velocity [m/s]")
            axs[i].legend(loc="best")

            # annotate plot with planet letter, median P and K values
            # TODO: add eccentricity and argument of periastron to annotations
            # TODO: add uncertainties to parameter values
            # TODO: display jitter value in plot title or legend
            _k = params[f"k_{letter}"]
            median_k = np.median(_k)
            s = f"Planet {letter}\nP={p_medians[letter]:.2f} d\nK={median_k:.2f} m/s"
            axs[i].annotate(s, xy=(0, 1), xycoords="axes fraction",
                            xytext=(+0.5, -0.5), textcoords="offset fontsize",
                            va="top")

            if i==0:
                axs[i].set_title("Posterior RV Phase plot")
            if i == len(self.planet_letters) - 1:
                axs[i].set_xlabel("Phase")

        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

        return


class LogPosterior:

    def __init__(
        self,
        planet_letters: list,
        parameterisation: Parameterisation,
        priors: dict,
        fixed_params: dict,
        free_params_names,
        time: np.ndarray,
        vel: np.ndarray,
        verr: np.ndarray,
        t0: float,
    ):
        self.planet_letters = planet_letters
        self.parameterisation = parameterisation
        self.priors = priors
        self.fixed_params = {key: fixed_params[key].value for key in fixed_params}
        self.free_params_names = free_params_names
        self.time = time
        self.vel = vel
        self.verr = verr
        self.t0 = t0

        # build expected params list - 5 pars per planet, then 3 trends, then jit
        self.expected_params = []
        for letter in planet_letters:
            for par in parameterisation.pars:
                self.expected_params.append(par + "_" + letter)
        self.expected_params += ["g", "gd", "gdd"]
        self.expected_params += ["jit"]

        # Create log-likelihood object and log-prior objects for later
        self.log_likelihood = LogLikelihood(time=self.time,
                                            vel=self.vel,
                                            verr=self.verr,
                                            t0=self.t0,
                                            planet_letters=self.planet_letters,
                                            parameterisation=self.parameterisation,
                                            )
        self.log_prior = LogPrior(self.priors)

    def log_probability(self, free_params_dict):
        # Evaluate priors on the free parameters. If any parameters are outside priors
        # (i.e. priors are infinite), then fail fast returning -infty early, so we
        # don't waste time calculating LogLikelihood when we know this step will be rejected.
        lp = self.log_prior(free_params_dict)
        if not np.isfinite(lp):
            return -np.inf

        # Calculate log-likelihood with all parameters
        _all_params_for_ll = self.fixed_params | free_params_dict
        ll = self.log_likelihood(_all_params_for_ll)

        # Return combined log-posterior (log-likelihood + log-prior)
        logprob = ll + lp
        return logprob

    def _negative_log_probability_for_MAP(self, free_params_vals):
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
        free_params_dict = dict(zip(self.free_params_names, free_params_vals))
        logprob = self.log_probability(free_params_dict)
        neg_logprob = -logprob

        # Handle -inf to prevent scipy RuntimeWarnings during optimization
        # scipy's optimizer can't handle -inf values in arithmetic operations
        if not np.isfinite(neg_logprob):
            return 1e30  # Very large finite number instead of inf

        return neg_logprob

    def _positive_log_probability_for_MCMC(self, free_params_vals):
        free_params_dict = dict(zip(self.free_params_names, free_params_vals))
        logprob = self.log_probability(free_params_dict)
        return logprob


class LogLikelihood:
    def __init__(
        self,
        time: np.ndarray,
        vel: np.ndarray,
        verr: np.ndarray,
        t0: float,
        planet_letters: list,
        parameterisation: Parameterisation,
    ):
        self.time = time
        self.vel = vel
        self.verr = verr
        self.t0 = t0

        # build expected params list - 5 pars per planet, then 3 trends, then jit
        self.expected_params = []
        self.planet_letters = planet_letters
        self.parameterisation = parameterisation
        for letter in self.planet_letters:
            for par in self.parameterisation.pars:
                self.expected_params.append(par + "_" + letter)
        self.expected_params += ["g", "gd", "gdd"]
        self.expected_params += ["jit"]

    def __call__(self, params: dict):
        rv_total = np.zeros(len(self.time))

        # Step 1: Calculate RV contributions from each planet
        # TODO: could we rely on dict maintaining order to just get each planet
        # by getting 5 params each time? rather than doing this loop yet again?
        # TODO: or is there a better way to use the list of keys that Parameterisation provides?
        for letter in self.planet_letters:
            _this_planet_keys = [par + "_" + letter for par in self.parameterisation.pars]
            _this_planet_params = {}
            for _this_planet_key in _this_planet_keys:
                _key_inside_dict = _this_planet_key[:-2]
                _this_planet_params[_key_inside_dict] = params[_this_planet_key]
                # Remove planet letter suffix since Planet class expects parameter names without it

            try:
                _this_planet = ravest.model.Planet(letter, self.parameterisation, _this_planet_params)
            except ValueError:
                # Planet.__init__ validates parameters and raises ValueError for invalid params
                return -np.inf

            rv_total += _this_planet.radial_velocity(self.time)

        # Step 2: Calculate RV contribution from trend parameters
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
    def __init__(self, priors: dict):
        self.priors = priors

    def __call__(self, params: dict):
        log_prior_probability = 0
        for param in params:
            # go into the `self.priors dict``, get the Prior object for this `param`
            # and call it with the value of said param, to get the prior probability
            log_prior_probability += self.priors[param](params[param])

        return log_prior_probability
