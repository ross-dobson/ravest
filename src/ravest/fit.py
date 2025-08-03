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
        self.params = {}

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

    def add_params(self, params):
        """Add the parameters, checking the correct parameters for the parameterisation are present.

        Parameters
        ----------
        params : dict
            Dictionary of Parameters, for each parameter in the fitting parameterisation, for each planet

        Raises
        ------
        ValueError
            If the expected parameters are not present in the params dict
        """
        # first check - do we have the right number of parameters?
        expected_length = 4 + (5 * len(self.planet_letters))  # 3 trend pars (g, gd, gdd) + jit, then 5 pars per planet
        if len(params) != expected_length:
            raise ValueError(
                f"Expected {expected_length} parameters, got {len(params)} parameters"
            )

        # second check - are all the parameters in the list?
        for planet_letter in self.planet_letters:  # for each planet
            for par_name in self.parameterisation.pars:  # and for each parameter we expect from the parameterisation
                expected_par = par_name + "_" + planet_letter
                if expected_par not in params:
                    raise ValueError(f"Parameter {expected_par} not found in parameter list")
                elif expected_par in params:
                    self.params[expected_par] = params[expected_par]
        if "g" not in params:
            raise ValueError("Parameter g not found in parameter list")
        self.params["g"] = params["g"]
        if "gd" not in params:
            raise ValueError("Parameter gd not found in parameter list")
        self.params["gd"] = params["gd"]
        if "gdd" not in params:
            raise ValueError("Parameter gdd not found in parameter list")
        self.params["gdd"] = params["gdd"]
        if "jit" not in params:
            raise ValueError("Parameter jit not found in parameter list")
        self.params["jit"] = params["jit"]

    def add_priors(self, priors: dict):
        """Add the priors for the free parameters, checking init value is valid.

        Given a dict of Prior functions, it checks that there is a prior for all
        of the free parameters (and none of the fixed parameters). It also calls
        each prior function with the initial value of the parameter to check
        that none of the starting positions are invalid (which can cause
        problems with the MCMC run later.)

        Parameters
        ----------
        priors : dict
            Dictionary of Prior functions for each of the free parameters

        Raises
        ------
        ValueError
            If the expected parameters are not present in the priors dict
        Warning
            If a parameter is fixed, but a prior is provided
        """
        if len(self.get_free_params_names()) < len(priors):
            raise Warning(
                "Too many priors provided. Have you accidentally provided a prior for a fixed parameter?"
                + f"\nReceived unexpected priors for {set(priors) - set(self.get_free_params_names())}."
            )
        if set(self.get_free_params_names()) != set(priors):
            raise ValueError(f"Priors must be provided for all free parameters. Missing priors for {set(self.get_free_params_names()) - set(priors)}.")

        # check that the initial parameter values are within the priors
        for par in self.get_free_params_names():
            prior_fn = priors[par]
            log_prior_prob = prior_fn(self.params[par].value)
            if not np.isfinite(log_prior_prob):
                raise ValueError(f"Initial value {self.params[par].value} of parameter {par} is invalid for prior {priors[par]}.")

        self.priors = priors

    def get_free_params_dict(self):
        """Return a dictionary of the free parameters."""
        self._free_pars = {}
        for par in self.params:
            if self.params[par].fixed is False:
                self._free_pars[par] = self.params[par]
        return self._free_pars

    def get_free_params_val(self):
        _list = list(self.get_free_params_dict().values())
        return [param.value for param in _list]

    def get_free_params_names(self):
        return list(self.get_free_params_dict().keys())

    def get_fixed_params_dict(self):
        """Return a dictionary of the fixed parameters."""
        self._fixed_pars = {}
        for par in self.params:
            if self.params[par].fixed is True:
                self._fixed_pars[par] = self.params[par]
        return self._fixed_pars

    def get_fixed_params_val(self):
        _list = list(self.get_fixed_params_dict().values())
        return [param.value for param in _list]

    def get_fixed_params_names(self):
        return list(self.get_fixed_params_dict().keys())

    def fit_model_to_data(self, nwalkers, nsteps=5000, progress=True):

        # create the log-posterior object
        lp = LogPosterior(
            self.planet_letters,
            self.parameterisation,
            self.priors,
            self.get_fixed_params_dict(),
            self.get_free_params_names(),
            self.time,
            self.vel,
            self.verr,
            self.t0,
        )

        # 1) Maximum A Posteriori to get the initial points for MCMC
        def negative_log_posterior(*args):
            return lp._negative_log_probability_for_MAP(*args)
        map_results = minimize(negative_log_posterior, self.get_free_params_val(), method="Powell")
        if map_results.success is False:
            print(map_results)
            raise Warning("MAP did not succeed. Check the initial values of the parameters, and the priors functions.")
        self.ndim = len(self.get_free_params_val())

        # zip the MAP results with the free parameter names to get a dict
        map_results_dict = dict(zip(self.get_free_params_names(), map_results.x))
        print("MAP results:", map_results_dict)

        # 2) MCMC
        print("Starting MCMC...")
        if nwalkers < 2 * self.ndim:
            print(f"Warning: nwalkers should be at least 2 * ndim. You have {nwalkers} walkers and {self.ndim} dimensions. Setting nwalkers to {2 * self.ndim}.")
            self.nwalkers = 2 * self.ndim
        else:
            self.nwalkers = nwalkers

        mcmc_init = map_results.x + 1e-5 * np.random.randn(self.nwalkers, self.ndim)
        # TODO: is passing in parameter_names causing slowdown?
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability,
                                            parameter_names=self.get_free_params_names())
        sampler.run_mcmc(initial_state=mcmc_init, nsteps=nsteps, progress=progress)

        # TODO: multiprocessing disabled for now as it's causing slowdown
        # I suspect something might be being pickled that shouldn't be, but
        # this requires further investigation.
        #
        # with Pool() as pool:
        #     sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability,
        #                                     parameter_names=self.get_free_params_names(),
        #                                     pool=pool)
        #     state = sampler.run_mcmc(mcmc_init, 10000, progress=True)
        print("...MCMC done.")
        self.sampler = sampler

    def get_samples_np(self, discard=0, thin=1, flat=False):
        """Returns a contiguous numpy array of MCMC samples.

        This is the foundational method for accessing MCMC samples. All other sample
        methods build on this for consistency and optimal performance.

        Parameters
        ----------
        discard : int, optional
            Discard the first `discard` steps as burn-in (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        flat : bool, optional
            If True, return flattened array shape (nsteps_after_discard_thin * nwalkers, ndim)
            If False, return unflattened array shape (nsteps_after_discard_thin, nwalkers, ndim) (default: False)

        Returns
        -------
        np.ndarray
            Contiguous array of MCMC samples. Shape depends on `flat` parameter:
            - flat=False: (nsteps_after_discard_thin, nwalkers, ndim)
            - flat=True: (nsteps_after_discard_thin * nwalkers, ndim)

        Notes
        -----
        This method wraps emcee's sampler.get_chain() but ensures the returned array
        is contiguous in memory for optimal performance in computational operations.
        The default shape (nsteps, nwalkers, ndim) matches emcee's convention.
        """
        samples = self.sampler.get_chain(discard=discard, thin=thin, flat=flat)
        return np.ascontiguousarray(samples)

    def get_samples_df(self, discard=0, thin=1):
        """Returns a pandas DataFrame of flattened MCMC samples.

        Each row represents one sample, columns are parameter names. Built on
        get_samples_np() for consistent performance.

        Parameters
        ----------
        discard : int, optional
            Discard the first `discard` steps as burn-in (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)

        Returns
        -------
        pd.DataFrame
            DataFrame with shape (nsteps_after_discard_thin * nwalkers, ndim).
            Columns are parameter names.
        """
        flat_samples = self.get_samples_np(discard=discard, thin=thin, flat=True)
        return pd.DataFrame(flat_samples, columns=self.get_free_params_names())

    def get_samples_dict(self, discard=0, thin=1):
        """Returns dict of MCMC samples for each free parameter.

        Each parameter gets a 1D contiguous array of all its samples. This is more
        efficient than the previous _get_samples_ndarray_dict method.

        Parameters
        ----------
        discard : int, optional
            Discard the first `discard` steps as burn-in (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)

        Returns
        -------
        dict
            Dictionary mapping parameter names to 1D arrays of samples.
            Each array has shape (nsteps_after_discard_thin * nwalkers,)

        Examples
        --------
        >>> samples_dict = fitter.get_samples_dict(discard=1000)
        >>> k_b_samples = samples_dict['k_b']  # All samples for k_b parameter
        """
        flat_samples = self.get_samples_np(discard=discard, thin=thin, flat=True)
        param_names = self.get_free_params_names()

        # Direct numpy slicing - much faster than pandas operations
        return {name: flat_samples[:, i] for i, name in enumerate(param_names)}

    def get_posterior_params_dict(self, discard=0, thin=1):
        """Returns dict combining fixed parameters and MCMC samples.

        This method creates a unified dictionary containing all model parameters:
        fixed parameters as single float values, and free parameters as arrays
        of MCMC samples. This format is ideal for functions like calculate_mpsini
        that need all parameters and should propagate uncertainties from free parameters.

        Parameters
        ----------
        discard : int, optional
            Discard the first `discard` steps as burn-in (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)

        Returns
        -------
        dict
            Dictionary of all parameters:
            - Fixed parameters: single float values
            - Free parameters: 1D arrays of MCMC samples with shape (nsteps_after_discard_thin * nwalkers,)

        Examples
        --------
        >>> params = fitter.get_posterior_params_dict(discard=1000)
        >>> params['per_b']  # Fixed parameter: single float (e.g., 20.8851)
        >>> params['k_b']    # Free parameter: array of samples (e.g., [10.1, 9.9, ...])
        """
        fixed_params_dict = dict(zip(self.get_fixed_params_names(), self.get_fixed_params_val()))
        free_samples_dict = self.get_samples_dict(discard=discard, thin=thin)
        return fixed_params_dict | free_samples_dict

    def plot_chains(self, discard=0, thin=1, save=False, fname="chains_plot.png", dpi=100):
        fig, axes = plt.subplots(self.ndim, figsize=(10,1+(self.ndim*2/3)), sharex=True)
        # TODO: scale the size of the figure with the number of parameters
        fig.suptitle("Chains plot")

        samples = self.sampler.get_chain(flat=False, thin=thin, discard=discard)
        for i in range(self.ndim):
            ax = axes[i]
            to_plot = samples[:, :, i]  # type: ignore

            ax.plot(to_plot, "k", alpha=0.3)
            ax.set_xlim(0, len(samples))  # type: ignore
            ax.set_ylabel(self.get_free_params_names()[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("Step number")
        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def plot_corner(self, discard=0, thin=1, save=False, fname="corner_plot.png", dpi=100):
        flat_samples = self.sampler.get_chain(flat=True, discard=discard, thin=thin)
        fig = corner.corner(
        flat_samples, labels=self.get_free_params_names(), show_titles=True,
        plot_datapoints=False, quantiles=[0.16, 0.5, 0.84],
        )
        fig.suptitle("Corner plots")
        if save:
            plt.savefig(fname=fname, dpi=dpi)
            print(f"Saved {fname}")
        plt.show()

    def _posterior_rv(self, discard=0, thin=1):
        """For each parameter sample in the MCMC chain, calculate the RV.

        The RVs are calculated at the times in `tlin`, which is calculated as
        uniform points between the minimum and maximum times in the data, with
        a 1% buffer on either side.

        Parameters
        ----------
        discard : int, optional
            Discard the first `discard` steps in the chain as burn-in. (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain. (default: 1)

        Returns
        -------
        np.ndarray
            Array of RVs for each sample in the chain, at the times in `tlin`.
        """
        samples = self.get_samples_np(discard=discard, thin=thin, flat=True)

        # get smooth time curve for plotting
        _tmin, _tmax = self.time.min(), self.time.max()
        _trange = _tmax - _tmin
        tlin = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # store the rv for each sample here
        rv_array = np.zeros((len(samples), len(tlin)))  # type: ignore

        # get the free parameter names and fixed parameter values
        # we don't need to call this repeatedly for each sample
        free_param_names = self.get_free_params_names()
        fixed_params_dict = dict(zip(self.get_fixed_params_names(), self.get_fixed_params_val()))

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

    def plot_posterior_rv(self, discard=0, thin=1, save=False, fname="posterior_rv.png", dpi=100):
        """Plot the posterior RV model (median & 16-84 percentiles).

        For each sample of parameters in the MCMC chain, calculate the RV. Plot the
        median RV, and the 16th and 84th percentiles as shaded region. The RVs are
        calculated at the times in `tlin`, which is calculated as uniform points
        between the minimum and maximum times in the data, with a 1% buffer on either
        side.

        Parameters
        ----------
        discard : int, optional
            Discard the first `discard` steps in the chain as burn-in (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        save : bool, optional
            Save the plot to path `fname` (default: False)
        fname : str, optional
            The path to save the plot to (default: "posterior_rv.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)
        """
        # TODO: should we combine this function with posterior phase, like in model?
        # That way, we could cut down on (re)calculating posterior RVs.

        # Get the posterior RVs, evaluated at each sample in the chains
        rv_array, tlin = self._posterior_rv(discard=discard, thin=thin)
        rv_percentiles = np.percentile(rv_array, [16, 50, 84], axis=0)

        # Get the new errorbars to include jit
        if "jit" in self.get_fixed_params_names():
            jit_median = self.get_fixed_params_dict()["jit"].value
        else:
            jit_median = np.median(self.get_samples_df()["jit"])
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

    def _posterior_rv_planet(self, planet_letter, times, discard=0, thin=1):
        """calculate the posterior rv for a planet, using all samples in the chain"""
        samples = self.get_samples_np(discard=discard, thin=thin, flat=True)
        this_planet_rvs = np.zeros((len(samples), len(times))) # type: ignore
        fixed_params_dict = dict(zip(self.get_fixed_params_names(), self.get_fixed_params_val()))

        for i, row in enumerate(samples):  # type: ignore
            # Combine fixed and free parameters
            free_params = dict(zip(self.get_free_params_names(), row))
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

    def _posterior_rv_trend(self, times, discard=0, thin=1):
        """calculate the posterior rv for the trend, using all samples in the chain"""
        samples = self.get_samples_np(discard=discard, thin=thin, flat=True)
        this_trend_rvs = np.zeros((len(samples), len(times))) # type: ignore
        fixed_params_dict = dict(zip(self.get_fixed_params_names(), self.get_fixed_params_val()))

        # for each sample in the chain, calculate the RV for the trend
        for i, row in enumerate(samples):  # type: ignore
            # Combine fixed and free parameters
            free_params = dict(zip(self.get_free_params_names(), row))
            params = fixed_params_dict | free_params

            this_trend = ravest.model.Trend(params={"g": params["g"], "gd": params["gd"], "gdd": params["gdd"]}, t0=self.t0)
            this_trend_rv = this_trend.radial_velocity(times)
            this_trend_rvs[i, :] = this_trend_rv

        return this_trend_rvs

    def plot_posterior_phase(self, discard=0, thin=1, save=False, fname="posterior_phase.png", dpi=100):
        """Plot the posterior RV model (median & 16-84 percentiles) in phase space.

        For each sample of parameters in the MCMC chain, calculate the RV. Fold the
        times around the median values of P and t_c for each planet. Plot the median RV
        and the 16th and 84th percentiles as shaded region. The RVs are calculated at
        the times in `tlin`, which is calculated as uniform points between the minimum
        and maximum times in the data, with a 1% buffer on either side.

        Parameters
        ----------
        discard : int, optional
            Discard the first `discard` steps in the chain as burn-in (default: 0)
        thin : int, optional
            Use only every `thin` steps from the chain (default: 1)
        save : bool, optional
            Save the plot to path `fname` (default: False)
        fname : str, optional
            The path to save the plot to (default: "posterior_phase.png")
        dpi : int, optional
            The dpi to save the image at (default: 100)
        """
        # TODO: should we combine this function with posterior phase, like in model?
        # That way, we could cut down on (re)calculating posterior RVs.

        # get smooth linear time curve for plotting
        _tmin, _tmax = self.time.min(), self.time.max()
        _trange = _tmax - _tmin
        tlin = np.linspace(_tmin - 0.01 * _trange, _tmax + 0.01 * _trange, 1000)

        # Each parameter is either fixed, or has a (flattened) chain of samples.
        # Construct a dict `params' to store whichever one we have.
        # Both work for taking a median, (the median of a fixed value is just
        # the fixed value) and the RV functions will propagate either a fixed
        # value or an array of values through the calculations.
        fixed_params_dict = dict(zip(self.get_fixed_params_names(), self.get_fixed_params_val()))
        samples_df = self.get_samples_df(discard=discard, thin=thin)
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
            # else, convert to default basis, giving us e and w, then get tc
            else:
                # get the parameterisation for this planet. Combine with planet letter
                _keys = [f"{par}_{letter}" for par in self.parameterisation.pars]
                _inpars = {key: samples_df[key] for key in _keys}
                _default_basis = self.parameterisation.convert_pars_to_default_basis(_inpars)  # dict of converted pars
                tc = _default_basis["tc"]

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

            ### 3) calculate the posterior rv for this planet at the data x times
            # this generates the matrix (x=times, y=samples) of RVs for this planet
            posterior_rvs[letter] = self._posterior_rv_planet(letter, times=self.time, discard=discard, thin=thin)

            ### 4) calculate the posterior rv for this planet at the smooth tlin times
            # this generates the matrix (x=tlin, y=samples) of RVs for this planet
            posterior_rvs_tlin[letter] = self._posterior_rv_planet(letter, times=tlin, discard=discard, thin=thin)

            # TODO: I'm not convinced that storing this all in dicts is ideal. It's a lot of writing and retrieving.
            # There might be a better way to do this without looping through planets twice.

        # For the plot, we need:
        # 1) the overall system trend RV
        # Then, for each planet:
        # 2) the posterior RVs for every other planet, at the data x times, summed
        # 3) add the system trend posterior RV matrix, at the data x times
        # 4) take the median of this
        # 5) subtract this median from the data, to get the residual RV for this planet
        # 6) overplot the jitter additional errorbars
        # 7) get this planet's posterior RVs at the tlin times (using the tlin_folds indices to fold them)
        # 8) calculate the 16,50,84 percentiles of the posterior RVs matrix
        # 9) plot the median RV, and the 16,50,84 percentiles as a shaded region

        fig, axs = plt.subplots(len(self.planet_letters), figsize=(8, len(self.planet_letters)*10/3), sharex=True)
        fig.subplots_adjust(hspace=0)

        ### 1) we need the system trend RV
        trend_rv = self._posterior_rv_trend(times=self.time, discard=discard, thin=thin)

        jit_median = np.median(params["jit"])
        verr_with_jit = np.sqrt(self.verr**2 + jit_median**2)

        for i, letter in enumerate(self.planet_letters):

            ### 2) sum all of the posterior_rv matrices for the OTHER planets
            all_other_rvs = np.zeros((len(samples_df), len(self.time)))
            other_letters = [pl for pl in self.planet_letters if pl != letter]
            for ol in other_letters:
                all_other_rvs += posterior_rvs[ol]

            ### 3) add on the posterior RV matrix for the system trend
            all_other_rvs += trend_rv

            ### 4) take the median of this
            median_all_other_rvs = np.median(all_other_rvs, axis=0)

            ### 5) subtract this median from the data, to get the residual RV for this planet
            # plot the t_fold, (data - median), observed errorbars
            axs[i].errorbar(t_folds[letter], self.vel-median_all_other_rvs, yerr=self.verr, marker=".", linestyle="None", color="black", zorder=4)

            ### 6) overplot the jitter errorbars
            axs[i].errorbar(t_folds[letter], self.vel-median_all_other_rvs, yerr=verr_with_jit, marker="None", linestyle="None", color="black", alpha=0.5, zorder=3)

            ### 7) get this planet's posterior RV matrix at smooth tlin times
            this_planet_rv = posterior_rvs_tlin[letter]

            # use the lin_inds to fold the RV. We do this so it's already in the
            # "correct" order to plot folded. Otherwise, matplotlib might
            # connect the first and last point with a horizontal line, and the
            # fill_between doesn't always work correctly.
            this_planet_inds = lin_inds[letter]
            this_planet_rv_folded = this_planet_rv[:,this_planet_inds] # get every sample's posterior RV, but in the order of the inds for folding (not in time order)

            ### 8) calculate the 16,50,84 percentiles of the posterior RVs matrix
            rv_percentiles = np.percentile(this_planet_rv_folded, [16, 50, 84], axis=0)

            ### 9) plot the median RV and the 16,50,84 percentiles as a shaded region
            axs[i].plot(tlin_folds[letter][lin_inds[letter]], rv_percentiles[1], label="median", color="tab:blue", zorder=2)
            axs[i].fill_between(tlin_folds[letter][lin_inds[letter]], rv_percentiles[0], rv_percentiles[2], color="tab:blue", alpha=0.3, zorder=1)
            axs[i].set_xlim(-0.5, 0.5)
            axs[i].set_ylabel("Radial velocity [m/s]")
            axs[i].legend(loc="best")

            # annotate plot with planet letter, median P and K values
            # TODO: e and omega (or parameterisation of) too?
            # TODO: uncertainties on these values
            # TODO: jitter in plot title or legend
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
        # one) sort out the fixed and free values so that we can actually pass them into the LL object

        # 2) Calculate priors. If any are infinite, return -inf (saves us wasting time calculating LL too)
        lp = self.log_prior(free_params_dict)
        if not np.isfinite(lp):
            return -np.inf

        # 3) Calculate the log-likelihood
        _all_params_for_ll = self.fixed_params | free_params_dict
        ll = self.log_likelihood(_all_params_for_ll)

        # 4) return log-likehood + log-prior
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
        return -logprob

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

        # one) calculate RV for each planet
        # TODO: could we rely on dict maintaining order to just get each planet
        # by getting 5 params each time? rather than doing this loop yet again?
        # TODO: or is there a better way to use the list of keys that Parameterisation provides?
        for letter in self.planet_letters:
            _this_planet_keys = [par + "_" + letter for par in self.parameterisation.pars]
            _this_planet_params = {}
            for _this_planet_key in _this_planet_keys:
                _key_inside_dict = _this_planet_key[:-2]
                _this_planet_params[_key_inside_dict] = params[_this_planet_key]
                # we do this because the Planet object doesn't want the planet letter in the key
            _this_planet = ravest.model.Planet(letter, self.parameterisation, _this_planet_params)

            # TODO - this isn't the best place to perform this check. Ideally,
            # we would do this check when the parameters are set/converted, so
            # we don't waste time calculating the RV when we know we have a bad
            # parameter (e.g.  e>=1). However, if we do it inside Planet, we can
            # raise an Exception, but we can't return -np.inf probability, as we
            # can't return from inside a class. So, we need to do it here, and
            # return -np.inf if the parameters are bad. Ideally, we should do
            # some kind of check (and perhaps conversion) when the parameters
            # are set, perhaps inside LogPrior. But then we might end up doing
            # parameter conversion twice.
            if not _this_planet.is_valid():
                return -np.inf

            rv_total += _this_planet.radial_velocity(self.time)

        # two) calculate RV for trend parameters
        _trend_keys = ["g", "gd", "gdd"]
        _trend_params = {key: params[key] for key in _trend_keys}
        _this_trend = ravest.model.Trend(params=_trend_params, t0=self.t0)
        _rv_trend = _this_trend.radial_velocity(self.time)
        rv_total += _rv_trend

        # three) do the log-likelihood calculation including jitter
        verr_jitter_squared = self.verr**2 + params["jit"]**2  # we don't sqrt here as we would square again in the next line anyway
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
