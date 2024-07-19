# fit.py
import ravest.model
from ravest.param import Parameter, Parameterisation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import emcee
import corner



class Fitter:

    def __init__(self, planet_letters: list[str], parameterisation: Parameterisation):
        self.planet_letters = planet_letters
        self.parameterisation = parameterisation
        self.params = {}

    def add_data(self, time, vel, verr):
        """Add the data to the Fitter object

        Parameters
        ----------
        time : array-like
            Time of each observation [days]
        vel : array-like
            Radial velocity at each time [m/s]
        verr : array-like
            Uncertainty on the radial velocity at each time [m/s]
        """
        if len(time) != len(vel) or len(time) != len(verr):
            raise ValueError(
                "Time, velocity, and uncertainty arrays must be the same length."
            )

        self.time = time
        self.vel = vel
        self.verr = verr

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
        expected_length = 4 + (5 * len(self.planet_letters))  # 3 trend pars + jit + 5 pars per planet
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
        """Add the priors for the parameters.

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
        if set(self.get_free_params_names()) != set(priors):
            raise ValueError(f"Priors must be provided for all free parameters. Missing priors for {set(self.get_free_params_names()) - set(priors)}.")
        if len(self.get_free_params_names()) < len(priors):
            raise Warning(
                f"There are fewer free parameters than priors provided. Have you accidentally provided a prior for a fixed parameter?"+
                f"\nReceived unexpected priors for {set(priors) - set(self.get_free_params_names())}."
            )
        self.priors = priors

    def get_free_params_dict(self):
        """Return a dictionary of the free parameters, i.e. those that are not fixed."""
        self._free_pars = {}
        for par in self.params:
            if self.params[par].fixed == False:
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
            if self.params[par].fixed == True:
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
            self.verr
        )

        # 1) Maximum A Posteriori to get the initial points for MCMC
        negative_log_posterior = lambda *args: lp._negative_log_probability_for_MAP(*args)
        map_results = minimize(negative_log_posterior, self.get_free_params_val(), method="Powell")
        if map_results.success == False:
            print(map_results)
            raise Warning("MAP did not converge. Check the initial values of the parameters, and the priors functions.")
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
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability, 
                                            parameter_names=self.get_free_params_names())
        state = sampler.run_mcmc(initial_state=mcmc_init, nsteps=nsteps, progress=progress)

        # TODO: multiprocessing disabled for now as it's causing slowdown
        # I suspect something might be being pickled that shouldn't be, but 
        # this requires further investigation.
        # with Pool() as pool:
        #     sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lp.log_probability, 
        #                                     parameter_names=self.get_free_params_names(),
        #                                     pool=pool)
        #     state = sampler.run_mcmc(mcmc_init, 10000, progress=True)
        print("...MCMC done.")
        self.sampler = sampler
    
    def get_samples_df(self, discard=0, thin=1):
        """Returns a dataframe of the (flattened) samples from the MCMC run."""
        # First, get the flat chains
        flat_samples = self.sampler.get_chain(flat=True, thin=thin, discard=discard)
        df = pd.DataFrame(flat_samples, columns=self.get_free_params_names())
        return df
    
    def plot_chains(self, discard=0, thin=1, save=False, fname="chains_plot.png", dpi=100):
        fig, axes = plt.subplots(self.ndim, figsize=(10,6), sharex=True)
        # TODO: scale the size of the figure with the number of parameters
        fig.suptitle("Chains plot")

        samples = self.sampler.get_chain(flat=False, thin=thin, discard=discard)
        for i in range(self.ndim):
            ax = axes[i]
            to_plot = samples[:, :, i] # type: ignore

            ax.plot(to_plot, "k", alpha=0.3)
            ax.set_xlim(0, len(samples)) # type: ignore
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
    ):
        self.planet_letters = planet_letters
        self.parameterisation = parameterisation
        self.priors = priors
        # self.fixed_params = fixed_params  # parameter objects
        self.fixed_params = {key: fixed_params[key].value for key in fixed_params}
        self.free_params_names = free_params_names
        self.time = time
        self.vel = vel
        self.verr = verr

        # build expected params list - 5 pars per planet, then 3 trends, then jit
        self.expected_params = []
        for letter in planet_letters:
            for par in parameterisation.pars:
                self.expected_params.append(par + "_" + letter)

        self.expected_params += ["g", "gd", "gdd"]
        self.expected_params += ["jit"]

        # Create log-likelihood object and log-prior objects for later
        self.log_likelihood = LogLikelihood(self.time, self.vel, self.verr, self.planet_letters, self.parameterisation)
        self.log_prior = LogPrior(self.priors)


    def log_probability(self, free_params_dict):
        # print("\nDEBUG: LOGPOSTERIOR.LOG_PROBABILITY FUNCTION")
        # one) sort out the fixed and free values so that we can actually pass them into the LL object

        # 2) Calculate priors. If any are infinite, return -inf (saves us wasting time calculating LL too)
        lp = self.log_prior(free_params_dict)
        if not np.isfinite(lp):
            # print("DEBUG DEBUG LogPosterior.log_probability: lp is not finite, returning -inf")
            return -np.inf
        # print("DEBUG LogPosterior.log_probability: lp", lp)
            
        # 3) Calculate the log-likelihood
        _all_params_for_ll = self.fixed_params | free_params_dict
        ll = self.log_likelihood(_all_params_for_ll)
        # print("DEBUG LogPosterior.log_probability: ll", ll)

        # 4) return log-likehood + log-prior
        logprob = ll+lp
        # print("DEBUG LogPosterior.log_probability: ll*lp", logprob)
        return logprob


    def _negative_log_probability_for_MAP(self, free_params_vals):
        """For MAP: run __call__ only passing in a list, not dict, of params.

        Because scipy.optimize.minimise only allows list of values, not a dict,
        we need to assign the values back to their corresponding keys, and pass
        that to __call__().

        This does not check that the values are in the correct order, it is
        assumed. As we're dealing with dicts, this hopefully is the case.
        TODO: add tests to check if this is the case!

        Parameters
        ----------
        free_params_vals : list
            float values of the free parameters
        """
        # print("\nDEBUG: LOGPOSTERIOR._NEGATIVE_LOG_PROBABILITY_FOR_MAP FUNCTION")
        free_params_dict = dict(zip(self.free_params_names, free_params_vals))
        logprob = self.log_probability(free_params_dict)
        # print("DEBUG LogPosterior._negative_log_probability_for_MAP function: logprob", logprob)
        neglogprob = -1 * logprob
        # print("DEBUG LogPosterior._negative_log_probability_for_MAP function: neglogprob", neglogprob)
        return neglogprob
    
    def _positive_log_probability_for_MCMC(self, free_params_vals):
        # print("\nDEBUG: LOGPOSTERIOR._NEGATIVE_LOG_PROBABILITY_FOR_MAP FUNCTION")
        free_params_dict = dict(zip(self.free_params_names, free_params_vals))
        logprob = self.log_probability(free_params_dict)
        # print("DEBUG LogPosterior._negative_log_probability_for_MAP function: logprob", logprob)
        return logprob
        

class LogLikelihood:
    def __init__(
        self,
        time: np.ndarray,
        vel: np.ndarray,
        verr: np.ndarray,
        planet_letters: list,
        parameterisation: Parameterisation,
    ):
        self.time = time
        self.vel = vel
        self.verr = verr

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
        # print("\nDEBUG: LOG-LIKELIHOOD OBJECT __CALL__ FUNCTION")
        rv_total = np.zeros(len(self.time))

        # one) calculate RV for each planet
        # TODO: could we rely on dict maintaining order to just get each planet
        # by getting 5 params each time? rather than doing this loop yet again?
        # TODO: or is there a better way to use the list of keys that Parameterisation provides?
        for letter in self.planet_letters:
            _this_planet_keys = [par + "_" + letter for par in self.parameterisation.pars]
            # _this_planet_params = {key[:-2]: params[key] for key in _this_planet_keys}
            _this_planet_params = {}
            for _this_planet_key in _this_planet_keys:
                # print("DEBUG: _this_planet_key", _this_planet_key)
                _key_inside_dict = _this_planet_key[:-2]
                # print("DBEUG: _key_inside_dict", _key_inside_dict)
                _this_planet_params[_key_inside_dict] = params[_this_planet_key]
                # we do this because the Planet object doesn't want the planet letter in the key
            _this_planet = ravest.model.Planet(
                letter, self.parameterisation, _this_planet_params
            )
            # print("DEBUG LogLikelihood.__call__: _this_planet", _this_planet)
            rv_total += _this_planet.radial_velocity(self.time)

        # two) calculate RV for trend parameters
        # TODO: this will later need updating to include multi-instrument support
        _trend_keys = ["g", "gd", "gdd"]
        _trend_params = {key: params[key] for key in _trend_keys}
        # print("DEBUG: _trend_params", _trend_params)
        _this_trend = ravest.model.Trend(_trend_params)
        # print("DEBUG: _this_trend", _this_trend)
        _rv_trend = _this_trend.radial_velocity(self.time)
        # print("DEBUG: _rv_trend", _rv_trend)
        rv_total += _rv_trend

        # three) do the log-likelihood calculation including jitter
        # print("DEBUG LogLikelihood.__call__: LOG-LIKELIHOOD CALCULATION")
        verr_jitter_quadrature = self.verr**2 + params["jit"]**2  # we don't sqrt here as we square again in the next line anyway
        jitter_penalty_term = np.sum(np.log(np.sqrt(2 * np.pi * verr_jitter_quadrature)))
        # print("DEBUG: jitter_penalty_term", jitter_penalty_term)
        residuals = rv_total - self.vel
        chi2 = np.sum(residuals**2 / verr_jitter_quadrature)
        ll = (-0.5 * chi2) - jitter_penalty_term
        # print("DEBUG LogLikelihood.__call__: ll", ll)
        return ll


class LogPrior:
    def __init__(self, priors: dict):
        self.priors = priors

    def __call__(self, params: dict):
        log_prior_probability = 0
        # print("DEBUG LogPrior.__call")
        # print("DEBUG: passed in params dict:", params)
        for param in params:
            # go into the priors dict, get the Prior object for this parameter
            # then Prior.__call__, passing in the passed-in value of said param
            # print("DEBUG: for param in params:", param, params)
            log_prior_probability += self.priors[param](params[param])

        return log_prior_probability
