# Changelog

## v0.4.1 (2026-09-29)

### Licence
- **Relicensed from MIT to GPL-3.0-or-later.** v0.4.0 and earlier remain MIT-licensed; code pinned to those versions is unaffected
- Licence declared the PEP 639 way (`license = "GPL-3.0-or-later"`, `license-files`), replacing the deprecated licence table and classifier

### Breaking changes
- Log-posterior corrections for the `secosw`/`sesinw` parameterisation (see the [log-posterior corrections](https://ravest.readthedocs.io/en/latest/logprob_corrections.html) page). They do not change parameter inference, but $\ln\mathcal{Z}$ estimates for fits in `secosw`/`sesinw` now differ from v0.4.0 by a constant per planet ($+\ln 2$ or $+\ln(4/\pi)$).
- `ecosw`/`esinw` parameterisations disabled, since their Jacobian is not constant
- A prior directly on `(secosw, sesinw)` other than `Uniform(-1, 1)` on both now raises `NotImplementedError`; put the prior on `e` instead
- Every free parameter must now have a prior, and every free GP hyperparameter a hyperprior. `find_map_estimate`, `run_mcmc` and the walker-initialisation methods raise `ValueError` naming what is missing, where previously a fit could run with no priors at all, silently sampling the likelihood alone
- `Fitter` and `GPFitter` raise `TypeError` if `parameterisation` is not a `Parameterisation` object
- `run_mcmc` raises `ValueError` if `check_convergence=True` but the first check could never happen within `max_steps`
- `param_key_to_latex` argument renamed `key` -> `param_key` (affects keyword callers only)

### Features
- `freeze_params` argument for `plot_posterior_phase` (both classes): fix e.g. `P` and `Tc` at their posterior medians, or given values, to stop the median model smearing out when phase-folding
- Overhauled RV and phase plots: `xlim`/`ylim`/`res_xlim`/`res_ylim` on the RV plots and `ylim`/`res_ylim` on the phase plots, `n_smooth` on the phase plots, phase-model curve spanning a full orbit, m s$^{-1}$ axis labels, consistent legend labels

### Fixes
- Walker initialisation: `Beta` hyperpriors drew from their shape parameters as if they were bounds, so every walker failed and the fit could not start; now drawn from `[0, 1]`
- Walker initialisation: `Normal`/`HalfNormal` parameters now drawn at one prior width, not two, matching the hyperparameters (measured: far fewer stuck walkers, no cost in convergence time)
- `GPFitter.ndim` was too small if `params` was set after `hyperparams`, which made BIC/AICc silently undercount the number of free parameters
- Parameter labels follow MNRAS style: roman subscripts for labels (planet letters, instruments, GP), capitalised $T_\mathrm{C}$/$T_\mathrm{P}$ subscripts, and a $\star$ subscript on every $\omega$ (the star's argument of periastron)
- Log-posterior correction cases logged at DEBUG rather than INFO
- Sphinx/RTD warnings from docstrings

### Performance
- `LogPosterior`/`GPLogPosterior` built once per walker-initialisation call rather than per walker

### Tests
- New tests for walker initialisation, prior and hyperprior presence checks, `GPFitter.ndim`, parameterisation type validation, and the log-posterior corrections
- `freeze_params` warnings asserted in tests instead of leaking
- `TestBeta` class-scoped fixture made a classmethod, ready for `pytest` 10

### Docs
- Example notebooks renumbered, renamed (`example_1_fitting`, `example_2_K2-24`, `example_3_GP`, `example_4_harmonic`) and retitled (Example 1: fitting a single planet; Example 2: fitting a two-planet system; Example 3: fitting with a Gaussian Process; Example 4: using the Learned Harmonic Mean Estimator), restructured with section headings for easier navigation, and cross-linked; the docs sidebar section is renamed from Tutorials to Examples, and the example pages have new URLs
- New Example 4 (`harmonic`) comparing 1- and 2-planet models for TOI-544, now using the Dobson et al. (2026) data and priors, with correct $\ln\mathcal{Z}$ error bars (they were $\ln\sigma_\mathcal{Z}$), evidence quality checks, and an error on $\Delta\ln\mathcal{Z}$
- Example notebooks now fix NumPy's random seed once, in a clearly labelled cell, so they reproduce exactly when re-run, with a note on the shortcuts (fewer walkers, shorter chains, fixed seed) not to copy into real fits.
- Removed the outdated `ravest.model` example notebook
- Enabled MyST strikethrough (single tilde) in the docs
- Example 2 (K2-24): information criteria labelled as AICc (what is computed), and its conclusion corrected: AICc and BIC both prefer the circular model
- New FAQ and log-posterior corrections pages; the latter has typeset maths, cites Appendix A of Dobson et al. (2026) for the full derivation, and is linked from Example 4
- Clarified that `GPFitter.params` holds free and fixed parameters, not hyperparameters (and is therefore not the chain's columns)
- README: citation section (Dobson et al. 2026), badges, GitHub link, JAX acknowledgement, links to all four examples (fixing a broken link to the GP example)
- Fixed the K2-24 example calling the removed `calculate_aic`
- Example notebooks updated for the new parameter labels

### Build
- Lock refreshed: `ml-dtypes` 0.6.0 clears 451 test DeprecationWarnings; `numpy` 2.4.6, `matplotlib` 3.11.2, `tinygp` 0.3.1, `corner` 2.3.0, `astropy` 7.2.2 and others, including security updates for `anyio`, `tornado`, `jupyterlab` and `soupsieve`
- Classifiers: Development Status Alpha, Python 3.11-3.13
- CI: checkout, setup-python and Codecov actions v7, coverage upload re-enabled, Poetry 2.3.2
- Pre-commit hooks: `nbstripout` 0.9.1, `pre-commit-hooks` v6.0.0, `ruff` v0.16.9

## v0.4.0 (2026-03-02)

### Features
- Multi-instrument RV fitting: separate gamma offset and jitter per instrument
- AIC replaced with AICc (small-sample correction)
- Automated LaTeX parameter labels on all plots (corner, chains, autocorrelation, phase)
- `param_key_to_latex()` and `param_key_to_unit()` public utilities for custom tables and plots
- Error handling for legacy single-instrument parameters

### Performance
- Numba JIT-compiled Kepler solver
- Vectorised log-likelihood with precomputed constants and integer index arrays
- Precomputed velerr squared, log(2*pi), and parameter key strings in LogLikelihood

### Fixes
- Consistent zorder in Fitter/GPFitter plots
- Fall back to absolute perturbation when generating random walker init positions if value=0

### Tests
- GPFitter tests added
- Multi-instrument test cases
- Warnings if all params fixed
- Coverage tests ensuring new parameterisations/kernels have label and unit entries

### Docs
- Restructured RTD sidebar with captioned sections (Tutorials, API, Project)
- Fixed LaTeX rendering in all docstrings (prior.py, fit.py, model.py)
- Fixed equation rendering in GP tutorial notebook
- Updated all example notebooks for multi-instrument support
- Synced docs/requirements.txt with pyproject.toml

### Build
- Added numba dependency
- Updated numpy and other dependencies
- Updated Poetry

## v0.3.0 (2026-01-03)
- Added full Gaussian Process (GP) functionality to handle stellar activity in RV fitting
- New GP classes: GPKernel, GPLogLikelihood, GPLogPosterior, and GPFitter for GP-based MCMC fitting
- New GP kernel: quasiperiodic
- New priors: Rayleigh and VanEylen19Mixture (for eccentricity)
- Multiple MCMC initialisation methods: random within priors, point estimates, MAP ball
- Autocorrelation convergence checking for MCMC
- Custom RV calculation methods for arbitrary parameter values
- AIC/BIC/chi-squared model comparison metrics
- Improved plotting: posterior RV and phase plots with median/MAP parameters, truth value overplotting
- Comprehensive parameter validation that converts to default parameterisation
- Better type hints throughout codebase
- Faster parameter validation, also now supporting arrays
- Progress bars on computationally heavy plotting
- Custom plot titles and axis labels on most plots
- Chain plots dynamically scale based on dimensionality
- New tutorial notebook showing GP usage with K2-229 HARPS data
- Updated README reflecting GP now available in the package
- Tests for all new GP classes and RV calculation methods
- Additional parameter and prior validation tests
- Multiprocessing re-enabled for MCMC
- Enhanced ruff configuration with pydocstyle and flake8-annotations
- Updated nbstripout config and sphinx config to preserve outputs on computationally heavy notebooks
- Suppressed non-interactive backend warnings in pytest
- Updated dependencies: main changes are numpy to >=2, JAX to >=0.8.2, harmonic>=1.3.1
- Updated dev dependencies

## v0.2.5 (2025-08-29)
- Fitter params and priors handling refactored, `add_params` and `add_priors` methods removed, getters (for validation) now act on attributes directly
- Added Beta distribution prior (and already refactored it too)
- Replaced BoundedNormal prior with TruncatedNormal prior (that now integrates correctly)
- Added HalfNormal prior
- EccentricityUniform prior (renamed EccentricityPrior) now half-open interval rather than closed (inclusive) bounds
- Uniform prior raises exception if bounds are not finite
- Add prior parameterisation flexibility - you can sample in a transformed parameterisation (e.g. in secosw sesinw) but have priors on the default e and w instead
- RV posterior phase plot now works for single planets

## v0.2.4 (2025-08-11)
- Add discard_start and discard_end arguments to samples, allowing user to focus on specific part of chain
- Add plot_lnprob method to easily inspect the log-probability at each step in the chain

## v0.2.3 (2025-08-07)
- Significant performance increase by ensuring contiguous arrays in RV data
- Replaced scipy Newton-Raphson/Halley with a vectorised Halley method that is a bit faster
- Split MAP and MCMC functions (so that users can choose their own initial positions)
- MCMC sampler and lbpron wrapped in more friendly and consistent wrapper functions
- Refactor parameter validation - now done in one place immediately upon generation (no more sqrt warnings!)
- More consistent pytests for parameter validation
- Free/fixed param getters now use properties
- performance increase in calculating eccentric anomaly (reverting a previous debugging step accidentally left in)
- performance increase in parameter conversion, sqrt(e) now just calculated once and re-used
- Fix runtime warning in MAP by converting inf to 1e30 (scipy doesn't like inf)
- Optimise log-likelihood by changing equation to one total sum rather than adding two sums
- Added more tests to try and increase coverage
- Update to dependencies (primarily adding JAX and harmonic as requirements)
- various backend build improvements (poetry 2, pre-commit hooks, updaitng rtd)

## v0.2.2 (2025-02-07)
- Refactor of constant/linear/quadratic Trend into separate object
- Trend now can have any reference time t0
- Refactor of parameterisation code to be clearer (both in UX and in code)
- Add checks when using non-default parameterisation that you have passed the correct parameters
- Add checks that initial parameters are within the prior functions
- Add EccentricityPrior as useful helper fn for the user
- Add BoundedNormal prior
- Add mpsini calculation
- Add fns to get samples and posterior params from Fitter - much clearer UX than before
- Fix various typos and minor bugs in plotting functions

## v0.2.1 (2024-06-20)

- Added support for different parameterisations to Planet model
- Model now converts parameterisations automatically for you
- Added support for velocity constant offset, linear and quadratic Trends
- MAP and MCMC model fitting now works
- Added normal and uniform priors for MCMC parameters
- Added chain plots and corner plots for results of MCMC
- Example notebook on fitting a model to some data

## v0.2.0 (2024-03-08)

- Added example notebook of modelling a system
- Complete sweep through most method docstrings
- Behind-the-scenes improvements to build and documentation processes

## v0.1.0 (2024-03-04)

- First release of `ravest`!
