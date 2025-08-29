# Changelog

## v0.2.5 (2025-08-29)
- Fitter params and priors handling refactored, `add_params` and `add_priors` methods removed, getters (for validation) now act on attributes directly
- Added Beta distribution prior (and already refactored it too)
- Replaced BoundedGaussian prior with TruncatedGaussian prior (that now integrates correctly)
- Added HalfGaussian prior
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
- Add BoundedGaussian prior
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
