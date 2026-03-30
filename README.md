# ravest

---

**Radial Velocity fitting with Bayesian model comparison**
Features:
- Model exoplanets and host stars, to simulate RV data for given orbital and instrumental parameters
- Fit RV data with MCMC to explore posterior distributions for parameters - including Gaussian Processes for stellar activity
- Bayesian Model Comparison using the Learned Harmonic Mean Estimator [harmonic](https://astro-informatics.github.io/harmonic/)
- Visualise/animate the star's orbit (coming soon!)

Check out the tutorial notebooks at the [online documentation](https://ravest.readthedocs.io/) to see examples of all of these!

## Installation

It should be as simple as
```bash
$ pip install ravest
```

JAX is a requirement for `ravest` (and `harmonic`), so you may want to consult [the JAX installation docs](https://docs.jax.dev/en/latest/installation.html) if you want GPU or TPU support (tl;dr: install JAX first according to those instructions, then install `ravest` on top).

## Usage

For an introduction to modelling planetary and stellar data, see the  [example modelling notebook](https://ravest.readthedocs.io/en/latest/Examples/example_model.html) for `ravest.model`.

For an example of how to fit a model to RV data, see the [example fitting notebook](https://ravest.readthedocs.io/en/latest/Examples/example_fitting.html) where we fit some ELODIE data for 51 Peg b.

For an example of how to use a Gaussian Process to mitigate stellar variability, see the [example GP notebook](https://ravest.readthedocs.io/en/latest/Examples/example_gp.html) where we use a quasiperiodic kernel on HARPS data for K2-229.

For an example on using the Learned Harmonic Mean Estimator from `harmonic` to compare two competing RV models by estimating the Bayesian evidence $\mathcal{Z}$ and Bayes Factors, see the [example harmonic notebook](https://ravest.readthedocs.io/en/latest/Examples/example_harmonic.html) where we compare a one-planet and two-planet fit for TOI-544.

If you have any questions, check the [Frequently Asked Questions](https://ravest.readthedocs.io/en/latest/Examples/example_gp.html), raise an issue on Github, or email me and I'll be happy to help.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## Acknowledgements

Ravest makes use of the following open-source packages:
- [NumPy](https://numpy.org/) for numerical computing
- [SciPy](https://scipy.org/) for scientific computing algorithms
- [Matplotlib](https://matplotlib.org/) for plotting and visualisation
- [Astropy](http://www.astropy.org/) for astronomical calculations and utilities
- [pandas](https://pandas.pydata.org/) for data manipulation
- [tqdm](https://tqdm.github.io/) for progress bars
- [emcee](https://emcee.readthedocs.io/) for MCMC sampling
- [corner](https://corner.readthedocs.io/) for visualising posterior distributions
- [tinygp](https://tinygp.readthedocs.io/) for Gaussian Process modelling, which requires [JAX](https://jax.readthedocs.io/)
- [harmonic](https://astro-informatics.github.io/harmonic/) for Bayesian evidence estimation via the Learned Harmonic Mean Estimator

## License

`ravest` was created by Ross Dobson. It is licensed under the terms of the MIT license.
