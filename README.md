# ravest

**Ravest is under active development, with lots of new features to be released soon for public use. This website will continue to be populated with new code and examples as they get released, so please check back soon for further updates! If you have any questions or ideas, please don't hesitate to contact me.**

---

**Radial Velocity fitting, simulation and mass precision estimation**
Features:
- Model exoplanets and host stars, to simulate RV data for given orbital and instrumental parameters
- Fit RV data with MCMC to explore posterior distributions for parameters - including Gaussian Processes for stellar activity (coming soon!)
- Estimate the precision on mass estimate $M_p\sin{i}$: what precision can you expect from N observations, or how many observations are required to reach a target precision?
- Visualise/animate the star's orbit (coming soon!)
- Bayesian Model Comparison using [harmonic](https://astro-informatics.github.io/harmonic/) (coming soon!)


## Installation

```bash
$ pip install ravest
```

## Usage

For an introduction to modelling planetary and stellar data, see the  [example modelling notebook](https://ravest.readthedocs.io/en/latest/Examples/example_model.html) for `ravest.model`.

For an example of how to fit a model to RV data, see the [example fitting notebook](https://ravest.readthedocs.io/en/latest/Examples/example_fitting.html) where we fit some ELODIE data for 51 Peg b.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`ravest` was created by Ross Dobson. It is licensed under the terms of the MIT license.
