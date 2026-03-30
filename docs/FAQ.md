# Frequently Asked Questions

## How can I speed up the fitting?

Try enabling `multiprocessing=True` in your `Fitter`/`GPFitter` objects. Although, as parallelisation is never a perfectly linear speedup due to overheads, if you're running a lot of fits/targets simultaneously it might actually be overall quicker to run them single-threaded, but run multiple fits at the same time in different `.py` files in parallel - I tend to do this using the GNU `parallel` command distributed across multiple cores.

Although `tinygp` is written in JAX, your `GPFitter` MCMC may actually still run faster if you run it on the CPU, rather than on GPU. However for `harmonic` model comparison, that normally benefits from being run on the GPU, so you may want to investigate shuffling your active devices around. (You can use `pickle` to write and read `Fitter`/`GPFitter` objects from disk if needed.)

Also, remember that if you want to run anything JAX-y on a GPU, you may need to install a specific version of JAX first in your environment, before then installing `ravest` and `harmonic` on top - see the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html) for details for your system.

## What does Ravest stand for?

Good question - to be completely honest, originally this package was going to be a completely different project, but that idea got scrapped and I didn't want to bother registering a new repo and website. So, it can stand for anything you want! Maybe "RAdial VElocity SuiTe"? "RAdial VElocity S*** Tool"? "Ross's Awful VElocity Software"? Best answers on a postcard.

## I have another question/compliment/complaint:

Please feel free to either open a Github issue, or email me at [ross.dobson@ucl.ac.uk](mailto:ross.dobson@ucl.ac.uk) and I'll be happy to help!
