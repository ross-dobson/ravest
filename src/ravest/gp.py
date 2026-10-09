"""Gaussian Process kernel management for radial velocity fitting."""
# gp.py
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from tinygp import kernels


@dataclass(frozen=True)
class _KernelEntry:
    """One GP kernel: its hyperparameter names, in order, and how to build it."""

    param_names: tuple[str, ...]
    build: Callable[[Mapping[str, ArrayLike]], kernels.Kernel]  # must work on traced JAX values


def _build_quasiperiodic(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Quasiperiodic kernel."""
    # LaTeX: exp{ - \frac{(x_i - x_j)^2}{2 {l}^2}}
    # where tinygp's "scale" l = our "gp_lambda_e"
    exp_squared = kernels.ExpSquared(scale=params["gp_lambda_e"])

    # LaTeX: exp{ - \Gamma \sin^2{\pi \frac{x_i-x_j}{P}}}
    # where tinygp's "scale" P = our "gp_P" P_GP
    # where tinygp's "gamma" = 1 / (2 {\lambda_p}^2)
    gamma = 1 / (2 * jnp.square(params["gp_lambda_p"]))
    exp_sine_squared = kernels.ExpSineSquared(scale=params["gp_P"], gamma=gamma)

    # LaTeX A^2 * \exp{ - \frac{(x_i - x_j)^2}{2 {\lambda_e}^2}} * exp{ - \frac{\sin^2{\pi \frac{x_i-x_j}{P}}} {2 {\lambda_p}^2}}
    return jnp.square(params["gp_A"]) * exp_sine_squared * exp_squared


def _build_squared_exponential(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the SquaredExponential kernel."""
    # LaTeX: A^2 \exp{ - \frac{(x_i - x_j)^2}{2 {\lambda_e}^2}}
    # where tinygp's "scale" = our "gp_lambda_e", as in the Quasiperiodic kernel
    return jnp.square(params["gp_A"]) * kernels.ExpSquared(scale=params["gp_lambda_e"])


def _build_exponential(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Exponential kernel."""
    # LaTeX: A^2 \exp{ - \frac{|x_i - x_j|}{\lambda}}
    return jnp.square(params["gp_A"]) * kernels.Exp(scale=params["gp_lambda"])


def _build_matern32(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Matern32 kernel."""
    # LaTeX: A^2 (1 + \frac{\sqrt{3} |x_i - x_j|}{\lambda}) \exp{ - \frac{\sqrt{3} |x_i - x_j|}{\lambda}}
    return jnp.square(params["gp_A"]) * kernels.Matern32(scale=params["gp_lambda"])


def _build_matern52(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Matern52 kernel."""
    # LaTeX: A^2 (1 + \frac{\sqrt{5} |x_i - x_j|}{\lambda} + \frac{5 (x_i - x_j)^2}{3 \lambda^2}) \exp{ - \frac{\sqrt{5} |x_i - x_j|}{\lambda}}
    return jnp.square(params["gp_A"]) * kernels.Matern52(scale=params["gp_lambda"])


_KERNELS: Dict[str, _KernelEntry] = {
    "Quasiperiodic": _KernelEntry(("gp_A", "gp_lambda_e", "gp_lambda_p", "gp_P"), _build_quasiperiodic),
    "SquaredExponential": _KernelEntry(("gp_A", "gp_lambda_e"), _build_squared_exponential),
    "Exponential": _KernelEntry(("gp_A", "gp_lambda"), _build_exponential),
    "Matern32": _KernelEntry(("gp_A", "gp_lambda"), _build_matern32),
    "Matern52": _KernelEntry(("gp_A", "gp_lambda"), _build_matern52),
}
SUPPORTED_KERNELS = list(_KERNELS)


class GPKernel:
    r"""A Gaussian Process kernel, for modelling correlated noise such as stellar activity.

    The kernel is chosen by name, which fixes its hyperparameters (``param_names``). They go
    into the fitter's ``params`` and ``priors`` alongside the other parameters. Every
    hyperparameter must be finite and positive.

    Notes
    -----
    The supported kernels, also listed in ``SUPPORTED_KERNELS``, are below. Each is a function
    of the time lag :math:`\tau = t_i - t_j` between two observations, and is built from tinygp's
    kernels.

    **Quasiperiodic** (Roberts et al. 2013, with slightly different notation)

    .. math::

        k(\tau) = A^2 \exp\left[- \frac{\tau^2}{2 \lambda_\mathrm{e}^2}
                  - \frac{\sin^2(\pi \tau / P)}{2 \lambda_\mathrm{p}^2}\right]

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_lambda_e`` (:math:`\lambda_\mathrm{e}`, d): evolutionary (exponential) length scale
    - ``gp_lambda_p`` (:math:`\lambda_\mathrm{p}`, dimensionless): periodic length scale
      (harmonic complexity)
    - ``gp_P`` (:math:`P`, d): period

    tinygp: ``ExpSquared(scale=gp_lambda_e) * ExpSineSquared(scale=gp_P,
    gamma=1 / (2 gp_lambda_p^2))``, times ``gp_A^2``. RadVel's ``QuasiPerKernel`` has no factor 2
    in its decay, so its ``gp_explength`` is :math:`\sqrt{2}` times ``gp_lambda_e``; its
    ``gp_perlength`` is ``gp_lambda_p``.

    **SquaredExponential**

    .. math::

        k(\tau) = A^2 \exp\left(- \frac{\tau^2}{2 \lambda_\mathrm{e}^2}\right)

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_lambda_e`` (:math:`\lambda_\mathrm{e}`, d): length scale; the Quasiperiodic kernel's
      decay is this same term, so the name and meaning are shared

    tinygp: ``ExpSquared(scale=gp_lambda_e)``, times ``gp_A^2``. RadVel's ``SqExpKernel`` and
    Pyaneti's ``SEK`` have no factor 2, so their length is :math:`\sqrt{2}` times
    ``gp_lambda_e``.

    **Exponential**

    .. math::

        k(\tau) = A^2 \exp\left(- \frac{|\tau|}{\lambda}\right)

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_lambda`` (:math:`\lambda`, d): length scale (the same value decays differently in
      Exponential, Matern32 and Matern52)

    tinygp: ``Exp(scale=gp_lambda)``, times ``gp_A^2``. celerite2's ``RealTerm`` is the same
    kernel, with ``a = gp_A^2`` and ``c = 1 / gp_lambda``.

    **Matern32**

    .. math::

        k(\tau) = A^2 \left(1 + \frac{\sqrt{3} |\tau|}{\lambda}\right)
                  \exp\left(- \frac{\sqrt{3} |\tau|}{\lambda}\right)

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_lambda`` (:math:`\lambda`, d): length scale (the same value decays differently in
      Exponential, Matern32 and Matern52)

    tinygp: ``Matern32(scale=gp_lambda)``, times ``gp_A^2``. celerite2's ``Matern32Term``
    approximates it, with ``sigma = gp_A`` and ``rho = gp_lambda``.

    **Matern52**

    .. math::

        k(\tau) = A^2 \left(1 + \frac{\sqrt{5} |\tau|}{\lambda} + \frac{5 \tau^2}{3 \lambda^2}\right)
                  \exp\left(- \frac{\sqrt{5} |\tau|}{\lambda}\right)

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_lambda`` (:math:`\lambda`, d): length scale (the same value decays differently in
      Exponential, Matern32 and Matern52)

    tinygp: ``Matern52(scale=gp_lambda)``, times ``gp_A^2``.
    """

    def __init__(self, kernel_type: str) -> None:
        """Initialize GP kernel.

        Parameters
        ----------
        kernel_type : str
            The kernel's name, exactly as in ``SUPPORTED_KERNELS`` (case matters),
            e.g. "Quasiperiodic".

        Raises
        ------
        ValueError
            If kernel_type is not supported
        """
        if kernel_type not in _KERNELS:
            raise ValueError(f"Unsupported kernel type: {kernel_type}. "
                             f"Supported kernels: {SUPPORTED_KERNELS}")
        self._kernel_type = kernel_type

    @property
    def kernel_type(self) -> str:
        """The kernel's name. Fixed when the GPKernel is created."""
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, value: str) -> None:
        """Refuse to change the kernel; make a new GPKernel instead."""
        raise AttributeError(
            "kernel_type is fixed when the GPKernel is created. Changing it can cause Bad Things "
            "to happen, so please make a new GPKernel instead."
        )

    @property
    def param_names(self) -> List[str]:
        """The kernel's hyperparameter names, in order (a new list each time)."""
        return list(_KERNELS[self._kernel_type].param_names)

    def __repr__(self) -> str:
        """Return e.g. ``GPKernel('Quasiperiodic')``."""
        return f"GPKernel({self._kernel_type!r})"

    def _validate_hyperparams_values(self, params_values: Mapping[str, float]) -> None:
        """Check that each of the kernel's hyperparameters is finite and positive.

        Parameters
        ----------
        params_values : dict
            Parameter values by name; must include every name in ``param_names``. Other keys
            are ignored.

        Raises
        ------
        ValueError
            If a hyperparameter is not finite, or not positive
        """
        for name in self.param_names:
            value = params_values[name]
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def build_kernel(self, params: Mapping[str, ArrayLike]) -> kernels.Kernel:
        """Build the tinygp kernel with the given hyperparameter values.

        Parameters
        ----------
        params : dict
            Parameter values by name; must include every name in ``param_names``. Other keys
            are ignored. Values may be traced JAX arrays.

        Returns
        -------
        tinygp kernel object
            Configured kernel (not full GaussianProcess)
        """
        return _KERNELS[self._kernel_type].build(params)
