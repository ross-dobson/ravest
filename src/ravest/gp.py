"""Gaussian Process kernel management for radial velocity fitting."""
# gp.py
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from tinygp import kernels
from tinygp.kernels import quasisep


@dataclass(frozen=True)
class _KernelEntry:
    """One GP kernel: its hyperparameter names, in order, and how to build it."""

    param_names: tuple[str, ...]
    build: Callable[[Mapping[str, ArrayLike]], kernels.Kernel]  # must work on traced JAX values


def _build_exponential(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Exponential kernel."""
    # LaTeX: A^2 \exp{ - \frac{|x_i - x_j|}{\lambda}}
    return jnp.square(params["gp_A"]) * kernels.Exp(scale=params["gp_lambda"])


def _build_exponential_quasiseparable(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Exponential_QuasiSeparable kernel."""
    # LaTeX: A^2 \exp{ - \frac{|x_i - x_j|}{\lambda}}, as the Exponential kernel
    return jnp.square(params["gp_A"]) * quasisep.Exp(scale=params["gp_lambda"])


def _build_matern32(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Matern32 kernel."""
    # LaTeX: A^2 (1 + \frac{\sqrt{3} |x_i - x_j|}{\lambda}) \exp{ - \frac{\sqrt{3} |x_i - x_j|}{\lambda}}
    return jnp.square(params["gp_A"]) * kernels.Matern32(scale=params["gp_lambda"])


def _build_matern32_quasiseparable(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Matern32_QuasiSeparable kernel."""
    # LaTeX: A^2 (1 + \frac{\sqrt{3} |x_i - x_j|}{\lambda}) \exp{ - \frac{\sqrt{3} |x_i - x_j|}{\lambda}}, as the Matern32 kernel
    return jnp.square(params["gp_A"]) * quasisep.Matern32(scale=params["gp_lambda"])


def _build_matern52(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Matern52 kernel."""
    # LaTeX: A^2 (1 + \frac{\sqrt{5} |x_i - x_j|}{\lambda} + \frac{5 (x_i - x_j)^2}{3 \lambda^2}) \exp{ - \frac{\sqrt{5} |x_i - x_j|}{\lambda}}
    return jnp.square(params["gp_A"]) * kernels.Matern52(scale=params["gp_lambda"])


def _build_matern52_quasiseparable(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Matern52_QuasiSeparable kernel."""
    # LaTeX: A^2 (1 + \frac{\sqrt{5} |x_i - x_j|}{\lambda} + \frac{5 (x_i - x_j)^2}{3 \lambda^2}) \exp{ - \frac{\sqrt{5} |x_i - x_j|}{\lambda}}, as the Matern52 kernel
    return jnp.square(params["gp_A"]) * quasisep.Matern52(scale=params["gp_lambda"])


def _build_squared_exponential(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the SquaredExponential kernel."""
    # LaTeX: A^2 \exp{ - \frac{(x_i - x_j)^2}{2 {\lambda_e}^2}}
    # where tinygp's "scale" = our "gp_lambda_e", as in the Quasiperiodic kernel
    return jnp.square(params["gp_A"]) * kernels.ExpSquared(scale=params["gp_lambda_e"])


def _build_periodic(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the Periodic kernel."""
    # LaTeX: A^2 \exp{ - \frac{\sin^2{\pi \frac{x_i-x_j}{P}}} {2 {\lambda_p}^2}}
    # where tinygp's "scale" P = our "gp_P" P_GP
    # where tinygp's "gamma" = 1 / (2 {\lambda_p}^2)
    gamma = 1 / (2 * jnp.square(params["gp_lambda_p"]))
    return jnp.square(params["gp_A"]) * kernels.ExpSineSquared(scale=params["gp_P"], gamma=gamma)


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


def _build_qpcosine(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the QPCosine kernel (Nicholson & Aigrain 2022 form)."""
    # LaTeX: exp{ - \frac{(x_i - x_j)^2}{2 {\lambda_e}^2}}
    exp_squared = kernels.ExpSquared(scale=params["gp_lambda_e"])

    # LaTeX: exp{ - \frac{\sin^2{\pi \frac{x_i-x_j}{P}}} {2 {\lambda_p}^2}}, as in the Quasiperiodic kernel
    gamma = 1 / (2 * jnp.square(params["gp_lambda_p"]))
    exp_sine_squared = kernels.ExpSineSquared(scale=params["gp_P"], gamma=gamma)

    # LaTeX: \cos{4 \pi \frac{x_i-x_j}{P}}
    # where tinygp's Cosine "scale" = P / 2
    cosine = kernels.Cosine(scale=params["gp_P"] / 2)

    # LaTeX: A^2 \exp{ - \frac{(x_i - x_j)^2}{2 {\lambda_e}^2}} [\exp{ - \frac{\sin^2{\pi \frac{x_i-x_j}{P}}} {2 {\lambda_p}^2}} + f \cos{4 \pi \frac{x_i-x_j}{P}}]
    return jnp.square(params["gp_A"]) * exp_squared * (exp_sine_squared + params["gp_f"] * cosine)


def _build_sho_quasiseparable(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the SHO_QuasiSeparable kernel."""
    # tinygp's omega = 2 \pi / \rho; its sigma = our gp_A, so k(0) = A^2
    return quasisep.SHO(omega=2 * jnp.pi / params["gp_rho"], quality=params["gp_Q"], sigma=params["gp_A"])


def _build_double_sho_quasiseparable(params: Mapping[str, ArrayLike]) -> kernels.Kernel:
    """Build the DoubleSHO_QuasiSeparable kernel (celerite2's RotationTerm)."""
    amp, period, f = params["gp_A"], params["gp_P"], params["gp_f"]
    # main mode: Q_1 = 1/2 + Q_0 + \delta Q, damped oscillations at P, variance A^2 / (1 + f)
    q1 = 0.5 + params["gp_Q0"] + params["gp_dQ"]
    w1 = 4 * jnp.pi * q1 / (period * jnp.sqrt(4 * jnp.square(q1) - 1))
    # half-period mode: Q_2 = 1/2 + Q_0 (also in w_2, as celerite2's code), at P / 2, variance f A^2 / (1 + f)
    q2 = 0.5 + params["gp_Q0"]
    w2 = 8 * jnp.pi * q2 / (period * jnp.sqrt(4 * jnp.square(q2) - 1))
    return (quasisep.SHO(omega=w1, quality=q1, sigma=amp / jnp.sqrt(1 + f))
            + quasisep.SHO(omega=w2, quality=q2, sigma=amp * jnp.sqrt(f / (1 + f))))


_KERNELS: Dict[str, _KernelEntry] = {
    "Exponential": _KernelEntry(("gp_A", "gp_lambda"), _build_exponential),
    "Exponential_QuasiSeparable": _KernelEntry(("gp_A", "gp_lambda"), _build_exponential_quasiseparable),
    "Matern32": _KernelEntry(("gp_A", "gp_lambda"), _build_matern32),
    "Matern32_QuasiSeparable": _KernelEntry(("gp_A", "gp_lambda"), _build_matern32_quasiseparable),
    "Matern52": _KernelEntry(("gp_A", "gp_lambda"), _build_matern52),
    "Matern52_QuasiSeparable": _KernelEntry(("gp_A", "gp_lambda"), _build_matern52_quasiseparable),
    "SquaredExponential": _KernelEntry(("gp_A", "gp_lambda_e"), _build_squared_exponential),
    "Periodic": _KernelEntry(("gp_A", "gp_P", "gp_lambda_p"), _build_periodic),
    "Quasiperiodic": _KernelEntry(("gp_A", "gp_P", "gp_lambda_e", "gp_lambda_p"), _build_quasiperiodic),
    "QPCosine": _KernelEntry(("gp_A", "gp_P", "gp_lambda_e", "gp_lambda_p", "gp_f"), _build_qpcosine),
    "SHO_QuasiSeparable": _KernelEntry(("gp_A", "gp_rho", "gp_Q"), _build_sho_quasiseparable),
    "DoubleSHO_QuasiSeparable": _KernelEntry(("gp_A", "gp_P", "gp_Q0", "gp_dQ", "gp_f"),
                                             _build_double_sho_quasiseparable),
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

    **Quasiseparable kernels.** Kernels whose name ends in ``_QuasiSeparable`` use tinygp's
    quasiseparable solver, whose cost grows linearly with the number of observations rather than
    with its cube. The solver needs the data in time order, which ``add_data`` ensures.
    ``Exponential_QuasiSeparable``, ``Matern32_QuasiSeparable`` and ``Matern52_QuasiSeparable`` are
    the same kernels as ``Exponential``, ``Matern32`` and ``Matern52``, and give the same results.

    **Exponential**

    .. math::

        k(\tau) = A^2 \exp\left(- \frac{|\tau|}{\lambda}\right)

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_lambda`` (:math:`\lambda`, d): length scale (the same value decays differently in
      Exponential, Matern32 and Matern52)

    tinygp: ``Exp(scale=gp_lambda)``, times ``gp_A^2``. celerite2's ``RealTerm`` is the same
    kernel, with ``a = gp_A^2`` and ``c = 1 / gp_lambda``.

    **Exponential_QuasiSeparable**

    The Exponential kernel, with the quasiseparable solver. ``param_names`` as Exponential:
    ``gp_A``, ``gp_lambda``.

    tinygp: ``quasisep.Exp(scale=gp_lambda)``, times ``gp_A^2``.

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

    **Matern32_QuasiSeparable**

    The Matern32 kernel, with the quasiseparable solver. ``param_names`` as Matern32:
    ``gp_A``, ``gp_lambda``.

    tinygp: ``quasisep.Matern32(scale=gp_lambda)``, times ``gp_A^2``.

    **Matern52**

    .. math::

        k(\tau) = A^2 \left(1 + \frac{\sqrt{5} |\tau|}{\lambda} + \frac{5 \tau^2}{3 \lambda^2}\right)
                  \exp\left(- \frac{\sqrt{5} |\tau|}{\lambda}\right)

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_lambda`` (:math:`\lambda`, d): length scale (the same value decays differently in
      Exponential, Matern32 and Matern52)

    tinygp: ``Matern52(scale=gp_lambda)``, times ``gp_A^2``.

    **Matern52_QuasiSeparable**

    The Matern52 kernel, with the quasiseparable solver. ``param_names`` as Matern52:
    ``gp_A``, ``gp_lambda``.

    tinygp: ``quasisep.Matern52(scale=gp_lambda)``, times ``gp_A^2``.

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

    **Periodic**

    .. math::

        k(\tau) = A^2 \exp\left[- \frac{\sin^2(\pi \tau / P)}{2 \lambda_\mathrm{p}^2}\right]

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_P`` (:math:`P`, d): period
    - ``gp_lambda_p`` (:math:`\lambda_\mathrm{p}`, dimensionless): periodic length scale
      (harmonic complexity)

    tinygp: ``ExpSineSquared(scale=gp_P, gamma=1 / (2 gp_lambda_p^2))``, times ``gp_A^2``.
    RadVel's ``PerKernel`` is the same kernel; its ``gp_length`` is ``gp_lambda_p``.

    **Quasiperiodic** (Roberts et al. 2013, with slightly different notation)

    .. math::

        k(\tau) = A^2 \exp\left[- \frac{\tau^2}{2 \lambda_\mathrm{e}^2}
                  - \frac{\sin^2(\pi \tau / P)}{2 \lambda_\mathrm{p}^2}\right]

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_P`` (:math:`P`, d): period
    - ``gp_lambda_e`` (:math:`\lambda_\mathrm{e}`, d): evolutionary (exponential) length scale
    - ``gp_lambda_p`` (:math:`\lambda_\mathrm{p}`, dimensionless): periodic length scale
      (harmonic complexity)

    tinygp: ``ExpSquared(scale=gp_lambda_e) * ExpSineSquared(scale=gp_P,
    gamma=1 / (2 gp_lambda_p^2))``, times ``gp_A^2``. RadVel's ``QuasiPerKernel`` has no factor 2
    in its decay, so its ``gp_explength`` is :math:`\sqrt{2}` times ``gp_lambda_e``; its
    ``gp_perlength`` is ``gp_lambda_p``.

    **QPCosine** (Perger et al. 2021, in the form of Nicholson & Aigrain 2022)

    .. math::

        k(\tau) = A^2 \exp\left(- \frac{\tau^2}{2 \lambda_\mathrm{e}^2}\right)
                  \left[\exp\left(- \frac{\sin^2(\pi \tau / P)}{2 \lambda_\mathrm{p}^2}\right)
                  + f \cos\left(\frac{4 \pi \tau}{P}\right)\right]

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude
    - ``gp_P``, ``gp_lambda_e``, ``gp_lambda_p``: as in the Quasiperiodic kernel
    - ``gp_f`` (:math:`f`, dimensionless): weight of the cosine term, at half the period; must
      be > 0. :math:`f \to 0` gives the Quasiperiodic kernel

    tinygp: ``ExpSquared(scale=gp_lambda_e) * (ExpSineSquared(scale=gp_P,
    gamma=1 / (2 gp_lambda_p^2)) + gp_f * Cosine(scale=gp_P / 2))``, times ``gp_A^2``.
    In Nicholson & Aigrain 2022's symbols, their :math:`A` is ``gp_A^2``, :math:`l` is
    ``gp_lambda_e`` and :math:`\Gamma` is ``1 / (2 gp_lambda_p^2)``. In Perger et al. 2021's,
    :math:`h_1` is ``gp_A``, :math:`h_2` is ``gp_A sqrt(gp_f)``, :math:`w` is ``gp_lambda_p`` and
    :math:`\lambda` is ``2 gp_lambda_e``; their recommendation of fixed :math:`w = 0.31` or priors
    between 0.2--0.5 therefore means fixing ``gp_lambda_p`` at 0.31 or using a prior between 0.2--0.5.

    **SHO_QuasiSeparable** (damped, driven simple harmonic oscillator; Foreman-Mackey et al. 2017)

    .. math::

        k(\tau) = A^2 \exp\left(-\frac{\omega_0 |\tau|}{2Q}\right) \times
        \begin{cases}
            \cosh(\eta\,\omega_0 |\tau|) + \frac{1}{2\eta Q} \sinh(\eta\,\omega_0 |\tau|),
                & \eta = \sqrt{1/(4Q^2) - 1}, \quad Q < 1/2 \\
            1 + \omega_0 |\tau|, & Q = 1/2 \\
            \cos(\eta\,\omega_0 |\tau|) + \frac{1}{2\eta Q} \sin(\eta\,\omega_0 |\tau|),
                & \eta = \sqrt{1 - 1/(4Q^2)}, \quad Q > 1/2
        \end{cases}

    with :math:`\omega_0 = 2\pi / \rho`.

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude; :math:`k(0) = A^2`
    - ``gp_rho`` (:math:`\rho`, d): undamped period, :math:`2\pi/\omega_0`. This is not the
      period of the oscillations: for :math:`Q > 1/2` they repeat every :math:`\rho / \eta`, and for
      :math:`Q \le 1/2` they do not oscillate
    - ``gp_Q`` (:math:`Q`, dimensionless): quality factor. :math:`Q < 1/2` is overdamped and
      :math:`Q > 1/2` underdamped (oscillating, for longer as :math:`Q` grows). At :math:`Q = 1/2`
      (critically damped) it is the Matern32 kernel with ``gp_lambda = sqrt(3) gp_rho / (2 pi)``

    tinygp: ``quasisep.SHO(omega=2 pi / gp_rho, quality=gp_Q, sigma=gp_A)``. celerite2's
    ``SHOTerm(sigma=gp_A, rho=gp_rho, Q=gp_Q)`` is the same kernel, with the same parameters.
    Foreman-Mackey et al. 2017, celerite and juliet use :math:`S_0`, :math:`\omega_0` and
    :math:`Q` instead, with :math:`k(0) = S_0 \omega_0 Q`; so :math:`S_0 = A^2 \rho / (2\pi Q)`,
    :math:`\omega_0 = 2\pi / \rho`, and their :math:`Q` is ``gp_Q`` (celerite takes the logarithms
    of all three). exoplanet up to version 0.3 also accepted :math:`S_\mathrm{tot} = S_0 \omega_0 Q`,
    which is ``gp_A^2``.

    **DoubleSHO_QuasiSeparable** (two damped SHOs, at the period and at half of it; celerite2's
    ``RotationTerm``)

    .. math::

        k(\tau) = k_\mathrm{SHO}(\tau;\ A_1, \omega_1, Q_1) + k_\mathrm{SHO}(\tau;\ A_2, \omega_2, Q_2)

    where :math:`k_\mathrm{SHO}` is the SHO_QuasiSeparable kernel (with :math:`\omega_0 = \omega_i`),
    and

    .. math::

        Q_1 = 1/2 + Q_0 + \delta Q, \quad
        \omega_1 = \frac{4\pi Q_1}{P\sqrt{4Q_1^2 - 1}}, \quad
        A_1^2 = \frac{A^2}{1 + f}

        Q_2 = 1/2 + Q_0, \quad
        \omega_2 = \frac{8\pi Q_2}{P\sqrt{4Q_2^2 - 1}}, \quad
        A_2^2 = \frac{f A^2}{1 + f}

    Both modes are underdamped, and oscillate at exactly :math:`P` and :math:`P/2`.

    ``param_names``, in order:

    - ``gp_A`` (:math:`A`, m/s): amplitude; :math:`k(0) = A^2`
    - ``gp_P`` (:math:`P`, d): period of the main mode; the second mode's is :math:`P/2`
    - ``gp_Q0`` (:math:`Q_0`, dimensionless): quality factor of the half-period mode, minus 1/2
    - ``gp_dQ`` (:math:`\delta Q`, dimensionless): how much higher the main mode's quality factor is
    - ``gp_f`` (:math:`f`, dimensionless): variance of the half-period mode relative to the main
      mode's (:math:`f : 1`); :math:`f \to 0` removes it

    tinygp: the sum of two ``quasisep.SHO(omega=omega_i, quality=Q_i, sigma=A_i)``. celerite2's
    ``RotationTerm(sigma=gp_A, period=gp_P, Q0=gp_Q0, dQ=gp_dQ, f=gp_f)`` is the same kernel.
    exoplanet up to version 0.3 had ``RotationTerm(amp, period, Q0, deltaQ, mix)``, with
    ``amp = gp_A^2 / (1 + gp_f)`` (the main mode's variance), ``mix = gp_f`` and
    ``deltaQ = gp_dQ``. juliet's ``CeleriteDoubleSHOKernel`` takes the same names but uses
    :math:`Q_1` in :math:`\omega_2`, so its second mode is not at exactly :math:`P/2`.
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
