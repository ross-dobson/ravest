"""Tests for GPKernel: every kernel's numbers, the kernel table, and GPFitter with each kernel."""
import re

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from tinygp import GaussianProcess
from tinygp.kernels import quasisep

# ravest.fit switches JAX to 64-bit floats; the number checks below need it
from ravest.fit import GPFitter
from ravest.gp import SUPPORTED_KERNELS, GPKernel
from ravest.param import Parameter, Parameterisation
from ravest.prior import Uniform

# Kernels expected in the table, in table order, with their hyperparameter names in order:
# amplitude, period or timescale, decay or coherence, shape
EXPECTED_PARAM_NAMES = {
    "Exponential": ["gp_A", "gp_lambda"],
    "Exponential_QuasiSeparable": ["gp_A", "gp_lambda"],
    "Matern32": ["gp_A", "gp_lambda"],
    "Matern32_QuasiSeparable": ["gp_A", "gp_lambda"],
    "Matern52": ["gp_A", "gp_lambda"],
    "Matern52_QuasiSeparable": ["gp_A", "gp_lambda"],
    "SquaredExponential": ["gp_A", "gp_lambda_e"],
    "Periodic": ["gp_A", "gp_P", "gp_lambda_p"],
    "Quasiperiodic": ["gp_A", "gp_P", "gp_lambda_e", "gp_lambda_p"],
    "QPCosine": ["gp_A", "gp_P", "gp_lambda_e", "gp_lambda_p", "gp_f"],
    "CeleriteQP_QuasiSeparable": ["gp_A", "gp_P", "gp_lambda", "gp_C"],
    "SHO_QuasiSeparable": ["gp_A", "gp_rho", "gp_Q"],
    "DoubleSHO_QuasiSeparable": ["gp_A", "gp_P", "gp_Q0", "gp_dQ", "gp_f"],
}
EXPECTED_KERNELS = list(EXPECTED_PARAM_NAMES)
# Each quasiseparable kernel with a dense twin, and that twin
QUASISEPARABLE_TWINS = {
    "Exponential_QuasiSeparable": "Exponential",
    "Matern32_QuasiSeparable": "Matern32",
    "Matern52_QuasiSeparable": "Matern52",
}
QUASISEPARABLE_KERNELS = [k for k in SUPPORTED_KERNELS if k.endswith("_QuasiSeparable")]

# Evaluation grid: tau from T[0] covers 0, P/2 (6.25) and P (12.5)
T = np.array([0.0, 0.4, 3.5, 6.25, 7.0, 12.5, 25.0, 31.3])
TAU = np.abs(T[:, None] - T[None, :])
# One value per hyperparameter name; each kernel takes its own names from here
# gp_rho = 4 pi and gp_Q = 3 are celerite2's own SHO test values (w0 = 0.5, Q = 3); gp_Q0 = 1.3 and
# gp_dQ = 1.05 are its RotationTerm test values; gp_C = 1 is RadVel's CeleriteKernel example value
VALUES = {"gp_A": 3.0, "gp_lambda_e": 30.0, "gp_lambda_p": 0.5, "gp_P": 12.5, "gp_lambda": 7.0, "gp_f": 0.4,
          "gp_rho": 4 * np.pi, "gp_Q": 3.0, "gp_Q0": 1.3, "gp_dQ": 1.05, "gp_C": 1.0}
A, LAMBDA_E, LAMBDA_P, P = VALUES["gp_A"], VALUES["gp_lambda_e"], VALUES["gp_lambda_p"], VALUES["gp_P"]
LAMBDA = VALUES["gp_lambda"]
F = VALUES["gp_f"]
RHO = VALUES["gp_rho"]

# External reference rows k(T[0], T), shared by a dense kernel and its quasiseparable twin
# celerite2 0.3.3, RealTerm(a=gp_A^2, c=1/gp_lambda)
EXPONENTIAL_CELERITE2 = [9.0, 8.500132232953828, 5.458775937413701, 3.6853571263712785, 3.310914970542981,
                         1.5090952387661742, 0.25304093774074843, 0.10287876795769818]
# george 0.4.4, gp_A^2 * Matern32Kernel(metric=gp_lambda^2)
MATERN32_GEORGE = [9.0, 8.95872183725947, 7.063988885617055, 4.881537705462964, 4.35021952136857,
                   1.6712003954530985, 0.1331137858279411, 0.03407932627465895]
# celerite2 0.3.3, Matern32Term(sigma=gp_A, rho=gp_lambda, eps=1e-8) (its approximation, exact here to rounding)
MATERN32_CELERITE2 = [9.0, 8.95872183725947, 7.063988885617054, 4.881537705462959, 4.3502195213685635,
                      1.671200395453092, 0.13311378582793945, 0.034079326274658296]
# george 0.4.4, gp_A^2 * Matern52Kernel(metric=gp_lambda^2)
MATERN52_GEORGE = [9.0, 8.975603614820384, 7.457842281763128, 5.286630814585738, 4.715946979486382,
                   1.7110854896589327, 0.09260432403135035, 0.018138285003575023]


def kernel_matrix(kernel_type: str, values: dict = VALUES) -> np.ndarray:
    """The kernel's covariance matrix on the grid T."""
    return np.asarray(GPKernel(kernel_type).build_kernel(values)(T, T))


def kernel_values(kernel_type: str) -> dict:
    """VALUES restricted to the kernel's own hyperparameters."""
    return {name: VALUES[name] for name in GPKernel(kernel_type).param_names}


def kernel_at(kernel_type: str, tau: float, values: dict = VALUES) -> float:
    """The kernel at a single lag tau."""
    return float(GPKernel(kernel_type).build_kernel(values)(np.array([0.0]), np.array([tau]))[0, 0])


def sho(tau: np.ndarray, rho: float, q: float) -> np.ndarray:
    """The SHO kernel over A^2 (Foreman-Mackey et al. 2017), in each damping regime."""
    x = 2 * np.pi * np.abs(tau) / rho  # omega_0 tau
    envelope = np.exp(-x / (2 * q))
    if q < 0.5:
        eta = np.sqrt(1 / (4 * q**2) - 1)
        return envelope * (np.cosh(eta * x) + np.sinh(eta * x) / (2 * eta * q))
    if q > 0.5:
        eta = np.sqrt(1 - 1 / (4 * q**2))
        return envelope * (np.cos(eta * x) + np.sin(eta * x) / (2 * eta * q))
    return envelope * (1 + x)


def double_sho(tau: np.ndarray, period: float, q0: float, dq: float, f: float) -> np.ndarray:
    """The DoubleSHO kernel over A^2: SHOs at the period and half of it (celerite2's RotationTerm)."""
    q1 = 0.5 + q0 + dq
    w1 = 4 * np.pi * q1 / (period * np.sqrt(4 * q1**2 - 1))
    q2 = 0.5 + q0
    w2 = 8 * np.pi * q2 / (period * np.sqrt(4 * q2**2 - 1))
    return (sho(tau, 2 * np.pi / w1, q1) + f * sho(tau, 2 * np.pi / w2, q2)) / (1 + f)


def celerite_qp(tau: np.ndarray, period: float, lam: float, c: float) -> np.ndarray:
    """The CeleriteQP kernel over A^2 (Foreman-Mackey et al. 2017, eq. 56)."""
    return np.exp(-np.abs(tau) / lam) * (np.cos(2 * np.pi * tau / period) + 1 + c) / (2 + c)


def squared_exponential(tau: np.ndarray, lambda_e: float) -> np.ndarray:
    """exp(-tau^2 / (2 lambda_e^2))."""
    return np.exp(-tau**2 / (2 * lambda_e**2))


def periodic(tau: np.ndarray, lambda_p: float, period: float) -> np.ndarray:
    """exp(-sin^2(pi tau / P) / (2 lambda_p^2))."""
    return np.exp(-np.sin(np.pi * tau / period)**2 / (2 * lambda_p**2))


class TestQuasiperiodic:
    """The Quasiperiodic kernel's numbers."""

    def test_closed_form(self) -> None:
        """Matches A^2 exp(-tau^2/(2 lambda_e^2)) exp(-sin^2(pi tau/P)/(2 lambda_p^2))."""
        expected = A**2 * squared_exponential(TAU, LAMBDA_E) * periodic(TAU, LAMBDA_P, P)
        np.testing.assert_allclose(kernel_matrix("Quasiperiodic"), expected, rtol=1e-12)

    def test_hand_values(self) -> None:
        """A^2 at tau = 0; at tau = P only the decay is left."""
        K = kernel_matrix("Quasiperiodic")
        assert K[0, 0] == pytest.approx(A**2, rel=1e-12)
        assert K[0, 5] == pytest.approx(A**2 * np.exp(-P**2 / (2 * LAMBDA_E**2)), rel=1e-12)

    def test_radvel(self) -> None:
        """Matches RadVel's QuasiPerKernel.

        RadVel 1.6.6 QuasiPerKernel, gp_amp = gp_A, gp_explength = sqrt(2) gp_lambda_e (RadVel has
        no factor 2 in its decay), gp_perlength = gp_lambda_p, gp_per = gp_P; row k(T[0], T).
        """
        radvel = [9.0, 8.819725444842941, 2.7265517271332596, 1.191869625936874, 1.2715363188127593,
                  8.25169820158826, 6.3598345007194474, 0.7069948947122298]
        np.testing.assert_allclose(kernel_matrix("Quasiperiodic")[0], radvel, rtol=1e-12)

    def test_george(self) -> None:
        """Matches george's ExpSquared x ExpSine2.

        george 0.4.4, gp_A^2 * ExpSquaredKernel(metric=gp_lambda_e^2)
        * ExpSine2Kernel(gamma=1/(2 gp_lambda_p^2), log_period=log(gp_P)); row k(T[0], T).
        """
        george = [9.0, 8.819725444842941, 2.7265517271332604, 1.191869625936874, 1.2715363188127593,
                  8.25169820158826, 6.3598345007194474, 0.7069948947122298]
        np.testing.assert_allclose(kernel_matrix("Quasiperiodic")[0], george, rtol=1e-12)


class TestSquaredExponential:
    """The SquaredExponential kernel's numbers."""

    def test_closed_form(self) -> None:
        """Matches A^2 exp(-tau^2/(2 lambda_e^2))."""
        expected = A**2 * squared_exponential(TAU, LAMBDA_E)
        np.testing.assert_allclose(kernel_matrix("SquaredExponential"), expected, rtol=1e-12)

    def test_hand_values(self) -> None:
        """A^2 at tau = 0; A^2 exp(-1/2) at tau = lambda_e."""
        assert kernel_at("SquaredExponential", 0.0) == pytest.approx(A**2, rel=1e-12)
        assert kernel_at("SquaredExponential", LAMBDA_E) == pytest.approx(A**2 * np.exp(-0.5), rel=1e-12)

    def test_radvel(self) -> None:
        """Matches RadVel's SqExpKernel.

        RadVel 1.6.6 SqExpKernel, gp_amp = gp_A, gp_length = sqrt(2) gp_lambda_e (RadVel has no
        factor 2); row k(T[0], T).
        """
        radvel = [9.0, 8.999200035554502, 8.938957948137276, 8.806791528659051, 8.75830466752246,
                  8.25169820158826, 6.3598345007194474, 5.222375396111611]
        np.testing.assert_allclose(kernel_matrix("SquaredExponential")[0], radvel, rtol=1e-12)

    def test_george(self) -> None:
        """Matches george's ExpSquared.

        george 0.4.4, gp_A^2 * ExpSquaredKernel(metric=gp_lambda_e^2); row k(T[0], T).
        """
        george = [9.0, 8.999200035554502, 8.938957948137276, 8.806791528659051, 8.75830466752246,
                  8.25169820158826, 6.3598345007194474, 5.222375396111611]
        np.testing.assert_allclose(kernel_matrix("SquaredExponential")[0], george, rtol=1e-12)


class TestExponential:
    """The Exponential kernel's numbers."""

    def test_closed_form(self) -> None:
        """Matches A^2 exp(-tau/lambda)."""
        expected = A**2 * np.exp(-TAU / LAMBDA)
        np.testing.assert_allclose(kernel_matrix("Exponential"), expected, rtol=1e-12)

    def test_hand_values(self) -> None:
        """A^2 at tau = 0; A^2 / e at tau = lambda (T[4] = 7)."""
        K = kernel_matrix("Exponential")
        assert K[0, 0] == pytest.approx(A**2, rel=1e-12)
        assert K[0, 4] == pytest.approx(A**2 * np.exp(-1), rel=1e-12)

    def test_celerite2(self) -> None:
        """Matches celerite2's RealTerm (EXPONENTIAL_CELERITE2)."""
        np.testing.assert_allclose(kernel_matrix("Exponential")[0], EXPONENTIAL_CELERITE2, rtol=1e-12)


class TestMatern32:
    """The Matern32 kernel's numbers."""

    def test_closed_form(self) -> None:
        """Matches A^2 (1 + sqrt(3) tau/lambda) exp(-sqrt(3) tau/lambda)."""
        r = np.sqrt(3) * TAU / LAMBDA
        np.testing.assert_allclose(kernel_matrix("Matern32"), A**2 * (1 + r) * np.exp(-r), rtol=1e-12)

    def test_hand_values(self) -> None:
        """A^2 at tau = 0; A^2 (1 + sqrt(3)) exp(-sqrt(3)) at tau = lambda (T[4] = 7)."""
        K = kernel_matrix("Matern32")
        assert K[0, 0] == pytest.approx(A**2, rel=1e-12)
        assert K[0, 4] == pytest.approx(A**2 * (1 + np.sqrt(3)) * np.exp(-np.sqrt(3)), rel=1e-12)

    def test_george(self) -> None:
        """Matches george's Matern32Kernel (MATERN32_GEORGE)."""
        np.testing.assert_allclose(kernel_matrix("Matern32")[0], MATERN32_GEORGE, rtol=1e-12)

    def test_celerite2(self) -> None:
        """Matches celerite2's Matern32Term (MATERN32_CELERITE2)."""
        np.testing.assert_allclose(kernel_matrix("Matern32")[0], MATERN32_CELERITE2, rtol=1e-12)


class TestMatern52:
    """The Matern52 kernel's numbers."""

    def test_closed_form(self) -> None:
        """Matches A^2 (1 + sqrt(5) tau/lambda + 5 tau^2/(3 lambda^2)) exp(-sqrt(5) tau/lambda)."""
        r = np.sqrt(5) * TAU / LAMBDA
        expected = A**2 * (1 + r + r**2 / 3) * np.exp(-r)
        np.testing.assert_allclose(kernel_matrix("Matern52"), expected, rtol=1e-12)

    def test_hand_values(self) -> None:
        """A^2 at tau = 0; A^2 (1 + sqrt(5) + 5/3) exp(-sqrt(5)) at tau = lambda (T[4] = 7)."""
        K = kernel_matrix("Matern52")
        assert K[0, 0] == pytest.approx(A**2, rel=1e-12)
        assert K[0, 4] == pytest.approx(A**2 * (1 + np.sqrt(5) + 5 / 3) * np.exp(-np.sqrt(5)), rel=1e-12)

    def test_george(self) -> None:
        """Matches george's Matern52Kernel (MATERN52_GEORGE)."""
        np.testing.assert_allclose(kernel_matrix("Matern52")[0], MATERN52_GEORGE, rtol=1e-12)


class TestPeriodic:
    """The Periodic kernel's numbers."""

    def test_closed_form(self) -> None:
        """Matches A^2 exp(-sin^2(pi tau/P)/(2 lambda_p^2))."""
        expected = A**2 * periodic(TAU, LAMBDA_P, P)
        np.testing.assert_allclose(kernel_matrix("Periodic"), expected, rtol=1e-12)

    def test_hand_values(self) -> None:
        """A^2 at tau = 0, P and 2P; A^2 exp(-1/(2 lambda_p^2)) at tau = P/2."""
        K = kernel_matrix("Periodic")
        for i in (0, 5, 6):  # T[5] - T[0] = P, T[6] - T[0] = 2P
            assert K[0, i] == pytest.approx(A**2, rel=1e-12)
        assert K[0, 3] == pytest.approx(A**2 * np.exp(-1 / (2 * LAMBDA_P**2)), rel=1e-12)

    def test_radvel(self) -> None:
        """Matches RadVel's PerKernel.

        RadVel 1.6.6 PerKernel, gp_amp = gp_A, gp_length = gp_lambda_p, gp_per = gp_P;
        row k(T[0], T).
        """
        radvel = [9.0, 8.82050945528243, 2.7451707107888152, 1.2180175491295144, 1.3066258030222249,
                  9.0, 9.0, 1.2184022728714008]
        np.testing.assert_allclose(kernel_matrix("Periodic")[0], radvel, rtol=1e-12)

    def test_george(self) -> None:
        """Matches george's ExpSine2Kernel.

        george 0.4.4, gp_A^2 * ExpSine2Kernel(gamma=1/(2 gp_lambda_p^2), log_period=log(gp_P));
        row k(T[0], T).
        """
        george = [9.0, 8.82050945528243, 2.745170710788816, 1.2180175491295144, 1.3066258030222249,
                  9.0, 9.0, 1.2184022728714008]
        np.testing.assert_allclose(kernel_matrix("Periodic")[0], george, rtol=1e-12)


class TestQPCosine:
    """The QPCosine kernel's numbers."""

    def test_closed_form(self) -> None:
        """Matches A^2 exp(-tau^2/(2 lambda_e^2)) [exp(-sin^2(pi tau/P)/(2 lambda_p^2)) + f cos(4 pi tau/P)]."""
        expected = (A**2 * squared_exponential(TAU, LAMBDA_E)
                    * (periodic(TAU, LAMBDA_P, P) + F * np.cos(4 * np.pi * TAU / P)))
        np.testing.assert_allclose(kernel_matrix("QPCosine"), expected, rtol=1e-12)

    def test_hand_values(self) -> None:
        """A^2 (1 + f) at tau = 0; at tau = P only the decay is left; tau = P/2 by hand."""
        K = kernel_matrix("QPCosine")
        assert K[0, 0] == pytest.approx(A**2 * (1 + F), rel=1e-12)
        assert K[0, 5] == pytest.approx(A**2 * np.exp(-P**2 / (2 * LAMBDA_E**2)) * (1 + F), rel=1e-12)
        # A^2 exp(-P^2/(8 lambda_e^2)) (exp(-1/(2 lambda_p^2)) + f)
        assert K[0, 3] == pytest.approx(4.7145862374004945, rel=1e-12)

    def test_perger_2021(self) -> None:
        """Matches Perger et al. 2021's QPC kernel in their own symbols.

        h1 = gp_A, h2 = gp_A sqrt(gp_f), w = gp_lambda_p, lambda = 2 gp_lambda_e.
        """
        h1, h2, w, lam = A, A * np.sqrt(F), LAMBDA_P, 2 * LAMBDA_E
        perger = np.exp(-2 * TAU**2 / lam**2) * (
            h1**2 * np.exp(-np.sin(np.pi * TAU / P)**2 / (2 * w**2)) + h2**2 * np.cos(4 * np.pi * TAU / P))
        np.testing.assert_allclose(kernel_matrix("QPCosine"), perger, rtol=1e-12)

    def test_nicholson_aigrain_2022(self) -> None:
        """Matches Nicholson & Aigrain 2022's form in their own symbols.

        A = gp_A^2, l = gp_lambda_e, Gamma = 1/(2 gp_lambda_p^2), f = gp_f.
        """
        amplitude, length, gamma = A**2, LAMBDA_E, 1 / (2 * LAMBDA_P**2)
        nicholson_aigrain = amplitude * np.exp(-TAU**2 / (2 * length**2)) * (
            np.exp(-gamma * np.sin(np.pi * TAU / P)**2) + F * np.cos(4 * np.pi * TAU / P))
        np.testing.assert_allclose(kernel_matrix("QPCosine"), nicholson_aigrain, rtol=1e-12)

    def test_george(self) -> None:
        """Matches george's ExpSquared x (ExpSine2 + f Cosine).

        george 0.4.4, gp_A^2 * ExpSquaredKernel(metric=gp_lambda_e^2)
        * (ExpSine2Kernel(gamma=1/(2 gp_lambda_p^2), log_period=log(gp_P))
        + gp_f * CosineKernel(log_period=log(gp_P/2))); row k(T[0], T).
        """
        george = [12.6, 12.132265634256273, -0.5979414362755127, 4.714586237400494, 3.8253480516217175,
                  11.552377482223562, 8.903768301007226, 2.7933066184588924]
        np.testing.assert_allclose(kernel_matrix("QPCosine")[0], george, rtol=1e-12)


class TestExponentialQuasiSeparable:
    """The Exponential_QuasiSeparable kernel's numbers (it also equals Exponential: TestQuasiseparableTwins)."""

    def test_closed_form(self) -> None:
        """Matches A^2 exp(-tau/lambda)."""
        expected = A**2 * np.exp(-TAU / LAMBDA)
        np.testing.assert_allclose(kernel_matrix("Exponential_QuasiSeparable"), expected, rtol=1e-12)

    def test_celerite2(self) -> None:
        """Matches celerite2's RealTerm (EXPONENTIAL_CELERITE2)."""
        np.testing.assert_allclose(kernel_matrix("Exponential_QuasiSeparable")[0], EXPONENTIAL_CELERITE2,
                                   rtol=1e-12)


class TestMatern32QuasiSeparable:
    """The Matern32_QuasiSeparable kernel's numbers (it also equals Matern32: TestQuasiseparableTwins)."""

    def test_closed_form(self) -> None:
        """Matches A^2 (1 + sqrt(3) tau/lambda) exp(-sqrt(3) tau/lambda)."""
        r = np.sqrt(3) * TAU / LAMBDA
        np.testing.assert_allclose(kernel_matrix("Matern32_QuasiSeparable"), A**2 * (1 + r) * np.exp(-r), rtol=1e-12)

    def test_george(self) -> None:
        """Matches george's Matern32Kernel (MATERN32_GEORGE)."""
        np.testing.assert_allclose(kernel_matrix("Matern32_QuasiSeparable")[0], MATERN32_GEORGE, rtol=1e-12)

    def test_celerite2(self) -> None:
        """Matches celerite2's Matern32Term (MATERN32_CELERITE2)."""
        np.testing.assert_allclose(kernel_matrix("Matern32_QuasiSeparable")[0], MATERN32_CELERITE2, rtol=1e-12)


class TestMatern52QuasiSeparable:
    """The Matern52_QuasiSeparable kernel's numbers (it also equals Matern52: TestQuasiseparableTwins)."""

    def test_closed_form(self) -> None:
        """Matches A^2 (1 + sqrt(5) tau/lambda + 5 tau^2/(3 lambda^2)) exp(-sqrt(5) tau/lambda)."""
        r = np.sqrt(5) * TAU / LAMBDA
        expected = A**2 * (1 + r + r**2 / 3) * np.exp(-r)
        np.testing.assert_allclose(kernel_matrix("Matern52_QuasiSeparable"), expected, rtol=1e-12)

    def test_george(self) -> None:
        """Matches george's Matern52Kernel (MATERN52_GEORGE)."""
        np.testing.assert_allclose(kernel_matrix("Matern52_QuasiSeparable")[0], MATERN52_GEORGE, rtol=1e-12)


class TestCeleriteQPQuasiSeparable:
    """The CeleriteQP_QuasiSeparable kernel's numbers, across its range."""

    # (gp_A, gp_P, gp_lambda, gp_C): RadVel's gp_C; RadVel's k2-131_celerite.py example; strong and
    # weak oscillation; decay within one period; near-periodic
    CASES = [(3.0, 12.5, 7.0, 1.0), (30.0, 9.0, 9.0, 1.0), (3.0, 12.5, 7.0, 0.01), (3.0, 12.5, 7.0, 100.0),
             (3.0, 12.5, 1.0, 1.0), (3.0, 12.5, 1000.0, 1.0)]

    # Rows k(T[0], T) by case. RadVel 1.6.6 CeleriteKernel's own coefficients
    # (compute_real_and_complex_hparams, with gp_B = gp_A^2, gp_C, gp_L = gp_lambda, gp_Prot = gp_P),
    # evaluated by celerite 0.4.3 RealTerm + ComplexTerm and by celerite2 0.3.3 RealTerm + ComplexTerm.
    RADVEL_CELERITE1 = {
        (3.0, 12.5, 7.0, 1.0): [9.0, 8.443053993433885, 3.2982264212153, 1.2284523754570926, 1.181139684899902,
                                1.5090952387661742, 0.25304093774074843, 0.0343037527250467],
        (30.0, 9.0, 9.0, 1.0): [900.0, 849.7595748509933, 250.91606858610183, 248.37456206756963,
                                299.58903155664586, 92.30686309239053, 40.544966440301145, 9.352929202312414],
        (3.0, 12.5, 7.0, 0.01): [9.0, 8.414940830685257, 2.234075166968326, 0.01833511008144865,
                                 0.13214588749360892, 1.5090952387661742, 0.2530409377407484,
                                 0.0005279989537407451],
        (3.0, 12.5, 7.0, 100.0): [8.999999999999998, 8.49845346120324, 5.395230363407864, 3.6130952219326256,
                                  3.248274520965243, 1.5090952387661738, 0.2530409377407483,
                                  0.10086185574497313],
        (3.0, 12.5, 1.0, 1.0): [9.0, 5.992369727681222, 0.16420902432574133, 0.005791362408683127,
                                0.0029277525648926483, 3.353987854870804e-05, 1.249914947846762e-10,
                                7.653165487322128e-14],
        (3.0, 12.5, 1000.0, 1.0): [9.0, 8.935990046159747, 5.418856828090284, 2.9813084718701837,
                                   3.1882743267446054, 8.888200204444933, 8.777789208254994, 2.908472558892724],
    }
    RADVEL_CELERITE2 = {
        (3.0, 12.5, 7.0, 1.0): [9.0, 8.443053993433885, 3.2982264212153, 1.2284523754570928, 1.181139684899902,
                                1.5090952387661742, 0.25304093774074843, 0.03430375272504671],
        (30.0, 9.0, 9.0, 1.0): [900.0, 849.7595748509934, 250.9160685861018, 248.37456206756957,
                                299.58903155664586, 92.30686309239047, 40.544966440301124, 9.352929202312405],
        (3.0, 12.5, 7.0, 0.01): [9.0, 8.414940830685257, 2.2340751669683265, 0.018335110081449058,
                                 0.13214588749360945, 1.5090952387661742, 0.25304093774074843,
                                 0.0005279989537407548],
        (3.0, 12.5, 7.0, 100.0): [9.0, 8.498453461203242, 5.395230363407865, 3.613095221932626,
                                  3.2482745209652437, 1.5090952387661742, 0.2530409377407484,
                                  0.10086185574497314],
        (3.0, 12.5, 1.0, 1.0): [9.0, 5.992369727681222, 0.16420902432574136, 0.005791362408683128,
                                0.0029277525648926488, 3.353987854870804e-05, 1.249914947846762e-10,
                                7.653165487322128e-14],
        (3.0, 12.5, 1000.0, 1.0): [9.0, 8.935990046159747, 5.418856828090284, 2.981308471870184,
                                   3.188274326744606, 8.888200204444933, 8.777789208254994, 2.9084725588927243],
    }

    @staticmethod
    def values(case: tuple) -> dict:
        """The CeleriteQP's hyperparameters (gp_A, gp_P, gp_lambda, gp_C) from case."""
        return dict(zip(EXPECTED_PARAM_NAMES["CeleriteQP_QuasiSeparable"], case))

    @pytest.mark.parametrize("case", CASES)
    def test_closed_form(self, case) -> None:
        """Matches A^2 / (2 + C) exp(-tau/lambda) [cos(2 pi tau/P) + 1 + C]."""
        amp, period, lam, c = case
        np.testing.assert_allclose(kernel_matrix("CeleriteQP_QuasiSeparable", self.values(case)),
                                   amp**2 * celerite_qp(TAU, period, lam, c), rtol=1e-12, atol=1e-12 * amp**2)

    @pytest.mark.parametrize("case", CASES)
    def test_hand_values(self, case) -> None:
        """A^2 at tau = 0; the peak A^2 exp(-P/lambda) at P; A^2 C exp(-P/(2 lambda)) / (2 + C) at P/2."""
        amp, period, lam, c = case
        values = self.values(case)
        assert kernel_at("CeleriteQP_QuasiSeparable", 0.0, values) == pytest.approx(amp**2, rel=1e-12)
        assert kernel_at("CeleriteQP_QuasiSeparable", period, values) == pytest.approx(
            amp**2 * np.exp(-period / lam), rel=1e-12)
        assert kernel_at("CeleriteQP_QuasiSeparable", period / 2, values) == pytest.approx(
            amp**2 * c * np.exp(-period / (2 * lam)) / (2 + c), rel=1e-12)

    @pytest.mark.parametrize("case", CASES)
    def test_never_negative(self, case) -> None:
        """The kernel is >= 0 at every lag from 0 to 10 periods."""
        lags = np.linspace(0, 10 * case[1], 2001)
        row = np.asarray(GPKernel("CeleriteQP_QuasiSeparable").build_kernel(self.values(case))(np.array([0.0]), lags))
        assert np.all(row >= 0)

    def test_large_c_is_exponential(self) -> None:
        """As C grows the cosine term vanishes, leaving the Exponential kernel with the same gp_lambda."""
        values = {"gp_A": A, "gp_P": P, "gp_lambda": LAMBDA, "gp_C": 1e8}
        np.testing.assert_allclose(kernel_matrix("CeleriteQP_QuasiSeparable", values),
                                   kernel_matrix("Exponential", {"gp_A": A, "gp_lambda": LAMBDA}), rtol=1e-7)

    @pytest.mark.parametrize("case", CASES)
    def test_radvel_celerite(self, case) -> None:
        """Matches RadVel's CeleriteKernel coefficients, evaluated by celerite (RADVEL_CELERITE1)."""
        np.testing.assert_allclose(kernel_matrix("CeleriteQP_QuasiSeparable", self.values(case))[0],
                                   self.RADVEL_CELERITE1[case], rtol=1e-12, atol=1e-12 * case[0]**2)

    @pytest.mark.parametrize("case", CASES)
    def test_radvel_celerite2(self, case) -> None:
        """Matches RadVel's CeleriteKernel coefficients, evaluated by celerite2 (RADVEL_CELERITE2)."""
        np.testing.assert_allclose(kernel_matrix("CeleriteQP_QuasiSeparable", self.values(case))[0],
                                   self.RADVEL_CELERITE2[case], rtol=1e-12, atol=1e-12 * case[0]**2)

    @pytest.mark.parametrize("case", CASES)
    def test_positive_semidefinite(self, case) -> None:
        """The covariance of 50 unsorted times over 100 days has no negative eigenvalues."""
        times = np.random.default_rng(1).uniform(0, 100, 50)
        K = np.asarray(GPKernel("CeleriteQP_QuasiSeparable").build_kernel(self.values(case))(times, times))
        eigenvalues = np.linalg.eigvalsh(K)
        assert eigenvalues.min() >= -1e-10 * eigenvalues.max()

    @pytest.mark.parametrize("case", CASES)
    def test_log_likelihood_matches_dense(self, case) -> None:
        """GPFitter's log-likelihood (quasiseparable solver) equals the dense Gaussian log-density."""
        fitter = make_gp_fitter("CeleriteQP_QuasiSeparable", gp_values=self.values(case), by_instrument=True)
        values = all_values(fitter)
        assert fitter.calculate_log_likelihood(values) == pytest.approx(dense_log_likelihood(fitter, values), rel=1e-12)


class TestSHOQuasiSeparable:
    """The SHO_QuasiSeparable kernel's numbers, in every damping regime."""

    # Quality factors where tinygp's SHO is exact: strongly overdamped, overdamped, just outside
    # tinygp's critical band on either side, exactly critical, and underdamped up to near-periodic
    QUALITIES = [0.05, 0.2, 0.5 - 1e-5, 0.5, 0.5 + 1e-5, 1.5, 3.0, 100.0]
    # Inside tinygp's critical band (jnp.allclose(Q, 0.5)) it uses the exact-critical formula
    BAND = [0.5 - 2e-6, 0.5 + 2e-6]

    # Rows k(T[0], T) with gp_A = 3, gp_rho = 4 pi, by Q. celerite2 0.3.3,
    # SHOTerm(sigma=gp_A, rho=gp_rho, Q=gp_Q); celerite 0.4.3, SHOTerm(log_S0, log_Q, log_omega0)
    # with S0 = gp_A^2 gp_rho / (2 pi gp_Q) and omega0 = 2 pi / gp_rho.
    CELERITE2 = {
        0.01: [9.0, 8.982914662410812, 8.844739291998597, 8.723944414155637, 8.691287626347775,
               8.455510433126497, 7.943167298752427, 7.696833028177737],
        0.05: [8.999999999999998, 8.932249255137219, 8.264923432801416, 7.7144695471970275, 7.570814190602953,
               6.5959454664878105, 4.82190923347377, 4.117615464355147],
        0.2: [9.0, 8.867973016644886, 6.530620734959786, 4.901479471618055, 4.532483652472385,
              2.5531085148255284, 0.6927131357077085, 0.35894626135277635],
        1.5: [9.0, 8.828308574277319, 1.3731534568523922, -2.8975025053712353, -2.9237560075650246,
              0.8853509193434265, 0.06433467247157039, -0.01421088364250517],
        3.0: [9.0, 8.824517698789727, 0.08673705930980115, -5.282051067032682, -5.042287190028831,
              3.0881436059490284, 1.0429416412285752, -0.6070432085546765],
        100.0: [9.0, 8.82071866151994, -1.546152484550615, -8.858505697423153, -8.297546822770572,
                8.71682314106366, 8.433204629179453, -8.306108647816629],
    }
    CELERITE1 = {
        0.01: [9.000000000000007, 8.982914662410819, 8.844739291998604, 8.723944414155646, 8.691287626347782,
               8.455510433126502, 7.943167298752433, 7.696833028177744],
        0.05: [9.000000000000002, 8.932249255137222, 8.26492343280142, 7.71446954719703, 7.570814190602958,
               6.595945466487814, 4.821909233473772, 4.117615464355149],
        0.2: [9.0, 8.867973016644886, 6.530620734959786, 4.901479471618055, 4.532483652472385,
              2.5531085148255284, 0.6927131357077085, 0.35894626135277635],
        1.5: [9.0, 8.828308574277319, 1.3731534568523922, -2.897502505371235, -2.9237560075650246,
              0.8853509193434266, 0.06433467247157039, -0.01421088364250517],
        3.0: [9.000000000000002, 8.824517698789728, 0.08673705930980087, -5.282051067032682, -5.042287190028831,
              3.088143605949028, 1.0429416412285748, -0.6070432085546764],
        100.0: [9.000000000000005, 8.820718661519944, -1.5461524845506158, -8.858505697423157, -8.297546822770576,
                8.716823141063664, 8.433204629179459, -8.306108647816636],
    }

    @staticmethod
    def values(q: float) -> dict:
        """The SHO's own VALUES, with quality factor q."""
        return kernel_values("SHO_QuasiSeparable") | {"gp_Q": q}

    @pytest.mark.parametrize("q", QUALITIES)
    def test_closed_form(self, q) -> None:
        """Matches the closed form in the kernel's own damping regime."""
        np.testing.assert_allclose(kernel_matrix("SHO_QuasiSeparable", self.values(q)), A**2 * sho(TAU, RHO, q),
                                   rtol=1e-12, atol=1e-12 * A**2)

    @pytest.mark.parametrize("q", BAND)
    def test_inside_critical_band(self, q) -> None:
        """Within tinygp's critical band the exact-critical formula stands in, to within 1e-5 A^2."""
        np.testing.assert_allclose(kernel_matrix("SHO_QuasiSeparable", self.values(q)), A**2 * sho(TAU, RHO, q),
                                   rtol=0, atol=1e-5 * A**2)

    @pytest.mark.parametrize("q", [0.5 - 1e-5, 0.5 - 2e-6, 0.5 + 2e-6, 0.5 + 1e-5])
    def test_continuous_across_critical(self, q) -> None:
        """Just off Q = 1/2 the kernel is within 1e-4 A^2 of the critical kernel: no jump at the branches."""
        np.testing.assert_allclose(kernel_matrix("SHO_QuasiSeparable", self.values(q)),
                                   kernel_matrix("SHO_QuasiSeparable", self.values(0.5)), rtol=0, atol=1e-4 * A**2)

    @pytest.mark.parametrize("q", QUALITIES + BAND)
    def test_hand_value(self, q) -> None:
        """A^2 at tau = 0."""
        assert kernel_at("SHO_QuasiSeparable", 0.0, self.values(q)) == pytest.approx(A**2, rel=1e-12)

    def test_critical_is_matern32(self) -> None:
        """At Q = 1/2 it is the Matern32 kernel with gp_lambda = sqrt(3) gp_rho / (2 pi)."""
        matern32 = {"gp_A": A, "gp_lambda": np.sqrt(3) * RHO / (2 * np.pi)}
        np.testing.assert_allclose(kernel_matrix("SHO_QuasiSeparable", self.values(0.5)),
                                   kernel_matrix("Matern32", matern32), rtol=1e-12)

    @pytest.mark.parametrize("q", [0.05, 0.2, 1.5, 3.0, 100.0])
    def test_celerite2(self, q) -> None:
        """Matches celerite2's SHOTerm (CELERITE2)."""
        np.testing.assert_allclose(kernel_matrix("SHO_QuasiSeparable", self.values(q))[0], self.CELERITE2[q],
                                   rtol=1e-12, atol=1e-12 * A**2)

    @pytest.mark.parametrize("q", [0.05, 0.2, 1.5, 3.0, 100.0])
    def test_celerite(self, q) -> None:
        """Matches celerite's SHOTerm (CELERITE1), checking the S0 and omega0 conversions."""
        np.testing.assert_allclose(kernel_matrix("SHO_QuasiSeparable", self.values(q))[0], self.CELERITE1[q],
                                   rtol=1e-12, atol=1e-12 * A**2)

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
        "tinygp's quasiseparable SHO returns NaN here: for Q < 1/2 at long lags its cosh and sinh "
        "terms overflow, where celerite and celerite2 stay finite"))
    def test_strongly_overdamped_at_long_lags(self) -> None:
        """At Q = 0.01 the lags up to 31.3 d match celerite2 and celerite."""
        row = kernel_matrix("SHO_QuasiSeparable", self.values(0.01))[0]
        np.testing.assert_allclose(row, self.CELERITE2[0.01], rtol=1e-12)
        np.testing.assert_allclose(row, self.CELERITE1[0.01], rtol=1e-12)

    @pytest.mark.parametrize("q", QUALITIES)
    def test_positive_semidefinite(self, q) -> None:
        """The covariance of 50 unsorted times over 100 days has no negative eigenvalues."""
        times = np.random.default_rng(1).uniform(0, 100, 50)
        K = np.asarray(GPKernel("SHO_QuasiSeparable").build_kernel(self.values(q))(times, times))
        eigenvalues = np.linalg.eigvalsh(K)
        assert eigenvalues.min() >= -1e-10 * eigenvalues.max()

    @pytest.mark.parametrize("q", QUALITIES)
    def test_log_likelihood_matches_dense(self, q) -> None:
        """GPFitter's log-likelihood equals the dense Gaussian log-density of the kernel's matrix.

        GPFitter goes through tinygp's quasiseparable solver; the expected value solves the
        kernel's own covariance matrix densely.
        """
        fitter = make_gp_fitter("SHO_QuasiSeparable", gp_values=self.values(q), by_instrument=True)
        values = all_values(fitter)
        assert fitter.calculate_log_likelihood(values) == pytest.approx(dense_log_likelihood(fitter, values), rel=1e-12)


class TestDoubleSHOQuasiSeparable:
    """The DoubleSHO_QuasiSeparable kernel's numbers, across its range."""

    # (gp_Q0, gp_dQ, gp_f): celerite2's test values; a heavily damped half-period mode; Q2 just
    # outside and inside tinygp's critical band; both modes inside it; nearly equal modes;
    # near-periodic; a weak and a strong (f > 1) harmonic
    CASES = [(1.3, 1.05, 0.4), (0.01, 1.05, 0.4), (1e-5, 1.05, 0.4), (2e-6, 1.05, 0.4), (1e-6, 1e-6, 0.4),
             (1.3, 1e-3, 0.4), (100.0, 100.0, 0.4), (1.3, 1.05, 0.01), (1.3, 1.05, 2.0)]

    # Rows k(T[0], T) with gp_A = 3, gp_P = 12.5, by (gp_Q0, gp_dQ, gp_f). celerite2 0.3.3,
    # RotationTerm(sigma=gp_A, period=gp_P, Q0=gp_Q0, dQ=gp_dQ, f=gp_f). (Inside tinygp's critical
    # band celerite2 regularises sqrt(4 Q^2 - 1), so those cases are checked against the closed form only.)
    CELERITE2 = {
        (1.3, 1.05, 0.4): [9.0, 8.66372065809558, -1.021266176490645, -3.2546871495147824, -3.1066251470679953,
                           2.1661217305904894, 0.6866103983853244, -0.3908172261104803],
        (0.01, 1.05, 0.4): [9.0, 7.287845394753511, 0.5139512460324246, -2.2205439094640607, -2.0608194587597906,
                            0.767015706155726, 0.09151537009837708, -0.03159998755548291],
        (1e-5, 1.05, 0.4): [9.0, 6.290459398037571, 0.520192135512206, -2.2036347104760066, -2.044987283941566,
                            0.7553787013445046, 0.08875953060254158, -0.030415004699495714],
        (1.3, 1e-3, 0.4): [9.0, 8.659186318833077, -0.5902566881693245, -2.175216843827629, -2.098075764128939,
                           1.1139794140252084, 0.17200495750458888, -0.06834519216087862],
        (100.0, 100.0, 0.4): [9.000000000000002, 8.665688667201639, -3.537634842937731, -3.8861150021887494,
                              -4.112312850908083, 8.744213417391242, 8.499425628977061, -3.9830952339897694],
        (1.3, 1.05, 0.01): [9.0, 8.812020210775717, -0.1137190301374656, -5.07631495123388, -4.727230078709436,
                            2.9107313118237035, 0.949311531783743, -0.542120579956591],
        (1.3, 1.05, 2.0): [9.0, 8.458889822940792, -2.2747671923083694, -0.7386610575677225, -0.8682511560145182,
                           1.1376695226649567, 0.3237684910589618, -0.1818375510546557],
    }

    @staticmethod
    def values(case: tuple) -> dict:
        """The DoubleSHO's own VALUES, with (gp_Q0, gp_dQ, gp_f) from case."""
        q0, dq, f = case
        return kernel_values("DoubleSHO_QuasiSeparable") | {"gp_Q0": q0, "gp_dQ": dq, "gp_f": f}

    @pytest.mark.parametrize("case", CASES)
    def test_closed_form(self, case) -> None:
        """Matches the sum of the two SHOs."""
        np.testing.assert_allclose(kernel_matrix("DoubleSHO_QuasiSeparable", self.values(case)),
                                   A**2 * double_sho(TAU, P, *case), rtol=1e-12, atol=1e-12 * A**2)

    @pytest.mark.parametrize("case", CASES)
    def test_hand_value(self, case) -> None:
        """A^2 at tau = 0."""
        assert kernel_at("DoubleSHO_QuasiSeparable", 0.0, self.values(case)) == pytest.approx(A**2, rel=1e-12)

    @pytest.mark.parametrize("case", CASES)
    def test_modes_at_period_and_half_period(self, case) -> None:
        """The modes oscillate at 2 pi / P and 4 pi / P, with variances A^2 / (1 + f) and f A^2 / (1 + f)."""
        f = case[2]
        kernel = GPKernel("DoubleSHO_QuasiSeparable").build_kernel(self.values(case))
        for mode, frequency, variance in ((kernel.kernel1, 2 * np.pi / P, A**2 / (1 + f)),
                                          (kernel.kernel2, 4 * np.pi / P, f * A**2 / (1 + f))):
            assert float(mode.omega * np.sqrt(1 - 1 / (4 * mode.quality**2))) == pytest.approx(frequency, rel=1e-9)
            assert float(mode.sigma**2) == pytest.approx(variance, rel=1e-12)

    def test_exoplanet_form(self) -> None:
        """Matches exoplanet's (up to 0.3) RotationTerm in its own symbols.

        amp = gp_A^2 / (1 + gp_f) is the main mode's variance, mix = gp_f, deltaQ = gp_dQ; each
        SHO in S0, w0, Q form, with k(0) = S0 w0 Q.
        """
        q0, delta_q, mix = VALUES["gp_Q0"], VALUES["gp_dQ"], F
        amp, period = A**2 / (1 + mix), P
        q1 = 0.5 + q0 + delta_q
        w1 = 4 * np.pi * q1 / (period * np.sqrt(4 * q1**2 - 1))
        s1 = amp / (w1 * q1)
        q2 = 0.5 + q0
        w2 = 8 * np.pi * q2 / (period * np.sqrt(4 * q2**2 - 1))
        s2 = mix * amp / (w2 * q2)
        exoplanet = s1 * w1 * q1 * sho(TAU, 2 * np.pi / w1, q1) + s2 * w2 * q2 * sho(TAU, 2 * np.pi / w2, q2)
        np.testing.assert_allclose(kernel_matrix("DoubleSHO_QuasiSeparable"), exoplanet, rtol=1e-12, atol=1e-12 * A**2)

    @pytest.mark.parametrize("case", list(CELERITE2))
    def test_celerite2(self, case) -> None:
        """Matches celerite2's RotationTerm (CELERITE2)."""
        np.testing.assert_allclose(kernel_matrix("DoubleSHO_QuasiSeparable", self.values(case))[0],
                                   self.CELERITE2[case], rtol=1e-12, atol=1e-12 * A**2)

    @pytest.mark.parametrize("case", CASES)
    def test_positive_semidefinite(self, case) -> None:
        """The covariance of 50 unsorted times over 100 days has no negative eigenvalues."""
        times = np.random.default_rng(1).uniform(0, 100, 50)
        K = np.asarray(GPKernel("DoubleSHO_QuasiSeparable").build_kernel(self.values(case))(times, times))
        eigenvalues = np.linalg.eigvalsh(K)
        assert eigenvalues.min() >= -1e-10 * eigenvalues.max()

    @pytest.mark.parametrize("case", CASES)
    def test_log_likelihood_matches_dense(self, case) -> None:
        """GPFitter's log-likelihood (quasiseparable solver) equals the dense Gaussian log-density."""
        fitter = make_gp_fitter("DoubleSHO_QuasiSeparable", gp_values=self.values(case), by_instrument=True)
        values = all_values(fitter)
        assert fitter.calculate_log_likelihood(values) == pytest.approx(dense_log_likelihood(fitter, values), rel=1e-12)


class TestKernelTable:
    """The kernel table and GPKernel's public surface, for every kernel."""

    def test_supported_kernels(self) -> None:
        """SUPPORTED_KERNELS lists every kernel, in table order."""
        assert SUPPORTED_KERNELS == EXPECTED_KERNELS

    @pytest.mark.parametrize("kernel_type", EXPECTED_KERNELS)
    def test_param_names(self, kernel_type) -> None:
        """Each kernel's hyperparameter names, in order."""
        assert GPKernel(kernel_type).param_names == EXPECTED_PARAM_NAMES[kernel_type]

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_quasiseparable_suffix(self, kernel_type) -> None:
        """A name ends in _QuasiSeparable exactly when the kernel uses tinygp's quasiseparable solver."""
        kernel = GPKernel(kernel_type).build_kernel(kernel_values(kernel_type))
        assert kernel_type.endswith("_QuasiSeparable") == isinstance(kernel, quasisep.Quasisep)

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_param_names_invariants(self, kernel_type) -> None:
        """Names start with gp_, are unique, and the amplitude gp_A comes first."""
        names = GPKernel(kernel_type).param_names
        assert all(name.startswith("gp_") for name in names)
        assert len(set(names)) == len(names)
        assert names[0] == "gp_A"

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_param_names_is_a_fresh_list(self, kernel_type) -> None:
        """Changing the returned list does not change the kernel."""
        kernel = GPKernel(kernel_type)
        names = kernel.param_names
        assert isinstance(names, list)
        names.append("gp_extra")
        assert "gp_extra" not in kernel.param_names

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_param_names_is_read_only(self, kernel_type) -> None:
        """param_names can't be assigned."""
        kernel = GPKernel(kernel_type)
        with pytest.raises(AttributeError, match="param_names"):
            kernel.param_names = ["gp_A"]

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_kernel_type_is_read_only(self, kernel_type) -> None:
        """kernel_type can't be changed after the kernel is created."""
        kernel = GPKernel(kernel_type)
        assert kernel.kernel_type == kernel_type
        message = ("kernel_type is fixed when the GPKernel is created. Changing it can cause Bad Things "
                   "to happen, so please make a new GPKernel instead.")
        with pytest.raises(AttributeError, match=re.escape(message)):
            kernel.kernel_type = "Quasiperiodic"
        assert kernel.kernel_type == kernel_type

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_repr(self, kernel_type) -> None:
        """Repr is quoted and evaluates back to the same kernel."""
        kernel = GPKernel(kernel_type)
        assert repr(kernel) == f"GPKernel('{kernel_type}')"
        assert eval(repr(kernel)).kernel_type == kernel_type

    @pytest.mark.parametrize("bad_name", ["invalid_kernel", *[k.lower() for k in SUPPORTED_KERNELS]])
    def test_unsupported_name_lists_every_kernel(self, bad_name) -> None:
        """An unknown name, or a known one in the wrong case, raises listing every kernel."""
        with pytest.raises(ValueError, match=f"Unsupported kernel type: {bad_name}") as excinfo:
            GPKernel(bad_name)
        for kernel_type in SUPPORTED_KERNELS:
            assert f"'{kernel_type}'" in str(excinfo.value)

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_valid_values_pass(self, kernel_type) -> None:
        """Finite, positive values for every name pass the value check."""
        GPKernel(kernel_type)._validate_hyperparams_values(kernel_values(kernel_type))

    @pytest.mark.parametrize("bad_value, problem", [
        (0.0, "positive"), (-1.0, "positive"), (np.nan, "finite"), (np.inf, "finite"), (-np.inf, "finite"),
    ])
    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_invalid_value_names_the_hyperparameter(self, kernel_type, bad_value, problem) -> None:
        """Each name, set to zero, negative or non-finite, raises naming it."""
        kernel = GPKernel(kernel_type)
        for name in kernel.param_names:
            values = kernel_values(kernel_type) | {name: bad_value}
            with pytest.raises(ValueError, match=re.escape(f"{name} must be {problem}, got {bad_value}")):
                kernel._validate_hyperparams_values(values)

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_matrix_is_positive_semidefinite(self, kernel_type) -> None:
        """The covariance of 50 unsorted times over 100 days has no negative eigenvalues."""
        times = np.random.default_rng(1).uniform(0, 100, 50)
        K = np.asarray(GPKernel(kernel_type).build_kernel(kernel_values(kernel_type))(times, times))
        eigenvalues = np.linalg.eigvalsh(K)
        assert eigenvalues.min() >= -1e-10 * eigenvalues.max()

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_build_kernel_traces_under_jit(self, kernel_type) -> None:
        """build_kernel works on traced JAX values and gives the same matrix."""
        kernel = GPKernel(kernel_type)

        @jax.jit
        def matrix(values):
            return kernel.build_kernel(values)(T, T)

        values = kernel_values(kernel_type)
        np.testing.assert_allclose(np.asarray(matrix(values)), kernel_matrix(kernel_type, values), rtol=1e-12)


def make_gp_fitter(kernel_type: str, gp_values: dict | None = None, planet: bool = False,
                   by_instrument: bool = False) -> GPFitter:
    """A GPFitter on two instruments, with its GP hyperparameters free.

    gp_values are the GP hyperparameters for the first params assignment (default: the
    kernel's own names, from VALUES). planet adds a planet b with only K_b free. by_instrument
    passes the data instrument by instrument, so out of time order.
    """
    rng = np.random.default_rng(2)
    time = np.sort(rng.uniform(0, 60, 24))
    instrument = np.array(["HARPS", "HIRES"] * 12)
    velerr = rng.uniform(0.8, 1.6, 24)
    vel = 3.0 * np.sin(2 * np.pi * time / 12.5) + rng.normal(0, 1, 24) + np.where(instrument == "HARPS", 5.0, -2.0)
    if planet:
        vel += 4.0 * np.sin(2 * np.pi * time / 7.3)
    if by_instrument:
        order = np.argsort(instrument, kind="stable")
        time, vel, velerr, instrument = time[order], vel[order], velerr[order], instrument[order]
    fitter = GPFitter(["b"] if planet else [], Parameterisation("P K e w Tp"), GPKernel(kernel_type))
    fitter.add_data(time, vel, velerr, instrument, t0=30.0)
    params = {
        "g_HARPS": Parameter(5.0, fixed=False), "jit_HARPS": Parameter(0.5, fixed=False),
        "g_HIRES": Parameter(-2.0, fixed=False), "jit_HIRES": Parameter(0.7, fixed=False),
        "gd": Parameter(0.0, fixed=True), "gdd": Parameter(0.0, fixed=True),
    }
    if planet:
        params |= {"P_b": Parameter(7.3, fixed=True), "K_b": Parameter(4.0, fixed=False),
                   "e_b": Parameter(0.0, fixed=True), "w_b": Parameter(np.pi / 2, fixed=True),
                   "Tp_b": Parameter(0.0, fixed=True)}
    gp_values = kernel_values(kernel_type) if gp_values is None else gp_values
    params |= {name: Parameter(value, fixed=False) for name, value in gp_values.items()}
    fitter.params = params
    # Uniform(0, 100) for the rest, widened for any value it would not cover
    fitter.priors = {name: Uniform(-50, 50) if name.startswith("g_")
                     else Uniform(0, max(100.0, 10 * fitter.params[name].value))
                     for name in fitter.free_params_names}
    return fitter


class TestGPFitterWithEachKernel:
    """GPFitter takes each kernel's names and uses its covariance."""

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_gp_names_come_last_in_kernel_order(self, kernel_type) -> None:
        """The GP block of free_params_names is the kernel's param_names."""
        fitter = make_gp_fitter(kernel_type)
        names = GPKernel(kernel_type).param_names
        assert fitter.free_params_names[-len(names):] == names

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_other_kernels_names_rejected(self, kernel_type) -> None:
        """A first params assignment with another kernel's hyperparameter names raises."""
        own = GPKernel(kernel_type).param_names
        others = [k for k in SUPPORTED_KERNELS if GPKernel(k).param_names != own]
        if not others:
            pytest.skip("no other kernel with different names in the table yet")
        for other in others:
            other_values = {name: VALUES[name] for name in GPKernel(other).param_names}
            with pytest.raises(ValueError, match="parameters"):
                make_gp_fitter(kernel_type, gp_values=other_values)

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_log_likelihood_matches_tinygp_by_hand(self, kernel_type) -> None:
        """calculate_log_likelihood is the tinygp GP log probability of the gamma-subtracted data."""
        fitter = make_gp_fitter(kernel_type)
        values = {name: param.value for name, param in fitter.params.items()}
        is_harps = fitter.instrument == "HARPS"
        gamma = np.where(is_harps, values["g_HARPS"], values["g_HIRES"])
        jitter = np.where(is_harps, values["jit_HARPS"], values["jit_HIRES"])
        gp = GaussianProcess(GPKernel(kernel_type).build_kernel(values), fitter.time,
                             diag=fitter.velerr**2 + jitter**2)
        expected = float(gp.log_probability(fitter.vel - gamma))
        assert fitter.calculate_log_likelihood(values) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("kernel_type", SUPPORTED_KERNELS)
    def test_log_posterior_rejects_invalid_values(self, kernel_type) -> None:
        """The log posterior is -inf when any hyperparameter is negative."""
        fitter = make_gp_fitter(kernel_type)
        log_posterior = fitter._build_log_posterior()
        free = {name: fitter.params[name].value for name in fitter.free_params_names}
        assert np.isfinite(log_posterior.log_probability(free))
        for name in GPKernel(kernel_type).param_names:
            assert log_posterior.log_probability(free | {name: -1.0}) == -np.inf


def all_values(fitter: GPFitter) -> dict:
    """Every parameter's value, by name."""
    return {name: param.value for name, param in fitter.params.items()}


def dense_log_likelihood(fitter: GPFitter, values: dict) -> float:
    """The Gaussian log-density of the fitter's data, solving the kernel's covariance matrix densely."""
    is_harps = fitter.instrument == "HARPS"
    residuals = fitter.vel - np.where(is_harps, values["g_HARPS"], values["g_HIRES"])
    jitter = np.where(is_harps, values["jit_HARPS"], values["jit_HIRES"])
    cov = (np.asarray(fitter.gp_kernel.build_kernel(values)(fitter.time, fitter.time))
           + np.diag(fitter.velerr**2 + jitter**2))
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (residuals @ np.linalg.solve(cov, residuals) + logdet + len(residuals) * np.log(2 * np.pi))


class TestQuasiseparableTwins:
    """Each quasiseparable twin gives its dense kernel's numbers, on data in any order."""

    @pytest.mark.parametrize("twin, dense", QUASISEPARABLE_TWINS.items())
    def test_matrix_equals_dense(self, twin, dense) -> None:
        """Same covariance as the dense kernel, on the grid and on unsorted times."""
        np.testing.assert_allclose(kernel_matrix(twin), kernel_matrix(dense), rtol=1e-12)
        times = np.random.default_rng(1).uniform(0, 100, 50)
        values = kernel_values(dense)
        np.testing.assert_allclose(np.asarray(GPKernel(twin).build_kernel(values)(times, times)),
                                   np.asarray(GPKernel(dense).build_kernel(values)(times, times)), rtol=1e-12)

    @pytest.mark.parametrize("twin, dense", QUASISEPARABLE_TWINS.items())
    def test_log_likelihood_equals_dense(self, twin, dense) -> None:
        """Data given instrument by instrument give the dense kernel's log-likelihood."""
        fitter, dense_fitter = (make_gp_fitter(k, by_instrument=True) for k in (twin, dense))
        values = all_values(fitter)
        assert fitter.calculate_log_likelihood(values) == pytest.approx(
            dense_fitter.calculate_log_likelihood(values), rel=1e-12)

    @pytest.mark.parametrize("twin, dense", QUASISEPARABLE_TWINS.items())
    def test_prediction_equals_dense(self, twin, dense) -> None:
        """calculate_rv_gp_custom at unsorted times gives the dense kernel's prediction."""
        fitter, dense_fitter = (make_gp_fitter(k, planet=True, by_instrument=True) for k in (twin, dense))
        times = np.random.default_rng(5).uniform(-10, 70, 40)
        values = all_values(fitter)
        np.testing.assert_allclose(fitter.calculate_rv_gp_custom(times, values),
                                   dense_fitter.calculate_rv_gp_custom(times, values), rtol=1e-10, atol=1e-10)


class TestQuasiseparablePredictions:
    """GPFitter's predictions and plots work with every quasiseparable kernel."""

    @pytest.mark.parametrize("kernel_type", QUASISEPARABLE_KERNELS)
    def test_prediction_at_unsorted_times(self, kernel_type) -> None:
        """calculate_rv_gp_custom at unsorted times is finite."""
        fitter = make_gp_fitter(kernel_type, planet=True, by_instrument=True)
        times = np.random.default_rng(5).uniform(-10, 70, 40)
        assert np.all(np.isfinite(fitter.calculate_rv_gp_custom(times, all_values(fitter))))

    @pytest.mark.parametrize("kernel_type", QUASISEPARABLE_KERNELS)
    def test_map_mcmc_and_plots_run(self, kernel_type) -> None:
        """MAP, a short MCMC, and the MAP, posterior and best-sample plots all run."""
        matplotlib.use("Agg")
        fitter = make_gp_fitter(kernel_type, planet=True, by_instrument=True)
        map_result = fitter.find_map_estimate()
        fitter.plot_MAP_rv(map_result, n_smooth=50)
        fitter.plot_MAP_phase("b", map_result, n_smooth=50)
        nwalkers = 2 * fitter.ndim + 2
        initial_positions = fitter.generate_initial_walker_positions_from_map(map_result, nwalkers=nwalkers)
        fitter.run_mcmc(initial_positions, nwalkers=nwalkers, max_steps=30, progress=False)
        # One kept step (of 30) keeps the per-sample GP predictions in the posterior plots few
        fitter.plot_posterior_rv(discard_start=25, thin=5, n_smooth=50)
        fitter.plot_posterior_phase("b", discard_start=25, thin=5, n_smooth=50)
        fitter.plot_best_sample_rv(discard_start=25, n_smooth=50)
        plt.close("all")
