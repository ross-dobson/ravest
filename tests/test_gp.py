"""Tests for GPKernel: every kernel's numbers, the kernel table, and GPFitter with each kernel."""
import re

import jax
import numpy as np
import pytest
from tinygp import GaussianProcess

# ravest.fit switches JAX to 64-bit floats; the number checks below need it
from ravest.fit import GPFitter
from ravest.gp import SUPPORTED_KERNELS, GPKernel
from ravest.param import Parameter, Parameterisation
from ravest.prior import Uniform

# Kernels expected in the table, in table order
EXPECTED_KERNELS = ["Quasiperiodic", "SquaredExponential", "Exponential"]

# Evaluation grid: tau from T[0] covers 0, P/2 (6.25) and P (12.5)
T = np.array([0.0, 0.4, 3.5, 6.25, 7.0, 12.5, 25.0, 31.3])
TAU = np.abs(T[:, None] - T[None, :])
# One value per hyperparameter name; each kernel takes its own names from here
VALUES = {"gp_A": 3.0, "gp_lambda_e": 30.0, "gp_lambda_p": 0.5, "gp_P": 12.5, "gp_lambda": 7.0, "gp_f": 0.4}
A, LAMBDA_E, LAMBDA_P, P = VALUES["gp_A"], VALUES["gp_lambda_e"], VALUES["gp_lambda_p"], VALUES["gp_P"]
LAMBDA = VALUES["gp_lambda"]


def kernel_matrix(kernel_type: str, values: dict = VALUES) -> np.ndarray:
    """The kernel's covariance matrix on the grid T."""
    return np.asarray(GPKernel(kernel_type).build_kernel(values)(T, T))


def kernel_values(kernel_type: str) -> dict:
    """VALUES restricted to the kernel's own hyperparameters."""
    return {name: VALUES[name] for name in GPKernel(kernel_type).param_names}


def kernel_at(kernel_type: str, tau: float, values: dict = VALUES) -> float:
    """The kernel at a single lag tau."""
    return float(GPKernel(kernel_type).build_kernel(values)(np.array([0.0]), np.array([tau]))[0, 0])


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
        """Matches celerite2's RealTerm.

        celerite2 0.3.3, RealTerm(a=gp_A^2, c=1/gp_lambda); row k(T[0], T).
        """
        celerite2 = [9.0, 8.500132232953828, 5.458775937413701, 3.6853571263712785, 3.310914970542981,
                     1.5090952387661742, 0.25304093774074843, 0.10287876795769818]
        np.testing.assert_allclose(kernel_matrix("Exponential")[0], celerite2, rtol=1e-12)


class TestKernelTable:
    """The kernel table and GPKernel's public surface, for every kernel."""

    def test_supported_kernels(self) -> None:
        """SUPPORTED_KERNELS lists every kernel, in table order."""
        assert SUPPORTED_KERNELS == EXPECTED_KERNELS

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


def make_gp_fitter(kernel_type: str, gp_values: dict | None = None) -> GPFitter:
    """A no-planet GPFitter on two instruments, with its GP hyperparameters free.

    gp_values are the GP hyperparameters for the first params assignment (default: the
    kernel's own names, from VALUES).
    """
    rng = np.random.default_rng(2)
    time = np.sort(rng.uniform(0, 60, 24))
    instrument = np.array(["HARPS", "HIRES"] * 12)
    velerr = rng.uniform(0.8, 1.6, 24)
    vel = 3.0 * np.sin(2 * np.pi * time / 12.5) + rng.normal(0, 1, 24) + np.where(instrument == "HARPS", 5.0, -2.0)
    fitter = GPFitter([], Parameterisation("P K e w Tp"), GPKernel(kernel_type))
    fitter.add_data(time, vel, velerr, instrument, t0=30.0)
    params = {
        "g_HARPS": Parameter(5.0, fixed=False), "jit_HARPS": Parameter(0.5, fixed=False),
        "g_HIRES": Parameter(-2.0, fixed=False), "jit_HIRES": Parameter(0.7, fixed=False),
        "gd": Parameter(0.0, fixed=True), "gdd": Parameter(0.0, fixed=True),
    }
    gp_values = kernel_values(kernel_type) if gp_values is None else gp_values
    params |= {name: Parameter(value, fixed=False) for name, value in gp_values.items()}
    fitter.params = params
    fitter.priors = {name: Uniform(-50, 50) if name.startswith("g_") else Uniform(0, 100)
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
