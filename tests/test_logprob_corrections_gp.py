"""Test per-planet log-posterior corrections for GPLogPosterior.

Mirrors tests/test_logprob_corrections.py but for the GP-enabled log-posterior
(GPLogPosterior), confirming the same per-planet Jacobian/prior-renormalisation
corrections apply regardless of the likelihood function (GP or otherwise) -
the corrections depend only on the parameterisation and priors.
"""
import numpy as np
import pytest

from ravest.fit import GPLogPosterior
from ravest.gp import GPKernel
from ravest.param import Parameterisation
from ravest.prior import HalfNormal, Normal, Uniform

LOG_4_OVER_PI = np.log(4.0 / np.pi)
LOG_2 = np.log(2.0)


@pytest.fixture
def simple_gp_test_data():
    """Simple fake data, single instrument: HARPS."""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    vel = np.array([5.0, -2.0, -5.0, 2.0, 3.0, -1.0])
    velerr = np.array([1.0, 1.1, 0.9, 0.85, 1.5, 1.0])
    instrument = np.array(["HARPS", "HARPS", "HARPS", "HARPS", "HARPS", "HARPS"])
    return time, vel, velerr, instrument


def _gp_kernel() -> GPKernel:
    return GPKernel("Quasiperiodic")


def _fixed_hyperparams_and_priors() -> tuple[dict, dict]:
    """All GP hyperparameters fixed.

    Isolates the corrections under test from the (unrelated) GP likelihood machinery.
    """
    fixed_hyperparams = {
        "gp_amp": 1.0,
        "gp_lambda_e": 50.0,
        "gp_lambda_p": 0.5,
        "gp_period": 10.0,
    }
    return fixed_hyperparams, {}


def _build_gp_log_posterior(
    planet_letters: list[str],
    parameterisation: Parameterisation,
    priors: dict,
    fixed_params: dict,
    free_params_names: list[str],
    data: tuple,
) -> GPLogPosterior:
    time, vel, velerr, instrument = data
    fixed_hyperparams, hyperpriors = _fixed_hyperparams_and_priors()
    return GPLogPosterior(
        planet_letters=planet_letters,
        parameterisation=parameterisation,
        gp_kernel=_gp_kernel(),
        priors=priors,
        hyperpriors=hyperpriors,
        fixed_params=fixed_params,
        fixed_hyperparams=fixed_hyperparams,
        free_params_names=free_params_names,
        free_hyperparams_names=[],
        time=time, vel=vel, velerr=velerr, t0=2.0,
        instrument=instrument, unique_instruments=list(np.unique(instrument)),
    )


def _common_fixed_params(letter: str) -> dict:
    return {
        "g_HARPS": 0.0,
        "gd": 0.0,
        "gdd": 0.0,
    }


class TestSinglePlanetCasesGP:
    """Single-planet log-posterior correction cases for GPLogPosterior."""

    def test_default_parameterisation_is_case_1(self, simple_gp_test_data) -> None:
        """Default (e, w) parameterisation always classifies as CASE_1."""
        fixed_params = _common_fixed_params("b") | {
            "P_b": 5.0, "e_b": 0.0, "w_b": 0.0, "Tp_b": 0.0, "jit_HARPS": 1.0,
        }
        priors = {"K_b": Uniform(0, 10)}
        lp = _build_gp_log_posterior(
            ["b"], Parameterisation("P K e w Tp"), priors, fixed_params, ["K_b"], simple_gp_test_data
        )

        assert lp._logprob_jacobian_correction == 0.0
        assert lp._logprob_prior_renorm_correction == 0.0
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_1"

    def test_transformed_with_fixed_secosw_sesinw_is_case_1(self, simple_gp_test_data) -> None:
        """Transformed parameterisation with fixed secosw/sesinw classifies as CASE_1."""
        fixed_params = _common_fixed_params("b") | {
            "P_b": 5.0, "secosw_b": 0.0, "sesinw_b": 0.0, "Tc_b": 0.0, "jit_HARPS": 1.0,
        }
        priors = {"K_b": Uniform(0, 10)}
        lp = _build_gp_log_posterior(
            ["b"], Parameterisation("P K secosw sesinw Tc"), priors, fixed_params, ["K_b"], simple_gp_test_data
        )

        assert lp._logprob_jacobian_correction == 0.0
        assert lp._logprob_prior_renorm_correction == 0.0
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_1"

    def test_transformed_with_uniform_uv_priors_is_case_2(self, simple_gp_test_data) -> None:
        """Transformed parameterisation with Uniform(-1, 1) (u, v) priors classifies as CASE_2."""
        fixed_params = _common_fixed_params("b") | {"P_b": 5.0, "Tc_b": 2.0, "jit_HARPS": 1.0}
        priors = {
            "K_b": Uniform(0, 10),
            "secosw_b": Uniform(-1, 1),
            "sesinw_b": Uniform(-1, 1),
        }
        free_params_names = ["K_b", "secosw_b", "sesinw_b"]
        lp = _build_gp_log_posterior(
            ["b"], Parameterisation("P K secosw sesinw Tc"), priors, fixed_params, free_params_names, simple_gp_test_data
        )

        assert lp._logprob_jacobian_correction == 0.0
        assert np.isclose(lp._logprob_prior_renorm_correction, LOG_4_OVER_PI)
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_2"

    @pytest.mark.parametrize("e_prior", [Uniform(0, 1), HalfNormal(0.3)])
    def test_transformed_with_e_w_priors_is_case_3(self, simple_gp_test_data, e_prior) -> None:
        """Transformed parameterisation with (e, w) priors classifies as CASE_3, for any e prior."""
        fixed_params = _common_fixed_params("b") | {"P_b": 5.0, "Tc_b": 2.0, "jit_HARPS": 1.0}
        priors = {
            "K_b": Uniform(0, 10),
            "e_b": e_prior,
            "w_b": Uniform(-np.pi, np.pi),
        }
        free_params_names = ["K_b", "secosw_b", "sesinw_b"]
        lp = _build_gp_log_posterior(
            ["b"], Parameterisation("P K secosw sesinw Tc"), priors, fixed_params, free_params_names, simple_gp_test_data
        )

        assert np.isclose(lp._logprob_jacobian_correction, LOG_2)
        assert lp._logprob_prior_renorm_correction == 0.0
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_3"


class TestMixedTwoPlanetCaseGP:
    """Mixed two-planet regression test for GPLogPosterior."""

    def test_mixed_case_2_and_case_3_regression(self, simple_gp_test_data) -> None:
        """Planet b on (u, v) priors (CASE_2), planet c on (e, w) priors (CASE_3).

        Confirms the per-planet classification and summation ported from
        LogPosterior also holds for GPLogPosterior, since the corrections
        depend only on the parameterisation and priors, not the likelihood.
        """
        fixed_params = {
            "g_HARPS": 0.0, "gd": 0.0, "gdd": 0.0,
            "P_b": 5.0, "Tc_b": 2.0,
            "P_c": 8.0, "Tc_c": 3.0,
        }
        priors = {
            "K_b": Uniform(0, 10),
            "secosw_b": Uniform(-1, 1),
            "sesinw_b": Uniform(-1, 1),
            "K_c": Uniform(0, 10),
            "e_c": Uniform(0, 1),
            "w_c": Uniform(-np.pi, np.pi),
            "jit_HARPS": Uniform(0, 5),
        }
        free_params_names = ["K_b", "secosw_b", "sesinw_b", "K_c", "secosw_c", "sesinw_c", "jit_HARPS"]
        lp = _build_gp_log_posterior(
            ["b", "c"], Parameterisation("P K secosw sesinw Tc"), priors, fixed_params, free_params_names, simple_gp_test_data
        )

        assert np.isclose(lp._logprob_jacobian_correction, LOG_2)
        assert np.isclose(lp._logprob_prior_renorm_correction, LOG_4_OVER_PI)
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_2"
        assert lp._logprob_correction_breakdown["c"]["case"] == "CASE_3"

        breakdown = lp._logprob_correction_breakdown
        total_jacobian = sum(v["jacobian"] for v in breakdown.values())
        total_renorm = sum(v["renorm"] for v in breakdown.values())
        assert np.isclose(total_jacobian, lp._logprob_jacobian_correction)
        assert np.isclose(total_renorm, lp._logprob_prior_renorm_correction)


class TestUnsupportedPriorRaisesGP:
    """Non-Uniform(-1, 1) priors on (secosw, sesinw) are unsupported and hard-raise."""

    @pytest.mark.parametrize("secosw_prior, sesinw_prior", [
        (Uniform(-0.5, 0.5), Uniform(-0.5, 0.5)),
        (Normal(0, 0.3), Normal(0, 0.3)),
    ])
    def test_non_uniform_uv_prior_raises_not_implemented(self, simple_gp_test_data, secosw_prior, sesinw_prior) -> None:
        """Non-Uniform(-1, 1) priors on (secosw, sesinw) raise NotImplementedError."""
        fixed_params = _common_fixed_params("b") | {"P_b": 5.0, "Tc_b": 2.0, "jit_HARPS": 1.0}
        priors = {
            "K_b": Uniform(0, 10),
            "secosw_b": secosw_prior,
            "sesinw_b": sesinw_prior,
        }
        free_params_names = ["K_b", "secosw_b", "sesinw_b"]

        with pytest.raises(NotImplementedError, match="e_b.*w_b"):
            _build_gp_log_posterior(
                ["b"], Parameterisation("P K secosw sesinw Tc"), priors, fixed_params, free_params_names, simple_gp_test_data
            )
