"""Test per-planet log-posterior corrections for the (u, v) parameterisation.

These constants (Jacobian + prior-renormalisation corrections) make Bayesian
evidence estimation (e.g. via harmonic/LHME) coordinate-invariant when sampling
in the secosw/sesinw parameterisation. They cancel in the MCMC acceptance
ratio, so they only matter for evidence, not parameter inference.
"""
import numpy as np
import pytest

from ravest.fit import Fitter, LogPosterior
from ravest.param import Parameter, Parameterisation
from ravest.prior import HalfNormal, Normal, Uniform

LOG_4_OVER_PI = np.log(4.0 / np.pi)
LOG_2 = np.log(2.0)


def _build_log_posterior(fitter: Fitter) -> LogPosterior:
    """Construct a LogPosterior from a configured Fitter.

    Mirrors the construction sites inside Fitter (e.g. find_map_estimate).
    """
    return LogPosterior(
        fitter.planet_letters,
        fitter.parameterisation,
        fitter.priors,
        fitter.fixed_params_values_dict,
        fitter.free_params_names,
        fitter.time,
        fitter.vel,
        fitter.velerr,
        fitter.instrument,
        fitter.unique_instruments,
        fitter.t0,
    )


@pytest.fixture
def simple_test_data():
    """Simple fake data, single instrument: HARPS."""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    vel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    velerr = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    instrument = np.array(["HARPS", "HARPS", "HARPS", "HARPS", "HARPS"])
    return time, vel, velerr, instrument


def _common_params(letter: str, fixed: bool = False) -> dict:
    return {
        f"P_{letter}": Parameter(5.0, "days", fixed=False),
        f"K_{letter}": Parameter(3.0, "m/s", fixed=False),
        "g_HARPS": Parameter(0.0, "m/s", fixed=True),
        "gd": Parameter(0.0, "m/s/day", fixed=True),
        "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
        "jit_HARPS": Parameter(1.0, "m/s", fixed=False),
    }


def _common_priors() -> dict:
    return {
        "jit_HARPS": Uniform(0, 5),
    }


class TestSinglePlanetCases:
    """Single-planet log-posterior correction cases."""

    def test_default_parameterisation_is_case_1(self, simple_test_data) -> None:
        """Default (e, w) parameterisation always classifies as CASE_1."""
        time, vel, velerr, instrument = simple_test_data
        params = _common_params("b") | {
            "P_b": Parameter(5.0, "days", fixed=False),
            "K_b": Parameter(3.0, "m/s", fixed=False),
            "e_b": Parameter(0.0, "", fixed=False),
            "w_b": Parameter(0.0, "rad", fixed=False),
            "Tp_b": Parameter(0.0, "days", fixed=False),
        }
        priors = _common_priors() | {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "e_b": Uniform(0, 1),
            "w_b": Uniform(-np.pi, np.pi),
            "Tp_b": Uniform(-5, 5),
        }
        fitter = Fitter(["b"], Parameterisation("P K e w Tp"))
        fitter.add_data(time=time, vel=vel, velerr=velerr, instrument=instrument, t0=2.0)
        fitter.params = params
        fitter.priors = priors

        lp = _build_log_posterior(fitter)

        assert lp._logprob_jacobian_correction == 0.0
        assert lp._logprob_prior_renorm_correction == 0.0
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_1"

    def test_transformed_with_fixed_secosw_sesinw_is_case_1(self, simple_test_data) -> None:
        """Transformed parameterisation with fixed secosw/sesinw classifies as CASE_1."""
        time, vel, velerr, instrument = simple_test_data
        params = _common_params("b") | {
            "secosw_b": Parameter(0.0, "", fixed=True),
            "sesinw_b": Parameter(0.0, "", fixed=True),
            "Tc_b": Parameter(0.0, "days", fixed=True),
        }
        priors = _common_priors() | {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.add_data(time=time, vel=vel, velerr=velerr, instrument=instrument, t0=2.0)
        fitter.params = params
        fitter.priors = priors

        lp = _build_log_posterior(fitter)

        assert lp._logprob_jacobian_correction == 0.0
        assert lp._logprob_prior_renorm_correction == 0.0
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_1"

    def test_transformed_with_uniform_uv_priors_is_case_2(self, simple_test_data) -> None:
        """Transformed parameterisation with Uniform(-1, 1) (u, v) priors classifies as CASE_2."""
        time, vel, velerr, instrument = simple_test_data
        params = _common_params("b") | {
            "secosw_b": Parameter(0.1, "", fixed=False),
            "sesinw_b": Parameter(0.2, "", fixed=False),
            "Tc_b": Parameter(2.0, "days", fixed=False),
        }
        priors = _common_priors() | {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "secosw_b": Uniform(-1, 1),
            "sesinw_b": Uniform(-1, 1),
            "Tc_b": Uniform(-5, 5),
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.add_data(time=time, vel=vel, velerr=velerr, instrument=instrument, t0=2.0)
        fitter.params = params
        fitter.priors = priors

        lp = _build_log_posterior(fitter)

        assert lp._logprob_jacobian_correction == 0.0
        assert np.isclose(lp._logprob_prior_renorm_correction, LOG_4_OVER_PI)
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_2"

    @pytest.mark.parametrize("e_prior", [Uniform(0, 1), HalfNormal(0.3)])
    def test_transformed_with_e_w_priors_is_case_3(self, simple_test_data, e_prior) -> None:
        """Transformed parameterisation with (e, w) priors classifies as CASE_3, for any e prior."""
        time, vel, velerr, instrument = simple_test_data
        params = _common_params("b") | {
            "secosw_b": Parameter(0.1, "", fixed=False),
            "sesinw_b": Parameter(0.2, "", fixed=False),
            "Tc_b": Parameter(2.0, "days", fixed=False),
        }
        priors = _common_priors() | {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "e_b": e_prior,
            "w_b": Uniform(-np.pi, np.pi),
            "Tp_b": Uniform(-5, 5),
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.add_data(time=time, vel=vel, velerr=velerr, instrument=instrument, t0=2.0)
        fitter.params = params
        fitter.priors = priors

        lp = _build_log_posterior(fitter)

        assert np.isclose(lp._logprob_jacobian_correction, LOG_2)
        assert lp._logprob_prior_renorm_correction == 0.0
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_3"


class TestTwoPlanetCases:
    """Two-planet log-posterior correction cases, including the mixed-case regression."""

    def _two_planet_params_and_priors(self, secosw_priors: dict, e_priors: dict, letters: list[str]) -> tuple[dict, dict]:
        params = {
            "g_HARPS": Parameter(0.0, "m/s", fixed=True),
            "gd": Parameter(0.0, "m/s/day", fixed=True),
            "gdd": Parameter(0.0, "m/s/day^2", fixed=True),
            "jit_HARPS": Parameter(1.0, "m/s", fixed=False),
        }
        priors = {"jit_HARPS": Uniform(0, 5)}

        for letter in letters:
            fixed = letter not in secosw_priors and letter not in e_priors
            params[f"P_{letter}"] = Parameter(5.0, "days", fixed=False)
            params[f"K_{letter}"] = Parameter(3.0, "m/s", fixed=False)
            params[f"secosw_{letter}"] = Parameter(0.1, "", fixed=fixed)
            params[f"sesinw_{letter}"] = Parameter(0.2, "", fixed=fixed)
            params[f"Tc_{letter}"] = Parameter(2.0, "days", fixed=fixed)

            priors[f"P_{letter}"] = Uniform(0, 10)
            priors[f"K_{letter}"] = Uniform(0, 10)
            if letter in secosw_priors:
                priors[f"secosw_{letter}"] = Uniform(-1, 1)
                priors[f"sesinw_{letter}"] = Uniform(-1, 1)
                priors[f"Tc_{letter}"] = Uniform(-5, 5)
            elif letter in e_priors:
                priors[f"e_{letter}"] = Uniform(0, 1)
                priors[f"w_{letter}"] = Uniform(-np.pi, np.pi)
                priors[f"Tp_{letter}"] = Uniform(-5, 5)

        return params, priors

    def _build_fitter(self, simple_test_data, letters: list[str], params: dict, priors: dict) -> Fitter:
        time, vel, velerr, instrument = simple_test_data
        fitter = Fitter(letters, Parameterisation("P K secosw sesinw Tc"))
        fitter.add_data(time=time, vel=vel, velerr=velerr, instrument=instrument, t0=2.0)
        fitter.params = params
        fitter.priors = priors
        return fitter

    def test_both_case_2(self, simple_test_data) -> None:
        """Both planets in CASE_2 sums to 2 x log(4/pi)."""
        params, priors = self._two_planet_params_and_priors(
            secosw_priors={"b", "c"}, e_priors=set(), letters=["b", "c"]
        )
        fitter = self._build_fitter(simple_test_data, ["b", "c"], params, priors)

        lp = _build_log_posterior(fitter)

        assert lp._logprob_jacobian_correction == 0.0
        assert np.isclose(lp._logprob_prior_renorm_correction, 2 * LOG_4_OVER_PI)

    def test_both_case_3(self, simple_test_data) -> None:
        """Both planets in CASE_3 sums to 2 x log(2)."""
        params, priors = self._two_planet_params_and_priors(
            secosw_priors=set(), e_priors={"b", "c"}, letters=["b", "c"]
        )
        fitter = self._build_fitter(simple_test_data, ["b", "c"], params, priors)

        lp = _build_log_posterior(fitter)

        assert np.isclose(lp._logprob_jacobian_correction, 2 * LOG_2)
        assert lp._logprob_prior_renorm_correction == 0.0

    def test_mixed_case_2_and_case_3_regression(self, simple_test_data) -> None:
        """Regression test: planet b on (u,v) priors, planet c on (e,w) priors.

        The stale prior implementation classified this globally via a single
        set-equality over all free params, so a mixed system like this would
        silently compute the wrong evidence correction. Each planet must be
        classified independently and the contributions summed.
        """
        params, priors = self._two_planet_params_and_priors(
            secosw_priors={"b"}, e_priors={"c"}, letters=["b", "c"]
        )
        fitter = self._build_fitter(simple_test_data, ["b", "c"], params, priors)

        lp = _build_log_posterior(fitter)

        assert np.isclose(lp._logprob_jacobian_correction, LOG_2)
        assert np.isclose(lp._logprob_prior_renorm_correction, LOG_4_OVER_PI)
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_2"
        assert lp._logprob_correction_breakdown["c"]["case"] == "CASE_3"

    def test_mixed_case_1_fixed_and_case_2(self, simple_test_data) -> None:
        """One planet fixed (CASE_1) and one in CASE_2 sums to only log(4/pi)."""
        params, priors = self._two_planet_params_and_priors(
            secosw_priors={"c"}, e_priors=set(), letters=["b", "c"]
        )
        fitter = self._build_fitter(simple_test_data, ["b", "c"], params, priors)

        lp = _build_log_posterior(fitter)

        assert lp._logprob_jacobian_correction == 0.0
        assert np.isclose(lp._logprob_prior_renorm_correction, LOG_4_OVER_PI)
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_1"
        assert lp._logprob_correction_breakdown["c"]["case"] == "CASE_2"

    def test_mixed_case_1_fixed_and_case_3(self, simple_test_data) -> None:
        """One planet fixed (CASE_1) and one in CASE_3 sums to only log(2)."""
        params, priors = self._two_planet_params_and_priors(
            secosw_priors=set(), e_priors={"c"}, letters=["b", "c"]
        )
        fitter = self._build_fitter(simple_test_data, ["b", "c"], params, priors)

        lp = _build_log_posterior(fitter)

        assert np.isclose(lp._logprob_jacobian_correction, LOG_2)
        assert lp._logprob_prior_renorm_correction == 0.0
        assert lp._logprob_correction_breakdown["b"]["case"] == "CASE_1"
        assert lp._logprob_correction_breakdown["c"]["case"] == "CASE_3"


class TestUnsupportedPriorRaises:
    """Non-Uniform(-1, 1) priors on (secosw, sesinw) are unsupported and hard-raise."""

    @pytest.mark.parametrize("secosw_prior, sesinw_prior", [
        (Uniform(-0.5, 0.5), Uniform(-0.5, 0.5)),
        (Normal(0, 0.3), Normal(0, 0.3)),
    ])
    def test_non_uniform_uv_prior_raises_not_implemented(self, simple_test_data, secosw_prior, sesinw_prior) -> None:
        """Non-Uniform(-1, 1) priors on (secosw, sesinw) raise NotImplementedError."""
        time, vel, velerr, instrument = simple_test_data
        params = _common_params("b") | {
            "secosw_b": Parameter(0.1, "", fixed=False),
            "sesinw_b": Parameter(0.2, "", fixed=False),
            "Tc_b": Parameter(2.0, "days", fixed=False),
        }
        priors = _common_priors() | {
            "P_b": Uniform(0, 10),
            "K_b": Uniform(0, 10),
            "secosw_b": secosw_prior,
            "sesinw_b": sesinw_prior,
            "Tc_b": Uniform(-5, 5),
        }
        fitter = Fitter(["b"], Parameterisation("P K secosw sesinw Tc"))
        fitter.add_data(time=time, vel=vel, velerr=velerr, instrument=instrument, t0=2.0)
        fitter.params = params
        fitter.priors = priors

        with pytest.raises(NotImplementedError, match="e_b.*w_b"):
            _build_log_posterior(fitter)


class TestCorrectionBreakdown:
    """Per-planet breakdown dict reports the right case and values sum to totals."""

    def test_breakdown_sums_to_totals(self, simple_test_data) -> None:
        """Per-planet breakdown values sum to the scalar total corrections."""
        params, priors = TestTwoPlanetCases()._two_planet_params_and_priors(
            secosw_priors={"b"}, e_priors={"c"}, letters=["b", "c"]
        )
        fitter = TestTwoPlanetCases()._build_fitter(simple_test_data, ["b", "c"], params, priors)

        lp = _build_log_posterior(fitter)

        breakdown = lp._logprob_correction_breakdown
        assert set(breakdown.keys()) == {"b", "c"}
        total_jacobian = sum(v["jacobian"] for v in breakdown.values())
        total_renorm = sum(v["renorm"] for v in breakdown.values())
        assert np.isclose(total_jacobian, lp._logprob_jacobian_correction)
        assert np.isclose(total_renorm, lp._logprob_prior_renorm_correction)
