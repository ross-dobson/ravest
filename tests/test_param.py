import numpy as np
import pytest

from ravest.gp import SUPPORTED_KERNELS, GPKernel
from ravest.param import (
    ALLOWED_PARAMETERISATIONS,
    Parameterisation,
    param_key_to_latex,
    param_key_to_unit,
)

# First, test the underlying conversion functions

@pytest.mark.parametrize("e, w", [
    (e, w) for e in np.arange(0, 1, 0.1) for w in np.arange(-np.pi/2, np.pi/2 + np.pi/4, np.pi/4)
])
def test_convert_e_w_to_secosw_sesinw(e, w) -> None:
    para = Parameterisation("P K e w Tp")
    expected_secosw = np.sqrt(e) * np.cos(w)
    expected_sesinw = np.sqrt(e) * np.sin(w)
    secosw, sesinw = para.convert_e_w_to_secosw_sesinw(e, w)
    assert np.isclose(secosw, expected_secosw)
    assert np.isclose(sesinw, expected_sesinw)

@pytest.mark.parametrize("e, w", [
    (e, w) for e in np.arange(0, 1, 0.1) for w in np.arange(-np.pi/2, np.pi/2 + np.pi/4, np.pi/4)
])
def test_convert_e_w_to_ecosw_esinw(e, w) -> None:
    para = Parameterisation("P K e w Tp")
    expected_ecosw = e * np.cos(w)
    expected_esinw = e * np.sin(w)
    ecosw, esinw = para.convert_e_w_to_ecosw_esinw(e, w)
    assert np.isclose(ecosw, expected_ecosw)
    assert np.isclose(esinw, expected_esinw)

@pytest.mark.parametrize("ecosw, esinw", [
    (ecosw, esinw) for ecosw in np.arange(0, 1, 0.1) for esinw in np.arange(0, 1, 0.1)
    if np.sqrt(ecosw**2 + esinw**2) < 1.0  # Only test valid combinations where e < 1
])
def test_convert_ecosw_esinw_to_e_w_valid(ecosw, esinw) -> None:
    para = Parameterisation("P K ecosw esinw Tp")
    expected_e = np.sqrt(ecosw**2 + esinw**2)
    expected_w = np.arctan2(esinw, ecosw)
    e, w = para.convert_ecosw_esinw_to_e_w(ecosw, esinw)
    assert np.isclose(e, expected_e)
    assert np.isclose(w, expected_w)

@pytest.mark.parametrize("ecosw, esinw", [
    (ecosw, esinw) for ecosw in np.arange(0, 1, 0.1) for esinw in np.arange(0, 1, 0.1)
    if np.sqrt(ecosw**2 + esinw**2) >= 1.0  # Test invalid combinations where e >= 1
])
def test_convert_ecosw_esinw_to_e_w_invalid(ecosw, esinw) -> None:
    para = Parameterisation("P K ecosw esinw Tp")
    with pytest.raises(ValueError, match="Invalid eccentricity.*>= 1.0"):
        para.convert_ecosw_esinw_to_e_w(ecosw, esinw)

@pytest.mark.parametrize("secosw, sesinw", [
    (secosw, sesinw) for secosw in np.arange(0, 1, 0.1) for sesinw in np.arange(0, 1, 0.1)
])
def test_convert_secosw_sesinw_to_e_w(secosw, sesinw) -> None:
    para = Parameterisation("P K secosw sesinw Tp")
    expected_e = secosw**2 + sesinw**2
    expected_w = np.arctan2(sesinw, secosw)
    e, w = para.convert_secosw_sesinw_to_e_w(secosw, sesinw)
    assert np.isclose(e, expected_e)
    assert np.isclose(w, expected_w)

@pytest.mark.parametrize("tp", [
    tp for tp in np.arange(0, 10, 0.1)
])
def test_convert_tp_to_tc_circular(tp) -> None: # if e = 0 & w = pi/2, then tp = tc
    para = Parameterisation("P K e w Tp")
    per = 10
    e = 0
    w = np.pi/2
    expected_tc = tp
    tc = para.convert_tp_to_tc(tp, per, e, w)
    assert expected_tc == tc

@pytest.mark.parametrize("tp, e, w, tc", [
    (0,    0.3, 3*np.pi/8, 0.32487717871429983),
    (3.33, 0.51, -np.pi/5, 5.200496945307864),
    (5,    0.69,        0, 5.493187444825672),
    (8.2, 0.8,    np.pi/7, 8.34625216953673)
])
def test_convert_tp_to_tc_eccentric(tp, e, w, tc) -> None:
    para = Parameterisation("P K e w Tc")
    per = 10
    assert np.isclose(tc, para.convert_tp_to_tc(tp, per, e, w))

@pytest.mark.parametrize("tc", [
    tc for tc in np.arange(0, 10, 0.1)
])
def test_convert_tc_to_tp_circular(tc) -> None: # if e = 0 & w = pi/2, then tc = tp
    para = Parameterisation("P K e w Tc")
    expected_tp = tc
    per = 10
    e = 0
    w = np.pi/2
    tp = para.convert_tc_to_tp(tc, per, e, w)
    assert expected_tp == tp

@pytest.mark.parametrize("tc, e, w, tp", [
    (0,    0.3, 3*np.pi/8, -0.32487717871429983),
    (3.33, 0.51, -np.pi/5, 1.459503054692136),
    (5,    0.69,        0, 4.506812555174328),
    (8.2, 0.8,    np.pi/7, 8.05374783046327)
])
def test_convert_tc_to_tp_eccentric(tc, e, w, tp) -> None:
    para = Parameterisation("P K e w Tc")
    per = 10
    assert np.isclose(tp, para.convert_tc_to_tp(tc, per, e, w))


# Second, test the automatic conversion function, for each of the
# parameterisations

def test_invalid_parameterisation() -> None:
    with pytest.raises(Exception):
        Parameterisation("not a valid parameterisation")


# ============================================================================
# Test convert_pars_to_default_parameterisation for all parameterisations
# ============================================================================


# Reference parameters in default parameterisation (P K e w Tp)
_DEFAULT_PARAMS = {"P": 10.0, "K": 25.0, "e": 0.3, "w": 0.5, "Tp": 5.0}


class TestConvertToDefault:
    """Test convert_pars_to_default_parameterisation for each parameterisation."""

    def test_pkewtc_to_default(self) -> None:
        """Test P K e w Tc -> P K e w Tp round-trip."""
        para_tc = Parameterisation("P K e w Tc")
        para_default = Parameterisation("P K e w Tp")

        # Convert default -> Tc parameterisation
        tc = para_default.convert_tp_to_tc(_DEFAULT_PARAMS["Tp"], _DEFAULT_PARAMS["P"],
                                           _DEFAULT_PARAMS["e"], _DEFAULT_PARAMS["w"])
        tc_params = {"P": 10.0, "K": 25.0, "e": 0.3, "w": 0.5, "Tc": tc}

        # Convert back to default
        result = para_tc.convert_pars_to_default_parameterisation(tc_params)

        for key in _DEFAULT_PARAMS:
            np.testing.assert_allclose(result[key], _DEFAULT_PARAMS[key], atol=1e-10,
                                       err_msg=f"Mismatch on {key}")

    def test_ecosw_esinw_tp_to_default(self) -> None:
        """Test P K ecosw esinw Tp -> P K e w Tp."""
        para = Parameterisation("P K ecosw esinw Tp")
        ecosw = _DEFAULT_PARAMS["e"] * np.cos(_DEFAULT_PARAMS["w"])
        esinw = _DEFAULT_PARAMS["e"] * np.sin(_DEFAULT_PARAMS["w"])
        inpars = {"P": 10.0, "K": 25.0, "ecosw": ecosw, "esinw": esinw, "Tp": 5.0}

        result = para.convert_pars_to_default_parameterisation(inpars)

        np.testing.assert_allclose(result["e"], _DEFAULT_PARAMS["e"], atol=1e-10)
        np.testing.assert_allclose(result["w"], _DEFAULT_PARAMS["w"], atol=1e-10)
        assert result["Tp"] == _DEFAULT_PARAMS["Tp"]

    def test_ecosw_esinw_tc_to_default(self) -> None:
        """Test P K ecosw esinw Tc -> P K e w Tp."""
        para = Parameterisation("P K ecosw esinw Tc")
        para_default = Parameterisation("P K e w Tp")

        ecosw = _DEFAULT_PARAMS["e"] * np.cos(_DEFAULT_PARAMS["w"])
        esinw = _DEFAULT_PARAMS["e"] * np.sin(_DEFAULT_PARAMS["w"])
        tc = para_default.convert_tp_to_tc(_DEFAULT_PARAMS["Tp"], _DEFAULT_PARAMS["P"],
                                           _DEFAULT_PARAMS["e"], _DEFAULT_PARAMS["w"])
        inpars = {"P": 10.0, "K": 25.0, "ecosw": ecosw, "esinw": esinw, "Tc": tc}

        result = para.convert_pars_to_default_parameterisation(inpars)

        np.testing.assert_allclose(result["e"], _DEFAULT_PARAMS["e"], atol=1e-10)
        np.testing.assert_allclose(result["w"], _DEFAULT_PARAMS["w"], atol=1e-10)
        np.testing.assert_allclose(result["Tp"], _DEFAULT_PARAMS["Tp"], atol=1e-10)

    def test_secosw_sesinw_tp_to_default(self) -> None:
        """Test P K secosw sesinw Tp -> P K e w Tp."""
        para = Parameterisation("P K secosw sesinw Tp")
        secosw = np.sqrt(_DEFAULT_PARAMS["e"]) * np.cos(_DEFAULT_PARAMS["w"])
        sesinw = np.sqrt(_DEFAULT_PARAMS["e"]) * np.sin(_DEFAULT_PARAMS["w"])
        inpars = {"P": 10.0, "K": 25.0, "secosw": secosw, "sesinw": sesinw, "Tp": 5.0}

        result = para.convert_pars_to_default_parameterisation(inpars)

        np.testing.assert_allclose(result["e"], _DEFAULT_PARAMS["e"], atol=1e-10)
        np.testing.assert_allclose(result["w"], _DEFAULT_PARAMS["w"], atol=1e-10)
        assert result["Tp"] == _DEFAULT_PARAMS["Tp"]

    def test_secosw_sesinw_tc_to_default(self) -> None:
        """Test P K secosw sesinw Tc -> P K e w Tp."""
        para = Parameterisation("P K secosw sesinw Tc")
        para_default = Parameterisation("P K e w Tp")

        secosw = np.sqrt(_DEFAULT_PARAMS["e"]) * np.cos(_DEFAULT_PARAMS["w"])
        sesinw = np.sqrt(_DEFAULT_PARAMS["e"]) * np.sin(_DEFAULT_PARAMS["w"])
        tc = para_default.convert_tp_to_tc(_DEFAULT_PARAMS["Tp"], _DEFAULT_PARAMS["P"],
                                           _DEFAULT_PARAMS["e"], _DEFAULT_PARAMS["w"])
        inpars = {"P": 10.0, "K": 25.0, "secosw": secosw, "sesinw": sesinw, "Tc": tc}

        result = para.convert_pars_to_default_parameterisation(inpars)

        np.testing.assert_allclose(result["e"], _DEFAULT_PARAMS["e"], atol=1e-10)
        np.testing.assert_allclose(result["w"], _DEFAULT_PARAMS["w"], atol=1e-10)
        np.testing.assert_allclose(result["Tp"], _DEFAULT_PARAMS["Tp"], atol=1e-10)


class TestConvertFromDefault:
    """Test convert_pars_from_default_parameterisation for each parameterisation."""

    def test_default_to_pkewtc(self) -> None:
        """Test default -> P K e w Tc."""
        para = Parameterisation("P K e w Tc")
        para_default = Parameterisation("P K e w Tp")

        result = para.convert_pars_from_default_parameterisation(_DEFAULT_PARAMS)

        assert result["P"] == _DEFAULT_PARAMS["P"]
        assert result["K"] == _DEFAULT_PARAMS["K"]
        assert result["e"] == _DEFAULT_PARAMS["e"]
        assert result["w"] == _DEFAULT_PARAMS["w"]
        # Tc should match the forward conversion
        expected_tc = para_default.convert_tp_to_tc(_DEFAULT_PARAMS["Tp"], _DEFAULT_PARAMS["P"],
                                                     _DEFAULT_PARAMS["e"], _DEFAULT_PARAMS["w"])
        np.testing.assert_allclose(result["Tc"], expected_tc, atol=1e-10)

    def test_default_to_ecosw_esinw_tp(self) -> None:
        """Test default -> P K ecosw esinw Tp."""
        para = Parameterisation("P K ecosw esinw Tp")

        result = para.convert_pars_from_default_parameterisation(_DEFAULT_PARAMS)

        expected_ecosw = _DEFAULT_PARAMS["e"] * np.cos(_DEFAULT_PARAMS["w"])
        expected_esinw = _DEFAULT_PARAMS["e"] * np.sin(_DEFAULT_PARAMS["w"])
        np.testing.assert_allclose(result["ecosw"], expected_ecosw, atol=1e-10)
        np.testing.assert_allclose(result["esinw"], expected_esinw, atol=1e-10)
        assert result["Tp"] == _DEFAULT_PARAMS["Tp"]

    def test_default_to_ecosw_esinw_tc(self) -> None:
        """Test default -> P K ecosw esinw Tc."""
        para = Parameterisation("P K ecosw esinw Tc")

        result = para.convert_pars_from_default_parameterisation(_DEFAULT_PARAMS)

        expected_ecosw = _DEFAULT_PARAMS["e"] * np.cos(_DEFAULT_PARAMS["w"])
        expected_esinw = _DEFAULT_PARAMS["e"] * np.sin(_DEFAULT_PARAMS["w"])
        np.testing.assert_allclose(result["ecosw"], expected_ecosw, atol=1e-10)
        np.testing.assert_allclose(result["esinw"], expected_esinw, atol=1e-10)
        assert "Tc" in result

    def test_default_to_secosw_sesinw_tp(self) -> None:
        """Test default -> P K secosw sesinw Tp."""
        para = Parameterisation("P K secosw sesinw Tp")

        result = para.convert_pars_from_default_parameterisation(_DEFAULT_PARAMS)

        expected_secosw = np.sqrt(_DEFAULT_PARAMS["e"]) * np.cos(_DEFAULT_PARAMS["w"])
        expected_sesinw = np.sqrt(_DEFAULT_PARAMS["e"]) * np.sin(_DEFAULT_PARAMS["w"])
        np.testing.assert_allclose(result["secosw"], expected_secosw, atol=1e-10)
        np.testing.assert_allclose(result["sesinw"], expected_sesinw, atol=1e-10)
        assert result["Tp"] == _DEFAULT_PARAMS["Tp"]

    def test_default_to_secosw_sesinw_tc(self) -> None:
        """Test default -> P K secosw sesinw Tc."""
        para = Parameterisation("P K secosw sesinw Tc")

        result = para.convert_pars_from_default_parameterisation(_DEFAULT_PARAMS)

        expected_secosw = np.sqrt(_DEFAULT_PARAMS["e"]) * np.cos(_DEFAULT_PARAMS["w"])
        expected_sesinw = np.sqrt(_DEFAULT_PARAMS["e"]) * np.sin(_DEFAULT_PARAMS["w"])
        np.testing.assert_allclose(result["secosw"], expected_secosw, atol=1e-10)
        np.testing.assert_allclose(result["sesinw"], expected_sesinw, atol=1e-10)
        assert "Tc" in result

    def test_default_identity(self) -> None:
        """Test P K e w Tp -> P K e w Tp is identity."""
        para = Parameterisation("P K e w Tp")

        result = para.convert_pars_from_default_parameterisation(_DEFAULT_PARAMS)

        for key in _DEFAULT_PARAMS:
            assert result[key] == _DEFAULT_PARAMS[key]


class TestRoundTripConversions:
    """Test that to_default -> from_default round-trips preserve values."""

    @pytest.mark.parametrize("param_str", [
        "P K e w Tc",
        "P K ecosw esinw Tp",
        "P K ecosw esinw Tc",
        "P K secosw sesinw Tp",
        "P K secosw sesinw Tc",
    ])
    def test_round_trip(self, param_str) -> None:
        """Test default -> parameterisation -> default round-trip."""
        para = Parameterisation(param_str)

        # Default -> this parameterisation
        converted = para.convert_pars_from_default_parameterisation(_DEFAULT_PARAMS)

        # Back to default
        restored = para.convert_pars_to_default_parameterisation(converted)

        for key in _DEFAULT_PARAMS:
            np.testing.assert_allclose(restored[key], _DEFAULT_PARAMS[key], atol=1e-10,
                                       err_msg=f"Round-trip failed for {param_str} on {key}")


# ============================================================================
# Test array-input validation paths
# ============================================================================


class TestArrayValidation:
    """Test validation methods with array inputs."""

    def test_validate_period_array_valid(self) -> None:
        """Test period validation with valid array."""
        para = Parameterisation("P K e w Tp")
        para._validate_period(np.array([1.0, 2.0, 3.0]))  # Should not raise

    def test_validate_period_array_invalid(self) -> None:
        """Test period validation with array containing invalid value."""
        para = Parameterisation("P K e w Tp")
        with pytest.raises(ValueError, match="some values <= 0"):
            para._validate_period(np.array([1.0, -0.5, 3.0]))

    def test_validate_semi_amplitude_array_valid(self) -> None:
        """Test semi-amplitude validation with valid array."""
        para = Parameterisation("P K e w Tp")
        para._validate_semi_amplitude(np.array([1.0, 5.0, 10.0]))  # Should not raise

    def test_validate_semi_amplitude_array_invalid(self) -> None:
        """Test semi-amplitude validation with array containing invalid value."""
        para = Parameterisation("P K e w Tp")
        with pytest.raises(ValueError, match="some values <= 0"):
            para._validate_semi_amplitude(np.array([1.0, 0.0, 3.0]))

    def test_validate_eccentricity_array_valid(self) -> None:
        """Test eccentricity validation with valid array."""
        para = Parameterisation("P K e w Tp")
        para._validate_eccentricity(np.array([0.0, 0.3, 0.9]))  # Should not raise

    def test_validate_eccentricity_array_negative(self) -> None:
        """Test eccentricity validation with negative array value."""
        para = Parameterisation("P K e w Tp")
        with pytest.raises(ValueError, match="some values < 0"):
            para._validate_eccentricity(np.array([0.0, -0.1, 0.5]))

    def test_validate_eccentricity_array_ge_one(self) -> None:
        """Test eccentricity validation with array value >= 1."""
        para = Parameterisation("P K e w Tp")
        with pytest.raises(ValueError, match="some values >= 1.0"):
            para._validate_eccentricity(np.array([0.0, 0.5, 1.0]))

    def test_validate_argument_periastron_array_valid(self) -> None:
        """Test argument of periastron validation with valid array."""
        para = Parameterisation("P K e w Tp")
        para._validate_argument_periastron(np.array([-np.pi + 0.01, 0.0, np.pi - 0.01]))

    def test_validate_argument_periastron_array_invalid(self) -> None:
        """Test argument of periastron validation with out-of-range array value."""
        para = Parameterisation("P K e w Tp")
        with pytest.raises(ValueError, match="some values not in"):
            para._validate_argument_periastron(np.array([0.0, np.pi]))  # pi is excluded


# ============================================================================
# Test Parameter and Parameterisation repr/str
# ============================================================================


class TestParameterReprStr:
    """Test string representations."""

    def test_parameterisation_str(self) -> None:
        """Test Parameterisation __str__ includes parameterisation string."""
        para = Parameterisation("P K e w Tp")
        assert "P K e w Tp" in str(para)

    def test_parameterisation_repr(self) -> None:
        """Test Parameterisation __repr__ includes parameterisation string."""
        para = Parameterisation("P K e w Tp")
        assert "P K e w Tp" in repr(para)


# ============================================================================
# Test param_key_to_latex
# ============================================================================


class TestParamKeyToLatex:
    """Test LaTeX label generation for parameter keys."""

    @pytest.mark.parametrize("key, expected", [
        ("P", r"$P$"),
        ("K", r"$K$"),
        ("e", r"$e$"),
        ("w", r"$\omega$"),
    ])
    def test_basic_params(self, key, expected) -> None:
        """Test basic orbital parameter keys."""
        assert param_key_to_latex(key) == expected

    @pytest.mark.parametrize("key, expected", [
        ("P_b", r"$P_b$"),
        ("K_c", r"$K_c$"),
        ("e_b", r"$e_b$"),
        ("w_c", r"$\omega_c$"),
    ])
    def test_planet_suffix(self, key, expected) -> None:
        """Test orbital parameters with planet suffixes."""
        assert param_key_to_latex(key) == expected

    @pytest.mark.parametrize("key, expected", [
        ("Tc", r"$T_{\rm c}$"),
        ("Tp", r"$T_{\rm p}$"),
        ("Tc_b", r"$T_{{\rm c},b}$"),
        ("Tp_c", r"$T_{{\rm p},c}$"),
    ])
    def test_time_params(self, key, expected) -> None:
        """Test Tc and Tp with and without planet suffix."""
        assert param_key_to_latex(key) == expected

    @pytest.mark.parametrize("key, expected", [
        ("secosw_b", r"$\sqrt{e}\cos\omega_b$"),
        ("sesinw_b", r"$\sqrt{e}\sin\omega_b$"),
        ("secosw", r"$\sqrt{e}\cos\omega$"),
        ("sesinw", r"$\sqrt{e}\sin\omega$"),
    ])
    def test_sqrt_parameterisation(self, key, expected) -> None:
        """Test sqrt(e)cos(w) and sqrt(e)sin(w) parameterisation."""
        assert param_key_to_latex(key) == expected

    @pytest.mark.parametrize("key, expected", [
        ("ecosw_b", r"$e\cos\omega_b$"),
        ("esinw_b", r"$e\sin\omega_b$"),
        ("ecosw", r"$e\cos\omega$"),
        ("esinw", r"$e\sin\omega$"),
    ])
    def test_ecosw_esinw_parameterisation(self, key, expected) -> None:
        """Test e*cos(w) and e*sin(w) parameterisation."""
        assert param_key_to_latex(key) == expected

    @pytest.mark.parametrize("key, expected", [
        ("jit_HARPS", r"$\sigma_{\rm HARPS}$"),
        ("jit_HARPS_15", r"$\sigma_{\rm HARPS_15}$"),
        ("g_ESPRESSO", r"$\gamma_{\rm ESPRESSO}$"),
    ])
    def test_instrument_params(self, key, expected) -> None:
        """Test jitter and gamma instrument parameters."""
        assert param_key_to_latex(key) == expected

    def test_trend_params(self) -> None:
        """Test linear and quadratic trend parameters."""
        assert param_key_to_latex("gd") == r"$\dot{\gamma}$"
        assert param_key_to_latex("gdd") == r"$\ddot{\gamma}$"

    @pytest.mark.parametrize("key, expected", [
        ("gp_amp", r"$A$"),
        ("gp_period", r"$P_{\rm GP}$"),
        ("gp_lambda_e", r"$\lambda_e$"),
        ("gp_lambda_p", r"$\lambda_p$"),
    ])
    def test_gp_hyperparams(self, key, expected) -> None:
        """Test GP kernel hyperparameters."""
        assert param_key_to_latex(key) == expected

class TestLabelCoverage:
    """Ensure all parameterisations and kernels are covered by label/unit functions."""

    def _get_all_known_keys(self) -> set[str]:
        """Collect all parameter keys that the model can produce."""
        keys = set()

        # Orbital params from all parameterisations (with a test suffix)
        for param_str in ALLOWED_PARAMETERISATIONS:
            para = Parameterisation(param_str)
            for base in para.pars:
                keys.add(base)            # bare key
                keys.add(f"{base}_b")     # with planet suffix

        # GP hyperparams from all supported kernels
        for kernel_type in SUPPORTED_KERNELS:
            kernel = GPKernel(kernel_type)
            for hp in kernel.expected_hyperparams:
                keys.add(hp)

        # Trend and instrument params (always present)
        keys.update(["gd", "gdd", "jit_TEST", "g_TEST"])

        return keys

    def test_all_keys_have_latex(self) -> None:
        """Every known parameter key must be handled by param_key_to_latex."""
        for key in self._get_all_known_keys():
            result = param_key_to_latex(key)
            assert result != key, f"param_key_to_latex returned key unchanged for '{key}'"

    def test_all_keys_have_unit(self) -> None:
        """Every known parameter key must be handled by param_key_to_unit."""
        for key in self._get_all_known_keys():
            result = param_key_to_unit(key)
            assert result is not None, f"param_key_to_unit returned None for '{key}'"


class TestLabelFallbacks:
    """Test that unrecognised keys hit the fallback correctly."""

    def test_latex_fallback_returns_key_unchanged(self) -> None:
        """Unrecognised key should be returned unchanged."""
        assert param_key_to_latex("foo_bar") == "foo_bar"
        assert param_key_to_latex("xyz") == "xyz"

    def test_unit_fallback_returns_none(self) -> None:
        """Unrecognised key should return None."""
        assert param_key_to_unit("foo_bar") is None
        assert param_key_to_unit("xyz") is None
