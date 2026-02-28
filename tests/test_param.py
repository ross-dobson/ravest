import numpy as np
import pytest

from ravest.param import Parameterisation

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

ALLOWED_PARAMETERISATIONS = ["P K e w Tp",
                             "P K e w Tc",
                             "P K ecosw esinw Tp",
                             "P K ecosw esinw Tc",
                             "P K secosw sesinw Tp",
                             "P K secosw sesinw Tc"]

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
