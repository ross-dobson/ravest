import numpy as np
import pytest

from ravest.model import Instrument, Planet, Star, Trend, _compute_rv, _njit_kepler_rv
from ravest.param import Parameterisation


def known_params1():
    return {"P": 13.2, "K": 27, "e": 0.2, "w": 0.9 * np.pi, "Tp": 2}


def known_params2():
    return {"P": 1.5, "K": 10, "e": 0, "w": np.pi / 2, "Tp": 0}


def data_rv1():
    return np.loadtxt("tests/data/rv1.txt")


def data_rv2():
    return np.loadtxt("tests/data/rv2.txt")


def data_tarr():
    return np.arange(0, 100, 0.1)


def good_planet1():
    return Planet(
        letter="b",
        parameterisation=Parameterisation("P K e w Tp"),
        params=known_params1(),
    )


def good_planet2():
    return Planet(
        letter="c",
        parameterisation=Parameterisation("P K e w Tp"),
        params=known_params2(),
    )


def good_trend():
    return Trend(t0=1000, params={"gd": 0, "gdd": 0})


def good_star():
    return Star(name="goodstar", mass=1)


def test_planet_letter_valueerror_if_multiple_letters() -> None:
    with pytest.raises(ValueError):
        Planet(
            letter="abcdefg",
            parameterisation=Parameterisation("P K e w Tp"),
            params=known_params1(),
        )


def test_planet_letter_valuerror_if_not_isalpha() -> None:
    with pytest.raises(ValueError):
        Planet(
            letter="!",
            parameterisation=Parameterisation("P K e w Tp"),
            params=known_params1(),
        )
    with pytest.raises(ValueError):
        Planet(
            letter="5",
            parameterisation=Parameterisation("P K e w Tp"),
            params=known_params1(),
        )


def test_planet_letter_good() -> None:
    assert (
        Planet(
            letter="b",
            parameterisation=Parameterisation("P K e w Tp"),
            params=known_params1(),
        ).letter
        == "b"
    )


def test_star_num_planets() -> None:
    star = good_star()
    star.add_planet(good_planet1())
    new_planet = Planet(letter="b", parameterisation=Parameterisation("P K e w Tp"), params=known_params2())

    # Test that overwriting triggers a warning
    with pytest.warns(UserWarning, match="Planet b already exists and will be overwritten"):
        star.add_planet(new_planet)

    assert star.num_planets == 1  # check the new planet "b" has replaced the old "b"


def test_rv_pkewtp_eccentric() -> None:
    b = good_planet1()
    rv1 = b.radial_velocity(data_tarr())
    assert list(data_rv1()) == pytest.approx(list(rv1))


def test_rv_pkewtp_circular() -> None:
    c = good_planet2()
    rv2 = c.radial_velocity(data_tarr())
    assert list(data_rv2()) == pytest.approx(list(rv2))


# def test_star_rv():
#     b = good_planet1()
#     c = good_planet2()
#     star = good_star()
#     star.add_planet(b)
#     star.add_planet(c)
# star.add_trend(good_trend())
# star_rv = star.radial_velocity(data_tarr())
# assert list(data_rv1()+data_rv2()) == pytest.approx(list(star_rv))


# ============================================================================
# Instrument class tests
# ============================================================================


def test_instrument_creation() -> None:
    """Test valid Instrument creation."""
    inst = Instrument("HARPS", g=5.0, jit=2.0)
    assert inst.name == "HARPS"
    assert inst.g == 5.0
    assert inst.jit == 2.0


def test_instrument_negative_jitter_raises() -> None:
    """Test that negative jitter raises ValueError."""
    with pytest.raises(ValueError, match="Jitter must be >= 0"):
        Instrument("HARPS", g=5.0, jit=-1.0)


def test_instrument_empty_name_raises() -> None:
    """Test that empty instrument name raises ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        Instrument("", g=5.0, jit=2.0)


def test_instrument_repr_str() -> None:
    """Test Instrument string representations."""
    inst = Instrument("HARPS", g=5.0, jit=2.0)
    assert "HARPS" in repr(inst)
    assert "5.0" in repr(inst)
    assert "HARPS" in str(inst)
    assert "γ=5.0" in str(inst)


def test_instrument_zero_jitter_allowed() -> None:
    """Test that zero jitter is allowed."""
    inst = Instrument("HARPS", g=0.0, jit=0.0)
    assert inst.jit == 0.0


# ============================================================================
# Star instrument tests
# ============================================================================


def test_star_add_instrument() -> None:
    """Test adding instruments to Star."""
    star = good_star()
    harps = Instrument("HARPS", g=5.0, jit=2.0)
    hires = Instrument("HIRES", g=-3.0, jit=1.5)

    star.add_instrument(harps)
    star.add_instrument(hires)

    assert "HARPS" in star.instruments
    assert "HIRES" in star.instruments
    assert star.instruments["HARPS"].g == 5.0
    assert star.instruments["HIRES"].jit == 1.5


def test_star_add_instrument_overwrite_warning() -> None:
    """Test that overwriting an instrument triggers a warning."""
    star = good_star()
    harps1 = Instrument("HARPS", g=5.0, jit=2.0)
    harps2 = Instrument("HARPS", g=10.0, jit=3.0)

    star.add_instrument(harps1)

    with pytest.warns(UserWarning, match="HARPS already exists"):
        star.add_instrument(harps2)

    # Check the new instrument replaced the old one
    assert star.instruments["HARPS"].g == 10.0


def test_star_gamma_offsets() -> None:
    """Test gamma_offsets returns correct array based on instrument column."""
    star = good_star()
    star.add_instrument(Instrument("HARPS", g=5.0, jit=2.0))
    star.add_instrument(Instrument("HIRES", g=-3.0, jit=1.5))

    instrument = np.array(["HARPS", "HARPS", "HIRES", "HARPS", "HIRES"])
    expected = np.array([5.0, 5.0, -3.0, 5.0, -3.0])

    result = star.gamma_offsets(instrument)
    np.testing.assert_array_equal(result, expected)


def test_star_jitter_values() -> None:
    """Test jitter_values returns correct array based on instrument column."""
    star = good_star()
    star.add_instrument(Instrument("HARPS", g=5.0, jit=2.0))
    star.add_instrument(Instrument("HIRES", g=-3.0, jit=1.5))

    instrument = np.array(["HIRES", "HARPS", "HIRES"])
    expected = np.array([1.5, 2.0, 1.5])

    result = star.jitter_values(instrument)
    np.testing.assert_array_equal(result, expected)


# ============================================================================
# Trend class tests
# ============================================================================


def test_trend_creation() -> None:
    """Test valid Trend creation with gd and gdd only."""
    trend = Trend(t0=2458000.0, params={"gd": 0.001, "gdd": 0.0})
    assert trend.gammadot == 0.001
    assert trend.gammadotdot == 0.0
    assert trend.t0 == 2458000.0


def test_trend_radial_velocity_linear() -> None:
    """Test Trend RV with linear term only."""
    trend = Trend(t0=100.0, params={"gd": 0.5, "gdd": 0.0})
    t = np.array([100.0, 101.0, 102.0, 99.0])
    expected = np.array([0.0, 0.5, 1.0, -0.5])

    result = trend.radial_velocity(t)
    np.testing.assert_array_almost_equal(result, expected)


def test_trend_radial_velocity_quadratic() -> None:
    """Test Trend RV with quadratic term only."""
    trend = Trend(t0=0.0, params={"gd": 0.0, "gdd": 1.0})
    t = np.array([0.0, 1.0, 2.0, -1.0])
    expected = np.array([0.0, 1.0, 4.0, 1.0])

    result = trend.radial_velocity(t)
    np.testing.assert_array_almost_equal(result, expected)


def test_trend_repr_str() -> None:
    """Test Trend string representations."""
    trend = Trend(t0=1000.0, params={"gd": 0.5, "gdd": 0.1})
    assert "gd" in repr(trend) or "0.5" in repr(trend)
    assert "1000" in repr(trend)


# ============================================================================
# Numba Kepler solver tests
# ============================================================================


def _numpy_reference_rv(M, e, K, w):
    """Reference numpy/scipy implementation for validation against numba solver."""
    from scipy.optimize import newton

    def kepler_eq(E, M_i, e):
        return E - e * np.sin(E) - M_i

    def kepler_eq_prime(E, M_i, e):
        return 1 - e * np.cos(E)

    E = np.array([newton(kepler_eq, Mi, fprime=kepler_eq_prime, args=(Mi, e))
                  for Mi in M])
    f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    return K * (np.cos(f + w) + e * np.cos(w))


def test_numba_kepler_rv_matches_scipy_eccentric() -> None:
    """Test that numba solver matches scipy newton for eccentric orbits."""
    M = np.linspace(0, 2 * np.pi, 200)
    e, K, w = 0.3, 25.0, 1.2

    rv_numba = _njit_kepler_rv(M, e, K, w)
    rv_scipy = _numpy_reference_rv(M, e, K, w)

    np.testing.assert_allclose(rv_numba, rv_scipy, atol=1e-6, rtol=1e-10)


def test_numba_kepler_rv_matches_scipy_high_eccentricity() -> None:
    """Test numba solver at high eccentricity (e=0.8)."""
    M = np.linspace(0, 2 * np.pi, 200)
    e, K, w = 0.8, 50.0, 2.5

    rv_numba = _njit_kepler_rv(M, e, K, w)
    rv_scipy = _numpy_reference_rv(M, e, K, w)

    np.testing.assert_allclose(rv_numba, rv_scipy, atol=1e-6, rtol=1e-10)


def test_compute_rv_circular_dispatch() -> None:
    """Test that _compute_rv returns correct RV for circular orbits (e=0)."""
    M = np.linspace(0, 2 * np.pi, 100)
    K, w = 10.0, np.pi / 4

    rv = _compute_rv(M, 0.0, K, w)
    expected = K * np.cos(M + w)

    np.testing.assert_allclose(rv, expected, atol=1e-12)
