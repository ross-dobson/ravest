from ravest.model import Planet, Star, Trend
from ravest.param import Parameterisation
import numpy as np
import pytest

def known_params1():
    return {"per": 13.2,
            "k": 27,
            "e": 0.2,
            "w": 0.9*np.pi,
            "tp": 2}
def known_params2():
    return {"per": 1.5,
            "k": 10,
            "e": 0,
            "w": np.pi/2,
            "tp": 0}

def data_rv1():
    return np.loadtxt("tests/data/rv1.txt")
def data_rv2():
    return np.loadtxt("tests/data/rv2.txt")
def data_tarr():
    return np.arange(0,100,0.1)


def good_planet1():
    return Planet(letter="b", parameterisation=Parameterisation("per k e w tp"), params=known_params1())
def good_planet2():
    return Planet(letter="c", parameterisation=Parameterisation("per k e w tp"), params=known_params2())

def good_trend():
    return Trend({"g":0, "gd":0, "gdd":0})

def good_star():
    return Star(name="goodstar", mass=1)

def test_planet_letter_valueerror_if_multiple_letters():
    with pytest.raises(ValueError):
        Planet(letter="abcdefg", parameterisation=Parameterisation("per k e w tp"), params=known_params1())
def test_planet_letter_valuerror_if_not_isalpha():
    with pytest.raises(ValueError):
        Planet(letter="!", parameterisation=Parameterisation("per k e w tp"), params=known_params1())
    with pytest.raises(ValueError):
        Planet(letter="5", parameterisation=Parameterisation("per k e w tp"), params=known_params1())
def test_planet_letter_good():
    assert Planet(letter="b", parameterisation=Parameterisation("per k e w tp"), params=known_params1()).letter == "b"

def test_star_num_planets():
    star = good_star()
    star.add_planet(good_planet1())
    new_planet = Planet(letter="b", parameterisation=Parameterisation("per k e w tp"), params=known_params2())
    star.add_planet(new_planet)
    assert star.num_planets == 1  # check the new planet "b" has replaced the old "b"

def test_rv_pkewtp_eccentric():
    b = good_planet1()
    rv1 = b.radial_velocity(data_tarr())
    assert list(data_rv1()) == pytest.approx(list(rv1))

def test_rv_pkewtp_circular():
    c = good_planet2()
    rv2 = c.radial_velocity(data_tarr())
    assert list(data_rv2()) == pytest.approx(list(rv2))

def test_star_rv():
    b = good_planet1()
    c = good_planet2()
    star = good_star()
    star.add_planet(b)
    star.add_planet(c)
    star.add_trend(good_trend())
    star_rv = star.radial_velocity(data_tarr())
    assert list(data_rv1()+data_rv2()) == pytest.approx(list(star_rv))