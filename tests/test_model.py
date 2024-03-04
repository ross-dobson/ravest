from ravest.model import Planet, Star
import numpy as np
import pytest

@pytest.fixture
def known_good_params():
    return {"p": 13.2,
            "k": 27,
            "e": 0.2,
            "w": np.pi/2,
            "tp": 4}

def test_planet_letter_valueerror_if_multiple_letters():
    with pytest.raises(ValueError):
        Planet(letter="abcdefg", basis="p k e w tp", params=known_good_params)
def test_planet_letter_valuerror_if_not_isalpha():
    with pytest.raises(ValueError):
        Planet(letter="!", basis="p k e w tp", params=known_good_params)
    with pytest.raises(ValueError):
        Planet(letter="5", basis="p k e w tp", params=known_good_params)
def test_planet_letter_good():
    assert Planet(letter="b", basis="p k e w tp", params=known_good_params).letter == "b"