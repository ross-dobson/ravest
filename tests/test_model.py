from ravest.model import Planet, Star
import numpy as np
import pytest

known_good_params = {"p": 13.2,
                     "k": 27,
                     "e": 0.2,
                     "w": np.pi/2,
                     "tp": 4}

def test_planet_letter_valueerror_if_multiple_letters():
    with pytest.raises(ValueError):
        Planet(letter="abcdefg", basis="p k e w tp", params=known_good_params)