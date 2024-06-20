# basis.py
import numpy as np

BASIS_STR = ["per k e w tp",
           "per k e w tc",
           "per k ecosw esinw tp",
           "per k ecosw esinw tc",
           "per k secosw sesinw tp",
           "per k secosw sesinw tc"]

class Basis:

    def __init__(self, parameterisation: str):
        if parameterisation not in BASIS_STR:
            raise ValueError(f"Paramterisation {parameterisation} not recognised. Must be one of {BASIS_STR}")
        self.parameterisation = parameterisation
        self.pars = parameterisation.split()


