# basis.py

BASIS_STR = ["per k e w tp",
           "per k e w tc",
           "per k ecosw esinw tp",
           "per k ecosw esinw tc",
           "per k secosw sesinw tp",
           "per k secosw sesinw tp"]

class Basis:

    def __init__(self, name: str):
        if name not in BASIS_STR:
            raise ValueError(f"Basis {name} not recognised")
        self.name = name
        self.pars = name.split()