# prior.py
import numpy as np


class Uniform:
    
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        # TODO: warnings if non-finite values?

    def __call__(self, value):
        if value < self.lower or value > self.upper:
            return -np.inf
        else:
            return 0.0
    def __repr__(self):
        return f"Uniform({self.lower}, {self.upper})"


class Gaussian:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, value):
        return -0.5 * ((value - self.mean) / self.std)**2 - 0.5*np.log((self.std**2)*2.*np.pi)
    
    def __repr__(self):
        return f"Gaussian({self.mean}, {self.std})"


class EccentricityPrior:
    
    def __init__(self, upper):
        self.upper = upper

    def __call__(self, value):
        if value < 0.0 or value > self.upper:
            return -np.inf
        else:
            return 0.0

    def __repr__(self):
        return f"EccentricityPrior({self.upper})"


class BoundedGaussian:

    def __init__(self, mean, std, lower, upper):
        self.mean = mean
        self.std = std
        self.lower = lower
        self.upper = upper
        # TODO: warnings if non-finite values?

    def __call__(self, value):
        if value < self.lower or value > self.upper:
            return -np.inf
        else:
            return -0.5 * ((value - self.mean) / self.std)**2 - 0.5*np.log((self.std**2)*2.*np.pi)
        
    def __repr__(self):
        return f"BoundedGaussian({self.mean}, {self.std}, {self.lower}, {self.upper})"
    