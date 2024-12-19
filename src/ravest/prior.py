# prior.py
import numpy as np

PRIOR_FUNCTIONS = ["Uniform", "Gaussian", "EccentricityPrior", "BoundedGaussian"]

class Uniform:
    r"""Log of uniform prior distribution.
        
    The log uniform prior function is defined as:
    .. math::
        -\log{b - a} \quad \text{for} \quad a \leq x \leq b \\
        -\inf \quad \text{otherwise} \\
    
    Parameters
    ----------
    lower : float
        Lower bound of the uniform distribution.
    upper : float
        Upper bound of the uniform distribution.
    
    Returns
    -------
    float
        Logarithm of the prior probability density function.
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        # TODO: warnings if non-finite values?

    def __call__(self, value):
        if value < self.lower or value > self.upper:
            return -np.inf
        else:
            return -np.log(self.upper-self.lower)
    def __repr__(self):
        return f"Uniform({self.lower}, {self.upper})"


class Gaussian:
    r"""Log of Gaussian prior distribution.

    The log Gaussian prior function is defined as:
    .. math::
        -0.5 \left( \frac{x - \mu}{\sigma} \right)^2 - 0.5 \log{2 \pi \sigma^2} \\

    Parameters
    ----------
    mean : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.
    
    Returns
    -------
    float
        Logarithm of the prior probability density function.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, value):
        return -0.5 * ((value - self.mean) / self.std)**2 - 0.5*np.log((self.std**2)*2.*np.pi)
    
    def __repr__(self):
        return f"Gaussian({self.mean}, {self.std})"


class EccentricityPrior:
    r"""Uniform prior for eccentricity. Lower bound must = 0, upper must < 1.

    The log eccentricity prior function is defined as:
    .. math::
        -\log{b} \quad \text{for} \quad 0 \leq x \leq b \\
        -\inf \quad \text{otherwise} \\

    Parameters
    ----------
    upper : float
        Upper bound of the uniform distribution.

    Returns
    -------
    float
        Logarithm of the prior probability density function.
    
    Notes
    -----
    This is useful for eccentricity because the normal Uniform prior lower bound 
    is exclusive <, whereas this is inclusive <=, allowing eccentricity to be 0.
    """
    def __init__(self, upper):
        if upper >= 1:
            raise ValueError("Upper bound of eccentricity must be less than 1.")
        if upper <= 0:
            raise ValueError("Upper bound of eccentricity must be greater than 0.")
        self.upper = upper

    def __call__(self, value):
        if value < 0 or value > self.upper:
            return -np.inf
        else:
            return -np.log(self.upper)

    def __repr__(self):
        return f"EccentricityPrior({self.upper})"


class BoundedGaussian:
    r"""Log of Gaussian prior distribution, with bounds.
    
    The log bounded Gaussian prior function is defined as:
    .. math::
        -0.5 \left( \frac{x - \mu}{\sigma} \right)^2 - 0.5 \log{2 \pi \sigma^2} \quad \text{for} \quad a \leq x \leq b
        -\inf \quad \text{otherwise}

    Use cases may include where you have a preferred value for a parameter, but
    you know it is bounded within a certain range due to physical constraints
    (e.g. ensuring a value stays positive).
        
    Parameters
    ----------
    mean : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.
    lower : float
        Lower bound of the Gaussian distribution.
    upper : float
        Upper bound of the Gaussian distribution.
        
    Returns
    -------
    float
        Logarithm of the prior probability density function.
    """
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
    