# prior.py
import numpy as np
from scipy.special import gammaln, xlog1py, xlogy
from scipy.stats import halfnorm, truncnorm

PRIOR_FUNCTIONS = ["Uniform", "Gaussian", "EccentricityPrior", "TruncatedGaussian", "Beta", "HalfGaussian"]


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
            return -np.log(self.upper - self.lower)

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


class TruncatedGaussian:
    r"""Log of properly normalized truncated Gaussian prior distribution.

    The log truncated Gaussian prior function is defined as:
    .. math::
        \log \left( \frac{1}{\sigma} \phi\left(\frac{x - \mu}{\sigma}\right) \right) - \log \left( \Phi\left(\frac{b - \mu}{\sigma}\right) - \Phi\left(\frac{a - \mu}{\sigma}\right) \right) \quad \text{for} \quad a \leq x \leq b \\
        -\inf \quad \text{otherwise}

    where lowercase phi is the standard normal PDF and uppercase Phi is the standard normal CDF.

    This provides a proper probability distribution that integrates to 1 over [a, b].
    Use cases include parameters with physical bounds, such as quantities that can't go negative
    or that are bounded between a lower and upper value.

    This implementation uses scipy.stats.truncnorm for proper normalisation,
    ensuring this integrates to 1 over the truncated interval.

    Parameters
    ----------
    mean : float
        Mean of the original (untruncated) Gaussian distribution.
    std : float
        Standard deviation of the original Gaussian distribution.
    lower : float
        Lower bound of the truncation.
    upper : float
        Upper bound of the truncation.

    Returns
    -------
    float
        Logarithm of the prior probability density function.

    """

    def __init__(self, mean, std, lower, upper):
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")

        self.mean = mean
        self.std = std
        self.lower = lower
        self.upper = upper

        # Convert to standard truncnorm parameters
        self._a = (lower - mean) / std  # Lower bound in standard units
        self._b = (upper - mean) / std  # Upper bound in standard units

    def __call__(self, value):
        if value < self.lower or value > self.upper:
            return -np.inf
        else:
            return truncnorm.logpdf(value, self._a, self._b, loc=self.mean, scale=self.std)

    def __repr__(self):
        return f"TruncatedGaussian({self.mean}, {self.std}, {self.lower}, {self.upper})"


class Beta:
    r"""Log of Beta prior distribution, for parameter x where 0 <= x <= 1.

    The log Beta prior function is defined as:
    .. math::
        (a - 1) \log{x} + (b - 1) \log{(1-x)} - \log{B(a, b)} \quad \text{for} \quad 0 \leq x \leq 1 \\
        -\inf \quad \text{otherwise} \\

    Parameters
    ----------
    a : float
        Shape parameter a of the Beta distribution. Must be > 0.
    b : float
        Shape parameter b of the Beta distribution. Must be > 0.

    Returns
    -------
    float
        Logarithm of the prior probability density function.
    """

    def __init__(self, a: float, b: float):
        if not a > 0:
            raise ValueError(f"Value of a > 0 required, got {a}")
        if not b > 0:
            raise ValueError(f"Value of b > 0 required, got {b}")
        self.a = float(a)
        self.b = float(b)
        # Pre-compute log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))
        self._log_beta = gammaln(self.a) + gammaln(self.b) - gammaln(self.a + self.b)

    def __call__(self, value):
        if value < 0.0 or value > 1.0:
            return -np.inf
        else:
            # Use xlogy and xlog1py for numerical stability
            # (a-1) * log(x) + (b-1) * log(1-x) - log(B(a,b))
            return xlogy(self.a - 1, value) + xlog1py(self.b - 1, -value) - self._log_beta

    def __repr__(self):
        return f"Beta({self.a}, {self.b})"


class HalfGaussian:
    r"""Log of half-Gaussian prior distribution.

    The log half-Gaussian prior function is defined as:
    .. math::
        \log \left( \frac{2}{\sigma \sqrt{2\pi}} \exp\left(-\frac{x^2}{2\sigma^2}\right) \right) \quad \text{for} \quad x \geq 0 \\
        -\inf \quad \text{otherwise}

    This is equivalent to a Gaussian distribution with mean=0 that has been
    folded about zero (or truncated at zero with the remaining mass redistributed).

    Commonly used for scale parameters that must be positive, such as
    standard deviations, measurement uncertainties, or jitter terms.

    Parameters
    ----------
    scale : float
        Scale parameter σ (sigma) of the half-Gaussian distribution. Must be > 0.

    Returns
    -------
    float
        Logarithm of the prior probability density function.
    """

    def __init__(self, scale: float):
        if scale <= 0:
            raise ValueError(f"Scale parameter must be positive, got {scale}")
        self.scale = float(scale)

    def __call__(self, value):
        if value < 0.0:
            return -np.inf
        else:
            return halfnorm.logpdf(value, scale=self.scale)

    def __repr__(self):
        return f"HalfGaussian({self.scale})"
