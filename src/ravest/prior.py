"""Prior probability distributions for Bayesian fitting."""
# prior.py
import numpy as np
from scipy.special import gammaln, logsumexp, xlog1py, xlogy
from scipy.stats import halfnorm, rayleigh, truncnorm

PRIOR_FUNCTIONS = ["Uniform", "EccentricityUniform", "Normal", "TruncatedNormal", "HalfNormal", "Rayleigh", "VanEylen19Mixture", "Beta"]

class Uniform:
    r"""Log of uniform prior distribution, with closed (inclusive) interval [a,b].

    The uniform probability distribution is:
    .. math::
        p(x) = \frac{1}{b - a} \quad \text{for} \quad a \leq x \leq b \\
        0 \quad \text{otherwise}

    The log probability is:
    .. math::
        \log p(x) = -\log{(b - a)} \quad \text{for} \quad a \leq x \leq b \\
        -\inf \quad \text{otherwise}

    Uses closed interval [a, b] - both boundary values are included. Note that
    for usage on eccentricity, we recommend EccentricityUniform instead, which
    uses a half-open interval [0, upper) to allow exactly e=0 (circular orbits)
    whilst excluding e=upper (to allow e up to, but not including, 1).

    Parameters
    ----------
    lower : float
        Lower bound of the uniform distribution.
    upper : float
        Upper bound of the uniform distribution.
    """

    def __init__(self, lower: float, upper: float) -> None:
        if not np.isfinite(lower):
            raise ValueError(f"Lower bound must be finite, got {lower}")
        if not np.isfinite(upper):
            raise ValueError(f"Upper bound must be finite, got {upper}")
        if lower >= upper:
            raise ValueError(f"Lower bound ({lower}) must be less than upper bound ({upper})")
        self.lower = lower
        self.upper = upper

    def __call__(self, value: float) -> float:
        """Calculate log uniform prior probability.

        Parameters
        ----------
        value : float
            Parameter value to evaluate

        Returns
        -------
        float
            Log prior probability
        """
        if value < self.lower or value > self.upper:
            return -np.inf
        else:
            return -np.log(self.upper - self.lower)

    def __repr__(self) -> str:
        return f"Uniform(lower={self.lower}, upper={self.upper})"


class EccentricityUniform:
    r"""Log Uniform prior for eccentricity. Uses half-open interval [0, upper).

    The uniform probability distribution is:
    .. math::
        p(x) = \frac{1}{b} \quad \text{for} \quad 0 \leq x < b \\
        0 \quad \text{otherwise}

    The log probability is:
    .. math::
        \log p(x) = -\log{b} \quad \text{for} \quad 0 \leq x < b \\
        -\inf \quad \text{otherwise}

    Parameters
    ----------
    upper : float
        Upper bound of the uniform distribution. Must satisfy 0 < upper <= 1.

    Notes
    -----
    Uses half-open interval [0, upper) to allow exactly e=0 (circular orbits)
    whilst excluding e=upper (to allow e up to, but not including, 1).
    """

    def __init__(self, upper: float) -> None:
        if upper > 1:
            raise ValueError("Upper bound of eccentricity must be less than or equal to 1.")
        if upper <= 0:
            raise ValueError("Upper bound of eccentricity must be greater than 0.")
        self.upper = upper

    def __call__(self, value: float) -> float:
        """Calculate log eccentricity uniform prior probability.

        Parameters
        ----------
        value : float
            Parameter value to evaluate

        Returns
        -------
        float
            Log prior probability
        """
        if value < 0 or value >= self.upper:
            return -np.inf
        else:
            return -np.log(self.upper)

    def __repr__(self) -> str:
        return f"EccentricityUniform(upper={self.upper})"


class Normal:
    r"""Log of Normal prior distribution.

    The Normal probability distribution is:
    .. math::
        p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    The log probability is:
    .. math::
        \log p(x) = -\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2 - \frac{1}{2}\log(2\pi\sigma^2)

    Parameters
    ----------
    mean : float
        Mean of the Normal distribution.
    std : float
        Standard deviation of the Normal distribution.
    """

    def __init__(self, mean: float, std: float) -> None:
        if std <= 0:
            raise ValueError(f"Standard deviation must be positive, got {std}")
        self.mean = mean
        self.std = std
        self._log_norm_const = 0.5 * np.log((self.std**2)*2.*np.pi)

    def __call__(self, value: float) -> float:
        """Calculate log Normal prior probability.

        Parameters
        ----------
        value : float
            Parameter value to evaluate

        Returns
        -------
        float
            Log prior probability
        """
        return -0.5 * ((value - self.mean) / self.std)**2 - self._log_norm_const


    def __repr__(self) -> str:
        return f"Normal(mean={self.mean}, std={self.std})"


class TruncatedNormal:
    r"""Log of properly normalized truncated Normal prior distribution.

    The truncated Normal probability distribution is:
    .. math::
        p(x) = \frac{\phi\left(\frac{x - \mu}{\sigma}\right)}{\sigma \left[\Phi\left(\frac{b - \mu}{\sigma}\right) - \Phi\left(\frac{a - \mu}{\sigma}\right)\right]} \quad \text{for} \quad a \leq x \leq b \\
        0 \quad \text{otherwise}

    where lowercase phi is the standard normal PDF and uppercase Phi is the
    standard normal CDF.

    The log probability is:
    .. math::
        \log p(x) = \log\phi\left(\frac{x - \mu}{\sigma}\right) - \log\sigma - \log\left[\Phi\left(\frac{b - \mu}{\sigma}\right) - \Phi\left(\frac{a - \mu}{\sigma}\right)\right] \quad \text{for} \quad a \leq x \leq b \\
        -\inf \quad \text{otherwise}

    This provides a proper probability distribution that integrates to 1 over
    [a, b]. This is useful for parameters e.g. that can't go negative or
    that are bounded between a lower and upper value, but where you want a
    more informative prior than a uniform distribution.

    Parameters
    ----------
    mean : float
        Mean of the original (untruncated) Normal distribution.
    std : float
        Standard deviation of the original Normal distribution.
    lower : float
        Lower bound of the truncation.
    upper : float
        Upper bound of the truncation.
    """

    def __init__(self, mean: float, std: float, lower: float, upper: float) -> None:
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

    def __call__(self, value: float) -> float:
        """Calculate log truncated Normal prior probability.

        Parameters
        ----------
        value : float
            Parameter value to evaluate

        Returns
        -------
        float
            Log prior probability
        """
        if value < self.lower or value > self.upper:
            return -np.inf
        else:
            return truncnorm.logpdf(value, self._a, self._b, loc=self.mean, scale=self.std)

    def __repr__(self) -> str:
        return f"TruncatedNormal(mean={self.mean}, std={self.std}, lower={self.lower}, upper={self.upper})"


class HalfNormal:
    r"""Log of half-Normal prior distribution.

    The half-Normal probability distribution is:
    .. math::
        p(x) = \frac{2}{\sigma\sqrt{2\pi}} \exp\left(-\frac{x^2}{2\sigma^2}\right) \quad \text{for} \quad x \geq 0 \\
        0 \quad \text{otherwise}

    The log probability is:
    .. math::
        \log p(x) = \log(2) - \log(\sigma) - \frac{1}{2}\log(2\pi) - \frac{x^2}{2\sigma^2} \quad \text{for} \quad x \geq 0 \\
        -\inf \quad \text{otherwise}

    This is equivalent to a Normal distribution with mean=0 that has been
    truncated at x=0 to only allow non-negative values (equivalently, the
    absolute value of a Normal distribution with mean=0).

    This can be useful for parameters that must be positive, such as
    standard deviations, measurement uncertainties, or jitter terms.

    Parameters
    ----------
    std : float
        Standard deviation (sigma) of the half-Normal distribution. Must be > 0.
    """

    def __init__(self, std: float) -> None:
        if std <= 0:
            raise ValueError(f"Standard deviation must be positive, got {std}")
        self.std = float(std)

    def __call__(self, value: float) -> float:
        """Calculate log half-Normal prior probability.

        Parameters
        ----------
        value : float
            Parameter value to evaluate

        Returns
        -------
        float
            Log prior probability
        """
        if value < 0.0:
            return -np.inf
        else:
            return halfnorm.logpdf(value, scale=self.std)

    def __repr__(self) -> str:
        return f"HalfNormal(std={self.std})"


class Rayleigh:
    r"""Log of Rayleigh prior distribution.

    The Rayleigh probability distribution is:
    .. math::
        p(x) = \frac{x}{\sigma^2} \exp\left(-\frac{x^2}{2\sigma^2}\right) \quad \text{for} \quad x \geq 0 \\
        0 \quad \text{otherwise}

    The log probability is:
    .. math::
        \log p(x) = \log(x) - 2\log(\sigma) - \frac{x^2}{2\sigma^2} \quad \text{for} \quad x \geq 0 \\
        -\inf \quad \text{otherwise}

    Parameters
    ----------
    scale : float
        Scale parameter (sigma) of the Rayleigh distribution. Must be > 0.

    Notes
    -----
    The Rayleigh prior is zero at x=0 (log prior is -inf). If you expect
    significant probability mass near zero, consider using another prior such as
    the HalfNormal or VanEylen19Mixture prior instead.
    """

    def __init__(self, scale: float) -> None:
        if scale <= 0:
            raise ValueError(f"Scale parameter must be positive, got {scale}")
        self.scale = float(scale)

    def __call__(self, value: float) -> float:
        """Calculate log Rayleigh prior probability.

        Parameters
        ----------
        value : float
            Parameter value to evaluate

        Returns
        -------
        float
            Log prior probability
        """
        if value < 0.0:
            return -np.inf
        else:
            return rayleigh.logpdf(value, scale=self.scale)

    def __repr__(self) -> str:
        return f"Rayleigh(scale={self.scale})"


class VanEylen19Mixture:
    r"""Log of a Rayleigh & Half-Normal mixture model prior distribution.

    The mixture model combines a half-Normal and a Rayleigh distribution,
    weighted by a mixing fraction f. This model is particularly useful for
    eccentricity priors in exoplanet systems, where the half-Normal component
    captures low eccentricities, and the Rayleigh component captures higher
    eccentricities, as described in Van Eylen et al. (2019).

    The mixture probability distribution is:
    .. math::
        p(x) = (1-f) \cdot p_{\text{HalfNormal}}(x; \sigma_{\text{normal}}) + f \cdot p_{\text{Rayleigh}}(x; \sigma_{\text{rayleigh}})

    The log probability is:
    .. math::
        \log p(x) = \log\left[(1-f) \cdot p_{\text{HalfNormal}}(x; \sigma_{\text{normal}}) + f \cdot p_{\text{Rayleigh}}(x; \sigma_{\text{rayleigh}})\right]

    where:
    - f = 0 indicates a pure half-Normal distribution (low eccentricities)
    - f = 1 indicates a pure Rayleigh distribution (higher eccentricities)
    - 0 < f < 1 represents a mixture of both components

    Parameters
    ----------
    sigma_normal : float
        Scale parameter for the half-Normal component. Must be > 0.
    sigma_rayleigh : float
        Scale parameter for the Rayleigh component. Must be > 0.
    f : float
        Mixing fraction between 0 and 1. f=0 gives pure half-Normal,
        f=1 gives pure Rayleigh.

    References
    ----------
    Vincent Van Eylen et al 2019 AJ 157 61 (https://doi.org/10.3847/1538-3881/aaf22f)
    """

    def __init__(self, sigma_normal: float, sigma_rayleigh: float, f: float) -> None:
        if sigma_normal <= 0:
            raise ValueError(f"sigma_normal must be positive, got {sigma_normal}")
        if sigma_rayleigh <= 0:
            raise ValueError(f"sigma_rayleigh must be positive, got {sigma_rayleigh}")
        if not (0 <= f <= 1):
            raise ValueError(f"Mixing fraction f must be between 0 and 1, got {f}")

        self.sigma_normal = float(sigma_normal)
        self.sigma_rayleigh = float(sigma_rayleigh)
        self.f = float(f)

    def __call__(self, value: float) -> float:
        """Calculate log mixture prior probability.

        Parameters
        ----------
        value : float
            Parameter value to evaluate

        Returns
        -------
        float
            Log prior probability
        """
        if value < 0.0:
            return -np.inf

        # Get log probabilities from each component
        log_halfnorm = halfnorm.logpdf(value, scale=self.sigma_normal)
        log_rayleigh = rayleigh.logpdf(value, scale=self.sigma_rayleigh)

        # Compute log of mixture: log((1-f)*p1 + f*p2)
        # Use logsumexp for numerical stability
        return logsumexp([log_halfnorm, log_rayleigh], b=[1 - self.f, self.f])

    def __repr__(self) -> str:
        return f"VanEylen19Mixture(sigma_normal={self.sigma_normal}, sigma_rayleigh={self.sigma_rayleigh}, f={self.f})"


class Beta:
    r"""Log of Beta prior distribution, for parameter x where 0 <= x <= 1.

    The Beta probability distribution is:
    .. math::
        p(x) = \frac{x^{a-1}(1-x)^{b-1}}{B(a,b)} \quad \text{for} \quad 0 \leq x \leq 1 \\
        0 \quad \text{otherwise}

    where B(a,b) is the beta function.

    The log probability is:
    .. math::
        \log p(x) = (a - 1)\log(x) + (b - 1)\log(1-x) - \log B(a,b) \quad \text{for} \quad 0 \leq x \leq 1 \\
        -\inf \quad \text{otherwise}

    Parameters
    ----------
    a : float
        Shape parameter a of the Beta distribution. Must be > 0.
    b : float
        Shape parameter b of the Beta distribution. Must be > 0.

    Notes
    -----
    The Beta distribution's behaviour at the boundaries depends on the shape
    parameters. Consider the shape of your Beta distribution if you expect to
    have significant probability mass near 0 or 1, where probability may be 0
    or infinite (depending on the shape parameters).
    """

    def __init__(self, a: float, b: float) -> None:
        if not a > 0:
            raise ValueError(f"Value of a > 0 required, got {a}")
        if not b > 0:
            raise ValueError(f"Value of b > 0 required, got {b}")
        self.a = float(a)
        self.b = float(b)
        # Pre-compute log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))
        self._log_beta = gammaln(self.a) + gammaln(self.b) - gammaln(self.a + self.b)

    def __call__(self, value: float) -> float:
        """Calculate log Beta prior probability.

        Parameters
        ----------
        value : float
            Parameter value to evaluate

        Returns
        -------
        float
            Log prior probability
        """
        if value < 0.0 or value > 1.0:
            return -np.inf
        else:
            # Use xlogy and xlog1py for numerical stability
            # (a-1) * log(x) + (b-1) * log(1-x) - log(B(a,b))
            return xlogy(self.a - 1, value) + xlog1py(self.b - 1, -value) - self._log_beta

    def __repr__(self) -> str:
        return f"Beta(a={self.a}, b={self.b})"
