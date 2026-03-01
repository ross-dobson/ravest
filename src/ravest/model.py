"""Radial velocity models for planets, stars, and trends.

Provides classes for modelling radial velocity signals from planetary orbits
and stellar trends (constant, linear, quadratic).
"""
# model.py

import math

import matplotlib.pyplot as plt
import numba
import numpy as np
from astropy import constants as const
from matplotlib.ticker import MultipleLocator
from scipy import constants

from ravest.param import Parameterisation

# Kepler solver functions are module-level rather than Planet methods because
# numba's @njit cannot compile methods on Python classes.
# They are called by Planet.radial_velocity() via _compute_rv().

@numba.njit(cache=True)
def _solve_kepler(Mi: float, e: float) -> tuple[float, float]:
    """Solve Kepler's equation for a single mean anomaly value.

    Kepler's equation E - e*sin(E) = M must be solved iteratively for the
    eccentric anomaly E. Uses Halley's method (cubic convergence) with the
    same tolerance and iteration limits as scipy's newton solver.

    Parameters
    ----------
    Mi : float
        Mean anomaly at a single time (radians).
    e : float
        Eccentricity of the orbit, 0 < e < 1 (dimensionless).

    Returns
    -------
    cos_E : float
        Cosine of the eccentric anomaly.
    sin_E : float
        Sine of the eccentric anomaly.
    """
    tol = 1.48e-08  # scipy default tolerance
    maxiter = 50    # scipy default max iterations

    # Initial guess E0 = M (good for low-moderate eccentricity)
    Ei = Mi
    sin_E = cos_E = 0.0

    for _ in range(maxiter):
        sin_E = math.sin(Ei)
        cos_E = math.cos(Ei)

        # f(E) = E - e*sin(E) - M and its first two derivatives
        f = Ei - e * sin_E - Mi        # f(E)
        fp = 1.0 - e * cos_E           # f'(E)
        fpp = e * sin_E                # f''(E)

        # Halley's method: E_new = E - f / (f' - f*f''/(2*f'))
        E_new = Ei - f / (fp - (f * fpp) / (2.0 * fp))

        if abs(E_new - Ei) < tol:
            sin_E = math.sin(E_new)
            cos_E = math.cos(E_new)
            break
        Ei = E_new

    return cos_E, sin_E


@numba.njit(cache=True)
def _true_anomaly(cos_E: float, sin_E: float, e: float, sqrt_1me2: float) -> tuple[float, float]:
    """Compute cos(f) and sin(f) of the true anomaly from the eccentric anomaly.

    Calculates the true anomaly directly from cos(E) and sin(E) rather than
    via the arctan formula, avoiding a trig call per element.

    Parameters
    ----------
    cos_E : float
        Cosine of the eccentric anomaly.
    sin_E : float
        Sine of the eccentric anomaly.
    e : float
        Eccentricity of the orbit (dimensionless).
    sqrt_1me2 : float
        Precomputed sqrt(1 - e^2).

    Returns
    -------
    cos_f : float
        Cosine of the true anomaly.
    sin_f : float
        Sine of the true anomaly.

    Notes
    -----
    Rather than computing f as an angle via the arctan formula
    f = 2*arctan(sqrt((1+e)/(1-e)) * tan(E/2)), we obtain cos(f) and sin(f)
    directly from cos(E) and sin(E):

        cos(f) = (cos(E) - e) / (1 - e*cos(E))
        sin(f) = sqrt(1 - e^2) * sin(E) / (1 - e*cos(E))

    The cos(f) identity is a standard result from the geometry of the
    auxiliary circle (Perryman, Exoplanet Handbook, eq. 2.6). sin(f)
    follows from sin^2(f) = 1 - cos^2(f), which simplifies to
    (1 - e^2)*sin^2(E) / (1 - e*cos(E))^2.

    Since cos(E) and sin(E) are already available from the Kepler solver,
    this avoids any additional trig calls.
    """
    denom = 1.0 - e * cos_E  # common denominator of cos(f) and sin(f)
    cos_f = (cos_E - e) / denom
    sin_f = sqrt_1me2 * sin_E / denom
    return cos_f, sin_f


@numba.njit(cache=True)
def _radial_velocity_from_f(cos_f: float, sin_f: float, K: float,
                            cos_w: float, sin_w: float, e_cos_w: float) -> float:
    """Compute radial velocity from the true anomaly of a single observation.

    Uses the trig identity cos(f + w) = cos(f)*cos(w) - sin(f)*sin(w) to
    avoid computing f explicitly.

    Parameters
    ----------
    cos_f : float
        Cosine of the true anomaly.
    sin_f : float
        Sine of the true anomaly.
    K : float
        Semi-amplitude of the stellar radial velocity (m/s).
    cos_w : float
        Precomputed cos(w).
    sin_w : float
        Precomputed sin(w).
    e_cos_w : float
        Precomputed e*cos(w).

    Returns
    -------
    float
        Radial velocity (m/s).

    Notes
    -----
    The RV equation is K * [cos(f + w) + e*cos(w)]. Rather than computing
    the angle f and evaluating cos(f + w), we expand via the addition
    identity cos(f + w) = cos(f)*cos(w) - sin(f)*sin(w), using cos(f) and
    sin(f) from _true_anomaly directly. cos(w), sin(w), and e*cos(w) are
    constant across all observations and precomputed by the caller.
    """
    # RV = K * [cos(f + w) + e*cos(w)]
    return K * (cos_f * cos_w - sin_f * sin_w + e_cos_w)


@numba.njit(cache=True)
def _njit_kepler_rv(M: np.ndarray, e: float, K: float, w: float) -> np.ndarray:
    """Solve Kepler's equation and compute radial velocity for an array of times.

    Loops over mean anomaly values element-by-element, calling the scalar
    Kepler solver, true anomaly, and RV functions for each. Numba inlines
    these calls at compile time, so performance is identical to a single
    monolithic loop while keeping each step readable.

    Parameters
    ----------
    M : np.ndarray
        Mean anomaly at each time (radians).
    e : float
        Eccentricity of the orbit, 0 < e < 1 (dimensionless).
    K : float
        Semi-amplitude of the stellar radial velocity (m/s).
    w : float
        Argument of periapsis of the star (radians).

    Returns
    -------
    np.ndarray
        Radial velocity of the star due to the planet at each time (m/s).
    """
    n = M.shape[0]
    rv = np.empty(n)

    # Precompute constants that are fixed across the time array within a single
    # call. These are recomputed each call, so they update between MCMC samples.
    sqrt_1me2 = math.sqrt(1.0 - e * e)
    cos_w = math.cos(w)
    sin_w = math.sin(w)
    e_cos_w = e * cos_w

    for i in range(n):
        cos_E, sin_E = _solve_kepler(M[i], e)
        cos_f, sin_f = _true_anomaly(cos_E, sin_E, e, sqrt_1me2)
        rv[i] = _radial_velocity_from_f(cos_f, sin_f, K, cos_w, sin_w, e_cos_w)

    return rv


def _compute_rv(M: np.ndarray, e: float, K: float, w: float) -> np.ndarray:
    """Compute radial velocity from mean anomaly, dispatching by eccentricity.

    For circular orbits (e=0) the RV simplifies to K*cos(M + w) and can be
    computed directly with numpy. For eccentric orbits, delegates to the
    numba-compiled solver.

    Parameters
    ----------
    M : np.ndarray
        Mean anomaly at each time (radians).
    e : float
        Eccentricity of the orbit, 0 <= e < 1 (dimensionless).
    K : float
        Semi-amplitude of the stellar radial velocity (m/s).
    w : float
        Argument of periapsis of the star (radians).

    Returns
    -------
    np.ndarray
        Radial velocity of the star due to the planet at each time (m/s).
    """
    if e == 0:
        # For circular orbits, E = M and f = M so the RV is just K*cos(M + w).
        # No need to iterate Kepler's equation — use vectorised numpy directly.
        return K * (np.cos(M + w) + e * np.cos(w))
    return _njit_kepler_rv(M, e, K, w)


class Planet:
    """Planet defined by its orbital parameters.

    Parameters
    ----------
    letter : `str`
        The label of the planet, e.g. "b", "c". Must be a single letter.
    parameterisation : `parameterisation`
        The set of planetary parameters used to define the planet.
    params : `dict`
        The orbital parameters, matching the parameterisation.
    """

    def __init__(self, letter: str, parameterisation: Parameterisation, params: dict[str, float]) -> None:
        if not (letter.isalpha() and (letter == letter[0] * len(letter))):
            raise ValueError(f"Letter {letter} is not a single alphabet character.")
        self.letter = letter
        self.parameterisation = parameterisation
        self.params = params

        # Check the input params and parameterisation match
        if not set(params.keys()) == set(parameterisation.pars):
            raise ValueError(f"Parameterisation {parameterisation} does not match input params {params}")

        # Convert to the default P K e w Tp parameterisation that we need for the RV equation
        self._rvparams = self.parameterisation.convert_pars_to_default_parameterisation(self.params)

        # Validate parameters immediately after conversion to avoid invalid parameters
        # Raises a ValueError if any parameter is invalid
        self.parameterisation.validate_default_parameterisation_params(self._rvparams)


    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}(letter={self.letter!r}, parameterisation={self.parameterisation!r}, params={self.params!r})"

    def __str__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name} {self.letter} {self.params}"

    def _calculate_mean_motion(self, period: float) -> float:
        """Calculate mean motion (mean angular rate of orbit in radians/day).

        This is the mean angular rate calculated by dividing the orbital period by
        2*pi radians.

        Parameters
        ----------
        period : `float`
            The orbital period in days.

        Returns
        -------
        `float`
            The mean motion, the average angular rate of orbit (radians/day).
        """
        return 2 * np.pi / period

    def _calculate_mean_anomaly(self, t: np.ndarray, n: float, time_peri: float) -> np.ndarray:
        """Calculate mean anomaly (radians).

        For an eccentric orbit with period P, mean anomaly is a fictitious
        angle that increases linearly with time ``t`` for an angular rate ``n``, the
        mean motion. It is the angle from the point of periapsis passage that would
        have been swept in a circular orbit of period ``P`` with fixed angular rate
        ``n``.

        Parameters
        ----------
        n : `float`
            The mean motion (radians/day).
        t : `float`
            The time to evaluate the mean anomaly at (day).
        time_peri : `float`
            The time of periapsis of the star's orbit (day).

        Returns
        -------
        `float`
            The mean anomaly (radians).
        """
        return n * (t - time_peri)

    def radial_velocity(self, t: np.ndarray) -> np.ndarray:
        """Calculate radial velocity of the star due to the planet, at time t.

        Uses a numba-compiled Kepler solver that solves Kepler's equation and
        computes the radial velocity in a single pass.

        Parameters
        ----------
        t : np.ndarray
            The time to calculate the radial velocity at (day).

        Returns
        -------
        np.ndarray
            Radial velocity of the reflex motion of star due to the planet (m/s).
        """
        P = self._rvparams["P"]
        K = self._rvparams["K"]
        e = self._rvparams["e"]
        w = self._rvparams["w"]
        tp = self._rvparams["Tp"]

        n = self._calculate_mean_motion(period=P)
        M = self._calculate_mean_anomaly(t=t, n=n, time_peri=tp)

        return _compute_rv(M, e, K, w)

    def mpsini(self, mass_star: float, unit: str = "kg") -> float:
        """Calculate the minimum mass of the planet.

        Parameters
        ----------
        mass_star : `float`
            The mass of the star in solar masses.
        unit : `str`
            The unit to return the planetary minimum mass in. Options are "kg", "M_earth", "M_jupiter".

        Returns
        -------
        `float`
            The minimum mass of the planet (solar masses).
        """
        period = self._rvparams["P"]
        semi_amplitude = self._rvparams["K"]
        eccentricity = self._rvparams["e"]

        # Convert stellar mass to kg and period to seconds for SI unit consistency
        # Formula requires SI units: M_s [kg], P [s], K [m/s]
        mpsini = calculate_mpsini(mass_star, period, semi_amplitude, eccentricity, unit)
        return mpsini


class Instrument:
    """Instrument-specific parameters for RV observations.

    Represents the measurement characteristics of a specific instrument/telescope,
    including its velocity offset (gamma) and additional noise (jitter).

    Parameters
    ----------
    name : str
        The name/identifier of the instrument (e.g., "HARPS").
        Must match the labels used in the data's instrument column.
    g : float
        Gamma offset - the constant RV offset for this instrument [m/s].
    jit : float
        Jitter - additional noise added in quadrature to uncertainties [m/s].
        Must be >= 0.

    Examples
    --------
    >>> harps = Instrument("HARPS", g=5.0, jit=2.0)
    >>> print(harps)
    Instrument HARPS: γ=5.0 m/s, jitter=2.0 m/s

    >>> hires = Instrument("HIRES", g=-3.6, jit=1.5)
    >>> hires.g
    -3.6
    """

    def __init__(self, name: str, g: float, jit: float) -> None:
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError(f"Instrument name must be a non-empty string, got: {name!r}")
        if jit < 0:
            raise ValueError(f"Jitter must be >= 0, got: {jit}")

        self.name = name
        self.g = g
        self.jit = jit

    def __repr__(self) -> str:
        return f"Instrument(name={self.name!r}, g={self.g}, jit={self.jit})"

    def __str__(self) -> str:
        return f"Instrument {self.name}: γ={self.g} m/s, jitter={self.jit} m/s"


class Trend:
    """System-wide trend in the radial velocity of the star.

    Represents long-term velocity trends that apply across all instruments,
    such as acceleration from a distant companion. The constant offset (gamma)
    is now handled per-instrument via the Instrument class.

    Parameters
    ----------
    t0 : float
        The reference zero-point time for the linear and quadratic trend.
        Recommended to be the mean of the input times.
    params : dict
        The parameters of the trend: linear and quadratic components.
        These must be named "gd" and "gdd" respectively (gamma-dot,
        gamma-dot-dot). These are in units of m/s/day and m/s/day^2 respectively.

    Returns
    -------
    float
        The radial velocity of the star due to the trend (m/s).

    Notes
    -----
    The radial velocity contribution from the trend is calculated as the sum of
    the linear and quadratic components: `gd*(t-t0)` and `gdd*((t-t0)**2)`.

    The constant offset (gamma) is now instrument-specific and handled via the
    Instrument class. Use `star.gamma_offsets(instrument)` to get per-instrument
    offsets.

    In general the trend is used to account for long-term effects such as
    acceleration from a distant companion. If you see a strong linear or
    quadratic trend in the data, it is worth investigating.

    Examples
    --------
    >>> trend = Trend(t0=2458000.0, params={"gd": 0.001, "gdd": 0.0})
    >>> rv_trend = trend.radial_velocity(times)
    """

    def __init__(self, t0: float, params: dict[str, float]) -> None:
        self.gammadot = params["gd"]
        self.gammadotdot = params["gdd"]

        # Validate and store reference time t0
        try:
            self.t0 = float(t0)
        except (TypeError, ValueError) as e:
            raise ValueError(f"t0 must be a numeric value (recommend mean or median of observation times), but got {type(t0).__name__}: {t0}") from e

    def __str__(self) -> str:
        return f"Trend: $\\dot\\gamma$={self.gammadot}, $\\ddot\\gamma$={self.gammadotdot}, $t_0$={self.t0:.2f}"

    def __repr__(self) -> str:
        return f"Trend(params={{'gd': {self.gammadot}, 'gdd': {self.gammadotdot}}}, t0={self.t0:.2f})"

    def _linear(self, t: np.ndarray, t0: float) -> np.ndarray:
        if self.gammadot == 0:
            return np.zeros(len(t))
        return self.gammadot * (t - t0)

    def _quadratic(self, t: np.ndarray, t0: float) -> np.ndarray:
        if self.gammadotdot == 0:
            return np.zeros(len(t))
        return self.gammadotdot * ((t - t0) ** 2)

    def radial_velocity(self, t: np.ndarray) -> np.ndarray:
        """Calculate radial velocity contribution from trend components.

        Parameters
        ----------
        t : array_like
            Time values

        Returns
        -------
        array_like
            RV trend values (linear + quadratic terms)
        """
        rv = 0
        rv += self._linear(t, self.t0)
        rv += self._quadratic(t, self.t0)
        return rv


class Star:
    """Star with orbiting planet(s).

    Parameters
    ----------
    name : `str`
        The name of the star system.
    mass : `float`
        The mass of the star (solar masses)

    Attributes
    ----------
    planets : `dict`
        The dict storing the `Planet` objects, the key is the `planet.letter`
        attribute.
    num_planets : int
        The number of `Planet` objects stored in the `Star` object.
    """

    def __init__(self, name: str, mass: float) -> None:
        self.name = name
        self.mass = mass
        self.planets = {}
        self.instruments = {}
        self.num_planets = 0
        if mass <= 0:
            raise ValueError(f"Stellar mass {self.mass} must be greater than zero")

    def __repr__(self) -> str:
        return f"Star(name={self.name!r}, mass={self.mass!r})"

    def __str__(self) -> str:
        if hasattr(self, "trend"):
            return f"Star {self.name}, {self.num_planets} planets: {[*self.planets]}, {self.trend}"
        else:
            return f"Star {self.name!r}, {self.num_planets!r} planets: {[*self.planets]!r}"

    def add_planet(self, planet: Planet) -> None:
        """Store `Planet` object in `planets` dict with the key `Planet.letter`.

        Planets cannot share letters; if two planets have the same letter then the
        second one will overwrite the first.

        Parameters
        ----------
        planet : `Planet`
            A `ravest.model.Planet` object
        """
        # Warn if planet letter already exists (will overwrite)
        if planet.letter in self.planets:
            import warnings
            warnings.warn(f"Planet {planet.letter} already exists and will be overwritten",
                         UserWarning, stacklevel=2)

        self.planets[planet.letter] = planet
        self.num_planets = len(self.planets)

    def add_trend(self, trend: Trend) -> None:
        """Store `Trend` object in `trend` attribute.

        Parameters
        ----------
        trend : `Trend`
            A `ravest.model.Trend` object
        """
        self.trend = trend

    def add_instrument(self, instrument: Instrument) -> None:
        """Store `Instrument` object in `instruments` dict with key `instrument.name`.

        Instruments cannot share names; if two instruments have the same name then
        the second one will overwrite the first.

        Parameters
        ----------
        instrument : `Instrument`
            A `ravest.model.Instrument` object
        """
        import warnings
        if instrument.name in self.instruments:
            warnings.warn(f"Instrument {instrument.name} already exists and will be overwritten",
                         UserWarning, stacklevel=2)
        self.instruments[instrument.name] = instrument

    def gamma_offsets(self, instrument: np.ndarray) -> np.ndarray:
        """Return gamma offset for each observation based on instrument column.

        Parameters
        ----------
        instrument : np.ndarray
            Array of instrument names for each observation.

        Returns
        -------
        np.ndarray
            Array of gamma offsets (m/s) for each observation.

        Raises
        ------
        ValueError
            If an instrument in the data is not found in the Star's instruments.
        """
        result = np.zeros(len(instrument))
        for name, inst in self.instruments.items():
            mask = (instrument == name)
            result[mask] = inst.g
        return result

    def jitter_values(self, instrument: np.ndarray) -> np.ndarray:
        """Return jitter value for each observation based on instrument column.

        Parameters
        ----------
        instrument : np.ndarray
            Array of instrument names for each observation.

        Returns
        -------
        np.ndarray
            Array of jitter values (m/s) for each observation.
        """
        result = np.zeros(len(instrument))
        for name, inst in self.instruments.items():
            mask = (instrument == name)
            result[mask] = inst.jit
        return result

    def radial_velocity(self, t: np.ndarray) -> np.ndarray:
        """Calculate the radial velocity of the star at time ``t`` due to the planets and trend.

        Calculate the radial velocity of the star at time ``t`` due to the
        planets and the trend (constant velocity, linear, and quadratic.)
        This is a linear sum of the RVs of each of the `Planet` objects stored in
        the `Star`, and the RV of the `Trend` object.

        Parameters
        ----------
        t : `float`
            The time at which to calculate the radial velocity (day)

        Returns
        -------
        `float`
            The radial velocity at time `t` (m/s)
        """
        rv = np.zeros(len(t))

        for planet in self.planets:
            rv += self.planets[planet].radial_velocity(t)

        rv += self.trend.radial_velocity(t)

        return rv

    def mpsini(self, planet_letter: str, unit: str = "kg") -> float:
        """Calculate the minimum mass of a planet in the star system.

        Parameters
        ----------
        planet_letter : `str`
            The letter of the planet to calculate the minimum mass of.
        unit : `str`
            The unit to return the planetary minimum mass in. Options are "kg", "M_earth", "M_jupiter".

        Returns
        -------
        `float`
            The minimum mass of the planet.
        """
        return self.planets[planet_letter].mpsini(self.mass, unit)

    def phase_plot(self, t: np.ndarray, ydata: np.ndarray, yerr: np.ndarray,
                   instrument: np.ndarray) -> None:
        """Generate a phase plot for each planet.

        Given RV ``ydata`` at time ``t`` with errorbars ``yerr``, generates a phase
        plot for each planet. Data is coloured by instrument and gamma offsets
        are subtracted before plotting.

        Parameters
        ----------
        t : np.ndarray
            The time of the observations ``ydata`` (day)
        ydata : np.ndarray
            The observed radial velocity at time ``t`` (m/s)
        yerr : np.ndarray
            The measurement error of the datapoint ``ydata`` (m/s)
        instrument : np.ndarray
            The instrument/telescope name for each observation.
        """
        # TODO use gridspec or subfigures to sort out figure spacing

        # Subtract gamma offsets from data
        gamma_offsets = self.gamma_offsets(instrument)
        ydata_corrected = ydata - gamma_offsets

        # Get unique instruments and assign colours
        unique_instruments = np.unique(instrument)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        inst_colors = {inst: colors[i % len(colors)]
                       for i, inst in enumerate(unique_instruments)}

        # Sort data by time for plotting
        sort_inds = np.argsort(t)
        t_sorted = t[sort_inds]
        tlin = np.linspace(t_sorted[0], t_sorted[-1], 1000)
        fig, axs = plt.subplots(2+self.num_planets, 1,
                                figsize=(10, (2*10/3)+(self.num_planets*10/3)),
                                constrained_layout=True)

        # Panel 1: Observed data with complete system model overlay
        axs[0].set_title("Stellar radial velocity")
        axs[0].set_ylabel("Radial Velocity [m/s]")
        axs[0].set_xlabel("Time [days]")
        axs[0].axhline(y=0, color="k", alpha=0.25, linestyle="--", zorder=1)

        # Model curve (planets + trend, no gamma)
        modelled_rv_tlin = self.radial_velocity(tlin)
        modelled_rv_tdata = self.radial_velocity(t)
        axs[0].plot(tlin, modelled_rv_tlin, color="k", zorder=2)

        # Plot data points coloured by instrument
        for inst in unique_instruments:
            mask = (instrument == inst)
            axs[0].errorbar(t[mask], ydata_corrected[mask], yerr=yerr[mask],
                           marker="o", color=inst_colors[inst], mfc="white",
                           ecolor=inst_colors[inst], markersize=8,
                           linestyle="None", zorder=3, label=inst, alpha=0.8)
        axs[0].legend()

        # Panel 2: Observed minus calculated (O-C) residuals
        axs[1].set_title("Observed-Calculated")
        axs[1].set_xlabel("Time [days]")
        axs[1].set_ylabel("Residual [m/s]")
        axs[1].axhline(y=0, color="k", linestyle="-")
        for inst in unique_instruments:
            mask = (instrument == inst)
            axs[1].errorbar(t[mask], ydata_corrected[mask] - modelled_rv_tdata[mask],
                           yerr=yerr[mask], marker="o", mfc="white",
                           color=inst_colors[inst], ecolor=inst_colors[inst],
                           markersize=8, linestyle="None", alpha=0.8)

        # Panels 3+: Individual planet phase plots
        for n, letter in enumerate(self.planets):
            n += 2  # we already have two subplots
            axs[n].set_title(f"Planet {letter}")
            axs[n].set_xlabel("Orbital phase")
            axs[n].set_ylabel("Radial velocity [m/s]")
            axs[n].set_xlim(-0.5, 0.5)
            axs[n].xaxis.set_major_locator(MultipleLocator(0.25))
            axs[n].axhline(y=0, color="k", alpha=0.25, linestyle="--", zorder=1)

            this_planet = self.planets[letter]
            p = this_planet._rvparams["P"]
            e = this_planet._rvparams["e"]
            w = this_planet._rvparams["w"]
            tp = this_planet._rvparams["Tp"]
            tc = this_planet.parameterisation.convert_tp_to_tc(tp, p, e, w)

            yplot = this_planet.radial_velocity(tlin)
            tlin_fold_sorted, tlin_inds = fold_time_series(tlin, p, tc)
            axs[n].plot(tlin_fold_sorted, yplot[tlin_inds], color="k")

            # Calculate RV contributions from all other planets and trend
            # Subtract from observed data to isolate the current planet's signal
            other_rv = np.zeros(len(t))
            for _letter in self.planets:
                if _letter != letter:
                    other_rv += self.planets[_letter].radial_velocity(t)
            other_rv += self.trend.radial_velocity(t)
            subtracted_data = ydata_corrected - other_rv

            # Plot phase-folded data coloured by instrument
            for inst in unique_instruments:
                mask = (instrument == inst)
                t_inst = t[mask]
                y_inst = subtracted_data[mask]
                yerr_inst = yerr[mask]
                tdata_fold_sorted, tdata_inds = fold_time_series(t_inst, p, tc)
                axs[n].errorbar(tdata_fold_sorted, y_inst[tdata_inds],
                               yerr=yerr_inst[tdata_inds], marker="o", mfc="white",
                               color=inst_colors[inst], ecolor=inst_colors[inst],
                               markersize=8, linestyle="None", alpha=0.8)

def calculate_mpsini(mass_star: float, period: float, semi_amplitude: float, eccentricity: float, unit: str = "kg") -> float:
    """Calculate the minimum mass of the planet.

    Parameters
    ----------
    mass_star : `float`
        The mass of the star in solar masses.
    period : `float`
        The orbital period of the planet in days.
    semi_amplitude : `float`
        The semi-amplitude of the radial velocity of the star in m/s.
    eccentricity : `float`
        The eccentricity of the orbit, 0 <= e < 1  (dimensionless).
    unit : `str`
        The unit to return the planetary minimum mass in. Options are "kg", "M_earth", "M_jupiter".

    Returns
    -------
    `float`
        The minimum mass mpsini of the planet.

    Raises
    ------
    ValueError
        If the unit is not one of "kg", "M_earth", "M_jupiter".
    """
    # convert M_s to kg, and period to s, as the formula is in SI units
    mass_star = mass_star * (1*const.M_sun).value  # type: ignore
    period = period * (1*constants.day)

    mpsini_kg = semi_amplitude * (period/(2*np.pi*constants.G))**(1/3) * (mass_star)**(2/3) * (1-(eccentricity**2))**(1/2)

    if unit=="kg":
        return mpsini_kg
    elif unit == "M_earth":
        return mpsini_kg / const.M_earth.value  # type: ignore
    elif unit == "M_jupiter":
        return mpsini_kg / const.M_jup.value  # type: ignore
    else:
        raise ValueError(f"Unit {unit} not valid. Use 'kg', 'M_Earth' or 'M_Jupiter'")


def fold_time_series(times: np.ndarray, period: float, t_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """Fold time series to orbital phase and return sorted arrays.

    Converts times to orbital phases in the range [-0.5, 0.5] and returns
    both the sorted phases and the indices needed to sort other arrays
    consistently.

    Parameters
    ----------
    times : np.ndarray
        Time values to fold
    period : float
        Orbital period
    t_ref : float
        Reference time (usually Tc - time of transit/conjunction)

    Returns
    -------
    phases_sorted : np.ndarray
        Phase-folded times in range [-0.5, 0.5], sorted ascending
    sort_indices : np.ndarray
        Indices that sort the original times by phase

    Examples
    --------
    >>> times = np.array([0, 1, 2, 3, 4])
    >>> period = 2.0
    >>> t_ref = 0.5  # Reference time
    >>> phases, indices = fold_time_series(times, period, t_ref)
    >>> # phases will be sorted from -0.5 to 0.5
    >>> # indices can be used to sort other arrays consistently
    """
    phases = ((times - t_ref + 0.5*period) % period - 0.5*period) / period
    sort_indices = np.argsort(phases)
    return phases[sort_indices], sort_indices
