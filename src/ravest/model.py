# model.py

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from scipy import constants
from scipy.optimize import newton

from ravest.param import Parameterisation


class Planet:
    """Planet defined by its orbital parameters.

    Parameters
    ----------
    letter : `str`
        The label of the planet, e.g. "b", "c". Must be a single letter.
    parameterisation : `parameterisation`
        The set of planetary parameters used to define the planet.
    params : `dict`
        The orbital parameters, matching the basis.
    """

    def __init__(self, letter: str, parameterisation: Parameterisation, params: dict):
        if not (letter.isalpha() and (letter == letter[0] * len(letter))):
            raise ValueError(f"Letter {letter} is not a single alphabet character.")
        self.letter = letter
        self.parameterisation = parameterisation
        self.params = params

        # Check the input params and parameterisation match
        if not set(params.keys()) == set(parameterisation.pars):
            raise ValueError(f"Parameterisation {parameterisation} does not match input params {params}")

        # Convert to the per k e w tp basis that we need for the RV equation
        self._rvparams = self.parameterisation.convert_pars_to_default_basis(self.params)

        # Validate parameters immediately after conversion to avoid invalid parameters
        # Raises a ValueError if any parameter is invalid
        self.parameterisation.validate_default_basis_params(self._rvparams)


    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}(letter={self.letter!r}, parameterisation={self.parameterisation!r}, params={self.params!r})"

    def __str__(self):
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
        """Calculate mean anomaly (radians)

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

    def _solve_keplerian_equation(self, eccentricity: float, M: np.ndarray) -> np.ndarray:
        """Solve the Keplerian equation for eccentric anomaly E.

        The eccentric anomaly is the corresponding angle for the true anomaly on
        the auxiliary circle, rather than the real elliptical orbit. Therefore, if
        the ``eccentricity`` e=0, then the eccentric anomaly E is equivalent to the
        mean anomaly ``M``. However, for eccentric cases, the equation is
        E(t) = M(t) + e*sin(E(t)), which requires solving iteratively. This
        function achieves this via Newton-Raphson iteration.

        Parameters
        ----------
        M : `float`
            The mean anomaly at time t.
        eccentricity : `float`
            The eccentricity of the orbit, 0 <= e < 1  (dimensionless).

        Returns
        -------
        `float`
            The eccentric anomaly at time t.
        """
        # TODO add Notes to docstring explaining Newton-Raphson, and choice of E0=M
        if eccentricity == 0:
            return M

        # Newton-Raphson finds roots, so solving E-(M+e*sinE) finds E
        E0 = M  # initial guess for E0. # TODO: reference for this initial choice?

        def f(E: float, eccentricity: float, M: float) -> float:
            return E - (eccentricity * np.sin(E)) - M

        def fp(E: float, eccentricity: float, M: float) -> float:
            return 1 - eccentricity * np.cos(E)

        def fpp(E: float, eccentricity: float, M: float) -> float:
            return eccentricity * np.sin(E)

        # TODO: fix typing here
        return newton(func=f, fprime=fp, fprime2=fpp, args=(eccentricity, M), x0=E0)  # type: ignore

    def _true_anomaly(self, E: np.ndarray, eccentricity: float) -> np.ndarray:
        """Calculate true anomaly at time t.

        Calculate true anomaly, the angle between periastron and planet, as
        measured from the system barycentre. This is the angle normally used to
        characterise an orbit. If orbit is circular, this is equal to mean anomaly.

        Parameters
        ----------
        E : `float`
            The Eccentric anomaly at time t (radian)
        eccentricity : `float`
            The eccentricity of the orbit, 0 <= e < 1  (dimensionless).

        Returns
        -------
        `float`
            The true anomaly (radians)
        """
        return 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(E / 2))

    def _radial_velocity(self, true_anomaly: np.ndarray, semi_amplitude: float, eccentricity: float, omega_star: float,) -> np.ndarray:
        """Calculate the radial velocity of the star due to the planet, at a given true anomaly.

        Parameters
        ----------
        true_anomaly : `float`
            The true anomaly at time t (radian).
        period : `float`
            The orbital period of planet (day).
        semi_amplitude : `float`
            The Semi-amplitude of the radial velocity of the star (m/s).
        eccentricity : `float`
            The eccentricity of the orbit, 0 <= e < 1  (dimensionless).
        omega_star : `float`
            The angle of periastron of the star (radians).
        time_peri : `float`
            The time of periastron passage (day).

        Returns
        -------
        `float`
            Radial velocity of the reflex motion of star due to the planet (m/s).
        """
        return semi_amplitude * (np.cos(true_anomaly + omega_star) + eccentricity * np.cos(omega_star))

    def radial_velocity(self, t: np.ndarray) -> np.ndarray:
        """Calculate radial velocity of the star due to the planet, at time t.

        Calculates the true anomaly at time t by solving the Keplerian equation,
        and uses that true anomaly to calculate the radial velocity.

        Parameters
        ----------
        t : `float`
            The time to calculate the radial velocity at (day)

        Returns
        -------
        `float`
            Radial velocity of the reflex motion of star due to the planet (m/s).
        """
        P = self._rvparams["per"]
        K = self._rvparams["k"]
        e = self._rvparams["e"]
        w = self._rvparams["w"]
        tp = self._rvparams["tp"]

        n = self._calculate_mean_motion(period=P)
        M = self._calculate_mean_anomaly(t=t, n=n, time_peri=tp)
        E = self._solve_keplerian_equation(M=M, eccentricity=e)
        f = self._true_anomaly(E=E, eccentricity=e)

        return self._radial_velocity(true_anomaly=f, semi_amplitude=K, eccentricity=e, omega_star=w)

    def mpsini(self, mass_star, unit="kg"):
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
        period = self._rvparams["per"]
        semi_amplitude = self._rvparams["k"]
        eccentricity = self._rvparams["e"]

        # Convert stellar mass to kg and period to seconds for SI unit consistency
        # Formula requires SI units: M_s [kg], P [s], K [m/s]
        mpsini = calculate_mpsini(mass_star, period, semi_amplitude, eccentricity, unit)
        return mpsini


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

    def __init__(self, name: str, mass: float):
        self.name = name
        self.mass = mass
        self.planets = {}
        self.num_planets = 0
        if mass <= 0:
            raise ValueError(f"Stellar mass {self.mass} must be greater than zero")

    def __repr__(self):
        return f"Star(name={self.name!r}, mass={self.mass!r})"

    def __str__(self):
        if hasattr(self, "trend"):
            return f"Star {self.name}, {self.num_planets} planets: {[*self.planets]}, {self.trend}"
        else:
            return f"Star {self.name!r}, {self.num_planets!r} planets: {[*self.planets]!r}"

    def add_planet(self, planet):
        """Store `Planet` object in `planets` dict with the key `Planet.letter`.

        Planets cannot share letters; if two planets have the same letter then the
        second one will overwrite the first.

        Parameters
        ----------
        planet : `Planet`
            A `ravest.model.Planet` object
        """
        # TODO validation of planet letter - warn for duplicates/overwrite?
        self.planets[planet.letter] = planet
        self.num_planets = len(self.planets)

    def add_trend(self, trend):
        """Store `Trend` object in `trend` attribute.

        Parameters
        ----------
        trend : `Trend`
            A `ravest.model.Trend` object
        """
        self.trend = trend

    def radial_velocity(self, t):
        """
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

    def mpsini(self, planet_letter, unit="kg"):
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

    def phase_plot(self, t, ydata, yerr):
        """Given RV ``ydata`` at time ``t`` with errorbars ``yerr``, generates a phase
        plot for each planet.

        Parameters
        ----------
        t : `float`
            The time of the observations ``ydata`` (day)
        ydata : `float`
            The observed radial velocity at time ``t`` (m/s)
        yerr : `float`
            The measurement error of the datapoint ``ydata`` (m/s)
        """
        # TODO use gridspec or subfigures to sort out figure spacing
        len(t)
        t = np.sort(t)
        tlin = np.linspace(t[0], t[-1], 1000)
        fig, axs = plt.subplots(2+self.num_planets,1, figsize=(10, (2*10/3)+(self.num_planets*10/3)), constrained_layout=True,)

        # Panel 1: Observed data with complete system model overlay
        axs[0].set_title("Stellar radial velocity")
        axs[0].set_ylabel("Radial Velocity [m/s]")
        axs[0].set_xlabel("Time [days]")
        axs[0].axhline(y=0, color="k", alpha=0.25, linestyle="--", zorder=1)

        modelled_rv_tlin = self.radial_velocity(tlin)
        modelled_rv_tdata = self.radial_velocity(t)
        axs[0].plot(tlin, modelled_rv_tlin, color="tab:blue", zorder=2)
        axs[0].errorbar(t, ydata, yerr=yerr, marker=".", color="k", mfc="white", ecolor="tab:gray", markersize=10, linestyle="None", zorder=3)

        # Panel 2: Observed minus calculated (O-C) residuals
        axs[1].set_title("Observed-Calculated")
        axs[1].set_xlabel("Time [days]")
        axs[1].set_ylabel("Residual [m/s]")
        axs[1].axhline(y=0, color="tab:blue", linestyle="-")
        axs[1].errorbar(t, ydata-modelled_rv_tdata, yerr=yerr, marker=".", mfc="white", color="k", ecolor="tab:gray", markersize=10, linestyle="None")

        # Panels 3+: Individual planet phase plots
        for n, letter in enumerate(self.planets):
            n += 2  # we already have two subplots
            axs[n].set_title(f"Planet {letter}")
            axs[n].set_xlabel("Orbital phase")
            axs[n].set_ylabel("Radial velocity [m/s]")
            axs[n].set_xlim(-0.5, 0.5)
            axs[n].axhline(y=0, color="k", alpha=0.25, linestyle="--", zorder=1)

            this_planet = self.planets[letter]
            p = this_planet._rvparams["per"]
            e = this_planet._rvparams["e"]
            w = this_planet._rvparams["w"]
            tp = this_planet._rvparams["tp"]
            tc = this_planet.parameterisation.convert_tp_to_tc(tp, p, e, w)

            yplot = this_planet.radial_velocity(tlin)
            tlin_fold = (tlin - tc + 0.5 * p) % p - 0.5 * p
            inds = np.argsort(tlin_fold)
            axs[n].plot(tlin_fold[inds]/p, yplot[inds], label=f"{n},{letter}, rvplot", color="tab:blue")

            # Calculate RV contributions from all other planets
            # Subtract from observed data to isolate the current planet's signal
            other_planets_modelled_rv_tdata = np.zeros(len(t))
            for _letter in self.planets:
                if _letter == letter:
                    continue  # don't do anything for this planet
                else:
                    other_planets_modelled_rv_tdata += self.planets[_letter].radial_velocity(t)
            subtracted_data = ydata - other_planets_modelled_rv_tdata
            tdata_fold = (t - tc + 0.5 * p) % p - 0.5 * p
            inds = np.argsort(tdata_fold)
            axs[n].errorbar(tdata_fold[inds]/p, subtracted_data[inds], yerr=yerr, marker=".", mfc="white", color="k", ecolor="tab:gray", markersize=10, linestyle="None")

class Trend:
    """Trend in the radial velocity of the star.

    Parameters
    ----------
    t0 : `float`
        The reference zero-point time for the linear and quadratic trend.
        Recommended to be the mean of the input times.
    params : `dict`
        The parameters of the trend: the constant, linear, and quadratic
        components. These must be named "g", "gd", and "gdd" respectively (which
        stands for gamma, gamma-dot, gamma-dot-dot). These are in units of m/s,
        m/s/day, and m/s/day^2 respectively.

    Returns
    -------
    `float`
        The radial velocity of the star due to the trend (m/s).

    Notes
    -----
    The radial velocity of the star due to the trend is calculated as the sum of
    the constant, linear, and quadratic components. The constant component is
    simply a constant offset of value gamma. The linear and quadratic components
    are calculated as `gd*(t-t0)` and `gdd*((t-t0)**2)` respectively.

    In general the trend is used to account for any unexpected effects. These
    could be due to instrumental effects, or for example a very long-term
    companion could show as a linear and/or quadratic trend in the data. If you
    see a strong linear or quadratic trend in the data, it is worth
    investigating.
    """

    def __init__(self, t0, params: dict):
        self.gamma = params["g"]
        self.gammadot = params["gd"]
        self.gammadotdot = params["gdd"]

        # Validate and store reference time t0
        try:
            self.t0 = float(t0)
        except (TypeError, ValueError) as e:
            raise ValueError(f"t0 must be a numeric value (recommend mean or median of observation times), but got {type(t0).__name__}: {t0}") from e

    def __str__(self):
        return f"Trend: $\\gamma$={self.gamma}, $\\dot\\gamma$={self.gammadot}, $\\ddot\\gamma$={self.gammadotdot}, $t_0$={self.t0:.2f}"

    def __repr__(self):
        return f"Trend(params={{'g': {self.gamma}, 'gd': {self.gammadot}, 'gdd': {self.gammadotdot} }}, t0={self.t0:.2f})"

    def _constant(self, t):
        return self.gamma

    def _linear(self, t, t0):
        if self.gammadot == 0:
            return np.zeros(len(t))
        return self.gammadot * (t - t0)

    def _quadratic(self, t, t0):
        if self.gammadotdot == 0:
            return np.zeros(len(t))
        return self.gammadotdot * ((t - t0) ** 2)

    def radial_velocity(self, t):
        rv = 0
        rv += self._constant(t)
        rv += self._linear(t, self.t0)
        rv += self._quadratic(t, self.t0)
        return rv


def calculate_mpsini(mass_star, period, semi_amplitude, eccentricity, unit="kg"):
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
