# model.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton


class Planet:
    """Planet defined by its orbital parameters.

    Parameters
    ----------
    letter : `str`
        The label of the planet, e.g. "b", "c". Must be a single letter.
    basis : `str`
        the set of planetary parameters used to define the planet.
    params : `dict`
        The orbital parameters, matching the basis.
    """

    def __init__(self, letter: str, basis: list[str], params: dict):
        # TODO: validation on letter and basis
        # TODO: implement basis detection and automatic conversion to synth basis? compare basis to params keys?
        if not (letter.isalpha() and (letter == letter[0] * len(letter))):
            raise ValueError(f"Letter {letter} is not a single alphabet character.")
        self.letter = letter
        self.basis = basis
        self.params = params

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}(letter={self.letter!r}, basis={self.basis!r}, params={self.params!r})"

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
        P = self.params["per"]
        K = self.params["k"]
        e = self.params["e"]
        w = self.params["w"]
        tp = self.params["tp"]

        n = self._calculate_mean_motion(period=P)
        M = self._calculate_mean_anomaly(t=t, n=n, time_peri=tp)
        E = self._solve_keplerian_equation(M=M, eccentricity=e)
        f = self._true_anomaly(E=E, eccentricity=e)

        return self._radial_velocity(true_anomaly=f, semi_amplitude=K, eccentricity=e, omega_star=w)

    def _time_given_true_anomaly(self, true_anomaly, period, eccentricity, time_peri):
        """Calculate the time that the star will be at a given true anomaly.

        Parameters
        ----------
        true_anomaly : `float`
            The true anomaly of the planet at the wanted time
        period : `float`
            The orbital period of the planet (day)
        eccentricity : `float`
            The eccentricity of the orbit, 0 <= e < 1  (dimensionless).
        time_peri : `float`
            The time of periastron (day).

        Returns
        -------
        `float`
            The time corresponding to the given true anomaly (days).
        """
        # TODO update this docstring to better reflect it is inverse of other equation (eastman et al)?
        # and also include in the notes a list of the angles we can get (Eastman et al equation 11)?
        eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(true_anomaly / 2))
        mean_anomaly = eccentric_anomaly - (eccentricity * np.sin(eccentric_anomaly))

        return mean_anomaly * (period / (2 * np.pi)) + time_peri

    def convert_tp_to_tc(self):
        """Calculate the time of transit center, given time of periastron passage.

        This is only a time of (primary) transit center if the planet is actually
        transiting the star from the observer's viewpoint/inclination. Therefore
        technically this is a time of (inferior) conjunction.

        Returns
        -------
        `float`
            Time of primary transit center/inferior conjunction (days)
        """
        arg_peri = self.params["w"]
        period = self.params["p"]
        eccentricity = self.params["e"]
        time_peri = self.params["tp"]

        theta_tc = (np.pi / 2) - arg_peri  # true anomaly at time t_c
        return self._time_given_true_anomaly(theta_tc, period, eccentricity, time_peri)

    def convert_tc_to_tp(self):
        """Calculate the time of periastron passage, given time of primary transit.

        Returns
        -------
        `float`
            Time of periastron passage (days).
        """
        time_conj = self.params["tc"]
        period = self.params["p"]
        eccentricity = self.params["e"]
        arg_peri = self.params["w"]

        theta_tc = (np.pi / 2) - arg_peri  # true anomaly at time t_c
        eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(theta_tc / 2))
        mean_anomaly = eccentric_anomaly - (eccentricity * np.sin(eccentric_anomaly))
        return time_conj - (period / (2 * np.pi)) * mean_anomaly


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

    def radial_velocity(self, t):
        """
        Calculate the radial velocity of the star at time ``t`` due to the planets.
        This is a linear sum of the RVs of each of the `Planet` objects stored in
        the `Star`.

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
        return rv

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
        N = len(t)
        t = np.sort(t)
        tplot = np.linspace(t[0], t[-1], 1000)
        fig, axs = plt.subplots(2+self.num_planets,1, figsize=(10, (2*10/3)+(self.num_planets*10/3)), constrained_layout=True,)

        # First panel: plot the observed data and the combined system model
        axs[0].set_title(f"Stellar radial velocity")
        axs[0].set_ylabel("Radial Velocity [m/s]")
        axs[0].set_xlabel("Time [days]")
        axs[0].axhline(y=0, color="k", alpha=0.25, linestyle="--", zorder=1)

        modelled_rv_tplot = self.radial_velocity(tplot)
        modelled_rv_tdata = self.radial_velocity(t)
        axs[0].plot(tplot, modelled_rv_tplot, color="tab:blue", zorder=2)
        axs[0].errorbar(t, ydata, yerr=yerr, marker=".", color="k", mfc="white", ecolor="tab:gray", markersize=10, linestyle="None", zorder=3)

        # Second panel: O-C residuals of the top panel
        axs[1].set_title(f"Observed-Calculated")
        axs[1].set_xlabel("Time [days]")
        axs[1].set_ylabel("Residual [m/s]")
        axs[1].axhline(y=0, color="tab:blue", linestyle="-")
        axs[1].errorbar(t, ydata-modelled_rv_tdata, yerr=yerr, marker=".", mfc="white", color="k", ecolor="tab:gray", markersize=10, linestyle="None")

        # Subsequent panels: phase plot, one per planet
        for n, l in enumerate(self.planets):
            n+=2  # we already have two subplots
            axs[n].set_title(f"Planet {l}")
            axs[n].set_xlabel("Phase [days]")
            axs[n].set_ylabel("Radial velocity [m/s]")

            this_planet = self.planets[l]
            tc = this_planet.convert_tp_to_tc()
            p = this_planet.params["p"]

            yplot = this_planet.radial_velocity(tplot)
            tplot_fold = (tplot - tc + 0.5 * p) % p - 0.5 * p
            inds = np.argsort(tplot_fold)
            axs[n].plot(tplot_fold[inds]/p, yplot[inds], label=f"{n},{l}, rvplot", color="tab:blue")

            # model the rv data for all the other planets, to subtract from the observed data
            other_planets_modelled_rv_tdata = np.zeros(len(t))
            for _l in self.planets:
                if _l == l:
                    continue  # don't do anything for this planet
                else:
                    other_planets_modelled_rv_tdata += self.planets[_l].radial_velocity(t)
            subtracted_data = ydata - other_planets_modelled_rv_tdata
            tdata_fold = (t - tc + 0.5 * p) % p - 0.5 * p
            inds = np.argsort(tdata_fold)
            axs[n].errorbar(tdata_fold[inds]/p, subtracted_data[inds], yerr=yerr, marker=".", mfc="white", color="k", ecolor="tab:gray", markersize=10, linestyle="None")

class Trend:
    def __init__(self, params: dict):
        self.gamma = params["g"]
        self.gammadot = params["gd"]
        self.gammadotdot = params["gdd"]

    def _constant(self, t):
        return self.gamma

    def _linear(self, t, t0):
        if self.gammadot == 0:
            return np.zeros(len(t))
        return self.gammadot * (t - t0)

    def _quadratic(self, t, t0):
        if self.gammadotdot == 0:
            return np.zeros(len(t))
        return self.gammadotdot * (t - t0) ** 2

    def radial_velocity(self, t):
        t0 = t[0]  # TODO: do we want this to be a separate parameter?
        rv = 0
        rv += self._constant(t)
        rv += self._linear(t, t0)
        rv += self._quadratic(t, t0)
        return rv

