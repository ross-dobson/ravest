# model.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

class Planet:
    """Planet defined by its orbital parameters.

    Attributes:
        letter: the label of the planet, e.g. "b", "c"
        basis: the set of planetary parameters used to define the planet.
        params: dict of planetary parameters, matching the basis
    """
    def __init__(self, letter: str, basis: str, params: dict):
        # TODO: validation on letter and basis
        # TODO: implement basis detection and automatic conversion to synth basis? compare basis to params keys?
        if not (letter.isalpha() and (letter == letter[0]*len(letter))):
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
        """Calculate mean motion (mean angular rate of orbit in radians/day)
        
        Returns:
            mean_motion: average angular rate of orbit (in radians/day)
        """
        return 2*np.pi/period

    def _calculate_mean_anomaly(self, t: float, n: float, time_peri: float) -> float:
        """Calculate mean anomaly (radians)
        
        Mean anomaly is a fictitious angle that you can think of as the equivalent
        angle swept at point t, for a planet with fixed velocity in a circular orbit
        of period P.
        
        Args:
            n: the mean motion (radians/day)
            t: the time to evaluate the mean anomaly at (day)
            time_peri: the time of periapsis of the orbit (day)
        Returns:
            Mean anomaly (radians)
        """
        # TODO this docstring isn't correct about fictitious angle. Find a better reference.
        return n * (t - time_peri)
    
    def _solve_keplerian_equation(self, eccentricity: float, M: float) -> float:
        """Solve the Keplerian equation for eccentric anomaly E.
        
        The eccentric anomaly is the corresponding angle for the true anomaly on the
        auxiliary circle, rather than the real elliptical orbit. Therefore, if the 
        eccentricity e=0, then the eccentric anomaly E is equivalent to the mean 
        anomaly M. However, for eccentric cases, the equation is
        E(t) = M(t) + e sin(E(t)) , which requires solving iteratively. This 
        function achieves this via the Newton-Raphson iteration method.
        
        Args:
            M: mean anomaly at time (t)
            eccentricity: eccentricity (dimensionless)

        Returns:
            E: Eccentric anomaly
        """
        # TODO: check this docstring's explanation (given that mean anomaly was wrong).
        if eccentricity == 0:
            return M
        
        # Newton-Raphson finds roots, so solving E-(M+e*sinE) finds E 
        E0 = M  # initial guess for E0 # TODO find reference for this choice
        def f(E: float, eccentricity: float, M: float) -> float:
            return (E - (eccentricity*np.sin(E)) - M)
        def fp(E: float, eccentricity: float, M: float) -> float:
            return 1 - eccentricity*np.cos(E)
        def fpp(E: float, eccentricity:float, M: float) -> float:
            return eccentricity*np.sin(E)
        
        return newton(func=f, fprime=fp, fprime2=fpp, args=(eccentricity, M), x0=E0)
    
    def _true_anomaly(self, E: float, eccentricity: float) -> float:
        """Calculate true anomaly, the angle between periastron and planet, as
        measured from the system barycentre. This is the angle normally used to 
        characterise an orbit.

        Args:
            E: Eccentric anomaly
            eccentricity: Orbital eccentricity

        Returns:
            True anomaly
        """
        return 2*np.arctan(np.sqrt((1+eccentricity)/(1-eccentricity))*np.tan(E/2))
    
    def _radial_velocity(self, true_anomaly: float, semi_amplitude: float, eccentricity: float, omega_star: float) -> float:
        """Calculate the radial velocity of the star due to the planet, at a given true anomaly.

        Args:
            true_anomaly: The true anomaly at time t
            period: orbital period of planet
            semi_amplitude: semi amplitude of radial velocity of star reflex motion due to the planet
            eccentricity: orbital eccentricity
            omega_star: the angle of periastron of the star
            time_peri: the time of periastron passage

        Returns:
            radial velocity in metres/second of the reflex motion of the star due to the planet
        """
        return semi_amplitude * (np.cos(true_anomaly+omega_star) + eccentricity*np.cos(omega_star))
    
    def radial_velocity(self, t: float) -> float:
        """Calculate radial velocity of the star due to the planet, at time t.

        Calculates the true anomaly at time t by solving the Keplerian equation, and uses the anomaly to calculate the radial velocity.
        
        Args:
            t: the time to calculate the radial velocity at (days)
            
        Returns:
            radial velocity at time t (m/s)
        """
        P = self.params['p']
        K = self.params['k']
        e = self.params['e']
        w = self.params['w']
        tp = self.params['tp']

        n = self._calculate_mean_motion(period=P)
        M = self._calculate_mean_anomaly(t=t, n=n, time_peri=tp)
        E = self._solve_keplerian_equation(M=M, eccentricity=e)
        f = self._true_anomaly(E=E, eccentricity=e)

        return self._radial_velocity(true_anomaly=f, semi_amplitude=K, eccentricity=e, omega_star=w)
    
    def _time_given_true_anomaly(self, true_anomaly, period, eccentricity, time_peri):
        """Calculate the time that the planet will be at a given true anomaly.

        Args:
            true_anomaly: the true anomaly of the planet at the wanted time
            period: length of period of orbit (days)
            eccentricity: eccentricity of orbit (dimensionless)
            time_peri: the time of periastron (days)

        Returns:
            time (days)
        """
        # TODO update this docstring to better reflect it is inverse of other equation (eastman et al)?
        eccentric_anomaly = 2*np.arctan(np.sqrt((1-eccentricity)/(1+eccentricity))*np.tan(true_anomaly/2))
        mean_anomaly = eccentric_anomaly - (eccentricity*np.sin(eccentric_anomaly))

        return mean_anomaly * (period / (2*np.pi)) + time_peri

    def convert_tp_to_tc(self):
        arg_peri = self.params['w']
        period = self.params['p']
        eccentricity = self.params['e']
        time_peri = self.params['tp']

        theta_tc = (np.pi/2) - arg_peri  # true anomaly at time t_c
        return self._time_given_true_anomaly(theta_tc, period, eccentricity, time_peri)

    def convert_tc_to_tp(self):
        time_conj = self.params['tc']
        period = self.params['p']
        eccentricity = self.params['e']
        arg_peri = self.params['w']
        
        theta_tc = (np.pi/2) - arg_peri # true anomaly at time t_c
        eccentric_anomaly = 2*np.arctan(np.sqrt((1-eccentricity)/(1+eccentricity))*np.tan(theta_tc/2))
        mean_anomaly = eccentric_anomaly - (eccentricity*np.sin(eccentric_anomaly))
        return time_conj - (period/(2*np.pi))*mean_anomaly
        

class Star:
    """Star with orbiting planet(s).
    
    Attributes:
        name: the name of the star system
        mass: mass of the star [solar mass]
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
        # TODO validation of planet letter - warn for duplicates/overwrite?
        self.planets[planet.letter] = planet
        self.num_planets = len(self.planets)

    def radial_velocity(self, t):
        rv = np.zeros(len(t))

        for planet in self.planets:
            rv += self.planets[planet].radial_velocity(t)
        return rv
    
    def phase_plot(self, t, ydata, yerr):
        # TODO use gridspec or subfigures to sort out figure spacing
        N = len(t)
        t = np.sort(t)
        tplot = np.linspace(t[0], t[-1], 1000)
        fig, axs = plt.subplots(2+self.num_planets, 1, 
                                figsize=(10,(2*10/3)+(self.num_planets*10/3)),
                                constrained_layout=True)
        
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
            n+=2 # we already have two subplots
            axs[n].set_title(f"Planet {l}")
            axs[n].set_xlabel("Phase [days]")
            axs[n].set_ylabel("Radial velocity [m/s]")
            
            this_planet = self.planets[l]
            tc = this_planet.convert_tp_to_tc()
            p = this_planet.params['p']

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
