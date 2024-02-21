# model.py

import numpy as np
from scipy.optimize import minimize, newton

class Planet:
    def __init__(self, letter: str, basis: str, params: dict):
        # TODO: validation on letter and basis
        # TODO: implement basis detection and automatic conversion to synth basis?
        self.letter = letter
        self.basis = basis
        self.params = params
        
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
    
    def _solve_keplerian_equation(M: float, eccentricity:float) -> float:
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
    
    def _true_anomaly(E: float, eccentricity: float) -> float:
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
    
    def _radial_velocity(true_anomaly: float, semi_amplitude: float, eccentricity: float, omega_star: float) -> float:
        """Calculate the radial velocity of the star due to the planet, at a given true anomaly.

        Args:
            true_anomaly: The true anomaly at time t
            period: orbital period of planet
            semi_amplitude: semi amplitude of radial velocity of star reflex motion due to the planet
            eccentricity: orbital eccentricity
            omega_star: the angle of periastron of the star, $\omega_* (=\omega_p + \pi/2)$
            time_peri: the time of periastron passage

        Returns:
            radial velocity in metres/second of the reflex motion of the star due to the planet
        """
        return semi_amplitude * (np.cos(true_anomaly+omega_star) + eccentricity*np.cos(omega_star))
    
    def radial_velocity(self, t: float) -> float:
        """Calculate radial velocity of the star due to the planet, at time t.

        Calculates the true anomaly at time t by solving the Keplerian equation, and uses that to calculate the radial velocity.
        
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