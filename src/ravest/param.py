# parameterisation.py
import numpy as np

ALLOWED_PARAMETERISATIONS = ["per k e w tp",
                             "per k e w tc",
                             "per k ecosw esinw tp",
                             "per k ecosw esinw tc",
                             "per k secosw sesinw tp",
                             "per k secosw sesinw tc"]


class Parameterisation:

    def __init__(self, parameterisation: str):
        """Parameterisation object handles parameter conversions

        Parameters
        ----------
        parameterisation : str
            The parameterisation you wish to use. Must be one of the following:
            - "per k e w tp"
            - "per k e w tc"
            - "per k ecosw esinw tp"
            - "per k ecosw esinw tc"
            - "per k secosw sesinw tp"
            - "per k secosw sesinw tc"

        Raises
        ------
        ValueError
            If the parameterisation is not one of the allowed parameterisations.
        """
        if parameterisation not in ALLOWED_PARAMETERISATIONS:
            raise ValueError(f"parameterisation {parameterisation} not recognised. Must be one of {ALLOWED_PARAMETERISATIONS}")
        self.parameterisation = parameterisation
        self.pars = parameterisation.split()

    def __str__(self) -> str:
        return f"Parameterisation: {self.parameterisation}"

    def __repr__(self) -> str:
        return f"Parameterisation({self.parameterisation})"

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

    def convert_tp_to_tc(self, time_peri, period, eccentricity, arg_peri):
        """Calculate the time of transit center, given time of periastron passage.

        This is only a time of (primary) transit center if the planet is actually
        transiting the star from the observer's viewpoint/inclination. Therefore
        technically this is a time of (inferior) conjunction.

        Returns
        -------
        `float`
            Time of primary transit center/inferior conjunction (days)
        """
        theta_tc = (np.pi / 2) - arg_peri  # true anomaly at time t_c (Eastman et. al. 2013)
        return self._time_given_true_anomaly(theta_tc, period, eccentricity, time_peri)

    def convert_tc_to_tp(self, time_conj, period, eccentricity, arg_peri):
        """Calculate the time of periastron passage, given time of primary transit.

        Returns
        -------
        `float`
            Time of periastron passage (days).
        """
        theta_tc = (np.pi / 2) - arg_peri  # true anomaly at time t_c

        # Because model.Planet.is_valid() check is performed after parameter
        # conversion, this occasionally will be called with invalid parameters
        # causing the sqrt to be negative. This is caught later on in is_valid()
        # so we can leave it as a warning and not Raise it here. This isn't
        # ideal as it will slow down the code, but it is the best we can do
        # without a refactor.
        _1me_over_1pe = (1 - eccentricity) / (1 + eccentricity)
        _sqrt_term = np.sqrt(_1me_over_1pe)
        # eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(theta_tc / 2))
        eccentric_anomaly = 2 * np.arctan(_sqrt_term * np.tan(theta_tc / 2))

        mean_anomaly = eccentric_anomaly - (eccentricity * np.sin(eccentric_anomaly))
        return time_conj - (period / (2 * np.pi)) * mean_anomaly

    def convert_secosw_sesinw_to_e_w(self, secosw, sesinw):
        e = secosw**2 + sesinw**2
        w = np.arctan2(sesinw, secosw)
        return e, w

    def convert_e_w_to_secosw_sesinw(self, e, w):
        secosw = np.sqrt(e) * np.cos(w)
        sesinw = np.sqrt(e) * np.sin(w)
        return secosw, sesinw

    def convert_ecosw_esinw_to_e_w(self, ecosw, esinw):
        e2 = ecosw**2 + esinw**2
        e = np.sqrt(e2)
        w = np.arctan2(esinw, ecosw)
        return e, w

    def convert_e_w_to_ecosw_esinw(self, e, w):
        ecosw = e * np.cos(w)
        esinw = e * np.sin(w)
        return ecosw, esinw

    def convert_pars_to_default_basis(self, inpars) -> dict:
        if self.parameterisation == "per k e w tp":
            return {"per": inpars["per"],
                    "k": inpars["k"],
                    "e": inpars["e"],
                    "w": inpars["w"],
                    "tp": inpars["tp"]}

        elif self.parameterisation == "per k e w tc":
            tp = self.convert_tc_to_tp(inpars["tc"], inpars["per"], inpars["e"], inpars["w"])
            return {"per": inpars["per"],
                    "k": inpars["k"],
                    "e": inpars["e"],
                    "w": inpars["w"],
                    "tp": tp}

        elif self.parameterisation == "per k ecosw esinw tp":
            e, w = self.convert_ecosw_esinw_to_e_w(inpars["ecosw"], inpars["esinw"])
            return {"per": inpars["per"],
                    "k": inpars["k"],
                    "e": e,
                    "w": w,
                    "tp": inpars["tp"]}

        elif self.parameterisation == "per k ecosw esinw tc":
            e, w = self.convert_ecosw_esinw_to_e_w(inpars["ecosw"], inpars["esinw"])
            tp = self.convert_tc_to_tp(inpars["tc"], inpars["per"], e, w)
            return {"per": inpars["per"],
                    "k": inpars["k"],
                    "e": e,
                    "w": w,
                    "tp": tp}

        elif self.parameterisation == "per k secosw sesinw tp":
            e, w = self.convert_secosw_sesinw_to_e_w(inpars["secosw"], inpars["sesinw"])
            return {"per": inpars["per"],
                    "k": inpars["k"],
                    "e": e,
                    "w": w,
                    "tp": inpars["tp"]}

        elif self.parameterisation == "per k secosw sesinw tc":
            e, w, = self.convert_secosw_sesinw_to_e_w(inpars["secosw"], inpars["sesinw"])
            tp = self.convert_tc_to_tp(inpars["tc"], inpars["per"], e, w)
            return {"per": inpars["per"],
                    "k": inpars["k"],
                    "e": e,
                    "w": w,
                    "tp": tp}

        else:
            raise Exception(f"parameterisation {self.parameterisation} not recognised")


class Parameter:

    def __init__(self, value: float, unit: str, fixed=False):
        """
        Initialize a parameter object.

        Parameters
        ----------
        value : float
            The value of the parameter.
        unit : str
            The unit of measurement for the parameter. This is only used for
            display purposes.
        fixed : bool
            Indicates whether the parameter is fixed or free to vary in fitting.
            Default is False.
        """
        self.value = value
        self.unit = unit
        self.fixed = fixed

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}(value={self.value!r}, unit={self.unit!r}, fixed={self.fixed!r})"

    def __str__(self):
        class_name = type(self).__name__
        return f"{class_name} {self.value} {self.unit}"
