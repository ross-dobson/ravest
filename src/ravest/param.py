# parameterisation.py
import numpy as np

ALLOWED_PARAMETERISATIONS = ["per k e w tp",
                             "per k e w tc",
                             "per k ecosw esinw tp",
                             "per k ecosw esinw tc",
                             "per k secosw sesinw tp",
                             "per k secosw sesinw tc"]


class Parameterisation:

    @staticmethod
    def _validate_period(per):
        """Validate orbital period."""
        if per <= 0:
            raise ValueError(f"Invalid period: {per} <= 0")

    @staticmethod
    def _validate_semi_amplitude(k):
        """Validate RV semi-amplitude."""
        if k <= 0:
            raise ValueError(f"Invalid semi-amplitude: {k} <= 0")

    @staticmethod
    def _validate_eccentricity(e):
        """Validate orbital eccentricity."""
        if e < 0:
            raise ValueError(f"Invalid eccentricity: {e} < 0")
        if e >= 1.0:
            raise ValueError(f"Invalid eccentricity: {e} >= 1.0")

    @staticmethod
    def _validate_argument_periastron(w):
        """Validate argument of periastron."""
        if not -np.pi <= w < np.pi:
            raise ValueError(f"Invalid argument of periastron: {w} not in [-pi, +pi)")

    def validate_default_parameterisation_params(self, params_dict):
        """Validate all parameters in default parameterisation (per k e w tp).

        Parameters
        ----------
        params_dict : dict
            Dictionary with keys: per, k, e, w, tp

        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        self._validate_period(params_dict["per"])
        self._validate_semi_amplitude(params_dict["k"])
        self._validate_eccentricity(params_dict["e"])
        self._validate_argument_periastron(params_dict["w"])
        # Note: tp (time of periastron) can be any real number, so no validation needed

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

        # Validate eccentricity before sqrt operations (prevents RuntimeWarnings)
        self._validate_eccentricity(eccentricity)

        # Calculate eccentric anomaly using the relation: E = 2*arctan(sqrt((1-e)/(1+e)) * tan(θ_tc/2))
        eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(theta_tc / 2))

        mean_anomaly = eccentric_anomaly - (eccentricity * np.sin(eccentric_anomaly))
        return time_conj - (period / (2 * np.pi)) * mean_anomaly

    def convert_secosw_sesinw_to_e_w(self, secosw, sesinw):
        e = secosw**2 + sesinw**2
        w = np.arctan2(sesinw, secosw)
        return e, w

    def convert_e_w_to_secosw_sesinw(self, e, w):
        # Validate eccentricity before sqrt operations (prevents RuntimeWarnings)
        self._validate_eccentricity(e)
        sqrt_e = np.sqrt(e)  # Calculate once, use twice
        secosw = sqrt_e * np.cos(w)
        sesinw = sqrt_e * np.sin(w)
        return secosw, sesinw

    def convert_ecosw_esinw_to_e_w(self, ecosw, esinw):
        e2 = ecosw**2 + esinw**2
        e = np.sqrt(e2)
        # Validate computed eccentricity is within valid range 0 <= e < 1
        self._validate_eccentricity(e)
        w = np.arctan2(esinw, ecosw)
        return e, w

    def convert_e_w_to_ecosw_esinw(self, e, w):
        ecosw = e * np.cos(w)
        esinw = e * np.sin(w)
        return ecosw, esinw

    def convert_pars_to_default_parameterisation(self, inpars) -> dict:
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
            raise ValueError(f"parameterisation {self.parameterisation} not recognised")

    def convert_pars_from_default_parameterisation(self, default_pars) -> dict:
        """Convert parameters from default (per k e w tp) to this parameterisation.

        Parameters
        ----------
        default_pars : dict
            Dictionary with keys: per, k, e, w, tp

        Returns
        -------
        dict
            Parameters in this parameterisation
        """
        if self.parameterisation == "per k e w tp":
            return {key: default_pars[key] for key in self.pars}

        elif self.parameterisation == "per k e w tc":
            tc = self.convert_tp_to_tc(default_pars["tp"], default_pars["per"],
                                      default_pars["e"], default_pars["w"])
            return {"per": default_pars["per"],
                    "k": default_pars["k"],
                    "e": default_pars["e"],
                    "w": default_pars["w"],
                    "tc": tc}

        elif self.parameterisation == "per k ecosw esinw tp":
            ecosw, esinw = self.convert_e_w_to_ecosw_esinw(default_pars["e"], default_pars["w"])
            return {"per": default_pars["per"],
                    "k": default_pars["k"],
                    "ecosw": ecosw,
                    "esinw": esinw,
                    "tp": default_pars["tp"]}

        elif self.parameterisation == "per k ecosw esinw tc":
            ecosw, esinw = self.convert_e_w_to_ecosw_esinw(default_pars["e"], default_pars["w"])
            tc = self.convert_tp_to_tc(default_pars["tp"], default_pars["per"],
                                      default_pars["e"], default_pars["w"])
            return {"per": default_pars["per"],
                    "k": default_pars["k"],
                    "ecosw": ecosw,
                    "esinw": esinw,
                    "tc": tc}

        elif self.parameterisation == "per k secosw sesinw tp":
            secosw, sesinw = self.convert_e_w_to_secosw_sesinw(default_pars["e"], default_pars["w"])
            return {"per": default_pars["per"],
                    "k": default_pars["k"],
                    "secosw": secosw,
                    "sesinw": sesinw,
                    "tp": default_pars["tp"]}

        elif self.parameterisation == "per k secosw sesinw tc":
            secosw, sesinw = self.convert_e_w_to_secosw_sesinw(default_pars["e"], default_pars["w"])
            tc = self.convert_tp_to_tc(default_pars["tp"], default_pars["per"],
                                      default_pars["e"], default_pars["w"])
            return {"per": default_pars["per"],
                    "k": default_pars["k"],
                    "secosw": secosw,
                    "sesinw": sesinw,
                    "tc": tc}

        else:
            raise ValueError(f"parameterisation {self.parameterisation} not recognised")


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
