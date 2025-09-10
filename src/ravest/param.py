"""Parameter handling and orbital parameterisation conversions."""
# parameterisation.py
import numpy as np

ALLOWED_PARAMETERISATIONS = ["per k e w tp",   # default - the one used in Keplerian RV equation
                             "per k e w tc",
                             "per k ecosw esinw tp",
                             "per k ecosw esinw tc",
                             "per k secosw sesinw tp",
                             "per k secosw sesinw tc"]


class Parameterisation:
    """Handle conversions between different orbital parameterisations."""

    @staticmethod
    def _validate_period(per) -> None:
        """Validate orbital period."""
        if per <= 0:
            raise ValueError(f"Invalid period: {per} <= 0")

    @staticmethod
    def _validate_semi_amplitude(k) -> None:
        """Validate RV semi-amplitude."""
        if k <= 0:
            raise ValueError(f"Invalid semi-amplitude: {k} <= 0")

    @staticmethod
    def _validate_eccentricity(e) -> None:
        """Validate orbital eccentricity."""
        if e < 0:
            raise ValueError(f"Invalid eccentricity: {e} < 0")
        if e >= 1.0:
            raise ValueError(f"Invalid eccentricity: {e} >= 1.0")

    @staticmethod
    def _validate_argument_periastron(w) -> None:
        """Validate argument of periastron."""
        if not -np.pi <= w < np.pi:
            raise ValueError(f"Invalid argument of periastron: {w} not in [-pi, +pi)")

    def validate_default_parameterisation_params(self, params_dict) -> None:
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

    def validate_planetary_params(self, params_dict) -> None:
        """Validate planetary parameters are astrophysically valid, in any parameterisation.

        Parameters
        ----------
        params_dict : dict
            Dictionary with planetary parameters in current parameterisation

        Raises
        ------
        ValueError
            If any parameter is invalid for this parameterisation
        """
        # Always validate period and semi-amplitude (present in all parameterisations)
        self._validate_period(params_dict["per"])
        self._validate_semi_amplitude(params_dict["k"])

        if self.parameterisation in ["per k e w tp", "per k e w tc"]:
            self._validate_eccentricity(params_dict["e"])
            self._validate_argument_periastron(params_dict["w"])

        elif self.parameterisation in ["per k ecosw esinw tp", "per k ecosw esinw tc"]:
            # For ecosw/esinw: check that ecosw² + esinw² < 1 (valid eccentricity)
            e_squared = params_dict["ecosw"]**2 + params_dict["esinw"]**2
            if e_squared >= 1.0:
                raise ValueError(f"Invalid ecosw/esinw: ecosw²+esinw² = {e_squared:.6f} >= 1.0")

        elif self.parameterisation in ["per k secosw sesinw tp", "per k secosw sesinw tc"]:
            # For secosw/sesinw: check that secosw² + sesinw² < 1 (valid eccentricity)
            e = params_dict["secosw"]**2 + params_dict["sesinw"]**2
            if e >= 1.0:
                raise ValueError(f"Invalid secosw/sesinw: secosw²+sesinw² = {e:.6f} >= 1.0")

        # Validate tc and tp are finite real numbers
        if "tc" in params_dict:
            if not np.isfinite(params_dict["tc"]):
                raise ValueError(f"Invalid tc: {params_dict['tc']} (must be finite)")
        if "tp" in params_dict:
            if not np.isfinite(params_dict["tp"]):
                raise ValueError(f"Invalid tp: {params_dict['tp']} (must be finite)")

    def __init__(self, parameterisation: str) -> None:
        """Parameterisation object handles parameter conversions.

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

    def _time_given_true_anomaly(self, true_anomaly, period, eccentricity, time_peri) -> float:
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

    def convert_tp_to_tc(self, time_peri, period, eccentricity, arg_peri) -> float:
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

    def convert_tc_to_tp(self, time_conj, period, eccentricity, arg_peri) -> float:
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

    def convert_secosw_sesinw_to_e_w(self, secosw, sesinw) -> tuple[float, float]:
        """Convert sqrt(e)cos(w), sqrt(e)sin(w) to eccentricity and argument of periastron.

        Parameters
        ----------
        secosw : float
            sqrt(e) * cos(w)
        sesinw : float
            sqrt(e) * sin(w)

        Returns
        -------
        float, float
            Eccentricity e and argument of periastron w
        """
        e = secosw**2 + sesinw**2
        w = np.arctan2(sesinw, secosw)
        return e, w

    def convert_e_w_to_secosw_sesinw(self, e, w) -> tuple[float, float]:
        """Convert eccentricity and argument of periastron to sqrt(e)cos(w), sqrt(e)sin(w).

        Parameters
        ----------
        e : float
            Eccentricity
        w : float
            Argument of periastron

        Returns
        -------
        float, float
            sqrt(e)*cos(w) and sqrt(e)*sin(w)
        """
        # Validate eccentricity before sqrt operations (prevents RuntimeWarnings)
        self._validate_eccentricity(e)
        sqrt_e = np.sqrt(e)  # Calculate once, use twice
        secosw = sqrt_e * np.cos(w)
        sesinw = sqrt_e * np.sin(w)
        return secosw, sesinw

    def convert_ecosw_esinw_to_e_w(self, ecosw, esinw) -> tuple[float, float]:
        """Convert e*cos(w), e*sin(w) to eccentricity and argument of periastron.

        Parameters
        ----------
        ecosw : float
            e * cos(w)
        esinw : float
            e * sin(w)

        Returns
        -------
        float, float
            Eccentricity e and argument of periastron w
        """
        e2 = ecosw**2 + esinw**2
        e = np.sqrt(e2)
        # Validate computed eccentricity is within valid range 0 <= e < 1
        self._validate_eccentricity(e)
        w = np.arctan2(esinw, ecosw)
        return e, w

    def convert_e_w_to_ecosw_esinw(self, e, w) -> tuple[float, float]:
        """Convert eccentricity and argument of periastron to e*cos(w), e*sin(w).

        Parameters
        ----------
        e : float
            Eccentricity
        w : float
            Argument of periastron

        Returns
        -------
        float, float
            e*cos(w) and e*sin(w)
        """
        ecosw = e * np.cos(w)
        esinw = e * np.sin(w)
        return ecosw, esinw

    def convert_pars_to_default_parameterisation(self, inpars) -> dict:
        """Convert parameters from this parameterisation to default (per k e w tp).

        Parameters
        ----------
        inpars : dict
            Parameters in this parameterisation

        Returns
        -------
        dict
            Parameters in default parameterisation (per k e w tp)
        """
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
    """Represents a model parameter with value, unit, and fixed/free status."""

    def __init__(self, value: float, unit: str, fixed=False) -> None:
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

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}(value={self.value!r}, unit={self.unit!r}, fixed={self.fixed!r})"

    def __str__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name} {self.value} {self.unit}"
