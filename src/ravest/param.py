"""Parameter handling and orbital parameterisation conversions."""
# parameterisation.py
import numpy as np

ALLOWED_PARAMETERISATIONS = ["P K e w Tp",   # default - the one used in Keplerian RV equation
                             "P K e w Tc",
                             "P K ecosw esinw Tp",
                             "P K ecosw esinw Tc",
                             "P K secosw sesinw Tp",
                             "P K secosw sesinw Tc"]


class Parameterisation:
    """Handle conversions between different orbital parameterisations."""

    @staticmethod
    def _validate_period(per: float) -> None:
        """Validate orbital period."""
        if per <= 0:
            raise ValueError(f"Invalid period: {per} <= 0")

    @staticmethod
    def _validate_semi_amplitude(k: float) -> None:
        """Validate RV semi-amplitude."""
        if k <= 0:
            raise ValueError(f"Invalid semi-amplitude: {k} <= 0")

    @staticmethod
    def _validate_eccentricity(e: float) -> None:
        """Validate orbital eccentricity."""
        if e < 0:
            raise ValueError(f"Invalid eccentricity: {e} < 0")
        if e >= 1.0:
            raise ValueError(f"Invalid eccentricity: {e} >= 1.0")

    @staticmethod
    def _validate_argument_periastron(w: float) -> None:
        """Validate argument of periastron."""
        if not -np.pi <= w < np.pi:
            raise ValueError(f"Invalid argument of periastron: {w} not in [-pi, +pi)")

    def validate_default_parameterisation_params(self, params_dict: dict[str, float]) -> None:
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
        self._validate_period(params_dict["P"])
        self._validate_semi_amplitude(params_dict["K"])
        self._validate_eccentricity(params_dict["e"])
        self._validate_argument_periastron(params_dict["w"])
        # Note: tp (time of periastron) can be any real number, so no validation needed
        # As by the time this is called, we've already validated that all parameters are at least finite real numbers

    def validate_planetary_params(self, params_dict: dict[str, float]) -> None:
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
        # convert the incoming params_dict to the default parameterisation (if we need to!)
        # check: are we in the default parameteirsation already?
        if self.parameterisation != "P K e w Tp":
            # convert to default parameterisation
            params_dict = self.convert_pars_to_default_parameterisation(params_dict)
        self.validate_default_parameterisation_params(params_dict)


    def __init__(self, parameterisation: str) -> None:
        """Parameterisation object handles parameter conversions.

        Parameters
        ----------
        parameterisation : str
            The parameterisation you wish to use. Must be one of the following:
            - "P K e w Tp"
            - "P K e w Tc"
            - "P K ecosw esinw Tp"
            - "P K ecosw esinw Tc"
            - "P K secosw sesinw Tp"
            - "P K secosw sesinw Tc"

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

    def _time_given_true_anomaly(self, true_anomaly: float, period: float, eccentricity: float, time_peri: float) -> float:
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
        eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(true_anomaly / 2))
        mean_anomaly = eccentric_anomaly - (eccentricity * np.sin(eccentric_anomaly))

        return mean_anomaly * (period / (2 * np.pi)) + time_peri

    def convert_tp_to_tc(self, time_peri: float, period: float, eccentricity: float, arg_peri: float) -> float:
        """Calculate the time of transit centre, given time of periastron passage.

        This is only a time of (primary) transit centre if the planet is actually
        transiting the star from the observer's viewpoint/inclination. Therefore
        technically this is a time of (inferior) conjunction.

        Returns
        -------
        `float`
            Time of primary transit centre/inferior conjunction (days)
        """
        theta_tc = (np.pi / 2) - arg_peri  # true anomaly at time t_c (Eastman et. al. 2013)
        return self._time_given_true_anomaly(theta_tc, period, eccentricity, time_peri)

    def convert_tc_to_tp(self, time_conj: float, period: float, eccentricity: float, arg_peri: float) -> float:
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

    def convert_secosw_sesinw_to_e_w(self, secosw: float, sesinw: float) -> tuple[float, float]:
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

    def convert_e_w_to_secosw_sesinw(self, e: float, w: float) -> tuple[float, float]:
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

    def convert_ecosw_esinw_to_e_w(self, ecosw: float, esinw: float) -> tuple[float, float]:
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

    def convert_e_w_to_ecosw_esinw(self, e: float, w: float) -> tuple[float, float]:
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

    def convert_pars_to_default_parameterisation(self, inpars: dict[str, float]) -> dict[str, float]:
        """Convert parameters from this parameterisation to default (per k e w tp).

        Parameters
        ----------
        inpars : dict
            Parameters in this parameterisation

        Returns
        -------
        dict
            Parameters in default parameterisation (P K e w Tp)
        """
        if self.parameterisation == "P K e w Tp":
            return {"P": inpars["P"],
                    "K": inpars["K"],
                    "e": inpars["e"],
                    "w": inpars["w"],
                    "Tp": inpars["Tp"]}

        elif self.parameterisation == "P K e w Tc":
            tp = self.convert_tc_to_tp(inpars["Tc"], inpars["P"], inpars["e"], inpars["w"])
            return {"P": inpars["P"],
                    "K": inpars["K"],
                    "e": inpars["e"],
                    "w": inpars["w"],
                    "Tp": tp}

        elif self.parameterisation == "P K ecosw esinw Tp":
            e, w = self.convert_ecosw_esinw_to_e_w(inpars["ecosw"], inpars["esinw"])
            return {"P": inpars["P"],
                    "K": inpars["K"],
                    "e": e,
                    "w": w,
                    "Tp": inpars["Tp"]}

        elif self.parameterisation == "P K ecosw esinw Tc":
            e, w = self.convert_ecosw_esinw_to_e_w(inpars["ecosw"], inpars["esinw"])
            tp = self.convert_tc_to_tp(inpars["Tc"], inpars["P"], e, w)
            return {"P": inpars["P"],
                    "K": inpars["K"],
                    "e": e,
                    "w": w,
                    "Tp": tp}

        elif self.parameterisation == "P K secosw sesinw Tp":
            e, w = self.convert_secosw_sesinw_to_e_w(inpars["secosw"], inpars["sesinw"])
            return {"P": inpars["P"],
                    "K": inpars["K"],
                    "e": e,
                    "w": w,
                    "Tp": inpars["Tp"]}

        elif self.parameterisation == "P K secosw sesinw Tc":
            e, w, = self.convert_secosw_sesinw_to_e_w(inpars["secosw"], inpars["sesinw"])
            tp = self.convert_tc_to_tp(inpars["Tc"], inpars["P"], e, w)
            return {"P": inpars["P"],
                    "K": inpars["K"],
                    "e": e,
                    "w": w,
                    "Tp": tp}

        else:
            raise ValueError(f"parameterisation {self.parameterisation} not recognised")

    def convert_pars_from_default_parameterisation(self, default_pars: dict[str, float]) -> dict[str, float]:
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
        if self.parameterisation == "P K e w Tp":
            return {key: default_pars[key] for key in self.pars}

        elif self.parameterisation == "P K e w Tc":
            tc = self.convert_tp_to_tc(default_pars["Tp"], default_pars["P"],
                                      default_pars["e"], default_pars["w"])
            return {"P": default_pars["P"],
                    "K": default_pars["K"],
                    "e": default_pars["e"],
                    "w": default_pars["w"],
                    "Tc": tc}

        elif self.parameterisation == "P K ecosw esinw Tp":
            ecosw, esinw = self.convert_e_w_to_ecosw_esinw(default_pars["e"], default_pars["w"])
            return {"P": default_pars["P"],
                    "K": default_pars["K"],
                    "ecosw": ecosw,
                    "esinw": esinw,
                    "Tp": default_pars["Tp"]}

        elif self.parameterisation == "P K ecosw esinw Tc":
            ecosw, esinw = self.convert_e_w_to_ecosw_esinw(default_pars["e"], default_pars["w"])
            tc = self.convert_tp_to_tc(default_pars["Tp"], default_pars["P"],
                                      default_pars["e"], default_pars["w"])
            return {"P": default_pars["P"],
                    "K": default_pars["K"],
                    "ecosw": ecosw,
                    "esinw": esinw,
                    "Tc": tc}

        elif self.parameterisation == "P K secosw sesinw Tp":
            secosw, sesinw = self.convert_e_w_to_secosw_sesinw(default_pars["e"], default_pars["w"])
            return {"P": default_pars["P"],
                    "K": default_pars["K"],
                    "secosw": secosw,
                    "sesinw": sesinw,
                    "Tp": default_pars["Tp"]}

        elif self.parameterisation == "P K secosw sesinw Tc":
            secosw, sesinw = self.convert_e_w_to_secosw_sesinw(default_pars["e"], default_pars["w"])
            tc = self.convert_tp_to_tc(default_pars["Tp"], default_pars["P"],
                                      default_pars["e"], default_pars["w"])
            return {"P": default_pars["P"],
                    "K": default_pars["K"],
                    "secosw": secosw,
                    "sesinw": sesinw,
                    "Tc": tc}

        else:
            raise ValueError(f"parameterisation {self.parameterisation} not recognised")


class Parameter:
    """Represents a model parameter with value, unit, and fixed/free status."""

    def __init__(self, value: float, unit: str, fixed: bool = False) -> None:
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
