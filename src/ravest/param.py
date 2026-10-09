"""Parameter handling and orbital parameterisation conversions."""
# parameterisation.py
import numpy as np

ALLOWED_PARAMETERISATIONS = ["P K e w Tp",   # default - the one used in Keplerian RV equation
                             "P K e w Tc",
                             # "P K ecosw esinw Tp",  # DISABLED, LIKELY TO BE REMOVED
                             # "P K ecosw esinw Tc",  # DISABLED, LIKELY TO BE REMOVED
                             "P K secosw sesinw Tp",
                             "P K secosw sesinw Tc"]


class Parameterisation:
    """Handle conversions between different orbital parameterisations."""

    @staticmethod
    def _validate_period(per: float | np.ndarray) -> None:
        """Validate orbital period.

        Parameters
        ----------
        per : float or array-like
            Orbital period(s) to validate.
        """
        if isinstance(per, (int, float)):
            if per <= 0:
                raise ValueError(f"Invalid period: {per} <= 0")
        else:
            per = np.asarray(per)
            if np.any(per <= 0):
                raise ValueError("Invalid period: some values <= 0")

    @staticmethod
    def _validate_semi_amplitude(k: float | np.ndarray) -> None:
        """Validate RV semi-amplitude.

        Parameters
        ----------
        k : float or array-like
            RV semi-amplitude(s) to validate.
        """
        if isinstance(k, (int, float)):
            if k <= 0:
                raise ValueError(f"Invalid semi-amplitude: {k} <= 0")
        else:
            k = np.asarray(k)
            if np.any(k <= 0):
                raise ValueError("Invalid semi-amplitude: some values <= 0")

    @staticmethod
    def _validate_eccentricity(e: float | np.ndarray) -> None:
        """Validate orbital eccentricity.

        Parameters
        ----------
        e : float or array-like
            Eccentricity value(s) to validate.
        """
        if isinstance(e, (int, float)):
            if e < 0:
                raise ValueError(f"Invalid eccentricity: {e} < 0")
            if e >= 1.0:
                raise ValueError(f"Invalid eccentricity: {e} >= 1.0")
        else:
            e = np.asarray(e)
            if np.any(e < 0):
                raise ValueError("Invalid eccentricity: some values < 0")
            if np.any(e >= 1.0):
                raise ValueError("Invalid eccentricity: some values >= 1.0")

    @staticmethod
    def _validate_argument_periastron(w: float | np.ndarray) -> None:
        """Validate argument of periastron.

        Parameters
        ----------
        w : float or array-like
            Argument of periastron value(s) to validate.
        """
        if isinstance(w, (int, float)):
            if not -np.pi <= w < np.pi:
                raise ValueError(f"Invalid argument of periastron: {w} not in [-pi, +pi)")
        else:
            w = np.asarray(w)
            if np.any(w < -np.pi) or np.any(w >= np.pi):
                raise ValueError("Invalid argument of periastron: some values not in [-pi, +pi)")

    def validate_default_parameterisation_params(self, params_dict: dict[str, float | np.ndarray]) -> None:
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

    def validate_planetary_params(self, params_dict: dict[str, float | np.ndarray]) -> None:
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

    def _time_given_true_anomaly(self, true_anomaly: float | np.ndarray, period: float | np.ndarray, eccentricity: float | np.ndarray, time_peri: float | np.ndarray) -> float | np.ndarray:
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

    def convert_tp_to_tc(self, time_peri: float | np.ndarray, period: float | np.ndarray, eccentricity: float | np.ndarray, arg_peri: float | np.ndarray) -> float | np.ndarray:
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

    def convert_tc_to_tp(self, time_conj: float | np.ndarray, period: float | np.ndarray, eccentricity: float | np.ndarray, arg_peri: float | np.ndarray) -> float | np.ndarray:
        """Calculate the time of periastron passage, given time of primary transit.

        Returns
        -------
        `float`
            Time of periastron passage (days).
        """
        theta_tc = (np.pi / 2) - arg_peri  # true anomaly at time t_c

        # Validate eccentricity before sqrt operations (prevents RuntimeWarnings)
        self._validate_eccentricity(eccentricity)

        # Calculate eccentric anomaly using the relation: E = 2*arctan(sqrt((1-e)/(1+e)) * tan(theta_tc/2))
        eccentric_anomaly = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(theta_tc / 2))

        mean_anomaly = eccentric_anomaly - (eccentricity * np.sin(eccentric_anomaly))
        return time_conj - (period / (2 * np.pi)) * mean_anomaly

    def convert_secosw_sesinw_to_e_w(self, secosw: float | np.ndarray, sesinw: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
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

    def convert_e_w_to_secosw_sesinw(self, e: float | np.ndarray, w: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
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

    def convert_ecosw_esinw_to_e_w(self, ecosw: float | np.ndarray, esinw: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
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

    def convert_e_w_to_ecosw_esinw(self, e: float | np.ndarray, w: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
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

    def convert_pars_to_default_parameterisation(self, inpars: dict[str, float | np.ndarray]) -> dict[str, float | np.ndarray]:
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

    def log_jacobian_determinant(self) -> float:
        """Log absolute Jacobian determinant ``|d(e,w)/d(u,v)|`` for this parameterisation.

        Returns log(2) for the secosw/sesinw parameterisation, else 0.0.
        """
        if "secosw" in self.parameterisation:
            return np.log(2)
        return 0.0


def _instrument_subscript_latex(inst: str) -> str:
    r"""Format an instrument name as the body of a LaTeX math subscript.

    Instrument names may carry a numeric suffix (e.g. ``HARPS_15``, used when
    data is split at a known instrument change). Wrapping the whole name in
    ``\rm`` puts the ``_15`` inside one roman group, so matplotlib mathtext only
    subscripts the first digit. Splitting on the first underscore and using
    ``\mathrm{...}`` with an explicit nested subscript renders the full suffix.

    Examples: ``HARPS`` -> ``\mathrm{HARPS}``; ``HARPS_15`` -> ``\mathrm{HARPS}_{15}``.
    """
    base, _, suffix = inst.partition("_")
    if suffix:
        return r"\mathrm{{{}}}_{{{}}}".format(base, suffix)
    return r"\mathrm{{{}}}".format(base)


def _planet_subscript_latex(planet_letter: str) -> str:
    r"""Format a planet letter as the body of a LaTeX math subscript.

    A planet letter is a label, not a physical variable, so MNRAS style sets it
    roman rather than italic (cf. ``T_{\mathrm{Eff}}``, ``b_{\mathrm{MAX}}``).

    Examples: ``b`` -> ``\mathrm{b}``.
    """
    return r"\mathrm{{{}}}".format(planet_letter)


def param_key_to_latex(param_key: str) -> str:
    r"""Convert a parameter key to a LaTeX-formatted label for plotting.

    Parameter keys are the flat strings Ravest uses to name parameters: the
    keys of ``Fitter.params`` and ``GPFitter.params``, and the column
    names of the sample dataframes. This function is the single source of the
    labels drawn on every Ravest plot.

    Most keys are a *base* plus an optional suffix, joined by an underscore.
    The base is the parameter's symbol name on its own: the base of
    ``secosw_b`` is ``secosw``, and the base of ``P_b`` is ``P``. No base
    contains an underscore, so the first underscore always ends the base. Keys
    take one of four shapes:

    - ``<base>`` alone, e.g. ``P``, ``secosw``.
    - ``<base>_<planet letter>``, e.g. ``P_b``, ``secosw_c``.
    - ``<prefix>_<instrument>``, e.g. ``g_HARPS``, ``jit_HARPS_15``. Unlike a
      planet letter, an instrument name may contain a further underscore.
    - a fixed whole-key name, not decomposed at all, e.g. ``gd``, ``gp_A``.

    Subscripts follow MNRAS style: a subscript that is a physical variable is
    italic, one that is merely a label is roman. Planet letters, instrument
    names, ``GP`` and the ``e``/``p`` on the GP length scales are all labels,
    so they are set with ``\mathrm``.

    Parameters
    ----------
    param_key : str
        Parameter key, e.g. 'P_b', 'w_c', 'jit_HARPS', 'gp_A'.

    Returns
    -------
    str
        LaTeX-formatted string suitable for matplotlib labels, wrapped in
        ``$...$``. Returns the key unchanged if the parameter is not
        recognised, so an unexpected parameter is still legible on the plot.

    Notes
    -----
    Those four key shapes need four ways of recognising a key, so the body is
    a run of guard clauses -- the first to match returns, and anything
    unrecognised falls through to the end:

    1. exact lookup for the GP hyperparameters;
    2. exact comparison for the trend parameters ``gd`` and ``gdd``;
    3. fixed-prefix match for ``Tc``/``Tp`` and ``jit_``/``g_``, parsing the
       remainder as a planet letter or an instrument name;
    4. split on the first underscore and look up the base, for the orbital
       parameters.

    Only the fallback's position matters; the four blocks match disjoint sets
    of keys.
    """
    # ---- Lookup tables ----------------------------------------------------

    # Orbital parameter bases -> the LaTeX symbol for that base on its own. A
    # planet suffix, where the key has one, is appended by section 4 below.
    _BASE_TO_LATEX = {
        "P": "P",
        "K": "K",
        "e": "e",
        "w": r"\omega",
        "secosw": r"\sqrt{e}\cos\omega",
        "sesinw": r"\sqrt{e}\sin\omega",
        "ecosw": r"e\cos\omega",
        "esinw": r"e\sin\omega",
    }

    # Param keys in _BASE_TO_LATEX where the latex label has omega in it. These take an extra \star
    # subscript naming the star's argument of periastron: the star's and the
    # planet's differ by pi, so the label has to say which one is meant.
    # For a good explanation see Householder & Weiss 2022 https://doi.org/10.48550/arXiv.2212.06966
    _OMEGA_PARAM_KEYS = frozenset({"w", "secosw", "sesinw", "ecosw", "esinw"})

    # GP hyperparameters. These are fixed whole-key names rather than bases, so
    # unlike _BASE_TO_LATEX the values are finished labels, $...$ delimiters and
    # all. 'GP', 'e' and 'p' are labels, not variables, so they are set roman.
    _GP_TO_LATEX = {
        "gp_A": r"$A_{\mathrm{GP}}$",
        "gp_P": r"$P_{\mathrm{GP}}$",
        "gp_lambda_e": r"$\lambda_{\mathrm{e}}$",
        "gp_lambda_p": r"$\lambda_{\mathrm{p}}$",
    }

    # ---- 1. Exact lookup: GP hyperparameters ------------------------------

    if param_key in _GP_TO_LATEX:
        return _GP_TO_LATEX[param_key]

    # ---- 2. Exact comparison: trend parameters ----------------------------
    # The linear and quadratic RV trend, written as time derivatives of gamma.

    if param_key == "gd":
        return r"$\dot{\gamma}$"
    if param_key == "gdd":
        return r"$\ddot{\gamma}$"

    # ---- 3a. Fixed prefix: the two characteristic times -------------------
    # Transit centre ('Tc', 'Tc_b') and periastron passage ('Tp', 'Tp_b'). Both
    # the C/P and the planet letter are labels, so the whole subscript is one
    # roman group. C and P are capitalised so that they cannot be misread as
    # the planet letters which are always lowercase.
    #
    # The doubled braces below are .format() escapes, not LaTeX: '{{' emits a
    # literal '{' and '{}' is a placeholder, so the format string
    # '$T_{{\mathrm{{{},{}}}}}$' emits '$T_{\mathrm{C,b}}$'.

    for time_prefix, time_letter in (("Tc", "C"), ("Tp", "P")):
        if param_key.startswith(time_prefix):
            planet_letter = param_key[len(time_prefix):].lstrip("_")

            # check if there's a planet letter
            if planet_letter:
                return r"$T_{{\mathrm{{{},{}}}}}$".format(time_letter, planet_letter)
            else:
                return r"$T_{{\mathrm{{{}}}}}$".format(time_letter)

    # ---- 3b. Fixed prefix: per-instrument parameters ----------------------
    # Jitter ('jit_HARPS') and RV offset ('g_HARPS'). Everything after the
    # prefix is an instrument name, which unlike a planet letter may contain a
    # further underscore ('HARPS_15'), hence the dedicated helper.

    if param_key.startswith("jit_"):
        instrument = param_key[len("jit_"):]
        return r"$\sigma_{{{}}}$".format(_instrument_subscript_latex(instrument))
    if param_key.startswith("g_"):
        instrument = param_key[len("g_"):]
        return r"$\gamma_{{{}}}$".format(_instrument_subscript_latex(instrument))

    # ---- 4. Table-driven: orbital parameters ------------------------------
    # A key here is '<base>' or '<base>_<planet letter>'. No base contains an
    # underscore, so the first underscore separates the two and a single lookup
    # settles it -- there is no need to try each base in turn.

    base, _, planet_letter = param_key.partition("_")
    if base in _BASE_TO_LATEX:
        symbol = _BASE_TO_LATEX[base]
        is_omega_parameter = base in _OMEGA_PARAM_KEYS
        if planet_letter:
            # \star shares the one subscript with the planet letter.
            subscript = (r"\star," if is_omega_parameter else "") + _planet_subscript_latex(planet_letter)
            return r"${}_{{{}}}$".format(symbol, subscript)
        # No planet letter, so \star is the entire subscript.
        return r"${}{}$".format(symbol, r"_\star" if is_omega_parameter else "")

    # ---- Fallback ---------------------------------------------------------
    # Nothing matched. Return the key as-is so that an unexpected parameter is
    # still legible on the plot, rather than raising part-way through a render.

    return param_key


# Every internal unit, plain-text form -> LaTeX form.
_UNIT_TO_LATEX = {
    "": "",
    "d": r"$\mathrm{d}$",
    "rad": r"$\mathrm{rad}$",
    "m/s": r"$\mathrm{m}\,\mathrm{s}^{-1}$",
    "m/s/d": r"$\mathrm{m}\,\mathrm{s}^{-1}\,\mathrm{d}^{-1}$",
    "m/s/d^2": r"$\mathrm{m}\,\mathrm{s}^{-1}\,\mathrm{d}^{-2}$",
}


def _param_key_to_plain_unit(key: str) -> str | None:
    """Return the plain-text unit for a parameter key, or None if unrecognised.

    Every unit returned is a key of ``_UNIT_TO_LATEX``.
    """
    # Orbital parameter base names -> units
    _BASE_UNITS = {
        "P": "d",
        "K": "m/s",
        "e": "",
        "w": "rad",
        "secosw": "",
        "sesinw": "",
        "ecosw": "",
        "esinw": "",
    }

    # GP hyperparameters
    _GP_UNITS = {
        "gp_A": "m/s",
        "gp_P": "d",
        "gp_lambda_e": "d",
        "gp_lambda_p": "",
    }
    if key in _GP_UNITS:
        return _GP_UNITS[key]

    # Trend parameters
    if key == "gd":
        return "m/s/d"
    if key == "gdd":
        return "m/s/d^2"

    # Tc and Tp (with or without planet suffix)
    if key.startswith("Tc") or key.startswith("Tp"):
        return "d"

    # Instrument parameters
    if key.startswith("jit_"):
        return "m/s"
    if key.startswith("g_"):
        return "m/s"

    # Orbital parameters with optional planet suffix
    for base in sorted(_BASE_UNITS.keys(), key=len, reverse=True):
        if key == base or key.startswith(base + "_"):
            return _BASE_UNITS[base]

    # Unrecognised key
    return None


def param_key_to_unit(key: str, *, latex: bool = False) -> str | None:
    r"""Return ravest's internal unit for a parameter key.

    ravest does not convert units: every parameter, prior and data value is
    in these units (see :doc:`/units`). This function is the single source of
    those units, for labelling plots and formatting results tables.

    Parameters
    ----------
    key : str
        Parameter key, e.g. 'P_b', 'K_c', 'jit_HARPS', 'gp_A'.
    latex : bool, optional
        If False (default), return plain text, e.g. 'm/s'. If True, return
        the LaTeX form for matplotlib labels, e.g.
        '$\mathrm{m}\,\mathrm{s}^{-1}$'. Must be passed by keyword.

    Returns
    -------
    str or None
        The unit. Returns '' for dimensionless parameters, in both forms.
        Returns None if the parameter is not recognised.

    Examples
    --------
    >>> param_key_to_unit("K_b")
    'm/s'
    >>> param_key_to_unit("K_b", latex=True)
    '$\\mathrm{m}\\,\\mathrm{s}^{-1}$'
    >>> param_key_to_unit("e_b")
    ''
    """
    unit = _param_key_to_plain_unit(key)
    if unit is None or not latex:
        return unit
    return _UNIT_TO_LATEX[unit]


class Parameter:
    """Represents a model parameter with value and fixed/free status."""

    def __init__(self, value: float, *, fixed: bool) -> None:
        """
        Initialize a parameter object.

        Parameters
        ----------
        value : float
            The value of the parameter, in ravest's internal units (days, m/s,
            radians). ravest does not convert units; see :doc:`/units`.
        fixed : bool
            Whether the parameter is fixed (True) or free to vary in fitting
            (False). Must be passed by keyword, as exactly True or False.

        Raises
        ------
        TypeError
            If `fixed` is not a Python bool.
        """
        if type(fixed) is not bool:
            raise TypeError(
                f"fixed must be True or False, not {fixed!r} of type {type(fixed)!r}."
            )
        self.value = value
        self.fixed = fixed

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}(value={self.value!r}, fixed={self.fixed!r})"
