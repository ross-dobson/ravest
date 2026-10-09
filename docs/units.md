# Units in Ravest

## Ravest does not convert units

Ravest works in one fixed set of units: **days, metres per second, and radians**. It does not read, check, or convert units anywhere. A number you pass in is used as-is, in the units listed on this page. That applies to:

- your data (`time`, `vel`, `velerr` and `t0` in `add_data`),
- every parameter value (`Parameter(value, fixed=...)`),
- every prior (the bounds, location and scale of a prior on a parameter are in that parameter's units).

If your data or literature values are in other units, convert them before they reach Ravest.

## Parameter units

| Parameter | Ravest key | Unit | Notes |
|-----------|------------|------|-------|
| $P$ | `P` | d | orbital period |
| $K$ | `K` | m/s | RV semi-amplitude |
| $e$ | `e` | dimensionless | eccentricity, $0 \le e < 1$ |
| $\omega_\star$ | `w` | rad | the star's argument of periastron, in $[-\pi, \pi)$ |
| $\sqrt{e}\cos\omega_\star$, $\sqrt{e}\sin\omega_\star$ | `secosw`, `sesinw` | dimensionless | eccentricity and argument of periastron combined |
| $T_{\mathrm{C}}$, $T_{\mathrm{P}}$ | `Tc`, `Tp` | d | time of transit centre / periastron passage, on the same time axis as your `time` data |
| $\gamma_{\mathrm{inst}}$ | `g_<instrument>` | m/s | RV offset for that instrument |
| $\sigma_{\mathrm{inst}}$ | `jit_<instrument>` | m/s | jitter for that instrument |
| $\dot{\gamma}$ | `gd` | m/s/d | linear trend, $\dot{\gamma}\,(t - t_0)$ |
| $\ddot{\gamma}$ | `gdd` | m/s/d^2 | quadratic trend, $\ddot{\gamma}\,(t - t_0)^2$ |
| $A_{\mathrm{GP}}$ | `gp_A` | m/s | GP amplitude |
| $f$ | `gp_f` | dimensionless | GP cosine-term weight (QPCosine kernel) |
| $\lambda$ | `gp_lambda` | d | GP length scale (Exponential, Matern32 and Matern52 kernels) |
| $\lambda_{\mathrm{e}}$ | `gp_lambda_e` | d | GP evolutionary (exponential) length scale |
| $\lambda_{\mathrm{p}}$ | `gp_lambda_p` | dimensionless | GP periodic (harmonic complexity) length scale |
| $P_{\mathrm{GP}}$ | `gp_P` | d | GP period |

Planet parameters take the planet letter as a suffix (`P_b`, `K_c`), and the units are the same for every planet. The symbols are the labels Ravest uses in its plots, where they gain the planet letter or instrument name as a subscript, e.g. $P_{\mathrm{b}}$, $\gamma_{\mathrm{HARPS}}$.

The data passed to `add_data` follow the same units: `time` and `t0` in days, `vel` and `velerr` in m/s. The stellar mass, given to `Star` or as `mass_star` to `Planet.mpsini` and `calculate_mpsini`, is in solar masses $M_\odot$.

## Priors

A prior is in the units of the parameter it is on. For example, `Uniform(0, 20)` on `K_b` means 0 to 20 m/s, and `Normal(12.3, 0.1)` on `P_b` means $12.3 \pm 0.1$ days. Eccentricity priors are dimensionless. A prior on `w_b` is in radians.

## Time systems

Ravest only ever uses time *differences* and periods: the phase of an orbit, and the trend relative to `t0`. So it never needs to know which time system or offset you use (JD, BJD, BJD - 2450000, BTJD, ...; UTC or TDB). What it does need is:

- times in **days**;
- **one shared time axis** for `time`, `t0` and every `Tc`/`Tp`, all in the same system and with the same offset;
- every instrument's data on that same axis.

If you mix systems or offsets (for example, `time` in BJD - 2450000 but a literature `Tc` in full BJD), the fit will run but the orbital phase will be wrong.

The RV plots label their time axis "Time [days]". To show your own system, pass e.g. `xlabel="BJD - 2450000 [days]"` to the plotting method.

## Converting your data

The most common slip is RV data in km/s. Convert both the velocities and their uncertainties:

```python
vel = vel_kms * 1000.0      # km/s -> m/s
velerr = velerr_kms * 1000.0
```

Do the same for any literature values you use for initial parameter values or priors (e.g. a semi-amplitude, offset or jitter quoted in km/s).


## Looking up units in code

`ravest.param.param_key_to_unit` returns the unit for any parameter key. It is the single source of the units in the table above.

```python
>>> from ravest.param import param_key_to_unit
>>> param_key_to_unit("K_b")
'm/s'
>>> param_key_to_unit("gdd")
'm/s/d^2'
>>> param_key_to_unit("e_b")
''
>>> param_key_to_unit("K_b", latex=True)
'$\\mathrm{m}\\,\\mathrm{s}^{-1}$'
```

It returns `''` for dimensionless parameters and `None` for keys it does not recognise. Use `latex=True` for a matplotlib-ready label.
