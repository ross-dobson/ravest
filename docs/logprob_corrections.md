# Log-posterior corrections for the (secosw, sesinw) parameterisation

## Overview

When fitting radial velocity data in the transformed parameterisation

```
u = secosw = sqrt(e) * cos(w)
v = sesinw = sqrt(e) * sin(w)
```

rather than sampling `(e, w)` directly, the log-posterior needs constant
corrections so that Bayesian evidence (via nested sampling or the Learned
Harmonic Mean Estimator, LHME, e.g. through the `harmonic` library) is
coordinate-invariant. These corrections are constants, so they cancel in the
Metropolis-Hastings acceptance ratio and have **no effect on MCMC parameter
inference** - they only matter when computing an absolute evidence value.

The inverse transform is `e = u^2 + v^2`, `w = atan2(v, u)`. The physical
constraint `0 <= e < 1` is equivalent to the unit disc `u^2 + v^2 < 1`.

## Per-planet classification

Ravest classifies **each planet independently**, then sums the per-planet
contributions. This matters for multi-planet systems where different planets
may have priors defined in different coordinate systems (e.g. planet b's
priors on `(u, v)`, planet c's priors on `(e, w)`) - each planet's correction
depends only on its own parameterisation and prior choice, not on any other
planet's.

### CASE_1: default parameterisation, or transformed with this planet's secosw/sesinw fixed

No transformation is in effect for this planet (either the whole fit uses
the default `(e, w)` parameterisation, or this planet's `secosw`/`sesinw` are
fixed rather than sampled). Contribution: `0`.

### CASE_2: transformed, free secosw/sesinw, priors on (u, v)

The sampler proposes `u` and `v` directly, and the priors are defined on `u`
and `v` too - no Jacobian is needed since sampling and prior space coincide.
However, the physical validity check truncates `(u, v)` to the unit disc.

With `Uniform(-1, 1)` priors on both `u` and `v`, the joint prior density is
`1/4` over the square `[-1,1] x [-1,1]` (area 4). Truncated to the unit disc
(area `pi`), the prior integrates to `pi/4` instead of `1`. Renormalising
requires multiplying by `4/pi`, i.e. adding `log(4/pi)` (~0.2416) in log
space:

```
log_posterior = log_likelihood + log_prior_u(u) + log_prior_v(v) + log(4/pi)
```

Ravest only supports this case for `Uniform(-1, 1)` priors on both `u` and
`v`. Any other prior shape on `(u, v)` would require computing

```
correction = -log( integral over u^2+v^2<1 of p_u(u) * p_v(v) du dv )
```

which has no closed form in general. **Ravest hard-raises
`NotImplementedError`** if it encounters such a prior, directing the user to
place their eccentricity belief on `e` instead (Case 3) - a separable,
rotationally-symmetric prior on `(u, v)` is always re-expressible as a prior
on `e` (e.g. iid Gaussians on `u` and `v` are exactly a `Rayleigh` prior on
`e`, which Ravest ships directly), so this is not a loss of expressiveness.

### CASE_3: transformed, free secosw/sesinw, priors on (e, w)

The sampler proposes `u` and `v`, but the priors are defined on `e` and `w`.
Evaluating the priors requires converting `e = u^2 + v^2`, `w = atan2(v, u)`,
which is a change of variables and therefore needs a Jacobian correction:

```
J = | de/du   de/dv |   =   | 2u     2v  |
    | dw/du   dw/dv |       | -v/e   u/e |

det(J) = (2u)(u/e) - (2v)(-v/e) = 2(u^2 + v^2)/e = 2e/e = 2
```

So `|d(e,w)/d(u,v)| = 2`, giving a correction of `+log(2)` (~0.6931):

```
log_posterior = log_likelihood + log_prior_e(e(u,v)) + log_prior_w(w(u,v)) + log(2)
```

No renormalisation is needed here: the prior on `e` handles its own support
(it returns `-inf` for `e >= 1`), so there is no external truncation, and the
Jacobian is a purely geometric property of the coordinate mapping -
independent of the prior shape. The same `+log(2)` applies whether the prior
on `e` is `Uniform`, `HalfNormal`, `Rayleigh`, `VanEylen19Mixture`, `Beta`, or
any other properly normalised distribution.

### Sanity check

With uniform priors (`e ~ Uniform(0,1)`, `w ~ Uniform(-pi, pi)`), the
effective log-prior density inside the supported region is:

- CASE_1: `log(1) + log(1/(2*pi)) = -log(2*pi)`
- CASE_2: `log(1/4) + log(4/pi) = -log(pi)`
- CASE_3: `log(1) + log(1/(2*pi)) + log(2) = -log(pi)`

CASE_2 and CASE_3 agree with each other (as they must - both express the
same physical prior on eccentricity, just in different coordinates). They
differ from CASE_1 by `log(2)`, which is expected since CASE_1 lives in a
different coordinate system.

## Worked example: mixed two-planet system

Consider a two-planet fit in the `(u, v)` parameterisation where planet b's
eccentricity prior is `Uniform(-1, 1)` on both `secosw_b`/`sesinw_b` (CASE_2),
and planet c's eccentricity prior is a `HalfNormal` on `e_c` with
`Uniform(-pi, pi)` on `w_c` (CASE_3):

```
total correction = log(4/pi) + log(2) ~= 0.2416 + 0.6931 = 0.9347
```

A classifier that inspects the *whole system's* free parameters and priors
in one pass (rather than per planet) cannot represent this: it would
misclassify the system as globally CASE_2 or globally CASE_3 and apply the
wrong per-planet constant (or the right constant to the wrong number of
planets), producing a systematic ln(Z) error of ~0.45 for this two-planet
example - far outside typical MCMC/LHME noise (~0.01). Classifying each
planet independently and summing avoids this.

## Implementation

- `Parameterisation.log_jacobian_determinant()` (`src/ravest/param.py`)
  returns `log(2)` for the `secosw`/`sesinw` parameterisation, `0.0`
  otherwise.
- `LogPosterior._classify_planet_case(letter)` classifies a single planet
  into `CASE_1`/`CASE_2`/`CASE_3` (or raises `NotImplementedError` for
  unsupported `(u, v)` priors).
- `LogPosterior._compute_logprob_corrections()` loops over all planets,
  classifies each, and sums the Jacobian and prior-renormalisation
  contributions. These are computed once at construction time (they depend
  only on the parameterisation, priors, and free/fixed status - all fixed at
  construction) and stored as `_logprob_jacobian_correction`,
  `_logprob_prior_renorm_correction`, and a per-planet
  `_logprob_correction_breakdown` dict.
- `LogPosterior.log_probability` adds both stored corrections to
  `log_likelihood + log_prior` before returning.
- `GPLogPosterior` mirrors `LogPosterior` exactly: the same
  `_classify_planet_case`/`_compute_logprob_corrections` methods, and
  `GPLogPosterior.log_probability` adds the same two stored corrections to
  `log_likelihood + log_prior + log_hyperprior` before returning. The
  corrections depend only on the parameterisation and priors, not the
  likelihood, so no GP-specific logic is needed.

## ecosw/esinw retirement

The `ecosw`/`esinw` parameterisation (`e cos w`, `e sin w`) has been disabled
(removed from `ALLOWED_PARAMETERISATIONS`). Its Jacobian is
`|d(e,w)/d(ecosw,esinw)| = 1/e`, which depends on the sample value and cannot
be precomputed as a constant correction - exactly why the field moved to
`secosw`/`sesinw` (whose Jacobian is the constant `2`) in the first place.
Re-enabling it would require a per-sample Jacobian evaluation, which is
outside the scope of this correction scheme.

## Empirical validation

The theory above was validated empirically (MCMC + `harmonic`/LHME evidence
estimation on the K2-24 system) for each case in isolation, and for the
mixed-case regression this fix addresses:

- Case 3 vs Case 1 (uniform prior on e): evidence agrees within MCMC noise
  once the `log(2)` correction is applied.
- Case 2 vs Case 1 (Uniform(-1,1) priors on secosw/sesinw): evidence agrees
  within MCMC noise once the `log(4/pi)` correction is applied.
- Case 3 with a HalfNormal prior on e: confirms the Jacobian correction is
  independent of the prior shape.
- **Mixed two-planet case** (planet b Case 2, planet c Case 3, same fit):
  per-planet classification and summation recovers the correct combined
  correction (`log(4/pi) + log(2)`), and the resulting evidence agrees with a
  reference fit sampled entirely in `(e, w)`.

## GPFitter

`GPFitter`/`GPLogPosterior` mirror `Fitter`/`LogPosterior` structurally, and
apply the same per-planet corrections (see `tests/test_logprob_corrections_gp.py`
for the mirrored unit-level coverage, including the mixed-case regression).
The correction values are unaffected by the likelihood function (GP or
otherwise), since they depend only on the parameterisation and priors.
