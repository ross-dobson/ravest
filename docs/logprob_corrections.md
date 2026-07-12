# Log-posterior corrections for the (secosw, sesinw) parameterisation

## You don't need to do anything

Ravest applies these corrections **automatically**. They are computed once when
the fit is set up (from your parameterisation, priors, and which parameters are
free) and added inside the log-posterior during sampling. There is no knob to
set and nothing to remember.

They also have **no effect on parameter inference**. The corrections are
constants, so they cancel in the sampler's acceptance ratio - Ravest uses
emcee's affine-invariant stretch move, in which any constant factor in the prior
cancels identically, just as it does in a Metropolis-Hastings ratio. If you only
care about parameter estimates and their uncertainties, you can ignore this page
entirely.

The corrections matter in exactly one situation: **Bayesian model comparison**,
where you use the posterior chains to estimate an evidence `ln(Z)` (for example
with the Learned Harmonic Mean Estimator, LHME, via the `harmonic` library). The
evidence is an integral of the likelihood against the prior, so the prior must be
correctly normalised in the space you actually sample in - otherwise the evidence
for that model is systematically biased and comparisons against other models are
unfair. Ravest handles this for you.

For the full derivation, see Appendix B of Dobson et al. (in prep.); this page is
a practical summary.

## The one choice that is yours

When you sample in the `(secosw, sesinw)` parameterisation, define your
eccentricity belief as a prior on **`e`** (Case 3 below), using one of Ravest's
eccentricity priors (`HalfNormal`, `Rayleigh`, `VanEylen19Mixture`, `Beta`,
`EccentricityUniform`, `TruncatedNormal`). The only prior on `(secosw, sesinw)`
directly that Ravest can normalise is `Uniform(-1, 1)` on both (Case 2); any
other prior on `(secosw, sesinw)` raises `NotImplementedError`, because its
renormalisation over the unit disc has no closed form in general. A separable,
rotationally-symmetric belief on `(secosw, sesinw)` can (probably) always be
re-expressed as a prior on `e` , yet you can still sample in `(secosw,sesinw)`
to speed up MCMC converge and avoid the Lucy--Sweeney bias.

## Background

In the transformed parameterisation

```
u = "secosw" = sqrt(e) * cos(w)
v = "sesinw" = sqrt(e) * sin(w)
```

the inverse transform is `e = u^2 + v^2`, `w = atan2(v, u)`, and the physical
constraint that `0 <= e < 1` is exactly the unit disc formed by `u^2 + v^2 < 1`.

Ravest classifies **each planet independently** and sums the per-planet
corrections. This matters for multi-planet systems, where different planets can
have priors in different coordinate systems (e.g. planet b on `(u, v)`, planet c
on `(e, w)`); each planet's correction depends only on its own choice.

## The three cases for sampling eccentricity as a free parameter

| Case | Sampling | Eccentricity prior | Correction (per planet) |
|------|----------|--------------------|-------------------------|
| 1 | `(e, w)` | any | `0` |
| 2 | `(u, v)`, priors on `(u, v)` | `Uniform(-1, 1)` on both | `+log(4/pi) ~= +0.242` |
| 3 | `(u, v)`, priors on `(e, w)` | any properly normalised prior on `e` | `+log(2) ~= +0.693` |

**Case 1** needs no correction: sampling and prior space coincide, and the prior
on `e` is already normalised over `[0, 1)` (provided you are using a proper prior...)

**Case 2** arises because `Uniform(-1, 1)` on each of `u` and `v` has joint
density `1/4` over the square `[-1, 1]^2` (area 4), but the physical validity
check `u^2 + v^2 < 1` truncates the support to the unit disc (area `pi`).
The prior then integrates to `pi/4` rather than `1`, so renormalising adds the 'missing'
`log(4/pi)` to the log-posterior to ensure all priors are normalised properly.

**Case 3** arises because the sampler proposes `(u, v)` but the prior is defined
on `(e, w)`, so evaluating it is a change of variables. The Jacobian determinant
of the `(u, v)` -> `(e, w)` mapping is the constant `2`, giving `+log(2)`. Being
purely geometric, it is independent of the prior shape - the same `+log(2)`
applies whether the prior on `e` is `Uniform`, `HalfNormal`, `Rayleigh`,
`VanEylen19Mixture`, `Beta`, or anything else, provided it is properly normalised
and truncated to `[0, 1)`.

## Multi-planet systems

Because classification is per-planet, the total correction to the log-prior
(and therefore log-posterior) is the sum of the per-planet contributions.
For a two-planet fit where planet b has `Uniform(-1, 1)` priors on
`(secosw_b, sesinw_b)` (Case 2) and planet c has a `HalfNormal` on `e_c` with
`Uniform(-pi, pi)` on `w_c` (Case 3):

```
total correction = log(4/pi) + log(2) ~= 0.242 + 0.693 = 0.935
```

Classifying the system as a whole rather than per planet would apply the wrong
constant (or the right one to the wrong number of planets), so Ravest always
classifies each planet independently and sums.

## The ecosw/esinw parameterisation is disabled

The `ecosw`/`esinw` parameterisation (`e cos w`, `e sin w`) is not available
(removed from `ALLOWED_PARAMETERISATIONS`). Its Jacobian is `1/e`, which depends
on the sample value and so cannot be precomputed as a constant correction - it
would need a per-sample evaluation during sampling, outside the scope of this
scheme. This is precisely why most people sample in `secosw`/`sesinw`, whose
Jacobian is the constant `2` (and avoids inducing an accidental prior on `e`.)

I really can't think of a reason you would want to sample in `ecosw/esinw` rather
than `secosw/sesinw` -- if you want a prior to incentivise lower values of `e`,
just fit in `secosw/sesinw` and use something like a `HalfNormal` prior on `e`
instead.

## Implementation and validation

The corrections live in `LogPosterior` (and mirrored in `GPLogPosterior`) in
`src/ravest/fit.py`: `_compute_logprob_corrections` classifies each planet and
sums the contributions at construction time, and `log_probability` adds the
stored total. The Jacobian value comes from
`Parameterisation.log_jacobian_determinant` in `src/ravest/param.py`. The
corrections depend only on the parameterisation and priors, not the likelihood,
so `GPFitter` uses the same values.

The corrections were validated against `harmonic`/LHME evidence estimates on the
K2-24 system for each case in isolation and for the mixed two-planet case; details
to be released as part of a paper featuring Ravest soon.
