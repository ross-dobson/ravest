# Log-posterior corrections for the (secosw, sesinw) parameterisation

## You don't need to do anything!

Ravest applies these corrections **automatically**. They are computed once when the fit is set up (from your parameterisation, priors, and which parameters are free) and added inside the log-posterior during sampling. There is nothing that needs to be set manually.

They also have **no effect on MCMC parameter inference**. The corrections are constants, so they cancel in the MCMC sampler's acceptance ratio - Ravest uses `emcee`'s affine-invariant stretch move, in which any constant factor in the prior cancels identically, just as it does in the classic Metropolis-Hastings ratio. If you only care about fitting RVs to get parameter estimates and their uncertainties, and not performing Bayesian model comparisons, then you can ignore this page entirely.

The corrections matter in exactly one situation: **Bayesian model comparison**, where you use the posterior chains to estimate an evidence $\ln\mathcal{Z}$ (for example with the Learned Harmonic Mean Estimator, LHME, via the `harmonic` package - see [Example 4: using the Learned Harmonic Mean Estimator](https://ravest.readthedocs.io/en/latest/Examples/example_4_harmonic.html)). The evidence $\mathcal{Z}$ is the integral of the likelihood with respect to the prior, so the prior must be correctly normalised in the space you actually sample in - otherwise, the evidence for the model with that improperly normalised prior is systematically biased, therefore comparisons against other models are unfair. Luckily, Ravest handles this for you behind the scenes.

For the full explanation and derivation, see Appendix A of [Dobson et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026MNRAS.551g1343D/abstract); this page is a practical summary.

## The one choice that is yours

When you sample in the `secosw sesinw` parameterisation, rather than defining priors on $\sqrt{e}\cos\omega_\star$ and $\sqrt{e}\sin\omega_\star$, you should instead define your eccentricity belief as a prior on $e$ (Case 3 below), using one of Ravest's eccentricity priors (`HalfNormal`, `Rayleigh`, `VanEylen19Mixture`, `Beta`, `EccentricityUniform`, `TruncatedNormal`). The only prior on $\sqrt{e}\cos\omega_\star$ and $\sqrt{e}\sin\omega_\star$ directly that Ravest can normalise is `Uniform(-1, 1)` on both (Case 2). Any other prior on $\sqrt{e}\cos\omega_\star$ and $\sqrt{e}\sin\omega_\star$ raises `NotImplementedError`, because its renormalisation over the unit disc has no closed form in general. A separable, rotationally-symmetric prior on $\sqrt{e}\cos\omega_\star$ and $\sqrt{e}\sin\omega_\star$ can (probably) always be re-expressed as a prior on $e$, yet you can still sample in `secosw sesinw` to speed up MCMC convergence and mitigate the Lucy--Sweeney bias.

(Also, you really should try to avoid using uniform priors on $e$ anyway - see [Dobson et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026MNRAS.551g1343D/abstract) and references therein as to why the Uniform prior can lead to spurious high-$e$ solutions, especially with sparse or poorly-sampled RV observations.)

## Background

We write the transformed parameterisation as

$$
u \equiv \texttt{secosw} = \sqrt{e}\cos\omega_\star, \qquad v \equiv \texttt{sesinw} = \sqrt{e}\sin\omega_\star,
$$

adopting the $(u, v)$ notation here to cut down on typing. The inverse transform is then $e = u^2 + v^2$, $\omega_\star = \operatorname{atan2}(v, u)$, and the physical constraint $0 \le e < 1$ forms the unit disc $u^2 + v^2 < 1$, with area $A_\text{disc}=\pi$.

## The three cases for sampling eccentricity as a free parameter

| Case | Sampling | Eccentricity prior | Correction (per planet) |
|------|----------|--------------------|-------------------------|
| 1 | $(e, \omega_\star)$ | any | $0$ |
| 2 | $(u, v)$, priors on $(u, v)$ | `Uniform(-1, 1)` on both | $+\ln(4/\pi) \approx +0.242$ |
| 3 | $(u, v)$, priors on $(e, \omega_\star)$ | any properly normalised prior on $e$ | $+\ln 2 \approx +0.693$ |

**Case 1** needs no correction: sampling and prior space coincide, and the prior on $e$ is already normalised over $[0, 1)$. (Well, provided that you are using a proper prior...)

**Case 2** arises because `Uniform(-1, 1)` on each of $u$ and $v$ has joint density $1/4$ over the square $[-1, 1]^2$ (area $A_\text{square}=4$), but the physical validity check $u^2 + v^2 < 1$ truncates the support to the unit disc (area $A_\text{disc}=\pi$). This mismatch between the support of the prior and the physically-allowed region means that the prior then integrates to $\pi/4$ rather than $1$. If this isn't corrected, any models with a Case 2 planet will have their log evidence $\ln\mathcal{Z}$ biased! So, we need to correct this by adding the 'missing' $\ln(4/\pi)$ to the log-posterior, to ensure all priors are normalised properly and that we don't bias any model comparisons.

**Case 3** arises because the sampler proposes $(u, v)$ but the prior is defined on $(e, \omega_\star)$, so evaluating it involves a change of variables. The Jacobian determinant of the $(u, v) \to (e, \omega_\star)$ mapping is the constant $2$, which again needs to be included to prevent biasing the evidence, giving a correction to the log-probability of $+\ln 2$. Being purely geometric and based on the parameterisation, it is independent of the prior shape - the same $+\ln 2$ applies whether the prior on $e$ is `Uniform`, `HalfNormal`, `Rayleigh`, `VanEylen19Mixture`, `Beta`, or anything else - provided that the prior is properly normalised and truncated to $[0, 1)$.

## Multi-planet systems

Because classification is per-planet, the total correction to the log-prior (and therefore log-posterior) is the sum of the per-planet contributions. For a two-planet fit where planet b has `Uniform(-1, 1)` priors on `secosw_b` and `sesinw_b` (Case 2), while planet c has a normalised `HalfNormal` prior on `e_c` with a `Uniform(-pi, pi)` prior on `w_c` (Case 3):

$$
\text{total correction} = \ln\frac{4}{\pi} + \ln 2 \approx 0.242 + 0.693 = 0.935
$$

Classifying the system as a whole rather than per planet would apply the wrong constant (or the right one to the wrong number of planets), so Ravest always classifies each planet independently and sums the required correction for the whole star system.

## The ecosw/esinw parameterisation is disabled

The `ecosw esinw` parameterisation ($e\cos\omega_\star$, $e\sin\omega_\star$) is no longer available (removed from `ALLOWED_PARAMETERISATIONS`). Its Jacobian is $1/e$, which depends on the sampled value of $e$, and so cannot be precomputed as a constant correction - it would need a per-sample evaluation during sampling, outside the scope of this scheme. Furthermore, as well as making model comparison tricky, that Jacobian factor also induces a non-constant linear prior on $e$ - this is one of the reasons most people sample in $\sqrt{e}\cos\omega_\star$ and $\sqrt{e}\sin\omega_\star$, whose Jacobian is the constant $2$ (and avoids inducing an accidental prior on $e$).

I really can't think of a reason you would ever want to sample in `ecosw esinw` rather than `secosw sesinw` -- if you did want the behaviour of that induced prior to incentivise lower values of $e$, just fit in `secosw sesinw` and use something like a `HalfNormal` or `VanEylen19Mixture` prior on $e$ instead.

## Implementation and validation

The corrections live in `LogPosterior` (and are mirrored in `GPLogPosterior`) in `src/ravest/fit.py`: `_compute_logprob_corrections` classifies each planet and sums the contributions at construction time, and `log_probability` adds the stored total. The Jacobian value comes from `Parameterisation.log_jacobian_determinant` in `src/ravest/param.py`. The corrections depend only on the parameterisation and priors, not the likelihood, so `GPFitter` uses the same values.

The analytical corrections were empirically validated against `harmonic`/LHME evidence estimates on single-planet and multiple-planet systems for each case in isolation and for the mixed two-planet case.
