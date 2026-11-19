# Architecture: built for speed and scale

## Design principles

1. **Write the model for one series; get thousands for free.** Every likelihood is a pure JAX function
   of one parameter vector and one data vector. `jax.vmap` turns it into a batched function and
   `jax.jit` compiles it with XLA for CPU or GPU.
2. **Second-order optimisation with exact derivatives.** EVA models have 2–10 parameters. A damped
   Newton (Levenberg–Marquardt) method with exact Hessians converges in 3–6 iterations from L-moment
   starting values, where quasi-Newton or Nelder–Mead needs dozens to hundreds of evaluations.
3. **Masks, not loops.** Missing values, ragged record lengths, threshold exceedances and unobserved
   IDF durations are all boolean masks on rectangular arrays, so heterogeneous problems still batch.
4. **Uncertainty is part of the batch.** Profile likelihoods (many series × covariate rows × return
   periods × grid points), POT and GEV bootstrap replicates and candidate thresholds are just more rows.

## Two model families, options not models

Every fit in `tails.core` is one of two likelihoods:

| Family | Likelihood | Entry points | Options |
|---|---|---|---|
| GEV | r largest values per block; `r = 1` is block maxima | `fit_gev`, `fit_gev(..., r=k)`, `fit_rlargest` | covariates on `loc`, `scale`, `shape`; `fixed_shape`; `method="gmle"` |
| GPD | excesses over a threshold, plus an optional exceedance-rate term (Bernoulli per observation, logit link) | `fit_gpd` (no rate), `fit_pot` (rate on) | covariates on `scale`, `shape`, `rate`; `fixed_shape`; `kind="block"` return levels |

The modules stack like this:

- `_engine.py` registers each likelihood as a small `Likelihood` record: the
  per-observation log-density in linear-predictor space, closed-form
  derivatives where we have them (forward-mode autodiff otherwise), the
  return-level map and its inverse. A `Spec` adds the design sizes, the prior
  and which coefficients are free, so a fixed shape is "this coefficient is
  not optimised", not a separate model. Nothing else in the package is
  family-specific.
- `model.py` has one fit driver (`fit_model`: L-moment start, feasibility
  check, stationary fit, then the covariate fit) and one result class
  (`ModelFit`) with `params`, `summary`, `confint`, `params_at`,
  `return_level` and `plot_diagnostics`.
- `uncertainty.py` has one profile likelihood for return levels (the level
  replaces the location or scale intercept at each covariate row), one for
  coefficients, and one parametric bootstrap (r-largest blocks are simulated
  from the limiting Poisson process; POT replicates redraw the exceedances).
- `fit.py` and `rlargest.py` are the public front ends: they prepare data
  (excesses, top-r blocks) and call `fit_model`.

A new model is a new `Likelihood`; it gets batching, covariates, fixed
parameters, profile and bootstrap intervals, and return levels from the
code above. Group-specific parameters with a shared shape are an indicator
covariate on that parameter today.

## Why JAX (and not Numba)?

| Need | JAX | Numba |
|---|---|---|
| Exact gradients/Hessians of arbitrary likelihoods (copulas, Matérn, t quantiles) | `grad`, `hessian`, `custom_jvp` | hand-written derivatives only |
| Vectorise a one-series model over a batch | `vmap` | manual loops (`prange`) |
| GPU | yes (same code) | limited CUDA subset |
| Composability with NumPyro (Bayesian), optax, ML models (planned: neural EVA) | native | no |
| Raw single-thread CPU speed on scalar loops | good after fusion | excellent |

JAX wins on generality — the IDF copula models, profile likelihoods and future spatial/ML models
need autodiff — while for the hottest kernels (GEV/GPD) we recover most of Numba's speed with
**closed-form derivatives** evaluated in one fused pass (`tails.core._derivs`, checked against
autodiff in the tests). A Numba/C++ kernel remains planned for JAX-free environments.

## The optimiser (`tails.core.optim.minimize`)

- Levenberg–Marquardt-damped Newton: solve $(H + \lambda (1+\max|H_{ii}|) I)s = -g$; accept if the
  objective decreases and stays finite (infeasible parameters return $+\infty$), else increase λ.
- Small systems (p ≤ 12) are solved with an **unrolled Cholesky** made of elementwise operations:
  under `vmap` this is pure SIMD arithmetic instead of 10⁵ tiny LAPACK calls.
- **Active-set compaction**: all problems run 8 iterations, then only the unconverged ones (padded to
  powers of two to bound recompilation) continue — most fits finish in 4 steps, so the stragglers no
  longer dictate the cost of the whole batch.

## Numerics

- $\log(1+\xi z)/\xi$ and $(e^{\xi w}-1)/\xi$ are evaluated with Taylor series for $|\xi z| < 10^{-5}$
  using the double-`where` trick, so values *and gradients* are exact through ξ = 0.
- Second derivatives use stable series of $r'(x), r''(x)$ with $r(x)=\log(1+x)/x$ for $|x|<10^{-3}$.
- Covariates are centred and scaled internally; coefficients and covariances are mapped back.
- Starting values: batched NaN-aware L-moments (Hosking), clipped, with a Gumbel fall-back whenever a
  start would place data outside the support. Non-stationary fits start from the stationary MLE.
- GMLE (Martins–Stedinger Beta(6,9) prior on ξ) for short records.
- Student-t quantiles use bisection on the incomplete beta function with an exact implicit derivative;
  the Matérn correlation uses a spectrally accurate quadrature of $K_\nu$ (JAX has no real-order
  `besselk`), accurate to 1e-14 vs SciPy and differentiable in ν.

## Data model and API

- Arrays: `(*batch, n)` with NaN = missing; covariates `(n, k)` shared or `(*batch, n, k)` per series.
- Results are dataclasses (`GEVFit`, `GPDFit`, `POTFit`, all sharing `ModelFit`, and `IDFFit`) holding coefficients, covariance (observed
  information), log-likelihood, convergence flags, and methods (`return_level(..., ci=...)`,
  `quantile_ci`, `summary`, `to_pandas`).
- `tails.xarray.fit_gev(da, dim="time")` and
  `tails.xarray.fit_idf(da, year="year", duration="duration")` map labelled,
  Dask-chunked arrays through `xarray.apply_ufunc(dask="parallelized")`; each
  chunk is one batched fit. IDF quantiles are `tails.xarray.idf_quantile`.
- 64-bit floats are enabled in JAX on import (tail probabilities need them).
