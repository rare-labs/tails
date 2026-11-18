# The extreme-value software landscape (late 2026)

## How does tails compare with other extreme-value packages?


This page surveys the packages `tails` is measured against, what they do well, and the gaps
`tails` is designed to fill. Coverage claims are based on each package's public documentation
and source. Timings mentioned here were measured on one 8-vCPU VM.

## Packages reviewed

### R

| Package | Scope | API style | Strengths | Gaps relevant to tails |
|---|---|---|---|---|
| **extRemes** (Gilleland & Katz) | GEV, GPD, PP, Gumbel, exponential; non-stationary via formulas; MLE, GMLE, Bayesian, L-moments; return levels with normal / profile / bootstrap CIs | `fevd(x, data, location.fun = ~t, type = "GEV")` | The reference for applied climate work; formula interface; good diagnostics | One series per call (loops over stations); profile CIs computed point by point; no gridded-data interface; single-threaded R optimiser (`optim`) |
| **ismev** (Coles 2001 companion) | `gev.fit`, `gpd.fit`, `pp.fit`, `rlarg.fit`; covariates via design matrices | functional | Canonical textbook implementation | Unmaintained API, no CIs for return levels beyond diagnostics plots, one series at a time |
| **evd** (Stephenson) | Univariate and *bivariate/multivariate* extreme distributions, `fgev`, `fpot`, `fbvevd`, profile likelihood | functional | Broad distribution coverage incl. bivariate logistic/asymmetric models, extremal index | No batching, older numerical style |
| **mev** (Belzile et al.) | Modern methods: threshold selection (Wadsworth, Northrop–Coleman, ...), penalised/bias-corrected MLE, r-largest, max-stable simulation, extremal coefficient estimation, PP | functional | Most complete collection of *recent* univariate/multivariate EVA methodology; careful numerics near ξ = 0 | Research-oriented; loops in R; no array/gridded interface |
| **texmex** (Southworth, Heffernan) | Conditional multivariate extremes (Heffernan–Tawn), GPD regression, Bayesian | formula | Only mature Heffernan–Tawn implementation | Narrow focus, slow MCMC |
| **POT** (Ribatet) | Peaks-over-threshold, threshold diagnostics, bivariate POT | functional | Good POT diagnostics | Largely unmaintained |

### Python

| Package | Scope | Strengths | Gaps |
|---|---|---|---|
| **scipy.stats.genextreme / genpareto** | Distributions + generic `fit` (Nelder–Mead / `fmin`) | Ubiquitous | Sign convention `c = -ξ`; generic optimiser is slow (~10–50 ms per fit), occasionally fails for ξ < −0.5 or short records; no covariates, no CIs |
| **pyextremes** (Bocharov) | Extraction (block maxima, POT with declustering), MLE (via scipy) and Bayesian (emcee) fits, return-value plots, threshold diagnostics | Friendly end-to-end workflow for *one* time series | One series at a time; stationary only; bootstrap CIs are slow (refit loops) |
| **xclim / xclim.indices.stats** (Ouranos) | `fit` of scipy distributions along a dimension of xarray objects, frequency analysis indices | Labelled, Dask-friendly, climate-services standard in Québec | Loops scipy fits per grid cell via `apply_ufunc(vectorize=True)`; stationary; no uncertainty for return levels beyond bootstrap via resampling utilities |
| **lmoments3** | L-moment estimation for many distributions | Fast, robust estimates | No likelihood, no covariates, no CIs |

### Julia

| Package | Scope | Strengths | Gaps |
|---|---|---|---|
| **Extremes.jl** | GEV, GPD, Gumbel; MLE, PWM, Bayesian; non-stationary covariates; return levels with CIs; diagnostics | Fast compiled code, coherent API | One series per call; Julia adoption in Canadian engineering/climate services is low |
| **IDFCurves.jl** | Simple and general (d-GEV) scaling IDF models, Gaussian/Student-t copulas across durations with exponential/Matérn correlation, scaling-model test, Wald CIs | Only open implementation of the copula-dependence IDF model and the scaling-model test | One station per call; complete records assumed; Student-t copula Hessian fails in v0.2.0 (fixed in our audit patch); Wald CIs only |

## Comparison

| Capability | extRemes | ismev | evd | mev | pyextremes | scipy | xclim | Extremes.jl | IDFCurves.jl | **tails** |
|---|---|---|---|---|---|---|---|---|---|---|
| GEV / GPD MLE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (scipy) | ✓ | (GEV) | ✓ |
| Non-stationary covariates | ✓ | ✓ | partial | ✓ | – | – | – | ✓ | – | ✓ (all params, per-series designs) |
| GMLE (shape prior) | ✓ | – | – | ✓ (penalised) | – | – | – | – | – | ✓ |
| Profile-likelihood return-level CI | ✓ | – | ✓ | ✓ | – | – | – | – | – | ✓ (batched) |
| Parametric bootstrap | ✓ | – | – | ✓ | ✓ | – | resampling | – | – | ✓ (all replicates in one batched fit) |
| Automated threshold selection | – | – | – | ✓ (several) | – | – | – | – | – | ✓ (ForwardStop + batched bootstrap AD) |
| Declustering / extremal index | ✓ | – | ✓ | ✓ | ✓ (runs) | – | – | – | – | ✓ |
| IDF scaling models + copulas | – | – | – | – | – | – | – | – | ✓ | ✓ (validated to ~1e-9) |
| Exact handling of missing values | partial | – | – | – | NaN drop | – | NaN skip | – | – | ✓ (masks everywhere, copula marginalisation) |
| **Fits many series in one call** | – | – | – | – | – | – | loop | – | – | ✓ (vmap) |
| xarray / Dask | – | – | – | – | – | – | ✓ | – | – | ✓ |
| GPU | – | – | – | – | – | – | – | – | – | ✓ (via JAX, untested in CI) |
| Autodiff (exact gradients/Hessians) | – | – | – | – | – | – | – | ForwardDiff | ForwardDiff | ✓ (JAX + closed forms) |
| Record statistics | – | – | – | – | – | – | – | – | – | ✓ |
| Bayesian | ✓ (MH) | – | – | – | ✓ (emcee) | – | – | ✓ | – | ✓ optional (NumPyro NUTS) |
| Multivariate / spatial extremes | – | – | ✓ (bivariate) | ✓ | – | – | – | – | – | planned |

## Where existing tools hurt in practice

1. **Scale.** National frequency analyses (e.g. ~150 Québec IDF stations × 9 durations, or 10⁵–10⁶ grid
   cells of a climate-model ensemble) are run as Python/R loops over one-series fitters. Each fit costs
   10–50 ms in scipy/extRemes, so a 100 000-cell map takes 20–80 minutes, *before* any bootstrap.
2. **Uncertainty is the expensive part.** Profile likelihood and bootstrap multiply the cost by 50–1000,
   which is why they are often skipped and Wald intervals reported even when they are known to
   under-cover for 100-year levels.
3. **Numerics near ξ = 0.** Generic optimisers stall or return ξ ≈ −1 solutions for short records;
   naive `log(1+ξz)/ξ` code loses digits near ξ = 0.
4. **Missing data and ragged records** are handled by dropping years (losing information) or ignored.
5. **IDF dependence models** exist only in Julia, one station at a time.

## tails' differentiators

- **Batched by design.** Every likelihood is written for one series and vectorised with `jax.vmap`;
  a damped Newton optimiser with exact Hessians advances thousands of problems in lock step, and
  compacts the active set so the slow few do not hold everyone back. Fitting 100 000 GEVs takes a few
  seconds on a laptop CPU.
- **Uncertainty is cheap**: profile likelihoods for many series and return periods are traced in one
  compiled scan; bootstrap replicates are just more rows in the batch; threshold selection with
  bootstrap p-values (the slow step of Bader et al. 2018) becomes interactive.
- **Stable numerics**: Taylor-series evaluation of the removable singularities at ξ = 0 with safe
  gradients, closed-form GEV/GPD derivatives (verified against autodiff), centred/scaled covariates,
  L-moment starting values with feasibility fall-backs, Martins–Stedinger GMLE.
- **One engine, many subjects**: the same masked, batched machinery serves block maxima, POT with
  time-varying thresholds, IDF copula models, and record statistics; it is the base for planned multivariate, spatial, attribution and ML-assisted work.
- **Labelled data**: an xarray/Dask front-end so a gridded dataset is one call, chunk-parallel.
  `import tails.xclim` adds `backend="tails"` to xclim's `frequency_analysis`
  (`xclim.compute.stats` on xclim 1.0, `xclim.indices.stats` on xclim 0.x),
  including a profile-likelihood interval.
- **Python, where Canadian hydrology and climate services work** (xclim, Ouranos, MELCCFP, ECCC), while
  crediting and numerically matching the Julia reference implementation of IDF models.
