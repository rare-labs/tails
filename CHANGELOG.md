# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `import tails.xclim` registers an optional backend on xclim frequency
  analysis (`xclim.compute.stats` in xclim 1.0 and later, `xclim.indices.stats`
  in xclim 0.x). `frequency_analysis(..., backend="tails", ci="profile")` and
  `fit(..., backend="tails")` then use tails' batched GEV and GPD fits.
  The parameter array stays in scipy order (`c = -ξ` for genextreme,
  `c = ξ` for genpareto), so `parametric_quantile` still applies. A GEV
  interval (`ci="delta"`, `"profile"` or `"bootstrap"`) comes back as a
  Dataset of `return_level`, `lower` and `upper`. Stationary models; methods
  ML, PWM and GMLE. Method of moments, maximum spacing and a constrained
  Gumbel are not on this backend. A GPD level is a quantile of the fitted
  sample (optional scalar `threshold`), not a peaks-over-threshold annual level.
- `tails.fit_block_maxima` and `tails.fit_peaks` turn one datetime series into
  block maxima or declustered peaks, fit them, and return a return-level table
  from `summary()`. `model="aic"` compares the GEV with the Gumbel model
  (shape ξ fixed at 0), or the GPD with the exponential model, and keeps the
  smaller AIC. `fit_gev(..., fixed_shape=0)` and `fit_pot(..., fixed_shape=0)`
  are the underlying fits; profile intervals keep that shape fixed.
- Persistent JAX compilation cache, on by default, in `$TAILS_CACHE/jax` (default
  `~/.cache/tails/jax`). A new Python session reuses kernels compiled by the previous one, so a
  first GEV fit with a profile interval takes under 2 s instead of about 5 s.
  `TAILS_JAX_CACHE=0` turns it off; a cache directory already set for JAX is left alone.
  `tails._jax.compilation_cache_dir()` reports the directory in use.

## [0.1.0] — 2026-11-18

First public release. The import is `import tails`; the distribution name on PyPI is `rare-tails`
because `tails` was already taken.

### Added

- Batched GEV, GPD and POT maximum likelihood with exact (closed-form) Hessians; non-stationary
  linear predictors on all parameters; GMLE; L-moments.
- Delta, profile-likelihood and parametric-bootstrap return-level intervals, all batched.
- Threshold diagnostics and ForwardStop selection with batched bootstrap Anderson–Darling p-values;
  runs declustering; Ferro–Segers extremal index; record statistics.
- `tails.idf`: a port of IDFCurves.jl (simple and general scaling, Gaussian and Student-t copulas,
  exponential and Matérn correlation, scaling test, Godambe sandwich intervals), validated against
  Julia, batched across stations, with missing-data handling and bootstrap intervals.
- xarray / Dask front-end, optional NumPyro posterior, and the ECCC IDF v3.40 Québec dataset plus
  a parser for any ECCC IDF file.
- Built-in diagnostic plots (`fit.plot_diagnostics()`, `ReturnLevels.plot()`, `IDFFit.plot()` with
  CI ribbons) behind the optional `plots` extra.
- `tails.how_rare(series, value_or_date)` — return period, confidence interval, and a
  plain-language sentence, with an optional chart.
- `tails.datasets.eccc_daily` / `eccc_hourly` — MSC GeoMet OGC API downloaders with on-disk caching.

[0.1.0]: https://github.com/rare-labs/tails/releases/tag/v0.1.0
