# AGENTS.md

Guidance for coding agents working in this repository or writing code that
*uses* `tails`. The import is `import tails`. The PyPI distribution name is
`rare-tails`.

## What this package is

A JAX-backed extreme-value library: GEV, GPD, peaks-over-threshold, and IDF
curves. One series or a batch (`(*batch, n)` with NaNs for missing years) in
a single call. Optional xarray/Dask, optional NumPyro, bundled ECCC IDF data
for Québec.

Docs: https://rare-labs.github.io/tails/
Source: https://github.com/rare-labs/tails

## API map

| Want | Call |
|---|---|
| Fit GEV to block maxima | `tails.fit_gev(y)` or `tails.fit_gev(Y)` with `Y.shape == (n_series, n)` |
| Return levels + CI | `fit.return_level([20, 100], ci="profile")` — `"delta"`, `"profile"`, `"bootstrap"` |
| Non-stationary GEV | `tails.fit_gev(y, loc=covariate, scale=..., shape=...)` |
| GPD / POT | `tails.fit_gpd(excesses)` / `tails.fit_pot(series, threshold=..., npy=365.25)` |
| Threshold choice | `tails.mean_residual_life`, `tails.parameter_stability`, `tails.select_threshold` |
| Declustering | `tails.decluster_runs`, `tails.extremal_index`, `tails.block_maxima` |
| IDF | `from tails.idf import IDFData, fit_idf, scaling_test` |
| Bundled data | `from tails.datasets import load_eccc_quebec, load_idfcurves, parse_eccc_idf` |
| ECCC / GeoMet | `from tails.datasets import eccc_daily, eccc_hourly` — `eccc_daily("701S001", 2020)` |
| How rare is this? | `tails.how_rare(series, value_or_date, ci="profile", unit="mm")` |
| Datetime series → extremes and a return-level table | `tails.fit_block_maxima(series)` / `tails.fit_peaks(series, threshold=...)`, then `result.summary()`. `model="aic"` compares GEV with Gumbel (ξ = 0), or GPD with the exponential |
| Plots | `fit.plot_diagnostics()`, `fit.return_level(...).plot()`, `IDFFit.plot()` (`rare-tails[plots]`) |
| xarray | `import tails.xarray as tx; tx.fit_gev(da, dim="time")`. Gridded IDF: `tx.fit_idf(intensity, year="year", duration="duration")`, then `tx.idf_quantile(ds, [1.0], [100])` |
| xclim frequency analysis | `import tails.xclim`, then `frequency_analysis(da, "max", [20, 100], "genextreme", backend="tails", ci="profile")` from `xclim.compute.stats` (xclim ≥ 1) or `xclim.indices.stats` (xclim 0.x). The returned `c` is scipy's: `c = -ξ` for genextreme, `c = ξ` for genpareto |
| Bayesian GEV | `from tails.bayes import fit_gev_bayes` (extra `rare-tails[bayes]`) |
| Records | `tails.records.record_breaks(x)` and `foster_stuart(x)`. `record_indicators` and `record_test` are the same counts |

Submodules `idf`, `xarray`, `xclim`, `bayes`, `datasets` are imported lazily. `import tails`
is enough for `fit_gev` / `fit_gpd` / `fit_pot` / `how_rare` / `fit_block_maxima`
/ `fit_peaks`.

## Common tasks

**Return period of an observed value** after `fit = tails.fit_gev(y)`:

```python
from tails.core.distributions import gev_cdf
loc, scale, shape = (float(fit.params[k]) for k in ("loc", "scale", "shape"))
p = float(gev_cdf(observed, loc, scale, shape))
return_period = 1.0 / (1.0 - p)
```

`fit.params` holds *natural* parameters (`loc`, `scale`, `shape`). The
internal coefficients use `log_scale`. `fit.summary()` is a small DataFrame
of estimates and standard errors.

**Worked numbers** (bundled ECCC, station `701S001`, 24-hour annual maxima,
64 years, 1961–2024): `fit_gev` gives loc ≈ 51.66 mm, scale ≈ 11.62 mm,
ξ ≈ −0.026. The 2005 total of 105.4 mm has a return period of about 138
years (`tails.how_rare` reports a 1-in-138-year event, 95% profile CI 30–1212 years). The 100-year level is 102.1 mm (95% profile CI 90.5–131.1 mm).
`fit_block_maxima` on the same series with `model="aic"` selects the Gumbel
model (ξ fixed at 0; AIC 517.9 against 519.8). That 100-year level is 104.7 mm
(95% profile CI 94.9–117.6 mm). `model="gev"` keeps the three-parameter fit.

**IDF, same station:** `scaling_test(data, "simple")` ≈ 0.0079 (reject);
`scaling_test(data, "general")` ≈ 0.78 (ok). `fit_idf(data, "general", "matern")`
then `quantile_ci([1.0], [100])` gives a 100-year 1-hour intensity of 43.6 mm/h
(Wald 95% interval 39.5–47.8).

## Pitfalls

- **Package name ≠ import name.** `pip install rare-tails`, then `import tails`.
- **GEV shape sign.** tails uses the usual ξ (Coles / extRemes / Extremes.jl).
 `scipy.stats.genextreme` uses `c = -ξ`. Copying a scipy `c` into `fit_gev`'s
 shape uses the wrong sign. The xclim backend returns scipy's `(c, loc, scale)`
 so `parametric_quantile` still works: genextreme `c = -ξ`, genpareto `c = ξ`.
- **First call is slow, once.** JAX compiles the likelihood and Hessian on
  first use (seconds). tails keeps those compiled kernels in a persistent
  cache (`$TAILS_CACHE/jax`, default `~/.cache/tails/jax`), so a new Python
  session reuses them:, a first GEV fit with a profile
  interval drops from about 5 s to under 2 s. Set `TAILS_JAX_CACHE=0` to turn
  it off; an existing `JAX_COMPILATION_CACHE_DIR` is respected.
- **Batch axis.** The last axis is the record. Everything before it is a
  batch. Pad short series with `np.nan`; do not drop them from a stacked
  array by shortening the time axis.
- **One datetime series at a time.** `fit_block_maxima` and `fit_peaks` are the
  convenience path. A batch is still `block_maxima` or `decluster_runs`, then
  `fit_gev` / `fit_pot`. A shape standard error of 0 means ξ was fixed, not
  that it was estimated with no error. `model="aic"` can select that Gumbel
  (or exponential) model when the free shape is close to 0.
- **Short records.** A batch will happily fit 11-year series. Cap Tourmente
  (`7041177`) is an example: ξ ≈ 1.24 and a nominal 100-year 24-hour level
  that you must not quote. Filter on `nobs` and ξ.
- **Bootstrap with a fixed shape.** `fixed_shape` (Gumbel, exponential) stays
  fixed for delta and profile intervals, including POT return levels and
  non-stationary GEV profiles. Bootstrap refits a free ξ, so `ci="bootstrap"`
  is rejected; use `"delta"` or `"profile"`.
- **Plots.** `fit.plot_diagnostics()` (GEV/GPD/POT, stationary) and
  `IDFFit.plot()` (curves + CI ribbons + empirical return-period points)
  require `pip install 'rare-tails[plots]'`. There is no `GEVFit.plot()`;
  use `plot_diagnostics()` or `ReturnLevels.plot()`.
- **Covariates** are centred and scaled internally; reported coefficients
  are on the user’s scale. Shape covariates are usually a bad idea unless
  you have a long record and a good reason.
- **Float64.** `import tails` enables JAX float64. Do not silently mix
  float32 models with these fits.
- **Do not add author names** of the IDFCurves.jl project to public docs or
  comments. Licence text stays in `NOTICE` only.
- Do not add a public planning document or a benchmark suite to this tree.

## Tests and docs

```bash
pip install -e ".[test,docs]"
ruff check src tests
pytest -q
mkdocs build --strict
```

Python 3.10–3.13. CI is `.github/workflows/ci.yml`. Releases are tagged
`v*` (the tag is `v` plus `project.version`) and published with Trusted
Publishing (`.github/workflows/release.yml`). conda-forge is a feedstock
pull request opened after that sdist is on PyPI.
