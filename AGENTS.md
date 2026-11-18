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
| Plots | `fit.plot_diagnostics()`, `fit.return_level(...).plot()`, `IDFFit.plot()` (`rare-tails[plots]`) |
| xarray | `import tails.xarray as tx; tx.fit_gev(da, dim="time")` |
| Bayesian GEV | `from tails.bayes import fit_gev_bayes` (extra `rare-tails[bayes]`) |
| Record indicators | `tails.records.record_indicators`, `record_test` |

Submodules `idf`, `xarray`, `bayes`, `datasets` are imported lazily. `import tails`
is enough for `fit_gev` / `fit_gpd` / `fit_pot` / `how_rare`.

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

**IDF, same station:** `scaling_test(data, "simple")` ≈ 0.0079 (reject);
`scaling_test(data, "general")` ≈ 0.78 (ok). `fit_idf(data, "general", "matern")`
then `quantile_ci([1.0], [100])` gives a 100-year 1-hour intensity of 43.6 mm/h
(Wald 95% interval 39.5–47.8).

## Pitfalls

- **Package name ≠ import name.** `pip install rare-tails`, then `import tails`.
- **GEV shape sign.** tails uses the usual ξ (Coles / extRemes / Extremes.jl).
  `scipy.stats.genextreme` uses `c = -ξ`. Do not copy scipy shape values in
  blindly.
- **First call is slow.** JAX compiles the likelihood and Hessian on first
  use (seconds). Later calls, including large batches, are fast.
- **Batch axis.** The last axis is the record. Everything before it is a
  batch. Pad short series with `np.nan`; do not drop them from a stacked
  array by shortening the time axis.
- **Short records.** A batch will happily fit 11-year series. Cap Tourmente
  (`7041177`) is an example: ξ ≈ 1.24 and a nominal 100-year 24-hour level
  that you must not quote. Filter on `nobs` and ξ.
- **POT intervals.** `POTFit.return_level` currently supports `ci=None` or
  `"delta"` only, not profile or bootstrap.
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
`v*` and published with Trusted Publishing (`.github/workflows/release.yml`).
