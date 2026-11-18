# Quickstart

```bash
pip install "rare-tails[xarray]"
```

The import is `import tails`.

## How do I fit one series?

This uses the bundled ECCC 24-hour annual maxima at Québec City (Jean Lesage,
station `701S001`, 64 years).

```python
import numpy as np, tails
from tails.datasets import load_eccc_quebec

df = load_eccc_quebec()
y = df.loc[df.station_id == "701S001", "24h"].to_numpy()   # Québec City, 24-h annual max (mm)

fit = tails.fit_gev(y)
print(fit)                       # coefficients ± standard errors, log-likelihood
rl = fit.return_level([20, 100], ci="profile")
rl.estimate, rl.lower, rl.upper
```

`ci` can be `"delta"` (Wald), `"profile"` (profile likelihood) or
`"bootstrap"`. Profile is a good default for a 100-year level. On this series
the printed fit is `loc=51.66±1.6`, `log_scale=2.453±0.097`,
`shape=-0.02595±0.078`, and the 20- and 100-year 24-hour levels are 84.9 mm
and 102.1 mm (95% profile intervals 77.8–98.0 mm and 90.5–131.1 mm).

If those annual maxima are still a series with timestamps,
`tails.fit_block_maxima(series)` does the blocking and `summary()` is the
return-level table. `model="aic"` compares this GEV with a Gumbel model
(ξ fixed at 0). On this station the Gumbel model wins by AIC; the how-to
shows that table. `model="gev"` keeps the fit above.

## How do I fit thousands of series in one call?

The last axis is time; every axis before that is a batch. This stacks all 151
bundled Québec stations; missing years stay as NaN and are masked.

```python
wide = df.pivot_table(index="station_id", columns="year", values="24h")   # 151 stations × years, NaN gaps
fits = tails.fit_gev(wide.to_numpy())             # one batched optimisation
fits.params["shape"]                              # (151,)
fits.return_level([100], ci="delta").estimate     # (151, 1)
```

Missing years are NaN and are simply masked. All 151 of those fits converge.
Québec City’s 100-year level in the batch matches the single-station fit
(102.06 mm). Short records can still extrapolate badly: look at `nobs` and ξ
before you quote a number.

## How do I add a trend or a covariate?

tails centres and scales covariates internally and reports coefficients on
the original scale.

```python
years = np.arange(len(y)) + 1961.0
fit = tails.fit_gev(y, loc=years - 2000)          # mu(t) = b0 + b1 (t - 2000)
fit.param_names                                    # ['loc', 'loc:x0', 'log_scale', 'shape']
fit.return_level([100], covariates={"loc": np.array([[-30.0], [24.0]])}, ci="profile")
fit.plot_diagnostics()             # observations, fitted 2-year level, delta ribbon
```

`ci` is `"delta"`, `"profile"` or `"bootstrap"` here as well. The profile
replaces the location intercept with the return level at each covariate row.
`plot_diagnostics` draws the maxima against the covariate (`years - 2000`
here), the fitted 2-year level along those rows, and a delta-method ribbon.
Pass `ci="profile"` or `"bootstrap"` for that interval, and `period` for a
level other than the median. An unnamed array is labelled "Covariate"; a
pandas column name is used when you pass one. A temperature index, or a
covariate on the scale or the shape, is the same figure.

Covariates can be passed for `loc`, `scale` (log link) and `shape`, shared `(n, k)` or per series
`(*batch, n, k)`. Use `method="gmle"` for short records.

## How do I do peaks over a threshold?

```python
sel = tails.select_threshold(daily, np.quantile(daily, np.linspace(0.90, 0.99, 15)))
pot = tails.fit_pot(daily, sel.threshold, npy=365.25, decluster=2)
pot.return_level([10, 50], ci="profile")  # or "delta", "bootstrap"
```

`select_threshold` runs a batched ForwardStop procedure with Anderson–Darling
p-values. `plot_threshold(daily, unit="mm")` draws that threshold on the
mean-excess ribbon and the modified-scale and shape plots; `unit` is the
axis label, and the call needs `pip install "rare-tails[plots]"`. See the
[how-to](howto.md#how-do-i-choose-a-threshold). `POTFit.return_level` takes
the same `ci` values as a GEV fit.
`"profile"` is the joint profile of the excesses and the exceedance rate;
`"bootstrap"` resamples both and refits. An exponential model
(`fixed_shape=0`) keeps ξ fixed in the profile interval.

## How do I work from an xarray Dataset?

```python
import tails.xarray as txr
ds = txr.fit_gev(annual_max.chunk({"lat": 50}), dim="year")   # Dataset of loc/log_scale/shape (+ se)
rl100 = txr.return_level(ds, 100)

idf = txr.fit_idf(intensity.chunk({"lat": 50}), year="year", duration="duration")
q100 = txr.idf_quantile(idf, [1.0], [100])   # 100-year intensity at 1 hour
```

`intensity` is annual-maximum intensity (mm/h). The `duration` coordinate is
hours, a timedelta, or an ECCC tag (`1h`, `24h`, …). Pass `depth=True` when
the array stores depth in mm instead. Chunks must not split `year` or
`duration`. A worked example on three bundled Québec stations is in the
[Québec City tutorial](tutorial_quebec_idf.md#8-the-same-fit-on-an-xarray-grid).

## How do I fit an IDF curve?

```python
from tails.idf import IDFData, fit_idf
data = IDFData.from_eccc("701S001")                 # intensities (mm/h), 9 durations
fit = fit_idf(data, "general", "matern")            # d-GEV + Matérn Gaussian copula
est, lo, hi = fit.quantile_ci([1.0], [100], method="bootstrap", n_boot=500)
fit.plot()
```

A longer walk-through, including a comparison with ECCC’s published curve, is
the [Québec City tutorial](tutorial_quebec_idf.md).
