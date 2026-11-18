# Quickstart

```bash
pip install "rare-tails[xarray]"
```

The import is `import tails`. Until the first release is on PyPI, use
`pip install "rare-tails[xarray] @ git+https://github.com/rare-labs/tails"`.

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
fit.return_level([100], covariates={"loc": np.array([[-30.0], [24.0]])}, ci="delta")
```

Covariates can be passed for `loc`, `scale` (log link) and `shape`, shared `(n, k)` or per series
`(*batch, n, k)`. Use `method="gmle"` for short records.

## How do I do peaks over a threshold?

```python
sel = tails.select_threshold(daily, np.quantile(daily, np.linspace(0.90, 0.99, 15)))
pot = tails.fit_pot(daily, sel.threshold, npy=365.25, decluster=2)
pot.return_level([10, 50], ci="delta")
```

`select_threshold` runs a batched ForwardStop procedure with Anderson–Darling
p-values. `POTFit.return_level` currently supports `ci=None` or `"delta"`.

## How do I work from an xarray Dataset?

```python
import tails.xarray as txr
ds = txr.fit_gev(annual_max.chunk({"lat": 50}), dim="year")   # Dataset of loc/log_scale/shape (+ se)
rl100 = txr.return_level(ds, 100)
```

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
