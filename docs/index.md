# tails

**tails** is a Python library for extreme-value analysis: return levels, return
periods, and intensity–duration–frequency (IDF) curves, for one series or for
thousands at a time.

Install with `pip install rare-tails` (the import is `import tails`). Until
the first release is on PyPI, use
`pip install "rare-tails @ git+https://github.com/rare-labs/tails"`.

## What is this, and when should I use it?

Use tails when you already have block maxima, threshold exceedances, or IDF
tables and you want a fitted model with honest uncertainty — especially if you
have many stations or grid cells. If you have a single pandas time series and
you mainly want diagnostic plots, [pyextremes](https://georgebv.github.io/pyextremes/)
may be a better first stop. If you already work in [xclim](https://xclim.readthedocs.io/),
`tails.xclim.frequency_analysis` follows the same block-extreme workflow on a
DataArray and can add a profile-likelihood interval. The shape it reports is
ξ. Non-stationary models and IDF curves stay on `tails.fit_gev` and `tails.idf`.

Start with the [Quickstart](quickstart.md) or the
[Québec City IDF tutorial](tutorial_quebec_idf.md).

## How rare was this storm?

The same Québec City 24-hour series used in the [Quickstart](quickstart.md)
has a wettest year of 105.4 mm, in 2005. After the GEV fit there, the return
period of that event is about **138 years**. The 100-year 24-hour level is
102.1 mm, with a 95% profile-likelihood interval of 90.5–131 mm, which still
covers 105.4 mm.

```python
from tails.core.distributions import gev_cdf

loc, scale, shape = (float(fit.params[k]) for k in ("loc", "scale", "shape"))
p = float(gev_cdf(105.4, loc, scale, shape))
1.0 / (1.0 - p)   # 138.3 years
```

The 1-hour annual maximum at the same station, 59.9 mm in 1979, has a
single-duration GEV return period of about 129 years. A joint IDF model
(d-GEV with a Matérn Gaussian copula) puts the 100-year 1-hour intensity at
43.6 mm/h, because it borrows strength from the other durations. See the
[tutorial](tutorial_quebec_idf.md) for that comparison.

## How do I fit 1,000 stations at once?

The same way as one series: pass an array whose last axis is time. The
[Quickstart](quickstart.md#how-do-i-fit-thousands-of-series-in-one-call)
stacks all 151 bundled Québec stations with a pivot table; every fit
converges, and Québec City’s 100-year level matches the single-station fit.
Look at record length and the shape parameter before you quote a number from
a short series.

## Which distribution should I use?

- **GEV** (`tails.fit_gev`) for block maxima: annual (or other block) maxima
  of rainfall, discharge, temperature, demand, and so on.
- **GPD / POT** (`tails.fit_gpd`, `tails.fit_pot`) when you have many
  exceedances above a threshold rather than one maximum per year. Use the
  Quickstart’s peaks-over-threshold example, plus `mean_residual_life` and
  `parameter_stability` if you need to choose the threshold.
- **IDF** (`tails.idf.fit_idf`) when you have annual maxima at several
  durations. Start with the general (d-GEV) scaling model; run `scaling_test`
  before you accept simple scaling. At Québec City the simple-scaling test
  p-value is 0.0079 (rejected) and the general-scaling p-value is 0.78
  (compatible).

tails does not try to replace a general-purpose distribution library. If you
need Wakeby, Kappa or Pearson III, look at scipy or lmoments3.

!!! note "Credit"
    `tails.idf` is a port of [IDFCurves.jl](https://github.com/JuliaExtremes/IDFCurves.jl).
    See [Credits](credits.md).
