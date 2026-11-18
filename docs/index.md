# tails

**tails** is a Python library for extreme-value analysis: return levels, return
periods, and intensity–duration–frequency (IDF) curves, for one series or for
thousands at a time.

Install with `pip install rare-tails` (the import is `import tails`).

## What is this, and when should I use it?

Use tails when you already have block maxima, threshold exceedances, or IDF
tables and you want a fitted model with honest uncertainty — especially if you
have many stations or grid cells. A single datetime series can start one step
earlier: `fit_block_maxima` or `fit_peaks` extracts the sample and
`summary()` is the return-level table. If you mainly want a single-site
diagnostic figure in the style of
[pyextremes](https://georgebv.github.io/pyextremes/), that package is still
the one built around those plots. If you already work in [xclim](https://xclim.readthedocs.io/),
`import tails.xclim` and keep calling `frequency_analysis` with
`backend="tails"` and, when you want an interval, `ci="profile"`. On xclim
1.0 the import is `xclim.compute.stats`; on xclim 0.x it is
`xclim.indices.stats`. Genextreme's `c` in the parameter array equals −ξ.
Non-stationary models stay on `tails.fit_gev`. The xclim backend does not fit
IDF curves; use `tails.idf` or, on a grid, `tails.xarray.fit_idf`.

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

That 138-year figure is what `tails.how_rare` reports for this storm
(`unit="mm"`). The same one-liner takes any block-maxima series, and `unit`
is the label in the sentence. The [how-to](howto.md#how-rare-was-this) runs
it on four public series outside rainfall, with the source and the units in
the text: a 39-basis-point rise in the US 10-year yield (4 April 1994, a
1-in-74-year event, 95% profile interval 13–758 years), a Henry Hub price of
23.86 USD per million Btu (17 February 2021, 1-in-24 years, 7.1–70), a week
of 6.137 million US initial unemployment claims (week ending 4 April 2020,
1-in-6654 years, 4251–50099), and the 1960 Valdivia earthquake at Mw 9.5
(1-in-123 years, 29–1004).

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
a short series. The [how-to](howto.md#how-do-i-map-a-batch-of-return-levels)
maps that batch at each station's latitude and longitude.

## Which distribution should I use?

- **GEV** (`tails.fit_gev`) for block maxima: annual (or other block) maxima
  of rainfall, discharge, temperature, demand, and so on.
- **GPD / POT** (`tails.fit_gpd`, `tails.fit_pot`) when you have many
  exceedances above a threshold rather than one maximum per year. Use the
  Quickstart’s peaks-over-threshold example. `plot_threshold(series, unit="mm")`
  draws the mean-excess ribbon, parameter stability and the ForwardStop
  threshold (`rare-tails[plots]`).
- **IDF** (`tails.idf.fit_idf`) when you have annual maxima at several
  durations. Start with the general (d-GEV) scaling model; run `scaling_test`
  before you accept simple scaling. At Québec City the simple-scaling test
  p-value is 0.0079 (rejected) and the general-scaling p-value is 0.78
  (compatible). On a DataArray, `tails.xarray.fit_idf` fits every other
  coordinate (station, lat/lon, …) in one batched call.

tails does not try to replace a general-purpose distribution library. If you
need Wakeby, Kappa or Pearson III, look at scipy or lmoments3.

!!! note "Credit"
    `tails.idf` is a port of [IDFCurves.jl](https://github.com/JuliaExtremes/IDFCurves.jl).
    See [Credits](credits.md).
