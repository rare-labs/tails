# How-to

Short answers to three common questions. The numbers below were produced by
running the snippets; they are not typed in.

## How do I plot diagnostics and IDF curves?

Install the optional extra, then call `plot_diagnostics()` on a GEV, GPD or
POT fit, or `plot()` on an IDF fit. Both need matplotlib:
`pip install "rare-tails[plots]"`.

```python
import tails
from tails.datasets import load_eccc_quebec

df = load_eccc_quebec()
y = df.loc[df.station_id == "701S001", "24h"].to_numpy()  # 24-h annual max, mm
fit = tails.fit_gev(y)
fig, axes = fit.plot_diagnostics(ci="profile")
```

That draws a 2×2 figure: QQ and PP plots with a simulated 95% envelope, a
return-level curve with a profile-likelihood ribbon, and a histogram plus the
fitted density. For a batched fit, `index` selects the series (default 0).
`ReturnLevels.plot()` draws just the return-level panel.

![Diagnostic plots for Québec City 24-h annual maxima](img/diagnostics_701S001_24h.png)

IDF curves use the same style — one line and a confidence ribbon per return
period, on log–log axes in mm/h against duration in hours. The points are the
empirical T-year intensities at each duration (Weibull plotting positions),
not a cloud of raw annual maxima.

```python
from tails.idf import IDFData, fit_idf

data = IDFData.from_eccc("701S001")
idf = fit_idf(data, "general", "matern")
ax = idf.plot(periods=(2, 10, 25, 100), ci="hessian")
```

At Québec City the 100-year 1-hour intensity on this fit is 43.6 mm/h. The
empirical 100-year point at 1 h is 59.9 mm/h — the 1979 storm, which sits
above the pooled curve. `ci` can be `"hessian"`, `"sandwich"`, `"bootstrap"`,
or `None` for the curves alone.

## How rare was this storm?

`tails.how_rare(series, value_or_date)` fits a GEV (or a POT model) and
returns the return period, a confidence interval, and a plain-language
sentence. If `series` has a DatetimeIndex, you can pass a date instead of a
number. Pass `plot=True` to mark the event on a return-level chart.

```python
import pandas as pd
import tails
from tails.idf import IDFData

data = IDFData.from_eccc("701S001")
# 24-h column is stored as intensity (mm/h); convert back to depth (mm).
depth = pd.Series(
    data.values[:, -1] * 24.0,
    index=pd.to_datetime(data.years.astype(str), format="%Y"),
)
result = tails.how_rare(depth, "2005-01-01", ci="profile", unit="mm", name="Québec City")
print(result.sentence)
print(result.return_period, result.lower, result.upper)
```

```
A value of 105.4 mm in 2005 at Québec City is a 1-in-138-year event (95% CI: 30–1212 years).
138.276...  29.63...  1212.48...
```

The 2005 24-hour total of 105.4 mm is a little above the 100-year level
(102.1 mm, 95% profile interval 90.5–131.1 mm), so the return-period interval
is wide — a 64-year record cannot pin down a 138-year event tightly, and the
sentence says so. `ci="delta"` or `"bootstrap"` are also accepted. For
peaks-over-threshold, pass `model="pot"` and a `threshold`.

![Return-level chart with the 2005 event marked](img/how_rare_2005.png)

## How do I download ECCC station data?

`tails.datasets.eccc_daily` and `eccc_hourly` talk to the official MSC GeoMet
OGC API at [api.weather.gc.ca](https://api.weather.gc.ca). Pass a climate
identifier or a station name, and a year or an inclusive `(start, end)`
range. Downloads are cached under `$TAILS_CACHE` or `~/.cache/tails`.

```python
from tails.datasets import eccc_daily, eccc_stations

eccc_stations("701S001")
#   station_id                 station_name   lat        lon
#      701S001   QUEBEC/JEAN LESAGE INTL   46.80   -71.38

df = eccc_daily("701S001", 2020)
df[["tmax", "tmin", "tmean", "precip"]].head()
```

```
            tmax  tmin  tmean  precip
time
2020-01-01  -0.7  -4.9   -2.8     1.5
2020-01-02  -0.2  -2.4   -1.3     0.7
2020-01-03   2.4  -1.3    0.5     0.2
2020-01-04   1.4  -3.1   -0.8     1.0
2020-01-05  -3.1 -14.3   -8.7     0.8
```

2020 has 366 daily rows at this station and 1,296.7 mm of total
precipitation. `df.attrs` holds the station id, name, coordinates, and a
link to the API. `eccc_hourly("701S001", (2019, 2020))` is the same idea for
hourly observations (`temp`, `precip`, wind, pressure). If a name matches
several sites, the call raises with the candidate climate identifiers so you
can pick one.
