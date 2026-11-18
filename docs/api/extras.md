# xarray, xclim, records, datasets and Bayes

`import tails.xclim` registers a tails backend on xclim. The call stays on the xclim side. Import it before importing `frequency_analysis`.

```python
import tails.xclim
from xclim.compute.stats import frequency_analysis, fit  # xclim 1.0 and later
# from xclim.indices.stats import frequency_analysis, fit  # xclim 0.x

frequency_analysis(da, "max", [20, 100], "genextreme", backend="tails", ci="profile")
fit(da, dist="genextreme", method="ML", backend="tails")
```

Gridded fits live on `tails.xarray`. `fit_gev` reduces a block-maxima dimension. `fit_idf` reduces the year and duration dimensions of a DataArray of intensities (mm/h, or depths in mm with `depth=True`).

```python
import tails.xarray as tx

ds = tx.fit_gev(annual_max, dim="year")
idf = tx.fit_idf(intensity, year="year", duration="duration")
tx.idf_quantile(idf, [1.0], [100])
```

::: tails.xarray
::: tails.xclim

`tails.records` is the record-breaking toolkit. `record_breaks` marks upper or lower records and compares the count with the harmonic number `H_n`. `foster_stuart` splits that into a drift (more highs than lows) and a change in spread (more records of both kinds). The last axis is the series. Missing values are skipped, and a tie is not a new record.

```python
from tails.records import foster_stuart, record_breaks

breaks = record_breaks(annual_max)          # or kind="lower"
breaks.summary()                            # one row per series
breaks.events()                             # one row per record
foster_stuart(annual_max).summary()
```

::: tails.records
::: tails.datasets
::: tails.bayes
