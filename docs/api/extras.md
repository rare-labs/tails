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
::: tails.records
::: tails.datasets
::: tails.bayes
