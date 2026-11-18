# xarray, xclim, records, datasets and Bayes

`import tails.xclim` registers a tails backend on xclim. The call stays on the xclim side. Import it before importing `frequency_analysis`.

```python
import tails.xclim
from xclim.compute.stats import frequency_analysis, fit  # xclim 1.0 and later
# from xclim.indices.stats import frequency_analysis, fit  # xclim 0.x

frequency_analysis(da, "max", [20, 100], "genextreme", backend="tails", ci="profile")
fit(da, dist="genextreme", method="ML", backend="tails")
```

::: tails.xarray
::: tails.xclim
::: tails.records
::: tails.datasets
::: tails.bayes
