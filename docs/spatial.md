# Joint risk on a labelled grid

`tails.xarray.how_rare_joint` applies the same joint-event model to arbitrary
labelled locations, rectangular grids or user-selected regions. It estimates
risk for the measured cells together. It does not fit a spatial covariance
model, interpolate between cells or predict at new locations.

For the [real precipitation application](applications.md), the three station
identifiers can be the spatial coordinate:

```python
import xarray as xr
import tails.xarray as tx
from application_data import rainfall  # examples/ on the Python path

frame = rainfall(cache)
rain = xr.DataArray(frame.to_numpy().T, dims=("station", "time"),
                    coords={"station": frame.columns.to_numpy(), "time": frame.index})
limits = rain.quantile(.99, dim="time")
risk = tx.how_rare_joint(rain, limits, spatial_dims=["station"],
                         event="kofn", k=2, npy=None, ci=None)
```

For a gridded field, pass `spatial_dims=["lat", "lon"]`. Different thresholds
per cell must have matching labels and coordinates; missing data stay missing.
Optional region masks are boolean DataArrays on those spatial dimensions:

```python
regions = {"first_two": rain.station.isin(["701S001", "7010565"])}
risk = tx.how_rare_joint(rain, limits, regions=regions, ci=None, npy=None)
```

Each region gets its own fitted score and exposure. Overlapping regions can
share observations, so their estimates are not independent. Thresholds, scales
and directions can be scalars or spatial DataArrays. The signed-limit rules
and event definitions are the same as the [array API](compound.md).

## Independent records and parallel computation

`replicate_dim="member"` pools independent records, keeping windows and blocks
within each member. Other dimensions not listed in `spatial_dims` index separate
analyses. Independent analyses may use Dask chunks; every joint spatial,
replicate and time core dimension must occupy one chunk. The wrapper rejects
split core chunks rather than silently changing which cells are fitted together.

Datetime and CFTime gaps are filled with missing values at an inferred regular
frequency; use `freq` for ambiguous cadences. Fixed-frequency calendars determine
observations per year unless `npy` is supplied. Daily no-leap and 360-day calendars
use 365 and 360, respectively. `npy=None` always reports periods in observations.

## Group labels and block intervals

```python
risk = tx.how_rare_joint(
    rain, limits, groups=season_labels, npy=None,
    ci="block_bootstrap", block_length=7, n_boot=200, seed=42,
)
```

`season_labels` must align exactly with the observed time coordinate. It can also
vary over independent batch or replicate dimensions. For each analysis, the
bootstrap shares indices across all selected cells. Grouped margins assume
stable residual dependence after probability transformation; they do not
automatically model seasonally changing spatial dependence. Grouped runs remain
unsupported, and bootstrap coverage still needs application-specific assessment.

The Dataset includes expected-count and return-period bounds plus
`bootstrap_attempted`, `bootstrap_successful`, `bootstrap_failed`,
`bootstrap_body` and `bootstrap_nonregular`. Any failed refit makes the bounds
unavailable. To inspect individual draws and detailed failure reasons, use the
underlying array API for the selected region. These are per-analysis intervals,
not simultaneous uncertainty bands across a map.

::: tails.xarray.how_rare_joint
