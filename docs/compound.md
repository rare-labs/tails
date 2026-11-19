# Joint and conditional extremes

Use joint-event analysis to estimate how often several series exceed their
thresholds together. Use conditional extremes to ask how large another series
gets when a designated series is already extreme. Both accept aligned series
from any application; locations and the challenge's 5 by 5 grid are examples.

## Joint events and expected counts

```python
import numpy as np
import tails

x = np.random.default_rng(12).exponential(size=(3, 10000))  # site, daily observation
result = tails.how_rare_joint(x, [2, 2.5, 3], ci="profile")
result.return_period                    # years between events
result.expected_count                  # expected events over this record
result.count_lower, result.count_upper  # confidence interval for the expectation
result.observed_count                  # events actually observed

tails.how_rare_joint(x, 3, event="kofn", k=2)  # at least two sites above 3
tails.how_rare_joint(x, 10, event="sum")       # raw sum above 10
tails.how_rare_joint(x, 2, run_length=2)       # all sites, at least two consecutive days
```

Each site is divided by its positive threshold. The minimum of these ratios
exceeds 1 exactly when all sites exceed their thresholds. The k-th largest
ratio represents at least k exceedances. `event="sum"` adds the ratios; a
single common threshold U represents a raw sum greater than U. With unequal
thresholds it represents the sum of normalised values greater than 1.
`tails.compound_series` exposes this reduction without fitting.

Arrays use `(*replicate, site, time)`; all leading dimensions are pooled into
one fit. DataFrame columns are sites, aligned by their shared index. Missing
sites invalidate that day. Missing dates are inserted before rolling daily
windows, so gaps cannot become consecutive days. Arrays must already be on
a regular observation grid, with NaNs at missing times. Long runs count once
in `observed_count`, and no run crosses a replicate boundary.

The model fits a GPD above the **0.95 quantile of the combined score**, not
above the requested event threshold. Change it with `threshold_quantile`.
The event must lie above that fitting threshold. A joint profile likelihood
for the excesses and their rate is evaluated on a logarithmic return-period
grid. `ci=None` skips the intervals. Counts use the rate per original day,
corrected for the number of valid windows. The default record exposure is
the mean number of complete days per replicate. `record_length=60225` instead
asks for an expected count over a specified target exposure. Set `npy=365`
for the challenge's no-leap calendar, or the appropriate observations/year
for other regularly sampled data.

An interval bound is NaN when the likelihood search does not establish it.
If the event lies beyond a fitted finite endpoint, its estimated probability
is zero and its estimated return period is infinite; the finite lower period
bound and upper expected-count bound are unavailable. This does not prove
the event is impossible in the population.

For runs, rolling minima are declustered at the fitting threshold separately
within each replicate. As in ordinary POT declustering, the extrapolation
approximates target-level runs: a cluster above the fitting threshold may
contain more than one separate target-level run. Check this approximation
against simulated or observed events before interpreting a run estimate.

These intervals quantify uncertainty in an **expected count**, rather than
predict the count in a future record. They assume a stationary GPD tail,
independent excesses and a binomial exceedance rate. They do not include
threshold choice, residual temporal dependence or climate trends.

## Arbitrary grids and regions

```python
import tails.xarray as tx

# rain has dimensions member, time, lat, lon. `west` is a boolean spatial mask.
ds = tx.how_rare_joint(
    rain, 30, spatial_dims=["lat", "lon"], replicate_dim="member",
    regions={"west": west}, event="kofn", k=3, ci="profile",
)
ds.expected_count.sel(region="west")
```

Supply any spatial dimensions, including a single `station` dimension with
longitude/latitude coordinates. Thresholds can be a scalar or a spatial
DataArray whose labels match exactly. Region masks select the sites included
in each joint event. Dimensions outside the spatial, time and replicate
dimensions index separate analyses. Dask can parallelise those independent
analyses; the spatial, replicate and time dimensions must each be a single
chunk. The Dataset includes event rates, return periods and count intervals.
Missing dates are inserted for pandas and CFTime calendars. Unless `npy` is
supplied, no-leap and 360-day calendars use 365 and 360 observations/year.

## Conditional extremes

```python
fit = tails.fit_conditional(x, conditioning=0)
sample = fit.sample(5000, given=6, seed=7)
q = fit.quantile(.95, given=6, n_boot=100, size=5000, seed=7)
q.estimate[1], q.lower[1], q.upper[1]  # site 1's conditional 95th percentile

# Also require site 1 > 2 while sampling site 2's behaviour.
sample = fit.sample(1000, given=6, conditions={1: 2}, seed=7)
```

`given=6` means the designated site **exceeds** 6, rather than equals 6.
It must be at least that site's fitted conditioning threshold. Additional
site conditions use rejection sampling; combinations too rare to simulate
raise an error. With `given=None`, conditioning uses the fitted high quantile,
and bootstrap refits use that same quantile of their own fitted margins.

The [Heffernan--Tawn model](https://doi.org/10.1111/j.1467-9868.2004.02050.x)
uses empirical bodies and fitted GPD upper tails to transform each site to
Laplace margins. Above the conditioning threshold it fits
`Y_j = alpha_j X + X**beta_j Z_j`, with `-1 < alpha_j < 1` and `beta_j < 1`.
JAX fits the Gaussian working likelihood in batches; simulation resamples
whole residual vectors, preserving dependence between the other sites.
Perfect dependence has a degenerate likelihood and is rejected.
The marginal tail mass is the observed fraction strictly above its fitting
threshold, so tied zero rainfall values do not acquire extra positive mass.
Tied body observations use their midranks when fitting the conditional model.
Conditioning thresholds are strict exceedances of the fitted raw quantile;
with ties, their actual exceedance fraction can be below the nominal fraction.

Unlike joint-event pooling, leading dimensions in `fit_conditional` represent
**separate fits**. Results and draws retain them. Conditional fitting accepts
aligned observations at any frequency, including hourly or annual DataFrames;
it does not construct daily windows. Bootstrap intervals resample aligned
time vectors and refit both margins and normalisations; they assume
independent records. Serially dependent data need separate block-bootstrap
validation. At least 20 complete conditioning exceedances are required.
Simulation-based quantiles also have Monte Carlo error; increase `size` when
checking very high quantiles.

## Real-data checks

The optional validation script reads the four user-supplied challenge files
without copying them into the repository. Install `rdata` to read RData:

```bash
python validation/compound_events.py --runs /path/to/downloads --ci profile
python validation/compound_events.py --eccc --ci profile
```

The ECCC example downloads daily precipitation for Québec/Jean Lesage
(`701S001`), Beauport (`7010565`) and Château-Richer (`7041330`) for 2000--2022
from the [official MSC GeoMet API](https://api.weather.gc.ca/collections/climate-daily).
The retrieved records have 6,522 jointly complete days. All three exceed
20 mm on 108 observed days; the model estimates 111.7 occurrences over that
complete-day exposure (95% profile interval 94.7--129.6). At least two exceed
30 mm on 80 days; the estimate is 76.7 (62.9--91.9). Incomplete-day selection
can itself affect the representativeness of this example.

The challenge cases require much farther extrapolation. With the four
provided runs pooled, the all-25-above-1.7 expected count per 165-year record
is 2.33 at fitting quantile 0.95, 0.34 at 0.975 and 0.11 at 0.99. No such
event is observed in those runs. This large sensitivity demonstrates that
the default tail approximation is not stable for this case; its profile
interval alone should not be treated as a reliable challenge prediction.
The unavailable 46 runs are not used, and empirical counts are not ground
truth for the population expectation. Synthetic tests check independent
exponential minima, binomial k-of-n probabilities, gamma sums, and run counts.
