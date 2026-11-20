# Joint events across series

An event can involve different measurements and different limits. For example,
"at least two of these three stations exceed their own rainfall limit today"
is `event="kofn", k=2`. Here **k** is the required count and **n** is the number
of series. `event="all"` requires every series; `event="sum"` adds them.

```python
import tails

# DataFrame: time on rows, one series per column. Arrays: (..., series, time).
result = tails.how_rare_joint(
    frame, thresholds=[30, 25, 40], event="kofn", k=2, ci=None,
)
print(result.observed_count, result.expected_count)
```

Each measurement is divided by its limit. The minimum, k-th largest, or sum
becomes a single score; a score strictly above 1 is the event. The GPD is fitted
above the score's 0.95 quantile. Positive per-series limits can have different
units for `all` and `kofn`. For a sum in common units exceeding U, pass the
**same** scalar `thresholds=U`; unequal limits instead define a weighted sum of
ratios, with a different meaning.

Leading array axes pool independent replicate records. Expected and observed
counts are for one average record, not the total across all replicates. Incomplete
vectors are missing exposure, not zero events. Datetime DataFrames are reindexed
to their regular cadence; provide `freq="h"` or another explicit frequency when
it cannot be inferred. `npy` is observations per year (default 365.25), so hourly
data require a different value. Use `npy=None` for periods in observations when
an annual rate would have no useful interpretation.

`run_length=r` uses a rolling minimum and counts consecutive qualifying windows
once per run. No window crosses a gap or replicate boundary. Fitting-threshold
declustering can merge distinct runs at the target level: run estimates remain
an approximation. Inspect `observed_count` alongside fitted counts.

## Signed limits and lower extremes

Ratios require positive limits. For temperatures below zero, losses, pressure
drops, or other signed quantities, supply positive `scales` and `directions`:

```python
score = tails.compound_series(
    measurements, thresholds=[-5, 100], scales=[2, 20], directions=[-1, 1],
)
```

The first measurement must be below −5 and the second above 100. A positive
scale sets the score's distance from each limit; changing it preserves the
event but can change the tail approximation. This formulation supports `all`
and `kofn`; sum events retain their common-unit or weighted-ratio definition.

## Changing conditions and uncertainty

An `operating_regime` is simply a supplied label for the conditions when a
measurement was made: winter/summer, day/night, or a machine's recorded load
state. It is not a rainfall-specific model. `groups=operating_regime` fits a
separate GPD scale **and shape** for each label. Labels must be known for every
complete vector and each group needs ten tail excesses. This is a piecewise
constant model, not a continuously changing covariate model. Grouped runs are
currently unsupported.

The default `ci="profile"` profiles the stationary GPD and event rate on a
logarithmic return-period grid. Its assumptions include independent tail
excesses and the chosen threshold. For serially dependent records, opt into
full-refit aligned block resampling:

```python
result = tails.how_rare_joint(
    frame, limits, event="kofn", k=2, groups=operating_regime,
    ci="block_bootstrap", block_length=7, n_boot=200, seed=42,
)
print(result.bootstrap.diagnostics)
```

Block length is a mean in observations, chosen for the application. All series
share each draw, gaps stay gaps, and independent records stay separate. Grouped
block fits require `threshold_quantile > .5`. Nonwrapping blocks overweight
segment ends, especially in short or fragmented records; check block-length
sensitivity before interpreting the intervals. Grouped
fits transform each series to fitted probabilities, resample aligned blocks,
and invert at the original destination labels. This assumes that dependence
after transformation is stable across those labels. It does not model changing
dependence automatically.

Every draw refits thresholds and tails while holding the **event limits fixed**.
If a bootstrap fitting threshold crosses the event limit, the empirical body
gives the event rate. Diagnostics report these draws, shapes ≤ −0.5, and failed
refits. Any failed draw makes the interval unavailable (`NaN`); failures are never
discarded to make an interval look usable. Bounds describe uncertainty in an
expected count, not the random count in a future record. These implemented
intervals have not been certified for coverage across applications.

See [four measured-data applications](applications.md).

::: tails.compound_series
::: tails.how_rare_joint
::: tails.JointRarity
::: tails.fit_marginal
::: tails.resampling.bootstrap_statistic
