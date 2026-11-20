# What happens elsewhere when one series is extreme?

Joint risk asks how often a combination happens. A conditional model asks a
different question: **given that one series exceeds a large value, what values
should we expect in the other series?** This applies to stations, power loads,
or aligned sensor measurements in their original units.

```python
import tails

fit = tails.fit_conditional(frame, conditioning=0)
draws = fit.sample(5000, given=60, seed=42)
quantiles = fit.quantile([.5, .9], given=60, size=10000)
```

Here `frame` has time on rows and series on columns. `conditioning=0` selects
the first column. `given=60` means **above 60**, not exactly 60. The level must
be at or above the fitted conditioning threshold and inside the fitted support.
Draws contain every series, including the conditioning series.

For the [measured precipitation example](applications.md), load `frame` with
`rainfall(cache)` from `examples/application_data.py`. Then `given=60` means
60 mm of daily total precipitation at station `701S001`, and the other columns
remain in mm. This is a conditional distribution for the observed station set,
not spatial interpolation to an unobserved location.

## The model in plain terms

Each series has an empirical body and fitted GPD upper tail. Transforming these
to a common Laplace scale removes the differences in units and typical sizes.
The Heffernan–Tawn model then estimates how the other series grow as the
conditioning series becomes larger. A Gaussian working likelihood estimates
the scaling rules; simulations resample **whole observed residual vectors**,
preserving their dependence instead of assuming the other series independent.

It is one particular model of extreme dependence, not automatic selection
among copulas or spatial models. The reference is
[Heffernan and Tawn (2004)](https://doi.org/10.1111/j.1467-9868.2004.02050.x).
Exact perfect dependence is degenerate and rejected. At least ten marginal
tail excesses per series and twenty complete conditioning excesses are needed.

## Observed operating conditions

If the relationship changes with operating conditions, fit contexts explicitly:

```python
fit = tails.fit_conditional(frame, groups=operating_regime, context="winter")
```

`operating_regime` is one supplied label per observation. Other contexts become
missing vectors while their positions stay on the time axis. This prevents
block resampling from joining separate winters across excluded summers. The
fit describes only the selected context; separate contexts have separate
margins and conditional parameters. Enough data are required in each selection.
No regimes, calendar seasons or transitions are inferred automatically.

## Uncertainty without changing the question

```python
q = fit.quantile(
    [.5, .9], given=60, size=10000, n_boot=200,
    bootstrap="block", block_length=7, seed=42,
)
print(q.estimate, q.lower, q.upper)
print(q.bootstrap.diagnostics)
```

Each draw resamples aligned vectors in blocks and refits the margins, scaling
rules and residual distribution. Blocks stay inside complete segments and
independent batch records. `block_length` is a mean in observations; choose it
for the sampling cadence and examine sensitivity. Nonwrapping blocks overweight
segment ends, especially in short or fragmented records. `bootstrap="iid"`
(the default) is suitable only when aligned observations can be treated as
independent. Leading array axes are separate conditional fits, not pooled runs.

The original `given` stays fixed, including when omitted (then the original
fitted conditioning threshold is used). If a refit's conditioning threshold
exceeds it, that draw cannot answer the same tail question and fails explicitly.
Failed refits stay in diagnostics and any failure makes the bounds unavailable.
Choose a scientifically meaningful large `given`; do not move it just to make
an interval appear. These are intervals for estimated quantiles, not prediction
intervals for individual future measurements. Monte Carlo error also depends
on `size`. Implemented resampling is not evidence that nominal coverage holds.

::: tails.fit_conditional
::: tails.ConditionalFit
::: tails.ConditionalQuantiles
