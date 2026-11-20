# Four applications with measured data

The same event machinery works with precipitation, electricity demand, machine
sensors, and computing metrics. The measurements below are real. The event
limits are illustrative sample 0.99 quantiles (CPU: 90%), **not** engineering safety limits.
Choosing limits and evaluating them on the same record demonstrates the API;
it is not an independent predictive check.

The runnable loader and fitting example is
[`examples/application_data.py`](https://github.com/rare-labs/tails/blob/main/examples/application_data.py).
From a source checkout with `tails` installed:

```bash
python examples/application_data.py --example all --cache /path/outside/repository
```

The first run downloads about 200 MB, chiefly electricity and hydraulic records.
Downloads are cached outside the checkout; large input files are not bundled.
Each application prints complete-vector exposure, fixed event limits, the
observed count and a fitted expected count. It uses `ci=None`: these data have
dependence and changing conditions that require checking before reporting
uncertainty. Downloaded source files and their terms remain with their providers.

| Application | Measured variables | Event in the example | Time basis |
|---|---|---|---|
| Québec precipitation | Three stations, mm per day | At least two exceed their respective limits | Daily, 2000–2022 |
| Electricity | DE, FR, AT national actual load, MW | At least two countries exceed their own limits | Hourly UTC, 2015–2019 |
| Hydraulic machinery | Pressure, motor power, vibration | At least one cycle maximum exceeds its own limit | One 60-second experimental cycle |
| Computing | Average CPU use in one AWS autoscaling group, % | CPU use exceeds 90% | Five-minute readings |

All four examples report periods in **observations** (`npy=None`). This avoids
turning experimental machine cycles or a short changing server trace into an
unsupported annual risk estimate. Expected counts use complete observations
only. The CPU record is one series: it does not demonstrate dependence between
different services.

An execution of the loader gives the following rounded counts. These are
descriptions of the supplied records, not out-of-sample accuracy scores.

| Application | Complete observations | Observed events | Fitted expected events |
|---|---:|---:|---:|
| Precipitation | 6,522 | 50 | 48.9 |
| Electricity | 43,794 | 234 | 282 |
| Hydraulic cycles | 2,205 | 57 | 63.8 |
| CPU above 90% | 18,050 | 545 | 373 |

The CPU discrepancy is substantial. The ceiling and operating changes described
below are reasons to investigate the model rather than treat this fit as a
reliable alerting model. Sources such as ECCC can revise their observations.

## Precipitation around Québec City

The loader calls the official
[ECCC MSC GeoMet API](https://api.weather.gc.ca/collections/climate-daily)
through `tails.datasets.eccc_daily` for climate identifiers `701S001`, `7010565`
and `7041330`, then aligns their daily **total precipitation**. This can include
snow water equivalent; it is not a liquid-rain-only measurement. Missing station
days remain missing, and an event requires all three values to be observed.

```python
from application_data import rainfall  # run from examples/, or add it to sys.path
import tails

frame = rainfall(cache="/path/outside/repository")
limits = frame.quantile(.99).to_numpy()
risk = tails.how_rare_joint(frame, limits, event="kofn", k=2, npy=None, ci=None)
```

Shared storms and seasons matter. Supplied season labels can fit separate
marginal event tails, but the probability-transform bootstrap also assumes
stable residual dependence. It does not establish that this assumption is true.

## Electricity demand

[Open Power System Data, release 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06)
publishes hourly national load measurements from ENTSO-E Transparency. The loader
selects `DE_load_actual_entsoe_transparency`, `FR_load_actual_entsoe_transparency`
and `AT_load_actual_entsoe_transparency`, retaining 2015–2019. UTC timestamps
avoid artificial daylight-saving gaps. These are actual loads, not forecasts.

```python
from application_data import electricity

frame = electricity(cache)
risk = tails.how_rare_joint(frame, frame.quantile(.99).to_numpy(),
                           event="kofn", k=2, npy=None, ci=None, freq="h")
```

Demand varies with season, hour and economic conditions. An annual stationary
fit is illustrative; forecasting needs an appropriate reference period and
covariates. Respect the upstream source and reuse terms linked by OPSD.

## Machine sensors with different units

[UCI Condition Monitoring of Hydraulic Systems](https://doi.org/10.24432/C5CW21)
contains 2,205 measured 60-second cycles from an experimental hydraulic rig
(Helwig, Pignanelli and Schütze, 2015; CC BY 4.0). The loader takes the maximum
within each cycle for PS1 pressure (bar), EPS1 motor power (W), and VS1 vibration
(mm/s). Their raw sampling rates differ, but their **cycle summaries** align.
The maxima need not happen at the same instant within a cycle.

```python
from application_data import hydraulic

frame = hydraulic(cache)
risk = tails.how_rare_joint(frame, frame.quantile(.99).to_numpy(),
                           event="kofn", k=1, npy=None, ci=None)
operating_regime = frame.attrs["operating_regime"]
```

`operating_regime` joins the four recorded component-condition labels; it is
available for inspection, not automatically passed into a fit. Many combinations
have too few cycles for separate extreme-value fits. This changing fault
experiment is not a stationary sample of factory failures, and the fitted count
must not be described as a production failure probability. A useful next analysis
would choose a sufficiently sampled, physically comparable operating condition.

## Computing load

The [Numenta Anomaly Benchmark data documentation](https://github.com/numenta/NAB/blob/master/data/README.md)
describes `realKnownCause/cpu_utilization_asg_misconfiguration.csv` as average
CPU usage across an AWS autoscaling group. The example uses this measured
five-minute trace with its timestamps and original percentage values.

```python
from application_data import computing

frame = computing(cache)
risk = tails.how_rare_joint(frame, [90.],
                           npy=None, ci=None, freq="5min")
```

This trace contains a point mass at the physical 100% CPU ceiling: a continuous
GPD cannot reproduce that atom and must not be used to extrapolate beyond the
ceiling. The example evaluates 90% inside the observed range and exposes this
model limitation. NAB also documents anomalies and autoscaling changes. A pooled GPD can describe
this record; it cannot by itself predict a different fleet or establish an
alert false-positive rate. With contemporaneous metrics from several services,
the same `all`, `kofn` and run definitions would apply after checking alignment
and recording the operating conditions.
