---
name: extreme-value-foundations
description: Use when fitting, implementing, reviewing, or interpreting GEV, GPD, peaks-over-threshold, block maxima, extreme quantiles, return periods, or non-stationary univariate tails.
---

# Extreme Value Foundations

Connect the fitted tail to the actual event, units, exposure and population of interest. A numerical fit is one part of the statistical model.

Read [core reference](references/core.md) for formulas, support, threshold changes, non-stationarity and numerical edge cases. Consult its primary sources when an assumption or parameter convention matters.

## Establish the quantity

Record the measured variable and units, upper/lower event and strictness, sampling interval, valid exposure, target population/time/context, and whether the answer concerns a probability, expected count, annual maximum, cluster or future realised count. Resolve material ambiguity from data/docs before asking the user.

Choose block-maxima GEV for consistent block extremes, or GPD for excesses above a suitable high threshold. A GPD excess law alone does not specify how often excesses occur. Lower extremes can be analysed through a stated sign transform.

## Connect and check

- Distinguish conditional tail survival, per-observation exceedance probability, cluster intensity, inverse event rate and annual exceedance probability. State the independence/Poisson assumption only where it is actually used.
- Derive exposure from cadence, calendars, missingness and recording opportunities. Pool comparable independent records without inventing windows across boundaries. Seasonal/trend forecasts need a stated destination context and exposure weighting.
- Check the parameter convention at the library boundary. SciPy GEV uses c=-xi; its GPD uses c=xi. Scale and log-scale are different parameters.
- Verify support, finite endpoints, the xi=0 limit and stable survival arithmetic before extrapolation. A fitted finite endpoint is not proof of a physical bound.
- Inspect threshold/parameter stability, QQ or tail-probability diagnostics, effective independent tail information, clustering, held-out support violations and sensitivity. Compare predictive scores at common physical cutoffs, not likelihoods based on different retained observations.
- Treat likelihood convergence, model adequacy and uncertainty calibration as separate evidence. Do not repair failures by silently restricting shape, dropping data, jittering ties or changing the event.

## Explain the result

Lead with the answer in ordinary language, name the variable and time unit, then give the supporting numbers and the most relevant limitation. Define a technical term when first used. Explain whether a proposed model change improves predictions, uncertainty, computation, or only the interface; those are different achievements.

For joint/conditional/spatial events, use the separate extreme-dependence-models skill when available. For validation or source-method replication, use extreme-value-validation when available. These references do not choose a universal best family or certify a fitted model.
