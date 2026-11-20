---
name: extreme-dependence-models
description: Use when analysing joint, conditional, spatial, compound, k-of-n, or consecutive-event extremes; tail copulas; conditional-extremes simulation; or dependence-preserving resampling.
---

# Extreme Dependence Models

Match the probability model to the exact multivariate event. Marginal tail fits, spatial wrappers and a complete space-time generator provide different capabilities.

Read [dependence reference](references/dependence.md) for event/run identities, conditional models, anchor weighting and transformation diagnostics. Read a source method before implementing its particular construction.

## Define and cover the event

Write the event as a set of aligned observations, with thresholds, directions, participating sites, time windows and run-count convention. For k-of-n on successive days, distinguish changing participants from the same k sites remaining extreme. Preserve gaps and independent-record boundaries.

A combined min/kth-largest/sum score can estimate that event directly with a univariate tail model. It does not identify a spatial stochastic process or supply predictions at unobserved locations. Event-preserving transformations can still change a finite-threshold tail approximation.

## Select and check the dependence model

Inspect simultaneous high-quantile exceedances, separation, temporal persistence and changes by context. Ordinary correlation and pairwise fit quality do not determine rare high-dimensional event probabilities. Distinguish finite-level diagnostics from assertions about an asymptotic limit.

Conditional extremes model the remaining series given a designated series is extreme. A Gaussian working likelihood does not imply Gaussian simulated residuals. Empirical residual vectors, fitted residual distributions and a residual copula are distinct modelling choices. Check residual dependence on the conditioning magnitude and the assumptions behind extrapolation.

A single anchor estimates an unconditional event only when that event lies inside the anchor's conditioning region and the anchor probability is supplied. General k-of-n/any-site events require complete region coverage and justified overlap correction. Do not treat several separately fitted conditionals as automatically compatible with one joint distribution.

## Resampling and simulation

Keep marginal transformation, joint simulation, time ordering and uncertainty refitting distinct. Shared block indices preserve aligned vectors in the distribution being resampled; they do not validate that distribution.

If probability-transformed observations are pooled across seasons or regimes, assess whether the transformed joint dependence is transferable. Exact forward/inverse agreement checks arithmetic only. Compare original event rates with rates induced by donor/context remapping, then examine refitted estimates and resampling-distribution shifts. A bootstrap interval excluding its own point estimate is a warning to investigate; exclusion alone is not a mathematical proof of invalidity.

For simulation, distinguish integration error, parameter uncertainty and future-event variability. Retain zero-hit and failed-fit diagnostics. Validate code with independent cases and validate the scientific approximation with appropriate data or known-truth experiments.

Explain findings as what improved, what did not, and what remains unsupported. For claims of a published-method reproduction or a model winner, consult extreme-value-validation when available; a shared model name is insufficient evidence.
