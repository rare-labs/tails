---
name: extreme-value-validation
description: Use when comparing, scientifically validating, reproducing, or reporting extreme-value models, uncertainty intervals, rare-event simulations, or published competition methods.
---

# Extreme Value Validation

Separate what was implemented, what was tested, and what the evidence supports. A partial adaptation, a close fitted count and numerical convergence answer different questions from a faithful reproduction or a calibrated forecast.

Use [comparison and uncertainty reference](references/validation.md) for experimental design, scoring, failure handling and interval interpretation. For published methods, read [reproduction reference](references/reproduction.md) before claiming equivalent implementation or performance.

## Scope the comparison

Record the exact target, units/exposure, data identity, train/test units, methods, physical event limits, score, relevant assumptions and stopping point. Reuse already authorised choices. Surface a materially different scientific assumption or a missing required component before adopting it; ordinary reversible execution does not need repeated approval.

Fit preprocessing, thresholds and tuning using training data. Preserve held-out event definitions. Report the original target set, unavailable cases and matched comparisons separately; an easier common subset does not erase missing capability. Overlapping folds and dependent locations are not independent experiments.

When comparing a paper, map each important source component to implemented, altered, missing or unverified. Confirm that target and scoring match the paper. Use 'partial adaptation' until the required components and reference results have actually been reproduced.

## Validate the right claim

- Numerical checks test formulas and software. Known-truth experiments test an estimator under stated generators. Held-out records test prediction on those populations. None automatically certifies all future extremes.
- Define the score's denominator and eligibility in advance. Keep zero-event and failed/unsupported cases visible. Do not change shape, thresholds, inputs or exclusions merely to obtain a favourable result.
- Distinguish confidence bounds for an unknown expectation from prediction bounds for a future realised count. One observed count is not the true expectation. Nominal coverage needs repeated appropriate experiments and Monte Carlo uncertainty.
- Refit every component included in the claimed uncertainty procedure. Keep physical event limits fixed when those define the estimand; for a quantile-defined event, specify and resample that functional consistently.
- Inspect distribution shifts, support failures, zero-hit simulations, refit availability, dependence preservation and sensitivity. A narrow interval or successful optimiser is not evidence of calibration.

## Explain before elaborating

Answer the user's actual question first: better on which task, by how much, and what prevents a broader conclusion. Use ordinary words before a model name or formula. Define 'observed' as counted in the data and 'fitted' as estimated by the model. Say whether the comparison used the same training record or unseen observations.

For a nonexpert, use a short verdict, at most the decisive numbers, and the practical limitation. Only add a graph when it clarifies that verdict; label units, populations and uncertainty types. Preserve the full scientific record in the authorised location and honour the agreed stopping point. A negative result can complete a study.
