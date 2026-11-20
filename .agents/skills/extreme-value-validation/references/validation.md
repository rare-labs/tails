# Scientific comparison and uncertainty reference

## Contents

Estimands and splits; fair scores; failures and support; uncertainty and coverage; resampling diagnostics; known-truth design; reporting and sources.

## Specify the claim and its target

Record variable/support, event set, observation cadence, valid exposure, threshold fitting rule, conditioning context, target horizon and population. Separate:

- a conditional excess survival from an unconditional probability;
- daily counts from storm/run counts;
- an estimated expected count from the future random count;
- a return level from a return period;
- marginal accuracy from high-dimensional joint-event accuracy;
- implementation equivalence from scientific model equivalence.

Observed counts are noisy measurements of an unknown expected count. A model can fit a historical total closely while predicting more extreme thresholds or future conditions poorly. Missing/zero measurements, censoring and discrete caps affect the observation model.

For a bounded experiment, state aims, generators/data, estimands, methods and performance measures before execution. Record a stopping condition. An unflattering result does not by itself authorise another family, a new threshold or a larger simulation campaign.

## Splits and leakage

Choose the independent observational unit: entire runs, years, storms, locations or time blocks as appropriate. Preserve aligned variables and record boundaries. Leaving out one ensemble run over the same calendar period evaluates that population transfer; it does not establish a forecast under a future trend.

Fit transformations, season boundaries chosen from data, thresholds, covariates and model selection only on training data. A supplied calendar convention is an assumption to verify or disclose. Reusing all folds repeatedly for method search makes later results exploratory; it does not turn the same data into pristine confirmation.

Do not count25 dependent sites times4 overlapping holdouts as100 independent experiments. Show per-fold or per-independent-unit results and choose uncertainty units that match the sampling design.

## Fair comparisons and scores

Compare methods on the same physical events, horizons, exposure and held-out observations. A shared nominal percentile does not imply shared physical cutoffs when each method uses its own percentile. For censored tail scores, choose a common training-defined evaluation cutoff with a valid predictive law above it.

Report both the original eligible target set and the common estimable subset. If an old score uses limits{a,b,c} and a new score uses{b,c}, recompute both on{b,c} before judging improvement; preserve the missing-a capability separately.

Define a count error's denominator explicitly: individual observed count, mean observed count per threshold, record exposure or known truth. These are not interchangeable. Percent error is undefined at zero observed counts. Avoid dropping those cases after seeing outcomes; use a prespecified appropriate measure and report zero-count forecasts directly.

Select a loss matching the prediction: squared/Bregman mean losses for expected-value forecasts, pinball loss for quantiles, and a proper predictive score for full distributions. Descriptive absolute count errors against single realised counts include event variability; they are not direct measurements of bias against the unknown population expectation.

For quantile q with prediction a and observation y, pinball loss is max(q*(y-a),(q-1)*(y-a)). A censored log score at cutoff c uses log F(c) for values at/below c and the appropriate predictive log density/mass for values above c. Include occurrence probabilities; a conditional tail density alone cannot fairly score unconditional events. Mixed atoms/censoring need the correct observation likelihood.

Pairwise composite likelihood and its held-out score assess selected lower-dimensional information. They are not full joint likelihoods and do not support ordinary AIC with an unadjusted parameter-count penalty. Better pairwise scores need not improve25-way probabilities.

## Failures, fitted support and integration error

Retain input identities, all forecasts, cutoffs, status, error causes, shapes, convergence/regularity, exposure and numerical tolerances. A fitted finite endpoint below a held-out observation is a support failure; a log score of negative infinity is evidence, not a row to remove.

Define estimator eligibility and failure handling before evaluation. Report availability and all-attempt operational performance alongside any successful-only diagnostic. If an interval is unavailable on an otherwise eligible dataset, do not describe coverage conditional on successful intervals as overall coverage. When a procedure is outside its declared domain, show the applicability gap rather than silently shrinking the task.

Zero simulated hits do not establish a zero probability. Distinguish known fitted support restrictions from a simulation that did not find rare events. For importance sampling, use the actual weighted estimator and its diagnostics; a crude binomial bound is not automatically valid. Monte Carlo integration error excludes parameter uncertainty and model misspecification. An analytic probability calculation can have zero integration error and substantial statistical uncertainty.

## Confidence, prediction and calibration

A confidence interval for theta=E[N] targets the unknown mean; a prediction interval targets a future random N. Comparing one observed N with a confidence interval for theta does not estimate confidence coverage. An interval can legitimately exclude its point estimate, e.g. a percentile/bias-corrected interval; that fact alone does not prove a bug. An ordinary likelihood-ratio confidence set defined from the same likelihood must contain its defining MLE. Exclusion there signals an inconsistent target, numerical approximation/failure, or another implementation problem to investigate.

For R independent known-truth experiments, coverage is the fraction of intervals containing each experiment's target truth. State eligibility and treatment of failures. Monte Carlo standard error is approximately sqrt(c*(1-c)/R); attach a suitable binomial interval. This formula does not apply as if dependent rows were independent experiments.

An observed coverage of91% may pass a project screen of90-98% while still giving evidence against nominal95% if its Monte Carlo interval excludes95%. Practical acceptance and exact calibration are different claims. Width is useful in conjunction with availability and coverage; narrower is not intrinsically better.

A Bayesian credible interval is conditional on its likelihood and prior. Numerical sampling checks, prior sensitivity and predictive checks address different issues; Bayesian computation alone does not guarantee frequentist coverage. If using simulation-based calibration, state its generative/prior assumptions and what part of the implementation it tests.

## Audit resampling before interpreting bounds

Check the target, units, original versus resampled exposure, parameter transforms, event limits and conditioning context. Refit the threshold/rate/margins/dependence/reordering components that the claimed procedure includes. Holding a component fixed is a different uncertainty method and should be labelled.

For context-transformed blocks:

1. Check exact forward/inverse values and event flags.
2. Compare original context-specific joint-event rates with rates induced by donor/context remapping before refitting.
3. Inspect simultaneous exceedances, lagged dependence and run lengths.
4. Inspect the bootstrap mean/median, spread, support, failed draws and shift from the original statistic.

A large shift can reflect dependence-transfer assumptions, marginal misfit, finite-sample bias or a coding error. A successful numerical inverse excludes only some explanations. A controlled comparison changing one component helps locate the cause; adopting the modified method is a separate scientific decision.

Block length, nonwrapping/circular treatment, missing segments and independent records matter. Resampling entire years can preserve season labels but requires suitable independent-year assumptions. Grouping after compression can join times that were never adjacent. Repeated refits need deterministic seeds and explicit retained diagnostics, not silent redraw-until-success.

## Known-truth designs worth reusing

Use exact or independently calculated answers for a purpose:

- Independent exponential variables: their minimum has rate equal to the sum of rates.
- Independent Bernoulli exceedances: k-of-n probability is the binomial tail when probabilities are equal, otherwise Poisson-binomial.
- Sums of independent exponentials with a common rate: gamma law. Unequal rates do not give the same gamma distribution.
- Stationary iid daily event probability p: expected maximal r-day-run count is p^r*[1+(N-r)*(1-p)] for N>=r, with the record-boundary convention explicitly stated. At r=1 this counts clusters, not event days.
- Conditional/spatial simulation: start with independently computable low-dimensional laws, dependence limits and anchor-coverage/overlap identities before sparse high-dimensional targets.

Separate parameter recovery, numerical equivalence, estimation bias, prediction loss and interval calibration. Include relevant misspecification, serial dependence, changing margins and sparse targets in a larger study only when that expansion is authorised. Use new independent data/seeds for confirmation rather than calling repeatedly tuned exploratory cases definitive.

## Compact reporting pattern

Give a plain-language verdict first. Then identify the comparison population and matched targets, decisive numbers, missing cases, and limitations. Explain what changed and whether the evidence concerns counts, quantiles, intervals or software. A component-fidelity table belongs with a paper replication, not hidden in implementation history. A plot should distinguish observed realised counts, fitted means, integration SEs, confidence limits and prediction limits.

Preserve the complete scientific record where the user authorised it. Do not publish local data/research or close project tasks as scientifically validated merely because software tests passed.

## Primary sources

- Morris, White and Crowther (2019), [Using simulation studies to evaluate statistical methods](https://doi.org/10.1002/sim.8086): simulation aims/design, failures, performance and Monte Carlo uncertainty. [Author copy](https://discovery.ucl.ac.uk/id/eprint/10066118/).
- Gneiting and Raftery (2007), [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437): predictive scoring and quantile/distribution evaluation. [Author manuscript](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
- Coles (2001), [An Introduction to Statistical Modeling of Extreme Values](https://doi.org/10.1007/978-1-4471-3675-0): extreme-value estimation, threshold diagnostics and inference.

These are targeted original summaries, not proof that any particular fitted model or interval has passed scientific validation.
