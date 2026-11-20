# Joint and space-time extreme-value reference

## Contents

Events and scores; temporal runs; tail dependence; conditional extremes; simulation and anchor coverage; block resampling; model checks; primary sources.

## Event algebra

For positive limits u_j, define R_j(t)=X_j(t)/u_j. Then:

- all sites above their limits: min_j R_j(t)>1;
- at least k above: kth-largest_j R_j(t)>1;
- sum in common units above U: sum_j X_j(t)/U>1.

Different denominators in a sum define a weighted sum of ratios, not the raw sum. For signed limits or mixed upper/lower directions, a score 1+d_j*(X_j-u_j)/s_j with s_j>0 and d_j in{-1,+1} gives the desired per-site strict event at score>1. A minimum or order statistic then combines the sites. These formulations preserve an event; different normalisations need not yield identical finite-threshold GPD approximations.

Specify whether missing any required site makes that observation ineligible. For k-of-n, partial vectors can only be used through an explicit missing-data/event-observability construction. Treating a missing value as a non-exceedance changes the event probability.

## Runs: order of operations matters

Let E_t denote the binary daily event, e.g. at least k sites exceed on day t. A qualifying r-day window is product_{i=0}^{r-1} E_{t+i}=1. Maximal runs lasting at least r days count once at their start:

N_r = sum_{t=1}^{N-r+1} I(E_t=...=E_{t+r-1}=1) * [1 if t=1 else I(E_{t-1}=0)].

Missing positions and replicate boundaries break eligible windows. Record-boundary treatment is part of the estimand: this counts an initially qualifying stretch even when its earlier history is unobserved.

For stationary binary E:

E[N_r] = P(E_1,...,E_r) + (N-r)*P(not E_0,E_1,...,E_r), N>=r.

For iid Bernoulli(p), E[N_r]=p^r*[1+(N-r)*(1-p)]. For N<r the expected count is0. At r=1 this formula counts clusters of consecutive events, not all event days; the expected daily-event count is N*p.

Changing participants and persistent participants are different. For daily sets A_t of exceeding sites:

- k-of-n each day: min_i |A_{t+i}|>=k;
- the same k sites throughout: |intersection_i A_{t+i}|>=k.

Example: k=3, A_1={1,2,3}, A_2={2,3,4}. The first event occurs and the second does not. Reduce spatially then take a rolling minimum for the first; switching those operations can implement the second. Declustering at a lower fitting threshold may merge distinct runs at the target threshold. Derive or disclose the approximation.

## Tail dependence and spatial structure

For continuous fitted margins U_j=F_j(X_j), chi_ij(q)=P(U_j>q | U_i>q). The limit as q approaches1 describes asymptotic tail dependence. Finite thresholds and scarce joint exceedances do not identify that limit reliably. Distance, lag, regimes and measurement support can all affect the diagnostic.

Gaussian copulas with nonperfect correlation are asymptotically independent; finite-df Student-t copulas at nonsingular pair correlations have asymptotic tail dependence. Neither property alone determines which approximates a finite event best. Nonindependent max-stable models typically impose asymptotic dependence and are naturally connected to block maxima. Conditional-tail models can accommodate a broader range of tail relationships, but add normalisation/residual assumptions. Select against the actual target, not complexity or popularity.

A spatial kernel needs meaningful locations/distances and positive-definite construction. Grid row/column units are not kilometres. Unstructured pair correlations may require a coherent matrix construction, whose changes to fitted correlations must be checked. A wrapper accepting arbitrary grids is not a fitted geographic process.

## Conditional extremes

After fitting suitable marginal bodies/tails, transform to standard exponential-tailed margins such as Laplace. On the positive Laplace tail, survival is0.5*exp(-x). For a large conditioning X=x, a common model is:

Y_j = a_j*x + x^(b_j)*Z_j, approximately for x>v>0.

Common practical restrictions include a_j in[-1,1] and b_j<1; precise admissibility and boundary cases depend on the model. Coefficients alone do not establish compatibility or every extremal property. Check whether fitted residuals Z change with X, whether residual margins/tails fit, and whether their joint structure matches the prediction event.

Drawing X conditional on X>v is different from fixing X=v. Keep the actual conditioning event in simulations and reported quantiles. A below-threshold conditioning limit is not answered by silently raising it to v. Separate context-specific dependence fits from an assumption of common dependence after marginal standardisation.

A Gaussian working likelihood can estimate normalisations even when simulations use empirical residual vectors. Resampling whole vectors preserves their observed cross-component association, but cannot invent unobserved residual support. Fitted asymmetric residual laws and vine copulas provide different extrapolation assumptions; substituting an empirical residual distribution is a model change, not a faithful reproduction of such a method.

## Single-anchor identities and overlap weighting

Let B be the target and A an anchor exceedance. If B is a subset of A:

P(B)=P(A)*P(B|A).

For all-site exceedances, any required site can supply this identity when its physical event threshold lies inside the fitted conditioning region. For at-least-k events, a specified anchor can remain nonextreme while other sites qualify; then the missing P(B intersect not A) cannot be discarded.

For a coherent parent density f and anchor sets A_i with positive masses p_i=P(A_i)>0, choose anchors with probabilities pi_i and draw from f conditional on A_i. The mixture proposal density on their union is

q(y)=f(y)*sum_i pi_i*I(y in A_i)/p_i.

For B covered by that union, use the reciprocal of the sum as an importance weight for I(B). If pi_i=p_i/C with C=sum_i p_i, it simplifies to C/m(y), where m is the number of qualifying anchors. Equal-anchor sampling gives this simple inverse-count form only with equal p_i. Self-normalised weights estimate conditional ratios, not an unconditional probability without its normalising mass.

This derivation assumes all conditional laws come from the same f. Separately fitted Heffernan-Tawn conditionals need a justified construction/approximation and compatibility checks; arithmetic overlap correction alone cannot establish exact unbiasedness. Check target-region coverage against conditioning thresholds before simulation.

### Finite-state overlap example

A common parent law assigns probabilities0.1 to both anchors,0.2 to only A,0.3 to only B, and0.4 to neither. Then p_A=0.3, p_B=0.4. With uniform anchor choice, the proposal probability of both is0.5*(0.1/0.3)+0.5*(0.1/0.4)=7/24. Its correct unconditional weight is2/(1/0.3+1/0.4)=12/35, giving (7/24)*(12/35)=0.1.

With anchor choice proportional to mass, the proposal probability of both is0.1*2/0.7=2/7 and the weight is0.7/2=0.35, again giving0.1. Bare inverse-overlap counts omit the normalising mass. Normalising those weights yields a probability conditional on at least one anchor, not the unconditional0.1.

Monitor weighted Monte Carlo error and effective sample size, as well as parameter uncertainty. With zero iid unweighted Bernoulli hits in m draws, a one-sided1-alpha probability bound is1-alpha^(1/m). This bound cannot be transplanted unmodified to arbitrary importance weights or dependent draws.

## Resampling and temporal simulation

Aligned blocks retain spatial vectors and some persistence while sampling their source distribution. Marginal standardisation does not generally make dependence stationary. Pooling transformed vectors over seasons and inverting at a destination season implicitly transfers the joint transformed distribution to that season.

Check this pipeline in layers:

1. Does transform/inverse recover original values and event flags?
2. What event rates arise when original transformed vectors are reassigned to destination contexts? Compare to original context-specific rates before fitting again. If this rate is estimated by simulation, report integration error; an exact empirical mixture calculation has no simulation error but still has sampling and model uncertainty.
3. What happens to simultaneous exceedance sizes, lagged extremes and maximal-run lengths?
4. How do bootstrap draws centre relative to the original estimate? Investigate large shifts rather than calling converged refits calibrated.

A shift can involve dependence nonstationarity, marginal approximation, finite-sample estimator bias or implementation errors; a successful round trip rules out only some numerical causes. Nonwrapping blocks can introduce boundary bias, especially in short fragmented segments. Circular blocks can instead invent end-to-start adjacency. Neither choice is universal; match the sampling design and assess sensitivity.

Simulating spatial vectors does not by itself simulate their time order. Rank reordering or an explicit temporal generator is another assumption. If time order is borrowed from observed data, evaluate what remains fixed and which future/persistence questions it can answer.

For discrete atoms, a deterministic CDF is not uniform even under a correct mixed distribution. Preserve mass and use explicit censoring/distributional-transform reasoning where needed; do not silently jitter real zeros, caps or rounded values.

## Primary sources

- Heffernan and Tawn (2004), [A conditional approach for multivariate extreme values](https://doi.org/10.1111/j.1467-9868.2004.02050.x): conditional normalisations, residual modelling and extrapolation framework.
- Wadsworth and Tawn (2022), [Higher-dimensional spatial extremes via single-site conditioning](https://doi.org/10.1016/j.spasta.2022.100677): joint inference and importance sampling over conditioning locations. [Author manuscript](https://eprints.lancs.ac.uk/id/eprint/171871/1/SpatialConditionalUpdated.pdf).
- Bratkova et al. (2026), [Estimating rare spatio-temporal precipitation events with conditional extremes](https://doi.org/10.1007/s10687-026-00548-5): one complete application combining non-stationary margins, asymmetric residual models, vines, weighted simulation, reordering and uncertainty. Its performance does not transfer automatically to a partial adaptation.

These summaries do not replace the exact source algorithm, parameter constraints or proposal-weight derivation for a claimed reproduction.
