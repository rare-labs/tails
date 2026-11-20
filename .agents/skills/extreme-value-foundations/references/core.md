# Univariate extreme-value reference

## Contents

GEV and GPD conventions; probability/exposure bridge; non-stationary quantities; fitting and diagnostics; executable arithmetic example; sources.

## GEV block maxima

With z=(x-mu)/sigma, sigma>0:

G(x)=exp(-(1+xi*z)^(-1/xi)) where 1+xi*z>0.

At xi=0, G(x)=exp(-exp(-z)). Outside the support, extend the CDF by 0 below the lower endpoint or 1 above the upper endpoint, according to the sign of xi. A T-block return level satisfies G(q_T)=1-1/T, T>1:

q_T = mu + sigma/xi * ((-log(1-1/T))^(-xi)-1).

At xi=0, q_T=mu-sigma*log(-log(1-1/T)). Use log1p/expm1 forms near zero. Blocks must represent comparable durations or an explicitly modelled exposure. Annual maxima already have annual probability units.

SciPy genextreme uses c=-xi. SciPy genpareto uses c=xi. Inspect another library's convention rather than generalising this sign conversion.

## GPD threshold excesses

For y=x-u>=0, scale sigma_u>0:

S_u(y)=Pr(X>u+y | X>u)=(1+xi*y/sigma_u)^(-1/xi).

At xi=0 it is exp(-y/sigma_u). For xi<0, the fitted endpoint is u-sigma_u/xi; survival at/beyond that endpoint is zero. For xi>=0 the upper support is unbounded. A fitted endpoint can be wrong; test held-out support and endpoint uncertainty. A known physical ceiling or measurement censoring is a separate part of the observation model.

Under the GPD approximation, moving from u to v>u preserves xi and changes scale to sigma_v=sigma_u+xi*(v-u), while sigma_v>0 and v remains in support. Threshold choice trades approximation bias against limited tail information. Stability plots are evidence, not an automatic proof of an appropriate threshold.

The severity mean is infinite at xi>=1 and variance at xi>=1/2. This does not make a finite-record expected event count infinite: that count is bounded by the number of eligible observations. Preserve the estimand when interpreting heavy shapes.

## Probability and exposure bridge

Let p_u be the threshold-exceedance probability per eligible observation. For x>u:

p_x = p_u*S_u(x-u).

For n eligible observations under stationarity, E[N_x]=n*p_x. With time/context-varying probabilities, E[N_x]=sum_t p_t(x). Linearity of expectation does not require independent observations. Missing observations are not non-events; extrapolation to missing periods needs an explicit missingness assumption. Counts of exceedances and counts of declustered storms/failures use different occurrence rates.

For an actual calendar year, use its eligible opportunities; an average-year cadence is a stated convention. If there are m equally spaced eligible opportunities per year, lambda_x=m*p_x is the expected annual event count. The inverse event rate 1/lambda_x is an intensity-based recurrence measure. It is not generally the mean wait from an arbitrary time origin under dependence.

For independent observations, the annual probability of any exceedance is 1-(1-p_x)^m; for independent varying probabilities it is 1-product_t(1-p_t). Under a justified Poisson event model it is 1-exp(-lambda_x). Neither follows merely from knowing an expected count. A probability-based annual return period is the reciprocal of the annual exceedance probability. It agrees approximately with inverse intensity for sufficiently rare events under the relevant assumptions.

A stationary independent annual-maxima process with annual exceedance probability 1/T has geometric waiting time of mean T years. This is not a schedule. The probability of at least one event over n independent years is 1-(1-1/T)^n.

## Seasonal and changing populations

An average over past seasons is not a forecast for a specified future season. For known destination exposures n_g, use sum_g n_g*p_g(x); do not automatically give equally sized weight to unequal seasons. Trends require a specified time/covariate path. There is no unique timeless return period when the process changes. Threshold, rate, scale and shape regressions represent different assumptions; use separate shapes, a shared shape, smooth covariates or pooling only with scientific justification and enough information.

GPD models can be applied outside precipitation. Check actual support, observation frequency, changing populations, rounding, zeros and censoring. A body with atoms and a continuous tail is a mixed model: preserve atom mass and use the corresponding probability or censored likelihood, rather than assuming continuous ranks or silently removing small values.

## Fitting, uncertainty and numerical checks

- Evaluate log survival directly for very rare events; 1-CDF can cancel to zero. Use inverse survival when a high quantile is represented through its small exceedance probability.
- Enforce support during optimisation; an optimiser's success flag does not show global optimality or model adequacy.
- Support depends on parameters. Standard MLE normal/Wald arguments require regularity; GPD/GEV shapes near/below -1/2 need particular care. For GPD shape below -1 an unrestricted likelihood can be unbounded as an endpoint approaches an observation. A finite local optimum is not proof that a global MLE exists.
- A finite Hessian is not a proof of regular asymptotics. Profile and bootstrap intervals also need their own validity checks; neither is automatically calibrated.
- Re-estimated thresholds, occurrence rates, covariates and serial dependence contribute uncertainty. State what a reported interval holds fixed.
- Declustering changes the process being counted. Select a separation rule that matches the event and sensitivity-check it; moving windows and cluster peaks can merge distinct target-level runs.

## Arithmetic example: hourly losses

This standalone calculation illustrates the probability bridge, not a fitted-model validation. The event is loss>15 USD; threshold10 USD; excess scale2 USD; xi=-0.2. There are900 threshold exceedances in230000 valid hours. Calculate expected events over80000 valid hours:

```python
import math

def gpd_survival(excess, scale, xi):
    if not all(map(math.isfinite, (excess, scale, xi))) or excess < 0 or scale <= 0:
        raise ValueError("Require finite excess>=0, scale>0 and finite xi")
    if xi == 0:
        return math.exp(-excess / scale)
    if xi < 0 and 1 + xi * excess / scale <= 0:
        return 0.0
    return math.exp(-math.log1p(xi * excess / scale) / xi)

u, x, scale, xi = 10.0, 15.0, 2.0, -0.2
p_u, valid_hours = 900 / 230000, 80000
survival = gpd_survival(x - u, scale, xi)
expected = valid_hours * p_u * survival
assert math.isclose(survival, 1 / 32)
assert math.isclose(expected, 9.782608695652174)
assert gpd_survival(25 - u, scale, xi) == 0
assert math.isclose(gpd_survival(5, scale, 0), math.exp(-2.5))
```

For xi=0 use exp(-excess/scale); for x>=20 in this particular fitted model survival is0, not evidence that losses above20 are physically impossible. Under a365.25-day hourly convention there are8766 opportunities/year, not365.25. Do not multiply an annual-maxima probability by8766.

## Primary sources and scope

- Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*: [book](https://doi.org/10.1007/978-1-4471-3675-0). Framework for block maxima, threshold tails, estimation and diagnostics.
- Davison and Smith (1990), *Models for Exceedances over High Thresholds*: [paper](https://doi.org/10.1111/j.2517-6161.1990.tb01796.x). Threshold modelling, covariates and occurrence/severity separation.
- Smith (1985), *Maximum likelihood estimation in a class of nonregular cases*: [paper](https://doi.org/10.1093/biomet/72.1.67). Parameter-dependent support and nonregular inference.
- SciPy's maintained [GEV](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html) and [GPD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html) documentation: software parameter conventions, not scientific validation of a data fit.

These are independent summaries and mathematical identities, not copied source text. Check current package documentation for implementation details and the original paper for a specific asymptotic claim.
