# Reproducing a published extreme-value method

## Establish a source contract

Identify the paper version/DOI, code commit, input data version/checksums, target definitions, time/exposure convention, score and reported reference outputs. Read the actual mathematics and code for the parts being reproduced; an abstract or a competition rank is insufficient. Do not invent unavailable benchmark truth.

Prepare a component-fidelity table with these columns:

| Source component and purpose | Source location | Implemented behaviour | Identical/altered/missing/unverified | Consequence for target/results |
|---|---|---|---|---|

Typical components are observation treatment, marginal bodies/tails, threshold selection, occurrence rates, covariates, residual model, dependence model, conditioning-anchor coverage, proposal/importance weights, temporal simulation/reordering, uncertainty refits, pooling weights and scoring.

Trace each resampled row's time/context through inverse transformation, ordering and refitting. Retaining the donor timestamp and mapping onto a fixed destination calendar are different procedures; both can be described loosely as probability-scale block resampling. Likewise check zero-length block handling, endpoint sampling, redraw policies and settings that differ between entry scripts.

A common model name such as GPD or Heffernan-Tawn is not algorithm equivalence. A replacement that simplifies computation may be useful, but it answers a separate model-comparison question. Match to reference outputs and documented numerical tolerance before labelling a reproduction. Separate exact reproduction, numerical port, scientifically modified implementation and partial adaptation.

## Example: LancasterConditional EVA2025 method

Primary source: Bratkova et al. (2026), [Estimating rare spatio-temporal precipitation events with conditional extremes](https://doi.org/10.1007/s10687-026-00548-5), published17October2026. Its paper reports joint second place. [Authors' code](https://github.com/Kika0/EVA25_STORM) is a separate versioned artifact; the inspected [model driver](https://github.com/Kika0/EVA25_STORM/blob/68ea65150427743b76bb500b2c0aab3da55dcf62/model_fit.R), [conditional/simulation functions](https://github.com/Kika0/EVA25_STORM/blob/68ea65150427743b76bb500b2c0aab3da55dcf62/conditional_model_fit.R) and [block implementation](https://github.com/Kika0/EVA25_STORM/blob/68ea65150427743b76bb500b2c0aab3da55dcf62/Block_Bootstrapping.R) are concrete code references, not evidence that this snapshot matches every paper setting; inspect and pin the relevant commit rather than assuming the latest branch equals the competition submission or the paper's refined analysis.

Verify these pieces against the particular version being reproduced:

| Piece | What to verify |
|---|---|
| Non-stationary marginal GPDs | Time-varying thresholds/scales and shape assumptions for each location; fitting quantile and occurrence treatment |
| Body/tail transformation | Empirical body, fitted tail, standard Laplace margins, inverse transformation and handling of atoms |
| Conditional extremes | Conditioning threshold and fitted models for conditioning locations, not only an arbitrary single anchor |
| Residual margins | Fitted asymmetric residual distribution and tail treatment, rather than automatically replacing with empirical vectors |
| Residual dependence | Vine-copula specification, fitting/selection and conditional simulation |
| Unconditional spatial simulations | Body/tail mixing, anchor selection and correction for overlapping extreme regions |
| Time ordering | The rank-based reordering step and which observed timing information it preserves |
| Uncertainty | Semi-parametric aligned block procedure, refitting of the complete specified model and simulation/parameter uncertainty |
| Ensemble pooling | What is averaged, with which weights; which parameters or simulations are pooled |
| Evaluation | Exact challenge targets and score; paper-versus-submission changes and benchmark access |

These pieces work as a system. Four separate seasonal GPDs plus a block sampler are not an implementation of the full method above. Testing a partial adaptation against a direct event-score GPD does not rank the source method. A successful result on moderate observable thresholds does not establish the competition's much rarer target accuracy.

## Stage evidence without expanding scope

Start with the source's executable pathway and a documented reference case. Identify unavailable dependencies/data and computational cost before starting large bootstrap/simulation jobs. Reuse authorised settings and small numerical cases to check a port. Do not replace missing components simply to obtain an executable result and still call it faithful reproduction.

When all requested evidence is gathered, report what matches, what differs and which claims remain untested. Ask about a material scientific change when needed; a negative or incomplete reproduction finding can complete the authorised task without launching another model search.
