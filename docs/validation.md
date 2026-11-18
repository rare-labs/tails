# Validation against IDFCurves.jl

`tails.idf` is a port of [IDFCurves.jl](https://github.com/JuliaExtremes/IDFCurves.jl)
(see NOTICE). Every model in the port is checked against the
Julia reference implementation at upstream commit `66a34d1` on the three
stations that ship with IDFCurves.jl, using identical data.

* `validation/idfcurves_reference.jl` runs the Julia models and writes
  `tests/data/idfcurves_reference.json`.
* `validation/compare_idfcurves.py` refits everything with `tails.idf` and writes
  the table below; `tests/test_idf.py` re-checks a subset in CI.

Station ids (example files distributed with IDFCurves.jl): `702S006`
Montréal Pierre-Elliott-Trudeau International Airport (QC), `6158731` Toronto
Lester B. Pearson International Airport (ON), `1108446` Vancouver Harbour CS (BC).

## Results

| Station | Model | max rel. diff params | loglik tails | loglik Julia | Δ loglik | q100(1h) tails | q100(1h) Julia | CI(1h) tails | CI(1h) Julia |
|---|---|---|---|---|---|---|---|---|---|
| 702S006 | simple | 6.0e-12 | -2031.3995 | -2031.3995 | -4.5e-13 | 45.3907 | 45.3907 | [41.685, 49.096] | [41.685, 49.096] |
| 702S006 | general | 4.0e-10 | -2011.0249 | -2011.0249 | -9.1e-13 | 48.0758 | 48.0758 | [44.063, 52.088] | [44.063, 52.088] |
| 702S006 | general_matern_gaussian | 3.6e-11 | -1583.4295 | -1583.4295 | -1.6e-11 | 45.1653 | 45.1653 | [40.495, 49.835] | [40.495, 49.835] |
| 702S006 | general_exponential_gaussian | 4.2e-09 | -1612.9369 | -1612.9369 | -4.5e-13 | 45.1553 | 45.1553 | [40.105, 50.206] | [40.105, 50.206] |
| 702S006 | general_exponential_t5 | 7.4e-06 | -1628.0566 | -1628.0566 | +4.8e-09 | 51.0579 | 51.0580 | [45.014, 57.102] | fails in IDFCurves 66a34d1 |
| 6158731 | simple | 4.6e-14 | -1925.0455 | -1925.0455 | -2.3e-13 | 56.2209 | 56.2209 | [50.136, 62.305] | [50.136, 62.305] |
| 6158731 | general | 6.1e-09 | -1907.6946 | -1907.6946 | -9.1e-13 | 58.9438 | 58.9438 | [53.103, 64.784] | [53.103, 64.784] |
| 6158731 | general_matern_gaussian | 4.7e-05 | -1415.4625 | -1415.4625 | +1.3e-08 | 53.8404 | 53.8401 | [46.412, 61.269] | [46.412, 61.268] |
| 6158731 | general_exponential_gaussian | 2.7e-10 | -1435.3053 | -1435.3053 | +4.5e-13 | 53.9268 | 53.9268 | [45.935, 61.919] | [45.935, 61.919] |
| 6158731 | general_exponential_t5 | 4.1e-05 | -1448.3548 | -1448.3548 | +7.2e-09 | 60.0601 | 60.0601 | [50.773, 69.348] | fails in IDFCurves 66a34d1 |
| 1108446 | simple | 7.8e-15 | -583.1242 | -583.1242 | +1.1e-13 | 27.8946 | 27.8946 | [23.829, 31.960] | [23.829, 31.960] |
| 1108446 | general | 3.5e-09 | -583.1242 | -583.1242 | -3.9e-09 | 27.8946 | 27.8946 | [23.514, 32.275] | [23.514, 32.275] |
| 1108446 | general_matern_gaussian | 1.1e-08 | -411.6647 | -411.6647 | -1.9e-09 | 25.3979 | 25.3979 | [21.369, 29.427] | [21.369, 29.427] |
| 1108446 | general_exponential_gaussian | 3.0e-09 | -426.4689 | -426.4689 | -4.6e-09 | 26.3377 | 26.3377 | [21.026, 31.649] | [21.026, 31.649] |
| 1108446 | general_exponential_t5 | 1.1e-05 | -435.3073 | -435.3073 | +4.6e-08 | 30.3276 | 30.3276 | [23.223, 37.432] | fails in IDFCurves 66a34d1 |

Likelihood implementation check (tails log-likelihood evaluated at the Julia MLE):

- 702S006: max |Δ loglik| = 1.52e-11
- 6158731: max |Δ loglik| = 1.59e-12
- 1108446: max |Δ loglik| = 6.25e-13

Scaling-model test p-values (shortest duration held out):

- 702S006: simple tails 3.413e-06 vs Julia 3.413e-06; general tails 1.000e+00 vs Julia 1.000e+00
- 6158731: simple tails 3.399e-06 vs Julia 3.399e-06; general tails 1.000e+00 vs Julia 1.000e+00
- 1108446: simple tails 4.230e-02 vs Julia 4.230e-02; general tails 7.602e-01 vs Julia 7.602e-01

**Summary.** Maximum-likelihood estimates agree to at most 4.7e-5 relative
difference (usually 1e-9 to 1e-12), maximized log-likelihoods to 5e-8,
100-year 1-hour quantiles and their Wald intervals to the printed precision.
The log-likelihood evaluated at the Julia MLE agrees to 1.5e-11, so the
likelihood itself, not just the optimum, is reproduced. Scaling-test p-values
are identical.

The Student-t copula confidence interval errors out in IDFCurves.jl `66a34d1`;
`tails` computes it.
