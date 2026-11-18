# tails

**tails** is a Python library for extreme-value analysis. Use it when you want
return levels, return periods, or intensity–duration–frequency (IDF) curves
from one station or from thousands of series in a single call.

It fits GEV, GPD and peaks-over-threshold models (including non-stationary
ones), plus duration-dependent IDF models, with uncertainty that is cheap
enough to always report. The import is `tails`. The package name on PyPI is
`rare-tails` (`tails` was already taken).

[![CI](https://github.com/rare-labs/tails/actions/workflows/ci.yml/badge.svg)](https://github.com/rare-labs/tails/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rare-tails.svg)](https://pypi.org/project/rare-tails/)
[![Docs](https://img.shields.io/badge/docs-rare--labs.github.io-blue)](https://rare-labs.github.io/tails/)
[Docs](https://rare-labs.github.io/tails/) · [Quickstart](docs/quickstart.md) · [Québec City IDF tutorial](docs/tutorial_quebec_idf.md) ·
[Landscape](docs/landscape.md)

## Install

```bash
pip install rare-tails
pip install "rare-tails[xarray,plots]"   # extras: xarray, plots, bayes (NumPyro), xclim, dev, docs, all

# current main, including changes that are not in a release yet
pip install "rare-tails[xarray,plots] @ git+https://github.com/rare-labs/tails"
```

Python ≥ 3.10. Runs on CPU; the same code runs on GPU with a CUDA build of `jax`.

## A first fit

`annual_maxima` can be one series `(n,)` or a stack `(*batch, n)`. The IDF
example uses real bundled ECCC data for Québec City (station `701S001`).

```python
import tails
fit = tails.fit_gev(annual_maxima)               # (n,) one series, or (*batch, n): thousands at once
fit.return_level([20, 100], ci="profile")        # profile-likelihood intervals, batched too

from tails.idf import IDFData, fit_idf           # IDF curves (port of IDFCurves.jl)
idf = fit_idf(IDFData.from_eccc("701S001"), "general", "matern")   # Québec City, real ECCC data
idf.quantile_ci([1.0], [100], method="bootstrap", n_boot=500)
```

A copy-pasteable version on the same Québec City 24-hour record (64 years) is
in the [Quickstart](docs/quickstart.md). That GEV has location 51.66 mm and
shape ξ ≈ −0.026; the 100-year 24-hour level is 102.1 mm (95% profile
interval 90.5–131.1 mm). The wettest year in the record, 105.4 mm in 2005,
has a return period of about 138 years.

## When should I use tails rather than pyextremes, scipy or xclim?

- **scipy.stats.genextreme / genpareto** are excellent distributions, not an
  extreme-value workflow. They fit one series with MLE and do not compute a
  100-year return-level interval. tails uses the same models (note: scipy’s
  GEV shape `c` has the *opposite* sign of the usual ξ), plus profile and
  bootstrap intervals, and can fit thousands of series in one call.
- **pyextremes** is a friendly univariate package, and its diagnostic figures
  for one site are worth using when that is the whole job. tails covers the
  extract-and-summarise steps for one datetime series as well
  (`fit_block_maxima`, `fit_peaks`, with an optional AIC choice between the
  GEV and the Gumbel model, shape ξ fixed at 0) and then the parts that go
  further: covariates, thousands of series in one call, profile-likelihood
  intervals, and IDF curves.
- **xclim** is where a lot of Canadian climate work already lives. After
  `import tails.xclim`, the same `frequency_analysis` call takes
  `backend="tails"` and `ci="profile"` (also `"delta"` or `"bootstrap"`).
  On xclim 1.0 the function is `xclim.compute.stats.frequency_analysis`;
  on xclim 0.x it is `xclim.indices.stats.frequency_analysis`. The fit is
  one batched call. The parameter array stays in scipy order, with
  genextreme `c` equal to −ξ. Non-stationary models stay on `fit_gev`.
  Station IDF curves stay on `tails.idf`; a DataArray of intensities is
  `tails.xarray.fit_idf`.

A wider comparison is in [docs/landscape.md](docs/landscape.md). A worked IDF
example on the same Québec City station is in
[docs/tutorial_quebec_idf.md](docs/tutorial_quebec_idf.md).

## What is in v0.1

- **`tails.core`** — GEV, GPD and POT (rate × GPD) maximum likelihood and GMLE;
  covariates on every parameter; L-moment starts; return levels with delta /
  profile / bootstrap intervals; mean residual life, parameter stability and
  automated threshold selection (ForwardStop with Anderson–Darling);
  block maxima, runs declustering and the Ferro–Segers extremal index.
- **`tails.idf`** — intensity–duration–frequency models ported from
  [IDFCurves.jl](https://github.com/JuliaExtremes/IDFCurves.jl) (see
  [NOTICE](NOTICE)): simple scaling and d-GEV, Gaussian and Student-t copulas,
  the scaling-model test, Wald / sandwich and bootstrap intervals. Station
  batches fit in one call.
- **`tails.records`**, **`tails.xarray`**, **`tails.bayes`** (NumPyro),
  **`tails.datasets`** (ECCC).

On the three stations shipped with IDFCurves.jl, `tails.idf` reproduces the
Julia estimates typically to 1e-9–1e-12 (see
[docs/validation.md](docs/validation.md)).

## Credits

`tails.idf` is a port of **IDFCurves.jl** (MIT, commit `66a34d1`); its models,
initialisation, interval formulas and scaling test follow that implementation,
and the bundled example CSVs come from it. See [NOTICE](NOTICE) and
[docs/credits.md](docs/credits.md). Québec data: Environment and Climate
Change Canada, Engineering Climate Datasets IDF v3.40, Open Government
Licence – Canada.

## Licence

MIT © 2026 Charles-Auguste Marois and rare-labs contributors.
