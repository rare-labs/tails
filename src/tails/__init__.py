"""tails: fast, batched extreme-value analysis in Python.

>>> import tails
>>> fit = tails.fit_gev(annual_maxima)          # one series or 100 000 at once
>>> fit.return_level([20, 100], ci="profile")

Submodules
----------
tails.core     GEV / GPD / POT models, thresholds, uncertainty, optimiser
tails.idf      Intensity-duration-frequency models (port of IDFCurves.jl)
tails.records  Distribution-free record statistics
tails.xarray   xarray / Dask wrappers for gridded and multi-station data
tails.xclim    registers backend="tails" on xclim frequency analysis
tails.bayes    Optional NumPyro posterior sampling
tails.datasets Bundled and downloadable real datasets (ECCC IDF and GeoMet)
"""

from . import (
    _jax,  # noqa: F401  (enables float64 in JAX)
    core,
    records,
)
from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .how_rare import HowRare, how_rare

__version__ = "0.1.0"

__all__ = ["core", "records", "__version__", "how_rare", "HowRare", *_core_all]


def __getattr__(name):  # lazy heavy submodules
    import importlib

    if name in {"idf", "xarray", "xclim", "bayes", "datasets"}:
        return importlib.import_module(f"tails.{name}")
    raise AttributeError(name)
