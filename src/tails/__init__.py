"""tails: fast, batched extreme-value analysis in Python.

>>> import tails
>>> fit = tails.fit_gev(annual_maxima)          # one series or 100 000 at once
>>> fit.return_level([20, 100], ci="profile")
>>> ext = tails.fit_block_maxima(daily_series)  # datetime series → block maxima → table
>>> ext.summary()

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
from .conditional import ConditionalFit, ConditionalQuantiles, fit_conditional
from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .how_rare import HowRare, how_rare
from .joint import JointRarity, compound_series, how_rare_joint
from .series import SeriesExtremes, fit_block_maxima, fit_peaks

__version__ = "0.1.0"

__all__ = [
    "core",
    "records",
    "__version__",
    "how_rare",
    "HowRare",
    "JointRarity",
    "compound_series",
    "how_rare_joint",
    "ConditionalFit",
    "ConditionalQuantiles",
    "fit_conditional",
    "fit_block_maxima",
    "fit_peaks",
    "SeriesExtremes",
    "plot_return_level_map",  # noqa: F405  — loaded lazily in __getattr__
    *_core_all,
]


def __getattr__(name):  # lazy heavy submodules
    import importlib

    if name == "plot_return_level_map":
        from .plotting import plot_return_level_map

        return plot_return_level_map
    if name in {"idf", "xarray", "xclim", "bayes", "datasets"}:
        return importlib.import_module(f"tails.{name}")
    raise AttributeError(name)
