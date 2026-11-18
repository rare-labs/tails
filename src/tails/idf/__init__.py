"""Intensity–duration–frequency (IDF) curves.

A Python/JAX port of `IDFCurves.jl
<https://github.com/JuliaExtremes/IDFCurves.jl>`_ (see the repository NOTICE),
extended with batched multi-station fitting, exact handling of missing
durations, Godambe/sandwich and parametric bootstrap intervals.
"""

from .data import IDFData, stack
from .models import IDFFit, IDFModel, fit_idf, loglikelihood
from .scalingtest import scaling_test

__all__ = ["IDFData", "IDFFit", "IDFModel", "fit_idf", "loglikelihood", "scaling_test", "stack"]
