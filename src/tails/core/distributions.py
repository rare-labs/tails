"""Vectorised, autodiff-safe GEV, Gumbel and GPD distribution functions.

All functions broadcast over their arguments and are pure JAX, so they can be
``jit``-ed, ``vmap``-ed and differentiated to any order. The shape parameter
``xi`` follows the climate/hydrology sign convention (``xi > 0`` is a heavy,
Fréchet-type tail), as in R ``extRemes``/``ismev``, Julia ``Extremes.jl`` and
``Distributions.jl``. Note that ``scipy.stats.genextreme`` uses ``c = -xi``.
"""

from __future__ import annotations

from .._jax import jax, jnp
from ._special import expm1_over_x, log1p_over_x

__all__ = [
    "gev_logpdf",
    "gev_cdf",
    "gev_logcdf",
    "gev_ppf",
    "gev_rvs",
    "gev_support",
    "gpd_logpdf",
    "gpd_cdf",
    "gpd_ppf",
    "gpd_rvs",
]


# --------------------------------------------------------------------------- GEV
def _gev_L(y, loc, scale, shape):
    """Return (feasible, x, z, L) with L = log(1 + xi z)/xi (stable)."""
    z = (y - loc) / scale
    x = shape * z
    feasible = (1.0 + x) > 0.0
    xs = jnp.where(feasible, x, 0.0)
    zs = jnp.where(feasible, z, 0.0)
    L = zs * log1p_over_x(xs)
    return feasible, xs, z, L


def gev_logpdf(y, loc, scale, shape):
    """Log-density of the GEV(loc, scale, shape) distribution.

    Returns ``-inf`` outside the support. Exact Gumbel limit at ``shape = 0``.
    """
    feasible, xs, _, L = _gev_L(y, loc, scale, shape)
    val = -jnp.log(scale) - jnp.log1p(xs) - L - jnp.exp(-L)
    return jnp.where(feasible, val, -jnp.inf)


def gev_logcdf(y, loc, scale, shape):
    """Log of the GEV cumulative distribution function."""
    feasible, _, z, L = _gev_L(y, loc, scale, shape)
    below = (shape > 0) & ~feasible  # below the lower endpoint
    val = -jnp.exp(-L)
    return jnp.where(feasible, val, jnp.where(below, -jnp.inf, 0.0))


def gev_cdf(y, loc, scale, shape):
    """GEV cumulative distribution function."""
    return jnp.exp(gev_logcdf(y, loc, scale, shape))


def gev_ppf(p, loc, scale, shape):
    """GEV quantile function. ``p`` is the non-exceedance probability."""
    w = -jnp.log(-jnp.log(p))  # Gumbel reduced variate
    return loc + scale * w * expm1_over_x(shape * w)


def gev_support(loc, scale, shape):
    """Lower and upper endpoints of the GEV support."""
    endpoint = loc - scale / jnp.where(shape == 0, 1.0, shape)
    lower = jnp.where(shape > 0, endpoint, -jnp.inf)
    upper = jnp.where(shape < 0, endpoint, jnp.inf)
    return lower, upper


def gev_rvs(key, loc, scale, shape, size=None):
    """Draw GEV variates by inversion. ``size`` defaults to the broadcast shape."""
    if size is None:
        size = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale), jnp.shape(shape))
    u = jax.random.uniform(key, size, minval=1e-300, maxval=1.0)
    return gev_ppf(u, loc, scale, shape)


# --------------------------------------------------------------------------- GPD
def gpd_logpdf(y, scale, shape):
    """Log-density of the generalised Pareto distribution for excesses ``y >= 0``."""
    x = shape * y / scale
    feasible = ((1.0 + x) > 0.0) & (y >= 0.0)
    xs = jnp.where(feasible, x, 0.0)
    ys = jnp.where(feasible, y, 0.0)
    val = -jnp.log(scale) - jnp.log1p(xs) - (ys / scale) * log1p_over_x(xs)
    return jnp.where(feasible, val, -jnp.inf)


def gpd_cdf(y, scale, shape):
    """GPD cumulative distribution function for excesses ``y``."""
    x = shape * y / scale
    feasible = ((1.0 + x) > 0.0) & (y >= 0.0)
    xs = jnp.where(feasible, x, 0.0)
    ys = jnp.where(feasible, y, 0.0)
    val = -jnp.expm1(-(ys / scale) * log1p_over_x(xs))
    return jnp.where(feasible, val, jnp.where(y < 0, 0.0, 1.0))


def gpd_ppf(p, scale, shape):
    """GPD quantile function for excesses."""
    w = -jnp.log1p(-p)
    return scale * w * expm1_over_x(shape * w)


def gpd_rvs(key, scale, shape, size=None):
    """Draw GPD excesses by inversion."""
    if size is None:
        size = jnp.broadcast_shapes(jnp.shape(scale), jnp.shape(shape))
    u = jax.random.uniform(key, size, minval=0.0, maxval=1.0 - 1e-16)
    return gpd_ppf(u, scale, shape)
