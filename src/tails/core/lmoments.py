"""Batched sample L-moments and L-moment (PWM) estimators.

Used for robust starting values and as a fast estimator in their own right
(Hosking, 1990; Hosking & Wallis, 1997). Missing values are encoded as NaN and
handled per series, so ragged records can be processed in one call.
"""

from __future__ import annotations

from functools import partial

from jax.scipy.special import gammaln

from .._jax import jax, jnp

__all__ = ["sample_lmoments", "gev_from_lmoments", "gpd_from_lmoments"]

_EULER = 0.5772156649015329


@partial(jax.jit, static_argnames=("axis",))
def sample_lmoments(y, axis=-1):
    """Unbiased sample L-moments ``(l1, l2, t3, t4)`` along ``axis`` (NaN-aware).

    Returns an array with a trailing dimension of size 4: ``l1``, ``l2``,
    L-skewness ``t3`` and L-kurtosis ``t4``.
    """
    y = jnp.moveaxis(jnp.asarray(y, dtype=jnp.float64), axis, -1)
    valid = jnp.isfinite(y)
    n = jnp.sum(valid, axis=-1, keepdims=True).astype(y.dtype)
    xs = jnp.sort(jnp.where(valid, y, jnp.inf), axis=-1)
    m = y.shape[-1]
    i = jnp.arange(m, dtype=y.dtype)  # 0-based rank among valid values
    keep = i < n
    xs = jnp.where(keep, xs, 0.0)

    def w(r):
        num = jnp.ones_like(i)
        den = jnp.ones_like(n)
        for k in range(1, r + 1):
            num = num * (i - k + 1)
            den = den * (n - k)
        return jnp.where(keep, num, 0.0) / den

    b0 = jnp.sum(xs, -1) / n[..., 0]
    b1 = jnp.sum(w(1) * xs, -1) / n[..., 0]
    b2 = jnp.sum(w(2) * xs, -1) / n[..., 0]
    b3 = jnp.sum(w(3) * xs, -1) / n[..., 0]
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return jnp.stack([l1, l2, l3 / l2, l4 / l2], axis=-1)


@jax.jit
def gev_from_lmoments(lm):
    """GEV ``(loc, scale, shape)`` from L-moments (Hosking, Wallis & Wood 1985).

    Shape uses the ``xi > 0`` heavy-tail convention.
    """
    l1, l2, t3 = lm[..., 0], lm[..., 1], lm[..., 2]
    c = 2.0 / (3.0 + t3) - jnp.log(2.0) / jnp.log(3.0)
    k = 7.8590 * c + 2.9554 * c * c
    k = jnp.clip(k, -0.95, 0.95)
    small = jnp.abs(k) < 1e-6
    ks = jnp.where(small, 1e-6, k)
    g = jnp.exp(gammaln(1.0 + ks))
    scale_k = l2 * ks / ((1.0 - 2.0 ** (-ks)) * g)
    loc_k = l1 - scale_k * (1.0 - g) / ks
    scale_g = l2 / jnp.log(2.0)
    loc_g = l1 - _EULER * scale_g
    scale = jnp.where(small, scale_g, scale_k)
    loc = jnp.where(small, loc_g, loc_k)
    return jnp.stack([loc, scale, -k], axis=-1)


@jax.jit
def gpd_from_lmoments(lm):
    """GPD ``(scale, shape)`` for excesses from L-moments (Hosking & Wallis 1987)."""
    l1, l2 = lm[..., 0], lm[..., 1]
    shape = 2.0 - l1 / l2
    shape = jnp.clip(shape, -0.9, 0.9)
    scale = l1 * (1.0 - shape)
    return jnp.stack([scale, shape], axis=-1)
