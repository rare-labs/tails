"""Closed-form first and second derivatives of GEV / GPD log-densities.

Generic ``jax.hessian`` of a summed likelihood costs 7-10x the likelihood on
CPU. These hand-derived expressions (verified against autodiff in the test
suite) are evaluated in one fused elementwise pass, making each Newton step
only ~3x the cost of a likelihood evaluation.

Parameterisation: GEV ``(mu, phi = log sigma, xi)``, GPD ``(phi, xi)``. With
``z`` the standardised value, ``x = xi z``, ``t = 1 + x`` and
``r(x) = log1p(x)/x`` we use ``L = log(t)/xi = z r(x)`` and the stable series
of ``r'(x)`` and ``r''(x)`` near ``x = 0``.
"""

from __future__ import annotations

from .._jax import jnp
from ._special import log1p_over_x

_TOL = 1e-3


def _r_derivs(x):
    small = jnp.abs(x) < _TOL
    xs = jnp.where(small, 0.5, x)
    t = 1.0 + xs
    r = log1p_over_x(xs)
    r1 = (1.0 / t - r) / xs
    r2 = (-1.0 / (t * t) - 2.0 * r1) / xs
    r1s = -0.5 + 2.0 * x / 3.0 - 0.75 * x * x + 0.8 * x**3 - 5.0 * x**4 / 6.0
    r2s = 2.0 / 3.0 - 1.5 * x + 2.4 * x * x - 10.0 * x**3 / 3.0 + 30.0 * x**4 / 7.0
    return jnp.where(small, r1s, r1), jnp.where(small, r2s, r2)


def gev_derivs(y, mu, phi, xi, stacked=True):
    """Return ``(lp, g, h)`` for the GEV log-density.

    ``g`` stacks first derivatives w.r.t. ``(mu, phi, xi)`` on the last axis;
    ``h`` is the ``(..., 3, 3)`` Hessian. Infeasible points give ``lp = -inf``
    and zero derivatives.
    """
    sigma = jnp.exp(phi)
    z = (y - mu) / sigma
    x = xi * z
    feas = (1.0 + x) > 0.0
    x = jnp.where(feas, x, 0.0)
    z = jnp.where(feas, z, 0.0)
    t = 1.0 + x
    lt = jnp.log1p(x)
    L = z * log1p_over_x(x)
    A = jnp.exp(-L)
    r1, r2 = _r_derivs(x)
    Lxi = z * z * r1
    Lxixi = z**3 * r2
    lp = -phi - lt - L - A

    lz = (A - 1.0 - xi) / t
    lzz = -(A + xi * (A - 1.0 - xi)) / (t * t)
    lxi = -z / t + (A - 1.0) * Lxi
    lxixi = z * z / (t * t) + (A - 1.0) * Lxixi - A * Lxi * Lxi
    lzxi = (-A * Lxi - 1.0) / t - (A - 1.0 - xi) * z / (t * t)

    g_mu = -lz / sigma
    g_phi = -1.0 - z * lz
    h_mumu = lzz / (sigma * sigma)
    h_muphi = (lzz * z + lz) / sigma
    h_phiphi = lzz * z * z + lz * z
    h_muxi = -lzxi / sigma
    h_phixi = -z * lzxi

    if not stacked:
        z0 = jnp.zeros_like(lp)
        w = lambda v: jnp.where(feas, v, z0)  # noqa: E731
        g = [w(g_mu), w(g_phi), w(lxi)]
        h = [[w(h_mumu), w(h_muphi), w(h_muxi)], [None, w(h_phiphi), w(h_phixi)], [None, None, w(lxixi)]]
        return jnp.where(feas, lp, -jnp.inf), g, h
    g = jnp.stack([g_mu, g_phi, lxi], axis=-1)
    h = jnp.stack(
        [
            jnp.stack([h_mumu, h_muphi, h_muxi], -1),
            jnp.stack([h_muphi, h_phiphi, h_phixi], -1),
            jnp.stack([h_muxi, h_phixi, lxixi], -1),
        ],
        -2,
    )
    lp = jnp.where(feas, lp, -jnp.inf)
    g = jnp.where(feas[..., None], g, 0.0)
    h = jnp.where(feas[..., None, None], h, 0.0)
    return lp, g, h


def gpd_derivs(y, phi, xi, stacked=True):
    """Return ``(lp, g, h)`` for the GPD log-density of excesses w.r.t. ``(phi, xi)``."""
    sigma = jnp.exp(phi)
    z = y / sigma
    x = xi * z
    feas = ((1.0 + x) > 0.0) & (y >= 0.0)
    x = jnp.where(feas, x, 0.0)
    z = jnp.where(feas, z, 0.0)
    t = 1.0 + x
    lt = jnp.log1p(x)
    L = z * log1p_over_x(x)
    r1, r2 = _r_derivs(x)
    Lxi = z * z * r1
    Lxixi = z**3 * r2
    lp = -phi - lt - L

    lz = -(1.0 + xi) / t
    lzz = (1.0 + xi) * xi / (t * t)
    lxi = -z / t - Lxi
    lxixi = z * z / (t * t) - Lxixi
    lzxi = -1.0 / t + (1.0 + xi) * z / (t * t)

    g_phi = -1.0 - z * lz
    h_phiphi = lzz * z * z + lz * z
    h_phixi = -z * lzxi
    if not stacked:
        z0 = jnp.zeros_like(lp)
        w = lambda v: jnp.where(feas, v, z0)  # noqa: E731
        return jnp.where(feas, lp, -jnp.inf), [w(g_phi), w(lxi)], [[w(h_phiphi), w(h_phixi)], [None, w(lxixi)]]
    g = jnp.stack([g_phi, lxi], axis=-1)
    h = jnp.stack([jnp.stack([h_phiphi, h_phixi], -1), jnp.stack([h_phixi, lxixi], -1)], -2)
    lp = jnp.where(feas, lp, -jnp.inf)
    g = jnp.where(feas[..., None], g, 0.0)
    h = jnp.where(feas[..., None, None], h, 0.0)
    return lp, g, h
