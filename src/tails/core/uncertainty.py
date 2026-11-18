"""Uncertainty quantification: profile likelihood and parametric bootstrap.

Both are batched: profile likelihoods for many series and return periods are
traced with a continuation ``lax.scan`` inside one ``vmap``; bootstrap
replicates of all series are refitted in a single batched optimisation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .._jax import jax, jnp
from ._engine import FAMILIES, back_transform_matrix, make_nll, make_vgh, natural_params
from ._special import expm1_over_x
from .distributions import gev_logpdf, gev_ppf
from .optim import _Settings, _single

__all__ = ["profile_return_level", "profile_fixed_shape_return_level", "bootstrap_coef"]


def _prof_nll(th, y, m, z, w):
    sc = jnp.exp(th[0])
    xi = th[1]
    mu = z - sc * w * expm1_over_x(xi * w)
    ysafe = jnp.where(m, y, mu)
    return -jnp.sum(jnp.where(m, gev_logpdf(ysafe, mu, sc, xi), 0.0))


_SET = _Settings(max_iter=100)


def _path(th0, y, m, zs, w):
    def step(prev, z):
        f_prev = _prof_nll(prev, y, m, z, w)
        alt = jnp.stack([prev[0], 0.0])
        start = jnp.where(jnp.isfinite(f_prev), prev, alt)
        r = _single(_prof_nll, start, (y, m, z, w), _SET)
        nxt = jnp.where(r.converged, r.x, prev)
        return nxt, jnp.where(r.converged, r.fun, jnp.nan)

    _, fs = jax.lax.scan(step, th0, zs)
    return fs


_path_batched = jax.jit(jax.vmap(_path, in_axes=(0, 0, 0, 0, 0)))


def _crossing(zs, dev, crit):
    """First crossing of ``dev = crit`` along each path (linear interpolation)."""
    above = dev > crit
    idx = np.argmax(above, axis=-1)
    found = above[np.arange(len(idx)), idx] & (idx > 0)
    i0 = np.clip(idx - 1, 0, None)
    r = np.arange(len(idx))
    d0, d1 = dev[r, i0], dev[r, idx]
    z0, z1 = zs[r, i0], zs[r, idx]
    frac = np.clip((crit - d0) / np.where(d1 != d0, d1 - d0, 1.0), 0, 1)
    out = z0 + frac * (z1 - z0)
    return np.where(found, out, np.nan)


def profile_return_level(fit, period, alpha=0.05, n_points=41):
    """Profile-likelihood intervals for stationary GEV return levels.

    Returns ``(lower, upper)`` arrays of shape ``(B, len(period))``. Bounds are
    NaN when the deviance does not cross the threshold inside the search grid
    (typically an extremely flat upper profile; widen with ``n_points``).
    """
    T = np.atleast_1d(np.asarray(period, float))
    B = int(np.prod(fit.batch_shape, dtype=int))
    rl = fit.return_level(T, ci="delta", alpha=alpha)
    est = rl.estimate.reshape(B, len(T))
    se = ((rl.upper - rl.lower) / 2.0 / 1.959963984540054).reshape(B, len(T))
    se = np.where(np.isfinite(se) & (se > 0), se, 0.1 * np.abs(est) + 1e-8)
    coef = fit.coef.reshape(B, 3)
    th0 = np.repeat(coef[:, None, 1:], len(T), axis=1).reshape(-1, 2)
    nll_hat = np.repeat(-fit.loglik.reshape(B)[:, None], len(T), axis=1).reshape(-1)
    n_dn = max(n_points // 3, 4)
    n_up = max(n_points - n_dn, 4)
    zs_up = est[..., None] + se[..., None] * np.linspace(0.0, 12.0, n_up)
    zs_dn = est[..., None] - se[..., None] * np.linspace(0.0, 6.0, n_dn)
    w = -np.log(-np.log(1.0 - 1.0 / T))
    w = np.broadcast_to(w, (B, len(T))).reshape(-1)
    y = np.repeat(fit._y[:, None, :], len(T), axis=1).reshape(B * len(T), -1)
    m = np.repeat(fit._mask[:, None, :], len(T), axis=1).reshape(B * len(T), -1)
    crit = chi2.ppf(1 - alpha, 1)
    out = []
    for zs in (zs_dn, zs_up):
        zs2 = zs.reshape(-1, zs.shape[-1])
        f = np.asarray(_path_batched(jnp.asarray(th0), jnp.asarray(y), jnp.asarray(m), jnp.asarray(zs2), jnp.asarray(w)))
        dev = 2.0 * (f - nll_hat[:, None])
        dev = np.where(np.isfinite(dev), dev, np.inf)
        out.append(_crossing(zs2, dev, crit).reshape(B, len(T)))
    return out[0], out[1]


def _fixed_prof_nll(th, y, m, z, w, xi):
    sc = jnp.exp(th[0])
    mu = z - sc * w * expm1_over_x(xi * w)
    ysafe = jnp.where(m, y, mu)
    return -jnp.sum(jnp.where(m, gev_logpdf(ysafe, mu, sc, xi), 0.0))


def _path_fixed(th0, y, m, zs, w, xi):
    def step(prev, z):
        r = _single(_fixed_prof_nll, prev, (y, m, z, w, xi), _SET)
        nxt = jnp.where(r.converged, r.x, prev)
        return nxt, jnp.where(r.converged, r.fun, jnp.nan)

    _, fs = jax.lax.scan(step, th0, zs)
    return fs


_path_fixed_batched = jax.jit(jax.vmap(_path_fixed, in_axes=(0, 0, 0, 0, 0, 0)))


def profile_fixed_shape_return_level(fit, period, alpha=0.05, n_points=41):
    """Profile-likelihood intervals for a stationary GEV with ξ held fixed.

    The deviance profiles the return level with location determined by the
    level and the scale free. Shape stays at the fitted constant, so a Gumbel
    fit (ξ = 0) is not quietly turned back into a three-parameter GEV.
    Returns ``(lower, upper)`` of shape ``(B, len(period))``.
    """
    T = np.atleast_1d(np.asarray(period, float))
    B = int(np.prod(fit.batch_shape, dtype=int))
    rl = fit.return_level(T, ci="delta", alpha=alpha)
    est = rl.estimate.reshape(B, len(T))
    se = ((rl.upper - rl.lower) / 2.0 / 1.959963984540054).reshape(B, len(T))
    se = np.where(np.isfinite(se) & (se > 0), se, 0.1 * np.abs(est) + 1e-8)
    coef = fit.coef.reshape(B, -1)
    if coef.shape[-1] != 3:
        raise ValueError("fixed-shape profile intervals are for the GEV")
    log_scale = np.repeat(coef[:, 1][:, None], len(T), axis=1).reshape(-1, 1)
    xi = np.repeat(coef[:, 2][:, None], len(T), axis=1).reshape(-1)
    nll_hat = np.repeat(-fit.loglik.reshape(B)[:, None], len(T), axis=1).reshape(-1)
    n_dn = max(n_points // 3, 4)
    n_up = max(n_points - n_dn, 4)
    zs_up = est[..., None] + se[..., None] * np.linspace(0.0, 12.0, n_up)
    zs_dn = est[..., None] - se[..., None] * np.linspace(0.0, 6.0, n_dn)
    w = -np.log(-np.log(1.0 - 1.0 / T))
    w = np.broadcast_to(w, (B, len(T))).reshape(-1)
    y = np.repeat(fit._y[:, None, :], len(T), axis=1).reshape(B * len(T), -1)
    m = np.repeat(fit._mask[:, None, :], len(T), axis=1).reshape(B * len(T), -1)
    crit = chi2.ppf(1 - alpha, 1)
    out = []
    for zs in (zs_dn, zs_up):
        zs2 = zs.reshape(-1, zs.shape[-1])
        f = np.asarray(
            _path_fixed_batched(
                jnp.asarray(log_scale),
                jnp.asarray(y),
                jnp.asarray(m),
                jnp.asarray(zs2),
                jnp.asarray(w),
                jnp.asarray(xi),
            )
        )
        dev = 2.0 * (f - nll_hat[:, None])
        dev = np.where(np.isfinite(dev), dev, np.inf)
        out.append(_crossing(zs2, dev, crit).reshape(B, len(T)))
    return out[0], out[1]


def bootstrap_coef(fit, n_boot=500, seed=0, chunk=50_000):
    """Parametric bootstrap of a fitted model's coefficients.

    Simulates ``n_boot`` datasets per series from the fitted model (keeping the
    missing-data pattern and covariates), refits them all in batched form and
    returns user-scale coefficients of shape ``(B, n_boot, p)``.
    """
    from .distributions import gpd_ppf

    fam = FAMILIES[fit.family]
    B = int(np.prod(fit.batch_shape, dtype=int))
    th = np.asarray(fit._theta_std).reshape(B, -1)
    ks = fit.ks
    Xs = [jnp.asarray(d.X) for d in fit._designs]
    axes = tuple(None if d.shared else 0 for d in fit._designs)
    nat = jax.vmap(lambda t, *X: natural_params(fam, t, X, ks), in_axes=(0, *axes))(jnp.asarray(th), *Xs)
    n = fit._y.shape[1]
    nat = [jnp.broadcast_to(v, (B, n)) for v in nat]
    key = jax.random.PRNGKey(seed)
    u = jax.random.uniform(key, (B, n_boot, n), minval=1e-12, maxval=1 - 1e-12)
    if fit.family == "gev":
        ysim = gev_ppf(u, nat[0][:, None, :], nat[1][:, None, :], nat[2][:, None, :])
    else:
        ysim = gpd_ppf(u, nat[0][:, None, :], nat[1][:, None, :])
    ysim = np.asarray(ysim).reshape(B * n_boot, n)
    mask = np.repeat(fit._mask, n_boot, axis=0)
    ysim = np.where(mask, ysim, 0.0)
    x0 = np.repeat(th, n_boot, axis=0)
    nll = make_nll(fit.family, ks, fit.prior)
    Xrep = [X if ax is None else jnp.repeat(X, n_boot, axis=0) for X, ax in zip(Xs, axes)]
    from .optim import minimize

    res_x = np.empty_like(x0)
    for s in range(0, x0.shape[0], chunk):
        sl = slice(s, s + chunk)
        Xc = [X if ax is None else X[sl] for X, ax in zip(Xrep, axes)]
        r = minimize(nll, jnp.asarray(x0[sl]), jnp.asarray(ysim[sl]), jnp.asarray(mask[sl]), *Xc, arg_axes=(0, 0, *axes), vgh=make_vgh(fit.family, ks, fit.prior))
        xs = np.array(r.x)
        xs[~np.asarray(r.converged)] = np.nan
        res_x[sl] = xs
    A = back_transform_matrix(fit._designs, B)
    A = np.repeat(A, n_boot, axis=0)
    coef = np.einsum("bij,bj->bi", A, res_x)
    return coef.reshape(B, n_boot, -1)
