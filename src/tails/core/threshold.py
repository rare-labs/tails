"""Threshold diagnostics and automated selection for peaks-over-threshold.

Everything is batched over candidate thresholds: exceedances of each
threshold are a *mask* on the same data vector, so fitting a GPD at 50
thresholds (or 50 thresholds × 200 bootstrap replicates) is one batched call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._jax import jax, jnp
from .distributions import gpd_cdf, gpd_ppf
from .fit import fit_gpd

__all__ = ["mean_residual_life", "parameter_stability", "select_threshold", "ThresholdSelection"]


def mean_residual_life(x, thresholds, alpha: float = 0.05):
    """Mean excess over each threshold with normal confidence bands.

    Linear in ``u`` above a valid GPD threshold. Returns a dict of arrays
    ``threshold``, ``mean_excess``, ``lower``, ``upper``, ``n_exceed``.
    """
    from scipy.stats import norm

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    u = np.asarray(thresholds, float)
    exc = x[None, :] - u[:, None]
    m = exc > 0
    n = m.sum(1)
    me = np.where(m, exc, 0).sum(1) / np.maximum(n, 1)
    var = np.where(m, (exc - me[:, None]) ** 2, 0).sum(1) / np.maximum(n - 1, 1)
    half = norm.ppf(1 - alpha / 2) * np.sqrt(var / np.maximum(n, 1))
    return {"threshold": u, "mean_excess": me, "lower": me - half, "upper": me + half, "n_exceed": n}


def parameter_stability(x, thresholds, alpha: float = 0.05):
    """GPD fits at many thresholds in one batched call.

    Returns modified scale ``sigma* = sigma - xi u`` and shape ``xi`` with Wald
    intervals; both should be constant above a valid threshold.
    """
    from scipy.stats import norm

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    u = np.asarray(thresholds, float)
    data = np.broadcast_to(x, (u.size, x.size))
    fit = fit_gpd(data, threshold=u)
    sc, xi = np.exp(fit.coef[:, 0]), fit.coef[:, 1]
    cov = fit.cov
    # delta method for sigma* = exp(phi) - xi u
    gphi, gxi = sc, -u
    var_s = gphi**2 * cov[:, 0, 0] + 2 * gphi * gxi * cov[:, 0, 1] + gxi**2 * cov[:, 1, 1]
    zq = norm.ppf(1 - alpha / 2)
    se_xi = np.sqrt(np.clip(cov[:, 1, 1], 0, None))
    se_s = np.sqrt(np.clip(var_s, 0, None))
    mstar = sc - xi * u
    return {
        "threshold": u,
        "shape": xi,
        "shape_lower": xi - zq * se_xi,
        "shape_upper": xi + zq * se_xi,
        "modified_scale": mstar,
        "modified_scale_lower": mstar - zq * se_s,
        "modified_scale_upper": mstar + zq * se_s,
        "n_exceed": fit.nobs,
        "converged": fit.converged,
    }


def _ad_stat(y, mask, scale, shape):
    """Anderson–Darling statistic of masked excesses (batched)."""
    n = mask.sum(-1)
    F = gpd_cdf(jnp.where(mask, y, 0.0), scale[..., None], shape[..., None])
    F = jnp.clip(F, 1e-12, 1 - 1e-12)
    F = jnp.where(mask, F, jnp.inf)
    Fs = jnp.sort(F, axis=-1)
    i = jnp.arange(1, y.shape[-1] + 1)
    valid = i <= n[..., None]
    Fs_rev = jnp.take_along_axis(Fs, jnp.clip(n[..., None] - i, 0, y.shape[-1] - 1), axis=-1)
    Fs = jnp.where(valid, Fs, 0.5)
    Fs_rev = jnp.where(valid, Fs_rev, 0.5)
    terms = (2 * i - 1) * (jnp.log(Fs) + jnp.log1p(-Fs_rev))
    return -n - jnp.sum(jnp.where(valid, terms, 0.0), -1) / n


@dataclass
class ThresholdSelection:
    """Result of :func:`select_threshold`."""

    threshold: float
    index: int
    thresholds: np.ndarray
    pvalues: np.ndarray
    statistics: np.ndarray
    n_exceed: np.ndarray
    forward_stop: np.ndarray


def select_threshold(x, thresholds, *, alpha: float = 0.05, n_boot: int = 199, seed: int = 0) -> ThresholdSelection:
    """Automated threshold choice with ForwardStop (Bader, Yan & Zhang, 2018).

    For each candidate threshold the GPD fit is tested with the
    Anderson–Darling statistic; p-values come from a *parametric bootstrap*
    (``n_boot`` refits per threshold, all done in one batched call — the
    expensive part of the original method, here typically < 1 s). ForwardStop
    then controls the false discovery rate of the ordered sequence of tests and
    returns the lowest threshold not rejected.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    u = np.sort(np.asarray(thresholds, float))
    K = u.size
    data = np.broadcast_to(x, (K, x.size))
    fit = fit_gpd(data, threshold=u)
    sc, xi = np.exp(fit.coef[:, 0]), fit.coef[:, 1]
    y = np.where(data > u[:, None], data - u[:, None], 0.0)
    m = data > u[:, None]
    stat = np.asarray(_ad_stat(jnp.asarray(y), jnp.asarray(m), jnp.asarray(sc), jnp.asarray(xi)))

    # parametric bootstrap: same number of exceedances, refit, recompute AD
    nk = m.sum(1)
    nmax = int(nk.max())
    key = jax.random.PRNGKey(seed)
    U = np.asarray(jax.random.uniform(key, (K, n_boot, nmax), minval=1e-12, maxval=1 - 1e-12))
    ys = np.asarray(gpd_ppf(U, sc[:, None, None], xi[:, None, None]))
    mb = np.arange(nmax)[None, None, :] < nk[:, None, None]
    ys = np.where(mb, ys, np.nan).reshape(K * n_boot, nmax)
    fb = fit_gpd(ys)
    sb = np.asarray(
        _ad_stat(
            jnp.asarray(np.where(np.isfinite(ys), ys, 0.0)),
            jnp.asarray(np.isfinite(ys)),
            jnp.asarray(np.exp(fb.coef[:, 0])),
            jnp.asarray(fb.coef[:, 1]),
        )
    ).reshape(K, n_boot)
    pv = (1 + (sb >= stat[:, None]).sum(1)) / (n_boot + 1)
    fs = -np.cumsum(np.log1p(-np.clip(pv, 0, 1 - 1e-12))) / np.arange(1, K + 1)
    rejected = np.flatnonzero(fs <= alpha)
    idx = int(rejected.max() + 1) if rejected.size else 0
    idx = min(idx, K - 1)
    return ThresholdSelection(float(u[idx]), idx, u, pv, stat, nk, fs)
