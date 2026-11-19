"""r-largest order statistics: a GEV fit that uses the top ``r`` values of each block.

Block maxima throw away everything but one value per year. When a record is
short, the second, third, ... largest values of each year carry real
information about the same tail. Under the usual extreme-value limit, the
``r`` largest values ``z_1 >= ... >= z_r`` of a block have the joint density
(Coles, 2001, section 3.5)

.. math::

    \\exp\\{-t(z_r)\\} \\prod_{k=1}^{r} \\sigma^{-1} t(z_k)^{\\xi + 1},
    \\qquad t(z) = \\{1 + \\xi (z - \\mu) / \\sigma\\}^{-1/\\xi},

with the same ``(mu, sigma, xi)`` as the GEV of the block maxima. Written with
GEV pieces, the log-likelihood of a block is
``sum_k [log f(z_k) - log F(z_k)] + log F(z_r)``, which is what is coded
below. ``r = 1`` is exactly the GEV likelihood.

Blocks may hold fewer than ``r`` values (NaN-padded): each block then uses the
values it has, with ``z_r`` its smallest one. The values within a block should
be separate events (declustered), otherwise the independence the likelihood
assumes does not hold. :func:`block_top_r` extracts them from a datetime
series with a minimum separation between events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .._jax import jax, jnp
from ._engine import Design
from ._special import expm1_over_x
from .distributions import gev_logcdf, gev_logpdf
from .fit import GEVFit, ReturnLevels, _inv_hessian, fit_gev
from .optim import _Settings, _single, minimize

__all__ = ["RLargestFit", "fit_rlargest", "rlargest_loglik", "block_top_r"]


def rlargest_loglik(z, mask, loc, scale, shape):
    """Log-likelihood of one series of r-largest blocks.

    ``z`` has shape ``(n_blocks, r)`` with each row sorted in decreasing order
    and ``mask`` marks the values present (a prefix of each row).
    """
    zs = jnp.where(mask, z, loc)
    lp = gev_logpdf(zs, loc, scale, shape)
    lc = gev_logcdf(zs, loc, scale, shape)
    feasible = jnp.all(jnp.where(mask, jnp.isfinite(lp), True))
    lps = jnp.where(mask & feasible, lp, 0.0)
    lcs = jnp.where(mask & feasible, lc, 0.0)
    nxt = jnp.concatenate([mask[:, 1:], jnp.zeros_like(mask[:, :1])], axis=1)
    last = mask & ~nxt
    ll = jnp.sum(lps - lcs) + jnp.sum(jnp.where(last, lcs, 0.0))
    return jnp.where(feasible, ll, -jnp.inf)


def _nll(theta, z, mask):
    return -rlargest_loglik(z, mask, theta[0], jnp.exp(theta[1]), theta[2])


def _prof_nll(th, z, mask, level, w):
    sc = jnp.exp(th[0])
    xi = th[1]
    mu = level - sc * w * expm1_over_x(xi * w)
    return -rlargest_loglik(z, mask, mu, sc, xi)


_SET = _Settings(max_iter=100)


def _path(th0, z, mask, levels, w):
    def step(prev, level):
        f_prev = _prof_nll(prev, z, mask, level, w)
        alt = jnp.stack([prev[0], 0.0])
        start = jnp.where(jnp.isfinite(f_prev), prev, alt)
        r = _single(_prof_nll, start, (z, mask, level, w), _SET)
        nxt = jnp.where(r.converged, r.x, prev)
        return nxt, jnp.where(r.converged, r.fun, jnp.nan)

    _, fs = jax.lax.scan(step, th0, levels)
    return fs


_path_batched = jax.jit(jax.vmap(_path, in_axes=(0, 0, 0, 0, 0)))


def _prepare_blocks(data, r):
    """Sort each block in decreasing order, NaN last; keep the first ``r``."""
    a = np.asarray(data, dtype=np.float64)
    if a.ndim < 2:
        raise ValueError("data must have shape (*batch, n_blocks, r)")
    batch_shape, n, rmax = a.shape[:-2], a.shape[-2], a.shape[-1]
    if r is None:
        r = rmax
    r = int(r)
    if not 1 <= r <= rmax:
        raise ValueError(f"r must be between 1 and {rmax}")
    a = a.reshape(-1, n, rmax)
    srt = -np.sort(np.where(np.isfinite(a), -a, np.inf), axis=-1)[..., :r]
    mask = np.isfinite(srt)
    z = np.where(mask, srt, 0.0)
    return z, mask, batch_shape, n, r


@dataclass(repr=False)
class RLargestFit(GEVFit):
    """GEV parameters fitted to the r largest values of each block.

    The parameters, standard errors and return levels are those of the GEV of
    the block maxima, so everything a :class:`~tails.core.fit.GEVFit` reports
    means the same here. ``r`` is the number of order statistics used.
    ``plot_diagnostics`` compares the fitted GEV with the block maxima.
    """

    r: int = 1
    _z: np.ndarray | None = None
    _zmask: np.ndarray | None = None

    def return_level(self, period, *, covariates=None, ci=None, alpha=0.05, profile_points=41, **kwargs):
        """T-block return levels of the fitted GEV, optionally with intervals.

        ``ci`` is ``None``, ``"delta"`` or ``"profile"``. The profile is the
        r-largest likelihood with the return level in place of the location.
        """
        if covariates is not None:
            raise ValueError("r-largest fits are stationary; covariates are not supported")
        if ci == "bootstrap":
            raise ValueError("ci='bootstrap' is not available for r-largest fits; use 'delta' or 'profile'")
        if ci != "profile":
            return super().return_level(period, ci=ci, alpha=alpha)
        T = np.atleast_1d(np.asarray(period, dtype=np.float64))
        rl = super().return_level(T, ci="delta", alpha=alpha)
        lo, hi = _profile(self, T, rl, alpha, profile_points)
        shp = (*self.batch_shape, len(T))
        return ReturnLevels(T, rl.estimate, lo.reshape(shp), hi.reshape(shp), "profile", alpha)


def _profile(fit, T, rl, alpha, n_points):
    from .uncertainty import _crossing

    B = int(np.prod(fit.batch_shape, dtype=int))
    nT = len(T)
    est = rl.estimate.reshape(B, nT)
    se = ((rl.upper - rl.lower) / 2.0 / 1.959963984540054).reshape(B, nT)
    se = np.where(np.isfinite(se) & (se > 0), se, 0.1 * np.abs(est) + 1e-8)
    coef = fit.coef.reshape(B, 3)
    th0 = np.repeat(coef[:, None, 1:], nT, axis=1).reshape(-1, 2)
    nll_hat = np.repeat(-fit.loglik.reshape(B)[:, None], nT, axis=1).reshape(-1)
    n_dn = max(n_points // 3, 4)
    n_up = max(n_points - n_dn, 4)
    zs_up = est[..., None] + se[..., None] * np.linspace(0.0, 12.0, n_up)
    zs_dn = est[..., None] - se[..., None] * np.linspace(0.0, 6.0, n_dn)
    w = np.broadcast_to(-np.log(-np.log(1.0 - 1.0 / T)), (B, nT)).reshape(-1)
    z = np.repeat(fit._z[:, None], nT, axis=1).reshape(B * nT, *fit._z.shape[1:])
    m = np.repeat(fit._zmask[:, None], nT, axis=1).reshape(B * nT, *fit._zmask.shape[1:])
    crit = chi2.ppf(1 - alpha, 1)
    out = []
    for zs in (zs_dn, zs_up):
        zs2 = zs.reshape(-1, zs.shape[-1])
        f = np.asarray(_path_batched(jnp.asarray(th0), jnp.asarray(z), jnp.asarray(m), jnp.asarray(zs2), jnp.asarray(w)))
        dev = 2.0 * (f - nll_hat[:, None])
        dev = np.where(np.isfinite(dev), dev, np.inf)
        out.append(_crossing(zs2, dev, crit).reshape(B, nT))
    return out[0], out[1]


def fit_rlargest(data, r: int | None = None, *, max_iter: int = 200) -> RLargestFit:
    """Fit a GEV to the ``r`` largest values of each block, for one or many series.

    Parameters
    ----------
    data
        Array of shape ``(*batch, n_blocks, r_max)``: for each block (usually a
        year), its largest values. Order within a block does not matter; NaN
        pads blocks with fewer values. Values in a block should be separate
        events (see :func:`block_top_r`).
    r
        Number of order statistics to use per block (default: all ``r_max``).
        ``r = 1`` gives the same fit as :func:`tails.fit_gev` on the maxima.
    max_iter
        Newton iterations.

    Returns
    -------
    RLargestFit
        A GEV fit: ``params``, ``summary()``, ``return_level(T, ci="delta" |
        "profile")`` and ``plot_diagnostics()`` work as for ``fit_gev``.

    Notes
    -----
    A larger ``r`` gives narrower intervals only while the GEV limit still
    holds for the r-th largest value. Compare the estimates across ``r`` and
    keep the largest ``r`` at which they are stable (Coles, 2001, section 3.5).
    """
    z, mask, batch_shape, n, r = _prepare_blocks(data, r)
    B = z.shape[0]
    has = mask[:, :, 0]
    nblocks = has.sum(-1)
    if np.any(nblocks < 2):
        raise ValueError("every series needs at least 2 blocks with data")
    maxima = np.where(has, z[:, :, 0], np.nan)
    start = fit_gev(maxima, max_iter=max_iter)
    x0 = np.asarray(start.coef).reshape(B, 3)
    x0 = np.where(np.isfinite(x0), x0, 0.0)
    res = minimize(_nll, jnp.asarray(x0), jnp.asarray(z), jnp.asarray(mask), max_iter=max_iter)
    theta = np.asarray(res.x)
    cov = _inv_hessian(np.asarray(res.hess))
    one = Design(np.ones((n, 1)), [""], np.zeros(0), np.ones(0), True)
    return RLargestFit(
        family="gev",
        coef=theta.reshape(*batch_shape, 3),
        cov=cov.reshape(*batch_shape, 3, 3),
        loglik=(-np.asarray(res.fun)).reshape(batch_shape),
        converged=np.asarray(res.converged).reshape(batch_shape),
        n_iter=np.asarray(res.n_iter).reshape(batch_shape),
        nobs=mask.sum((-1, -2)).reshape(batch_shape),
        param_names=["loc", "log_scale", "shape"],
        batch_shape=batch_shape,
        method="mle",
        prior=None,
        _theta_std=theta.reshape(*batch_shape, 3),
        _y=np.where(has, z[:, :, 0], 0.0),
        _mask=has,
        _designs=[one, one, one],
        _raw=[np.ones((n, 1))] * 3,
        r=r,
        _z=z,
        _zmask=mask,
    )


def block_top_r(series, r: int, *, freq: str = "YS", min_separation=None, min_count: int | None = None):
    """The ``r`` largest separate events of each block of a datetime series.

    Parameters
    ----------
    series
        pandas Series with a DatetimeIndex (e.g. daily rainfall).
    r
        Number of values to keep per block.
    freq
        Block length as a pandas frequency (default ``"YS"``, calendar years).
    min_separation
        Minimum time between two kept values of the same block, as a pandas
        Timedelta or string (e.g. ``"3D"``). A value closer than this to a
        larger one already kept is skipped, so one storm counts once.
        Default: no separation.
    min_count
        Drop blocks with fewer non-missing observations than this.

    Returns
    -------
    pandas.DataFrame
        One row per block (indexed by block start) and columns ``1 .. r``
        holding the values in decreasing order, NaN where a block has fewer
        than ``r`` events. ``df.to_numpy()`` goes straight into
        :func:`fit_rlargest`.
    """
    import pandas as pd

    s = series.dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    if int(r) < 1:
        raise ValueError("r must be >= 1")
    gap = pd.Timedelta(min_separation) if min_separation is not None else None
    rows = {}
    for start, block in s.groupby(pd.Grouper(freq=freq)):
        if len(block) == 0 or (min_count is not None and len(block) < min_count):
            continue
        kept_t, kept_v = [], []
        for t, v in block.sort_values(ascending=False, kind="stable").items():
            if gap is not None and any(abs(t - k) < gap for k in kept_t):
                continue
            kept_t.append(t)
            kept_v.append(float(v))
            if len(kept_v) == r:
                break
        rows[start] = kept_v + [np.nan] * (r - len(kept_v))
    out = pd.DataFrame.from_dict(rows, orient="index", columns=list(range(1, r + 1)))
    out.index.name = "block"
    return out
