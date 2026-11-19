"""Public entry points: one per model, two model families underneath.

* **GEV family** — :func:`fit_gev`. Block maxima by default; ``r=k`` uses the
  ``k`` largest values of each block (r-largest order statistics; extract
  them with :func:`~tails.core.blocks.block_top_r`).
* **GPD family** — :func:`fit_gpd` fits the excesses only; :func:`fit_pot`
  adds the threshold and the exceedance rate (a Bernoulli component that can
  take covariates), which gives return levels in years.

Each is a thin wrapper around :func:`tails.core.model.fit_model`; the fit
objects share one implementation (:class:`~tails.core.model.ModelFit`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._jax import jnp
from ._engine import Spec
from .model import ModelFit, ReturnLevels, _evaluate, check_method, fit_model, inv_hessian, prepare

__all__ = ["fit_gev", "fit_gpd", "fit_pot", "GEVFit", "GPDFit", "POTFit", "ReturnLevels"]

_inv_hessian = inv_hessian  # backward-compatible private alias


# --------------------------------------------------------------------- GEV
@dataclass(repr=False)
class GEVFit(ModelFit):
    """Fitted GEV model (block maxima, or the r largest values of each block).

    ``r`` is the number of order statistics per block (1 for block maxima).
    """

    r: int = 1


def _prepare_blocks(data, r):
    """``(*batch, n_blocks, r_max)`` → values sorted decreasing (NaN last), first ``r`` kept."""
    a = np.asarray(data, dtype=np.float64)
    if a.ndim < 2:
        raise ValueError("r-largest data must have shape (*batch, n_blocks, r_max): one row of top values per block")
    batch_shape, n, rmax = a.shape[:-2], a.shape[-2], a.shape[-1]
    r = rmax if r is None else int(r)
    if not 1 <= r <= rmax:
        raise ValueError(f"r must be between 1 and {rmax} (the number of values given per block)")
    srt = -np.sort(np.where(np.isfinite(a.reshape(-1, n, rmax)), -a.reshape(-1, n, rmax), np.inf), axis=-1)[..., :r]
    mask = np.isfinite(srt)
    return np.where(mask, srt, 0.0), mask, batch_shape, n, r


def _gev(data, r, covs, mask, method, prior, max_iter, fixed_shape):
    if r is None:
        y2, m2, batch_shape, n = prepare(data, mask)
        z, zm, rr, n_obs = y2, m2, 1, None
    else:
        if mask is not None:
            raise ValueError("mask is not supported with r-largest blocks; use NaN for missing values")
        z, zm, batch_shape, n, rr = _prepare_blocks(data, r)
        y2, m2 = z[..., 0], zm[..., 0]
        n_obs = zm.sum((-1, -2))
        if rr == 1:
            z, zm = y2, m2
    out = fit_model("gev", z, zm, batch_shape, covs, method=method, prior=check_method(method, prior),
                    max_iter=max_iter, fixed={"shape": fixed_shape}, start_values=np.where(m2, y2, np.nan), n_obs=n_obs)
    return dict(out, _y=y2, _mask=m2, r=rr)


def fit_gev(
    data,
    *,
    loc=None,
    scale=None,
    shape=None,
    mask=None,
    method: str = "mle",
    prior: str | None = None,
    max_iter: int = 200,
    fixed_shape: float | None = None,
    r: int | None = None,
) -> GEVFit:
    """Fit GEV distributions to one or many series of block maxima at once.

    Parameters
    ----------
    data
        Array of shape ``(*batch, n)``; the last axis holds the block maxima of
        each series. NaN marks missing values (ragged records are fine). With
        ``r``, shape ``(*batch, n_blocks, r_max)``: the largest values of each
        block (any order, NaN-padded), e.g. from :func:`tails.block_top_r`.
    loc, scale, shape
        Optional covariates for each parameter: ``(n,)``/``(n, k)`` shared by
        all series, or ``(*batch, n, k)`` per series (one row per block). A
        pandas object keeps its column names. Links: identity (loc, shape),
        log (scale).
    mask
        Optional boolean array broadcastable to ``data``; False excludes a value.
    method
        ``"mle"`` (default), ``"gmle"`` (MLE with the Martins–Stedinger
        Beta(6, 9) prior on the shape, robust for short records) or
        ``"lmom"`` (L-moments, stationary only).
    fixed_shape
        Hold ξ at this value instead of estimating it. ``0`` is the Gumbel
        model. Location and scale may still take covariates. AIC and BIC
        count the free parameters only; the shape is still reported in
        ``params`` and ``coef``.
    r
        Use the ``r`` largest values of each block (r-largest order
        statistics, Coles 2001 §3.5). The parameters are still those of the
        GEV of the block maxima; ``r = 1`` is the block-maxima fit.

    Returns
    -------
    GEVFit
        Coefficients, covariance, log-likelihood, convergence flags and
        methods for return levels with uncertainty.

    Examples
    --------
    >>> import numpy as np, tails
    >>> y = np.random.default_rng(1).gumbel(30, 8, size=(1000, 60))
    >>> fit = tails.fit_gev(y)            # 1000 stations in one call
    >>> fit.params["shape"].shape
    (1000,)
    """
    return GEVFit(**_gev(data, r, {"loc": loc, "scale": scale, "shape": shape}, mask, method, prior, max_iter, fixed_shape))


# --------------------------------------------------------------------- GPD
@dataclass(repr=False)
class GPDFit(ModelFit):
    """Fitted GPD model of threshold excesses (``threshold`` is ``None`` for raw excesses).

    Without the exceedance rate, ``return_level(T)`` is the level exceeded
    once every ``T`` exceedances. Use :func:`fit_pot` for return levels in years.
    """

    threshold: Any = None


@dataclass(repr=False)
class POTFit(GPDFit):
    """A GPD fit with the exceedance-rate component on (peaks over threshold).

    Same model and methods as :class:`GPDFit`. ``coef``, ``cov``, ``loglik``
    and ``n_params`` describe the GPD of the excesses, as before; the rate
    has its own ``rate_coef`` / ``rate_cov`` (logit of ``zeta``). ``zeta`` is
    P(X > u) and ``rate`` the exceedances per year (``zeta * npy``), per
    observation when the rate has covariates.
    """

    rate: np.ndarray | None = None  # exceedances per year
    zeta: np.ndarray | None = None  # P(X > u)
    n_total: np.ndarray | None = None
    npy: float = 365.25
    rate_coef: np.ndarray | None = None
    rate_cov: np.ndarray | None = None
    loglik_rate: np.ndarray | None = None


def _excesses(data, threshold):
    x = np.asarray(data, dtype=np.float64)
    u = np.asarray(threshold, dtype=np.float64)
    if u.ndim and u.shape != x.shape:
        u = u[..., None] if u.shape == x.shape[:-1] else np.broadcast_to(u, x.shape)
    y = x - u
    return np.where(np.isfinite(x) & (y > 0), y, np.nan)


def fit_gpd(
    data,
    threshold=None,
    *,
    scale=None,
    shape=None,
    mask=None,
    method: str = "mle",
    prior: str | None = None,
    max_iter: int = 200,
    fixed_shape: float | None = None,
) -> GPDFit:
    """Fit generalised Pareto distributions to threshold excesses (batched).

    If ``threshold`` is given, ``data`` are raw observations and only values
    above the threshold are used (as excesses). The threshold may be a scalar,
    one value per series (``batch`` shape) or one per observation (time-varying
    thresholds). Otherwise ``data`` are already excesses (NaN = unused).
    ``scale`` and ``shape`` take covariates with one row per observation.

    ``fixed_shape`` holds ξ. ``0`` is the exponential distribution of excesses.
    """
    y = _excesses(data, threshold) if threshold is not None else np.asarray(data, float)
    y2, m2, batch_shape, n = prepare(y, mask)
    out = fit_model("gpd", y2, m2, batch_shape, {"scale": scale, "shape": shape}, method=method,
                    prior=check_method(method, prior), max_iter=max_iter, fixed={"shape": fixed_shape},
                    start_values=np.where(m2, y2, np.nan))
    return GPDFit(**out, _y=y2, _mask=m2, threshold=threshold)


def fit_pot(
    data,
    threshold,
    *,
    npy: float = 365.25,
    decluster: int | None = None,
    scale=None,
    shape=None,
    rate=None,
    method: str = "mle",
    prior: str | None = None,
    max_iter: int = 200,
    fixed_shape: float | None = None,
) -> POTFit:
    """Peaks-over-threshold analysis of (batched) series: GPD excesses + exceedance rate.

    Parameters
    ----------
    data
        ``(*batch, n)`` observations (e.g. daily values); NaN = missing.
    threshold
        Scalar or one value per series.
    npy
        Observations per year (365.25 for daily data), or per season/block.
    decluster
        If given, apply runs declustering with this run length before fitting
        (cluster maxima only).
    scale, shape
        Covariates of the GPD, one row per observation (``(n,)``, ``(n, k)`` or
        ``(*batch, n, k)``). Only the rows of exceedances enter the GPD.
    rate
        Covariates of the exceedance probability (logit link), one row per
        observation, e.g. a season or temperature index.
    fixed_shape
        Hold the GPD shape ξ. ``0`` is an exponential excess model.
    """
    x = np.asarray(data, dtype=np.float64)
    valid = np.isfinite(x)
    if decluster is not None:
        from .blocks import decluster_runs

        x = decluster_runs(x, threshold, run_length=decluster)
    exc = _excesses(x, threshold)
    hit = np.isfinite(exc)
    y2, v2, batch_shape, n = prepare(np.where(hit, exc, 0.0), valid)
    e2 = hit.reshape(-1, n)
    out = fit_model("pot", y2, v2, batch_shape, {"scale": scale, "shape": shape, "rate": rate}, method=method,
                    prior=check_method(method, prior), max_iter=max_iter, fixed={"shape": fixed_shape},
                    start_values=np.where(e2, y2, np.nan))
    return _pot_from_fields(out, y2, e2, v2, threshold, npy)


def _pot_from_fields(out, y2, e2, v2, threshold, npy):
    """Split a joint GPD + rate fit into the backward-compatible POT fields."""
    bs, spec = out["batch_shape"], out["_spec"]
    kr = spec.ks[2]
    pg = spec.p - kr
    # Log-likelihood of the excesses alone: what loglik and AIC have always reported.
    designs = out["_designs"][:2]
    th = np.asarray(out["_theta_std"]).reshape(-1, spec.p)[:, :pg]
    Xs = [jnp.asarray(d.X) for d in designs]
    axes = tuple(None if d.shared else 0 for d in designs)
    ll_gpd = -_evaluate(Spec("gpd", spec.ks[:2]), th, y2, e2, Xs, axes).reshape(bs)
    fp = out["free_params"]
    fields = dict(
        out,
        coef=out["coef"][..., :pg],
        cov=out["cov"][..., :pg, :pg],
        loglik=ll_gpd,
        param_names=out["param_names"][:pg],
        free_params=None if fp is None else fp - kr,
    )
    fit = POTFit(**fields, _y=y2, _mask=e2, threshold=threshold, npy=float(npy), n_total=v2.sum(-1).reshape(bs),
                 rate_coef=out["coef"][..., pg:], rate_cov=out["cov"][..., pg:, pg:], loglik_rate=out["loglik"] - ll_gpd)
    z = np.asarray(fit._nat()[2])
    fit.zeta = z[:, 0].reshape(bs) if kr == 1 else z.reshape(*bs, -1)
    fit.rate = fit.zeta * float(npy)
    return fit
