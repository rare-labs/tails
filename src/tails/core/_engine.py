"""Generic batched likelihood engine for GEV / GPD regression models.

A model has one linear predictor per distribution parameter,
``eta_j = X_j @ beta_j`` with an identity link for location and shape and a log
link for scale. Stationary models are the special case ``X_j = 1``.

Internally covariates are centred and scaled (better conditioning, intercepts
that mean "parameter at the average covariate"); coefficients and covariances
are mapped back to the user's covariate scale before being returned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np

from .._jax import jax, jnp
from .distributions import gev_logpdf, gpd_logpdf

LINKS = {"identity": (lambda e: e), "log": jnp.exp}


@dataclass(frozen=True)
class Family:
    name: str
    params: tuple[str, ...]
    links: tuple[str, ...]
    logpdf: Callable


GEV = Family("gev", ("loc", "scale", "shape"), ("identity", "log", "identity"), gev_logpdf)
GPD = Family("gpd", ("scale", "shape"), ("log", "identity"), gpd_logpdf)
FAMILIES = {"gev": GEV, "gpd": GPD}


# ----------------------------------------------------------------- shape priors
def _martins_stedinger(xi):
    """Log Beta(6, 9) prior on -xi over (-0.5, 0.5) (Martins & Stedinger, 2000)."""
    inside = (xi > -0.5) & (xi < 0.5)
    xs = jnp.where(inside, xi, 0.0)
    val = 5.0 * jnp.log(0.5 - xs) + 8.0 * jnp.log(0.5 + xs) + 9.799126537211386
    return jnp.where(inside, val, -jnp.inf)


PRIORS = {None: None, "martins-stedinger": _martins_stedinger}


def natural_params(family: Family, theta, Xs, ks):
    out, off = [], 0
    for X, k, link in zip(Xs, ks, family.links):
        eta = X @ theta[off : off + k]
        out.append(LINKS[link](eta))
        off += k
    return out


@cache
def make_nll(family_name: str, ks: tuple[int, ...], prior: str | None):
    """Return the (cached) negative log-likelihood ``nll(theta, y, mask, *Xs)``."""
    family = FAMILIES[family_name]
    prior_fn = PRIORS[prior]

    def nll(theta, y, mask, *Xs):
        nat = natural_params(family, theta, Xs, ks)
        if family.name == "gev":
            ysafe = jnp.where(mask, y, nat[0])
        else:
            ysafe = jnp.where(mask, y, 0.0)
        lp = family.logpdf(ysafe, *nat)
        ll = jnp.sum(jnp.where(mask, lp, 0.0))
        if prior_fn is not None:
            off = sum(ks[:-1])
            ll = ll + prior_fn(theta[off])  # shape intercept = xi at mean covariates
        return -ll

    return nll


# ------------------------------------------------------------------- designs
@dataclass
class Design:
    """Standardised design matrices for one parameter."""

    X: np.ndarray  # (n, k) shared or (B, n, k) per series, intercept first
    names: list[str]
    mean: np.ndarray  # (k-1,) or (B, k-1)
    sd: np.ndarray
    shared: bool


def _as_cov_array(cov, n, batch_shape):
    names = None
    if hasattr(cov, "columns"):  # pandas DataFrame
        names = [str(c) for c in cov.columns]
        cov = cov.to_numpy()
    elif hasattr(cov, "name") and hasattr(cov, "to_numpy"):  # pandas Series
        names = [str(cov.name)]
        cov = cov.to_numpy()
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov[:, None]
    shared = cov.ndim == 2
    if shared and cov.shape[0] != n:
        raise ValueError(f"covariate has {cov.shape[0]} rows but data has {n} observations")
    if not shared:
        if cov.shape[:-1] != (*batch_shape, n):
            raise ValueError(
                f"per-series covariate shape {cov.shape} must be (*batch, n, k) = {(*batch_shape, n, '…')}"
            )
        cov = cov.reshape(-1, n, cov.shape[-1])
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariates must be finite everywhere")
    k = cov.shape[-1]
    names = names or [f"x{j}" for j in range(k)]
    return cov, names, shared


def build_design(cov, n: int, batch_shape: tuple, B: int, prefix: str) -> Design:
    if cov is None:
        return Design(np.ones((n, 1)), [f"{prefix}"], np.zeros(0), np.ones(0), True)
    arr, names, shared = _as_cov_array(cov, n, batch_shape)
    mean = arr.mean(axis=-2)
    sd = arr.std(axis=-2)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (arr - mean[..., None, :]) / sd[..., None, :]
    ones = np.ones((*Z.shape[:-1], 1))
    X = np.concatenate([ones, Z], axis=-1)
    return Design(X, [f"{prefix}"] + [f"{prefix}:{nm}" for nm in names], mean, sd, shared)


def back_transform_matrix(designs: list[Design], B: int) -> np.ndarray:
    """Matrix ``A`` (B, p, p) with ``theta_user = A @ theta_std``."""
    p = sum(d.X.shape[-1] for d in designs)
    A = np.zeros((B, p, p))
    off = 0
    for d in designs:
        k = d.X.shape[-1]
        A[:, off, off] = 1.0
        if k > 1:
            mean = np.broadcast_to(d.mean, (B, k - 1))
            sd = np.broadcast_to(d.sd, (B, k - 1))
            A[:, off, off + 1 : off + k] = -mean / sd
            idx = np.arange(1, k)
            A[:, off + idx, off + idx] = 1.0 / sd
        off += k
    return A


@cache
def make_vgh(family_name: str, ks: tuple[int, ...], prior: str | None):
    """Value, gradient and Hessian of the NLL via the "linear predictor" trick.

    The log-density of observation ``i`` depends on ``theta`` only through the
    linear predictors ``eta_ij = X_ij @ beta_j``. We therefore differentiate the
    scalar per-observation log-density with respect to its handful of
    ``eta``'s in forward mode (cheap, fully fused elementwise code) and
    assemble ``g = sum_i X_i' d_i`` and ``H = sum_i X_i' D_i X_i``. This is
    3-6x faster than a generic ``jax.hessian`` of the summed likelihood.
    """
    family = FAMILIES[family_name]
    prior_fn = PRIORS[prior]
    P = len(ks)
    from ._derivs import gev_derivs, gpd_derivs

    def vgh(theta, y, mask, *Xs):
        etas, off = [], 0
        for X, k in zip(Xs, ks):
            etas.append(X @ theta[off : off + k])
            off += k
        if family.name == "gev":
            ysafe = jnp.where(mask, y, etas[0])
            lp, G, Hh = gev_derivs(ysafe, *etas, stacked=False)
        else:
            ysafe = jnp.where(mask, y, 0.0)
            lp, G, Hh = gpd_derivs(ysafe, *etas, stacked=False)
        w = mask.astype(y.dtype)
        f = -jnp.sum(jnp.where(mask, lp, 0.0))
        gs = [-(Xs[a].T @ (G[a] * w)) for a in range(P)]
        Hb = [[None] * P for _ in range(P)]
        for a in range(P):
            for b in range(a, P):
                Hb[a][b] = -(Xs[a].T @ (Xs[b] * (Hh[a][b] * w)[:, None]))
                if b > a:
                    Hb[b][a] = Hb[a][b].T
        g = jnp.concatenate(gs)
        H = jnp.block(Hb)
        if prior_fn is not None:
            off = sum(ks[:-1])
            pv, pg, ph = prior_fn(theta[off]), jax.grad(prior_fn)(theta[off]), jax.hessian(prior_fn)(theta[off])
            f = f - pv
            g = g.at[off].add(-pg)
            H = H.at[off, off].add(-ph)
        return f, g, H

    return vgh
