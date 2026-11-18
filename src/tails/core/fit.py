"""User-facing GEV / GPD / POT fitting with batched maximum likelihood."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._jax import jax, jnp
from ._derivs import gev_derivs, gpd_derivs
from ._engine import FAMILIES, Design, back_transform_matrix, build_design, make_nll, make_vgh, natural_params
from .distributions import gev_logpdf, gev_ppf, gpd_logpdf, gpd_ppf
from .lmoments import gev_from_lmoments, gpd_from_lmoments, sample_lmoments
from .optim import minimize

__all__ = ["fit_gev", "fit_gpd", "fit_pot", "GEVFit", "GPDFit", "POTFit", "ReturnLevels"]


# ----------------------------------------------------------------- containers
@dataclass
class ReturnLevels:
    """Return-level estimates with optional confidence bounds.

    Arrays have shape ``(*batch, [m,] len(period))`` where ``m`` is the number
    of covariate rows for non-stationary models.
    """

    period: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    method: str | None = None
    alpha: float = 0.05

    def to_pandas(self):
        """Long-format :class:`pandas.DataFrame` (one row per series × period)."""
        import pandas as pd

        est = np.asarray(self.estimate)
        idx = np.indices(est.shape).reshape(est.ndim, -1).T
        df = pd.DataFrame(idx[:, :-1], columns=[f"dim{i}" for i in range(est.ndim - 1)])
        df["period"] = self.period[idx[:, -1]]
        df["estimate"] = est.ravel()
        if self.lower is not None:
            df["lower"] = np.asarray(self.lower).ravel()
            df["upper"] = np.asarray(self.upper).ravel()
        return df

    def plot(self, ax=None, index: int = 0, **kwargs):
        """Plot the estimate against return period, with a CI ribbon if present."""
        from ..plotting import plot_return_levels

        return plot_return_levels(self, ax=ax, index=index, **kwargs)


@dataclass
class _FitBase:
    family: str
    coef: np.ndarray
    cov: np.ndarray
    loglik: np.ndarray
    converged: np.ndarray
    n_iter: np.ndarray
    nobs: np.ndarray
    param_names: list[str]
    batch_shape: tuple
    method: str
    prior: str | None
    _theta_std: np.ndarray = field(repr=False)
    _y: np.ndarray = field(repr=False)
    _mask: np.ndarray = field(repr=False)
    _designs: list[Design] = field(repr=False)
    _raw: list[np.ndarray] = field(repr=False)
    # When shape is held fixed, the stored coefficient vector still contains it
    # (so return levels keep working) but AIC/BIC must not count it as free.
    free_params: int | None = None

    # ---------------------------------------------------------------- basics
    @property
    def ks(self) -> tuple[int, ...]:
        return tuple(d.X.shape[-1] for d in self._designs)

    @property
    def stationary(self) -> bool:
        return all(k == 1 for k in self.ks)

    @property
    def n_params(self) -> int:
        if self.free_params is not None:
            return int(self.free_params)
        return int(sum(self.ks))

    @property
    def se(self) -> np.ndarray:
        """Standard errors of ``coef`` (observed information)."""
        return np.sqrt(np.clip(np.diagonal(self.cov, axis1=-2, axis2=-1), 0, None))

    @property
    def aic(self) -> np.ndarray:
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self) -> np.ndarray:
        return -2 * self.loglik + np.log(self.nobs) * self.n_params

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Natural parameters. Stationary: shape ``batch``; else ``(*batch, n)``."""
        fam = FAMILIES[self.family]
        B = int(np.prod(self.batch_shape, dtype=int))
        th = jnp.asarray(self._theta_std.reshape(B, -1))
        Xs = [jnp.asarray(d.X) for d in self._designs]
        axes = tuple(None if d.shared else 0 for d in self._designs)
        nat = jax.vmap(lambda t, *X: natural_params(fam, t, X, self.ks), in_axes=(0, *axes))(th, *Xs)
        out = {}
        for name, v in zip(fam.params, nat):
            v = np.asarray(v)
            if self.stationary:
                v = v[:, 0]
                out[name] = v.reshape(self.batch_shape)
            else:
                out[name] = v.reshape(*self.batch_shape, -1)
        return out

    def summary(self):
        """Tidy :class:`pandas.DataFrame` of coefficients, standard errors and fit info."""
        import pandas as pd

        B = int(np.prod(self.batch_shape, dtype=int))
        coef = self.coef.reshape(B, -1)
        se = self.se.reshape(B, -1)
        rows = []
        for b in range(B):
            for j, nm in enumerate(self.param_names):
                rows.append((b, nm, coef[b, j], se[b, j]))
        df = pd.DataFrame(rows, columns=["series", "term", "estimate", "std_error"])
        return df

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        B = int(np.prod(self.batch_shape, dtype=int))
        head = f"{type(self).__name__}(family={self.family!r}, batch_shape={self.batch_shape}, terms={self.param_names}"
        if B == 1:
            c = ", ".join(f"{n}={v:.4g}±{s:.2g}" for n, v, s in zip(self.param_names, self.coef.ravel(), self.se.ravel()))
            return head + f", {c}, loglik={float(self.loglik.ravel()[0]):.4f})"
        return head + f", converged={float(np.mean(self.converged)):.1%})"

    def plot_diagnostics(self, index: int = 0, **kwargs):
        """Diagnostic figure for one fitted series.

        A stationary GEV, GPD or POT fit draws a 2×2 figure: QQ and PP plots
        with a simulated pointwise envelope, a return-level curve with a
        confidence ribbon, and a histogram with the fitted density.

        A non-stationary GEV draws the observations against the covariate, the
        fitted return level along that covariate, and a confidence ribbon.
        The covariate is whatever was passed to ``fit_gev`` — a year, a
        temperature index, or any other series — on the location, the scale,
        or the shape. ``period`` (default 2, the median) selects the level.
        ``ci`` is ``"delta"`` (default), ``"profile"``, ``"bootstrap"`` or
        ``None``.

        Parameters
        ----------
        index
            Series to plot from a batched fit. The default is 0.
        **kwargs
            ``ci``, ``period``, ``alpha``, ``n_boot``, ``seed`` and
            ``profile_points`` are the return-level interval on a trend fit.
            ``periods``, ``n_band`` and ``seed`` control the stationary panels.
            ``figsize`` and ``fig`` override the canvas.

        Returns
        -------
        tuple
            ``(fig, axes)``. ``axes`` has shape (2, 2) for a stationary fit
            and is the single trend axes for a non-stationary GEV.

        Notes
        -----
        Requires the ``plots`` extra (``pip install 'rare-tails[plots]'``).
        """
        from ..plotting import plot_fit_diagnostics

        return plot_fit_diagnostics(self, index=index, **kwargs)


# ------------------------------------------------------------- preparation
def _prepare(data, mask):
    y = np.asarray(data, dtype=np.float64)
    if y.ndim == 0:
        raise ValueError("data must have at least one dimension")
    batch_shape, n = y.shape[:-1], y.shape[-1]
    y2 = y.reshape(-1, n)
    m2 = np.isfinite(y2)
    if mask is not None:
        m2 &= np.broadcast_to(np.asarray(mask, bool), y.shape).reshape(-1, n)
    y2 = np.where(m2, y2, 0.0)
    return y2, m2, batch_shape, n


def _raw_design(cov, n, batch_shape):
    """User-scale design (intercept + raw covariates) for prediction."""
    if cov is None:
        return np.ones((n, 1))
    arr = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov, float)
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim > 2:
        arr = arr.reshape(-1, n, arr.shape[-1])
    ones = np.ones((*arr.shape[:-1], 1))
    return np.concatenate([ones, arr], axis=-1)


_VNLL: dict = {}


def _vmapped_nll(nll, designs):
    axes = tuple(None if d.shared else 0 for d in designs)
    key = (nll, axes)
    if key not in _VNLL:
        _VNLL[key] = jax.jit(jax.vmap(nll, in_axes=(0, 0, 0, *axes)))
    return _VNLL[key]


def _run(family, y2, m2, designs, prior, x0, max_iter):
    ks = tuple(d.X.shape[-1] for d in designs)
    nll = make_nll(family, ks, prior)
    Xs = [jnp.asarray(d.X) for d in designs]
    axes = tuple(None if d.shared else 0 for d in designs)
    vgh = make_vgh(family, ks, prior)
    res = minimize(nll, x0, jnp.asarray(y2), jnp.asarray(m2), *Xs, arg_axes=(0, 0, *axes), max_iter=max_iter, vgh=vgh)
    return res


def _inv_hessian(H):
    H = np.asarray(H)
    H = 0.5 * (H + np.swapaxes(H, -1, -2))
    out = np.full_like(H, np.nan)
    w, V = np.linalg.eigh(np.where(np.isfinite(H), H, 0.0))
    ok = np.all(w > 0, axis=-1) & np.all(np.isfinite(H), axis=(-1, -2))
    inv = (V / np.where(w > 0, w, 1.0)[..., None, :]) @ np.swapaxes(V, -1, -2)
    out[ok] = inv[ok]
    return out


def _stationary_start(family, y2, m2, prior):
    yy = np.where(m2, y2, np.nan)
    lm = sample_lmoments(jnp.asarray(yy))
    if family == "gev":
        p = np.array(gev_from_lmoments(lm))
        if prior is not None:
            p[:, 2] = np.clip(p[:, 2], -0.45, 0.45)
        x0 = np.stack([p[:, 0], np.log(p[:, 1]), p[:, 2]], axis=-1)
        # Gumbel fallback (always feasible)
        l1, l2 = np.asarray(lm[:, 0]), np.asarray(lm[:, 1])
        sc = l2 / np.log(2.0)
        fb = np.stack([l1 - 0.5772156649015329 * sc, np.log(sc), np.zeros_like(sc)], axis=-1)
    else:
        p = np.array(gpd_from_lmoments(lm))
        ymax = np.nanmax(np.where(m2, y2, -np.inf), axis=-1)
        neg = p[:, 1] < 0
        p[:, 0] = np.where(neg, np.maximum(p[:, 0], 1.05 * np.abs(p[:, 1]) * ymax), p[:, 0])
        if prior is not None:
            p[:, 1] = np.clip(p[:, 1], -0.45, 0.45)
        x0 = np.stack([np.log(p[:, 0]), p[:, 1]], axis=-1)
        mean = np.nanmean(yy, axis=-1)
        fb = np.stack([np.log(mean), np.zeros_like(mean)], axis=-1)
    bad = ~np.all(np.isfinite(x0), axis=-1)
    x0 = np.where(bad[:, None], fb, x0)
    d1 = [Design(np.ones((y2.shape[1], 1)), [""], np.zeros(0), np.ones(0), True)] * len(FAMILIES[family].params)
    f = _vmapped_nll(make_nll(family, (1,) * len(d1), prior), d1)
    vals = np.asarray(f(jnp.asarray(x0), jnp.asarray(y2), jnp.asarray(m2), *[jnp.asarray(d.X) for d in d1]))
    infeas = ~np.isfinite(vals)
    x0 = np.where(infeas[:, None], fb, x0)
    return x0


def _fit(family, data, covs: Sequence[Any], mask, prior, method, max_iter, x0=None):
    fam = FAMILIES[family]
    y2, m2, batch_shape, n = _prepare(data, mask)
    B = y2.shape[0]
    designs = [build_design(c, n, batch_shape, B, name) for c, name in zip(covs, fam.params)]
    raws = [_raw_design(c, n, batch_shape) for c in covs]
    ks = tuple(d.X.shape[-1] for d in designs)
    nobs = m2.sum(-1)
    if np.any(nobs < 2):
        raise ValueError("every series needs at least 2 finite observations")

    start = _stationary_start(family, y2, m2, prior)
    if method == "lmom":
        if any(k > 1 for k in ks):
            raise ValueError("method='lmom' only supports stationary models")
        theta_std = start
        nll = _vmapped_nll(make_nll(family, ks, None), designs)
        Xs = [jnp.asarray(d.X) for d in designs]
        f = np.asarray(nll(jnp.asarray(theta_std), jnp.asarray(y2), jnp.asarray(m2), *Xs))
        H = np.full((B, 3, 3) if family == "gev" else (B, 2, 2), np.nan)
        res_conv, res_it = np.isfinite(f), np.zeros(B, int)
    else:
        if x0 is None:
            s1 = (1,) * len(ks)
            d1 = [Design(np.ones((n, 1)), [""], np.zeros(0), np.ones(0), True)] * len(ks)
            r1 = _run(family, y2, m2, d1, prior, start, max_iter)
            st = np.where(np.asarray(r1.converged)[:, None], np.asarray(r1.x), start)
            if ks == s1:
                res = r1
                x0 = None
            else:
                x0 = np.zeros((B, sum(ks)))
                off = 0
                for j, k in enumerate(ks):
                    x0[:, off] = st[:, j]
                    off += k
        if x0 is not None:
            res = _run(family, y2, m2, designs, prior, jnp.asarray(x0), max_iter)
        theta_std = np.asarray(res.x)
        f = np.asarray(res.fun)
        H = np.asarray(res.hess)
        res_conv, res_it = np.asarray(res.converged), np.asarray(res.n_iter)

    A = back_transform_matrix(designs, B)
    coef = np.einsum("bij,bj->bi", A, theta_std)
    cov_std = _inv_hessian(H)
    cov = A @ cov_std @ np.swapaxes(A, -1, -2)
    names = [nm for d in designs for nm in d.names]
    names = [nm.replace("scale", "log_scale", 1) if nm.startswith("scale") else nm for nm in names]
    loglik = -f
    if prior is not None and method != "lmom":
        loglik = -np.asarray(_vmapped_nll(make_nll(family, ks, None), designs)(
            jnp.asarray(theta_std), jnp.asarray(y2), jnp.asarray(m2), *[jnp.asarray(d.X) for d in designs]))
    p = coef.shape[-1]
    return dict(
        family=family,
        coef=coef.reshape(*batch_shape, p),
        cov=cov.reshape(*batch_shape, p, p),
        loglik=loglik.reshape(batch_shape),
        converged=res_conv.reshape(batch_shape),
        n_iter=res_it.reshape(batch_shape),
        nobs=nobs.reshape(batch_shape),
        param_names=names,
        batch_shape=batch_shape,
        method=method,
        prior=prior,
        _theta_std=theta_std.reshape(*batch_shape, p),
        _y=y2,
        _mask=m2,
        _designs=designs,
        _raw=raws,
    )


def _check_method(method, prior):
    if method not in ("mle", "gmle", "lmom"):
        raise ValueError("method must be 'mle', 'gmle' or 'lmom'")
    if method == "gmle":
        prior = prior or "martins-stedinger"
    return prior


# --------------------------------------------------------------- fixed shape
def _fixed_nll_gev(theta, y, mask, xi):
    loc, phi = theta[0], theta[1]
    ysafe = jnp.where(mask, y, loc)
    lp = gev_logpdf(ysafe, loc, jnp.exp(phi), xi)
    return -jnp.sum(jnp.where(mask, lp, 0.0))


def _fixed_vgh_gev(theta, y, mask, xi):
    loc, phi = theta[0], theta[1]
    ysafe = jnp.where(mask, y, loc)
    lp, g, h = gev_derivs(ysafe, loc, phi, xi, stacked=False)
    w = mask.astype(y.dtype)
    f = -jnp.sum(jnp.where(mask, lp, 0.0))
    grad = jnp.stack([jnp.sum(-(g[0] * w)), jnp.sum(-(g[1] * w))])
    h00 = jnp.sum(-(h[0][0] * w))
    h01 = jnp.sum(-(h[0][1] * w))
    h11 = jnp.sum(-(h[1][1] * w))
    hess = jnp.array([[h00, h01], [h01, h11]])
    return f, grad, hess


def _fixed_nll_gpd(theta, y, mask, xi):
    phi = theta[0]
    ysafe = jnp.where(mask, y, 0.0)
    lp = gpd_logpdf(ysafe, jnp.exp(phi), xi)
    return -jnp.sum(jnp.where(mask, lp, 0.0))


def _fixed_vgh_gpd(theta, y, mask, xi):
    phi = theta[0]
    ysafe = jnp.where(mask, y, 0.0)
    lp, g, h = gpd_derivs(ysafe, phi, xi, stacked=False)
    w = mask.astype(y.dtype)
    f = -jnp.sum(jnp.where(mask, lp, 0.0))
    grad = jnp.stack([jnp.sum(-(g[0] * w))])
    hess = jnp.array([[jnp.sum(-(h[0][0] * w))]])
    return f, grad, hess


def _repair_fixed_start(family, x0, y2, m2, xi):
    """Move an infeasible L-moment start inside the support of a fixed shape."""
    nll = _fixed_nll_gev if family == "gev" else _fixed_nll_gpd
    fun = jax.jit(jax.vmap(nll, in_axes=(0, 0, 0, None)))
    vals = np.asarray(fun(jnp.asarray(x0), jnp.asarray(y2), jnp.asarray(m2), jnp.asarray(xi)))
    bad = ~np.isfinite(vals)
    if not np.any(bad):
        return x0
    yy = np.where(m2, y2, np.nan)
    ymin = np.nanmin(yy, axis=1)
    ymax = np.nanmax(yy, axis=1)
    span = np.maximum(ymax - ymin, 1e-3)
    x0 = np.array(x0, copy=True)
    if family == "gev":
        sc = np.maximum(span, np.exp(x0[:, 1]))
        if xi < 0:
            loc = ymax + sc - sc / abs(xi)
        elif xi > 0:
            loc = ymin - sc
        else:
            loc = 0.5 * (ymin + ymax)
        x0[bad, 0] = loc[bad]
        x0[bad, 1] = np.log(sc[bad])
    else:
        x0[bad, 0] = np.log(np.maximum(span[bad], 1e-3))
    return x0


def _fit_fixed_shape(family, data, fixed_shape, mask, max_iter):
    """Stationary GEV or GPD with ξ held at ``fixed_shape``.

    Coefficients still include the fixed shape so return levels and ``params``
    look like a free fit. ``free_params`` is 2 (GEV: loc, log scale) or 1
    (GPD: log scale), which is what AIC and BIC use. ξ = 0 is the Gumbel
    distribution for block maxima and the exponential distribution for excesses.
    """
    if family not in ("gev", "gpd"):
        raise ValueError("fixed shape is implemented for the GEV and the GPD")
    xi = float(fixed_shape)
    if not np.isfinite(xi):
        raise ValueError("fixed_shape must be finite")
    y2, m2, batch_shape, n = _prepare(data, mask)
    B = y2.shape[0]
    nobs = m2.sum(-1)
    if np.any(nobs < 2):
        raise ValueError("every series needs at least 2 finite observations")
    yy = np.where(m2, y2, np.nan)
    if family == "gev":
        lm = np.asarray(sample_lmoments(jnp.asarray(yy)))
        l1, l2 = lm[:, 0], lm[:, 1]
        sc = np.maximum(l2 / np.log(2.0), 1e-8)
        x0 = np.stack([l1 - 0.5772156649015329 * sc, np.log(sc)], axis=-1)
        nll, vgh = _fixed_nll_gev, _fixed_vgh_gev
        n_free, prefixes = 2, ("loc", "scale", "shape")
    else:
        mean = np.nanmean(yy, axis=-1)
        mean = np.where(np.isfinite(mean) & (mean > 0), mean, 1.0)
        x0 = np.log(mean)[:, None]
        nll, vgh = _fixed_nll_gpd, _fixed_vgh_gpd
        n_free, prefixes = 1, ("scale", "shape")
    # ξ = 0 (Gumbel / exponential) is supported on the whole line, so the
    # L-moment start is feasible and does not need a second compiled check.
    if xi != 0.0:
        x0 = _repair_fixed_start(family, x0, y2, m2, xi)
    res = minimize(
        nll,
        jnp.asarray(x0),
        jnp.asarray(y2),
        jnp.asarray(m2),
        jnp.asarray(xi),
        arg_axes=(0, 0, None),
        max_iter=max_iter,
        vgh=vgh,
    )
    theta = np.asarray(res.x)
    p_full = 3 if family == "gev" else 2
    coef = np.zeros((B, p_full))
    cov = np.zeros((B, p_full, p_full))
    cov_free = _inv_hessian(np.asarray(res.hess))
    if family == "gev":
        coef[:, 0:2] = theta
        coef[:, 2] = xi
        cov[:, :2, :2] = cov_free
    else:
        coef[:, 0] = theta[:, 0]
        coef[:, 1] = xi
        cov[:, 0, 0] = cov_free[:, 0, 0]
    designs = [build_design(None, n, batch_shape, B, name) for name in prefixes]
    names = [nm.replace("scale", "log_scale", 1) if nm.startswith("scale") else nm for nm in prefixes]
    loglik = -np.asarray(res.fun)
    return dict(
        family=family,
        coef=coef.reshape(*batch_shape, p_full),
        cov=cov.reshape(*batch_shape, p_full, p_full),
        loglik=loglik.reshape(batch_shape),
        converged=np.asarray(res.converged).reshape(batch_shape),
        n_iter=np.asarray(res.n_iter).reshape(batch_shape),
        nobs=nobs.reshape(batch_shape),
        param_names=names,
        batch_shape=batch_shape,
        method="mle",
        prior=None,
        _theta_std=coef.reshape(*batch_shape, p_full),
        _y=y2,
        _mask=m2,
        _designs=designs,
        _raw=[_raw_design(None, n, batch_shape) for _ in prefixes],
        free_params=n_free,
    )


def _reject_fixed_shape_extras(fixed_shape, method, covariates):
    if fixed_shape is None:
        return
    if method != "mle":
        raise ValueError("fixed_shape requires method='mle'")
    if any(c is not None for c in covariates):
        raise ValueError("fixed_shape is only implemented for stationary models without covariates")


# --------------------------------------------------------------------- GEV
@dataclass(repr=False)
class GEVFit(_FitBase):
    """Fitted (possibly non-stationary, possibly batched) GEV model."""

    def _nat_user(self, covariates: Mapping | None):
        """Natural params (B, m, 3) from user-scale coefficients and covariates."""
        B = int(np.prod(self.batch_shape, dtype=int))
        out, off = [], 0
        for j, (name, k) in enumerate(zip(("loc", "scale", "shape"), self.ks)):
            if covariates is not None and name in covariates and k > 1:
                Xn = np.asarray(covariates[name], float)
                Xn = Xn[:, None] if Xn.ndim == 1 else Xn
                X = np.concatenate([np.ones((Xn.shape[0], 1)), Xn], axis=-1)
            elif k == 1:
                X = np.ones((1, 1))
            else:
                X = self._raw[j]
            X = np.broadcast_to(X, (B, *X.shape[-2:])) if X.ndim == 2 else X
            out.append((X, off, k))
            off += k
        return out

    def _rl_function(self, covariates):
        spec = self._nat_user(covariates)
        m = max(s[0].shape[1] for s in spec)
        Xs = [np.broadcast_to(X, (X.shape[0], m, X.shape[2])) if X.shape[1] == 1 else X for X, _, _ in spec]
        offs = [(o, k) for _, o, k in spec]

        def rl(theta, X0, X1, X2, periods):
            eta = [X @ theta[o : o + k] for X, (o, k) in zip((X0, X1, X2), offs)]
            loc, scale, shape = eta[0], jnp.exp(eta[1]), eta[2]
            p = 1.0 - 1.0 / periods
            return gev_ppf(p[None, :], loc[:, None], scale[:, None], shape[:, None])  # (m, T)

        return rl, [jnp.asarray(X) for X in Xs], m

    def return_level(
        self,
        period,
        *,
        covariates: Mapping | None = None,
        ci: str | None = None,
        alpha: float = 0.05,
        n_boot: int = 500,
        seed: int = 0,
        profile_points: int = 41,
    ) -> ReturnLevels:
        """T-year return levels, optionally with confidence intervals.

        Parameters
        ----------
        period
            Return period(s) in blocks (years for annual maxima), each > 1.
        covariates
            For non-stationary models: ``{"loc": X, "scale": X, "shape": X}``
            with new covariate rows (user scale). Default: the fitted covariates
            (effective return level per observation).
        ci
            ``None``, ``"delta"`` (Wald, observed information), ``"profile"``
            (profile likelihood) or ``"bootstrap"`` (parametric bootstrap,
            percentile intervals). Profile intervals are available for
            stationary models and for models with covariates: the return level
            at each covariate row replaces the location intercept. A fit with
            ``fixed_shape`` keeps that shape fixed in the profile interval;
            bootstrap intervals are not available for it.
        """
        T = np.atleast_1d(np.asarray(period, dtype=np.float64))
        if np.any(T <= 1):
            raise ValueError("return periods must be > 1")
        rl, Xs, m = self._rl_function(covariates)
        B = int(np.prod(self.batch_shape, dtype=int))
        coef = jnp.asarray(self.coef.reshape(B, -1))
        Tj = jnp.asarray(T)
        est = np.asarray(jax.vmap(rl, in_axes=(0, 0, 0, 0, None))(coef, *Xs, Tj))
        squeeze = self.stationary and covariates is None
        shape_out = (*self.batch_shape, len(T)) if squeeze else (*self.batch_shape, m, len(T))
        lo = hi = None
        if ci == "delta":
            J = jax.vmap(jax.jacfwd(rl), in_axes=(0, 0, 0, 0, None))(coef, *Xs, Tj)  # (B, m, T, p)
            J = np.asarray(J)
            cov = self.cov.reshape(B, *self.cov.shape[-2:])
            var = np.einsum("bmtp,bpq,bmtq->bmt", J, cov, J)
            from scipy.stats import norm

            zq = norm.ppf(1 - alpha / 2)
            sd = np.sqrt(np.clip(var, 0, None))
            lo, hi = est - zq * sd, est + zq * sd
        elif ci == "profile":
            if self.free_params is not None:
                from .uncertainty import profile_fixed_shape_return_level

                lo, hi = profile_fixed_shape_return_level(self, T, alpha=alpha, n_points=profile_points)
                lo, hi = lo[:, None, :], hi[:, None, :]
            elif self.stationary:
                from .uncertainty import profile_return_level

                lo, hi = profile_return_level(self, T, alpha=alpha, n_points=profile_points)
                lo, hi = lo[:, None, :], hi[:, None, :]
            else:
                from .uncertainty import profile_covariate_return_level

                lo, hi = profile_covariate_return_level(
                    self, T, covariates=covariates, alpha=alpha, n_points=profile_points
                )
        elif ci == "bootstrap":
            if self.free_params is not None:
                raise ValueError(
                    "bootstrap intervals refit a free shape; with fixed_shape use ci='delta' or 'profile'"
                )
            from .uncertainty import bootstrap_coef

            boot = bootstrap_coef(self, n_boot=n_boot, seed=seed)  # (B, R, p)
            R = boot.shape[1]
            f = jax.vmap(jax.vmap(rl, in_axes=(0, None, None, None, None)), in_axes=(0, 0, 0, 0, None))
            draws = np.asarray(f(jnp.asarray(boot), *Xs, Tj))  # (B, R, m, T)
            lo = np.nanquantile(draws, alpha / 2, axis=1)
            hi = np.nanquantile(draws, 1 - alpha / 2, axis=1)
            del R
        elif ci is not None:
            raise ValueError("ci must be None, 'delta', 'profile' or 'bootstrap'")
        rs = lambda a: None if a is None else np.asarray(a).reshape(shape_out)  # noqa: E731
        return ReturnLevels(T, rs(est), rs(lo), rs(hi), ci, alpha)


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
) -> GEVFit:
    """Fit GEV distributions to one or many series of block maxima at once.

    Parameters
    ----------
    data
        Array of shape ``(*batch, n)``; the last axis holds the block maxima of
        each series. NaN marks missing values (ragged records are fine).
    loc, scale, shape
        Optional covariates for each parameter: ``(n,)``/``(n, k)`` shared by
        all series, or ``(*batch, n, k)`` per series. A pandas object keeps its
        column names. Links: identity (loc, shape), log (scale).
    mask
        Optional boolean array broadcastable to ``data``; False excludes a value.
    method
        ``"mle"`` (default), ``"gmle"`` (MLE with the Martins–Stedinger
        Beta(6, 9) prior on the shape, robust for short records) or
        ``"lmom"`` (L-moments, stationary only).
    fixed_shape
        Hold ξ at this value instead of estimating it. ``0`` is the Gumbel
        model (two free parameters: location and log scale). Stationary models
        only. AIC and BIC then count the free parameters, not the fixed shape.
        The shape is still reported in ``params`` and ``coef``.

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
    _reject_fixed_shape_extras(fixed_shape, method, (loc, scale, shape))
    if fixed_shape is not None:
        return GEVFit(**_fit_fixed_shape("gev", data, fixed_shape, mask, max_iter))
    prior = _check_method(method, prior)
    return GEVFit(**_fit("gev", data, (loc, scale, shape), mask, prior, method, max_iter))


# --------------------------------------------------------------------- GPD
@dataclass(repr=False)
class GPDFit(_FitBase):
    """Fitted GPD model for threshold excesses."""

    threshold: Any = None


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

    ``fixed_shape`` holds ξ. ``0`` is the exponential distribution of excesses
    (one free parameter, the log scale). Stationary models only.
    """
    y = _excesses(data, threshold) if threshold is not None else np.asarray(data, float)
    _reject_fixed_shape_extras(fixed_shape, method, (scale, shape))
    if fixed_shape is not None:
        out = _fit_fixed_shape("gpd", y, fixed_shape, mask, max_iter)
    else:
        prior = _check_method(method, prior)
        out = _fit("gpd", y, (scale, shape), mask, prior, method, max_iter)
    return GPDFit(**out, threshold=threshold)


# --------------------------------------------------------------------- POT
@dataclass(repr=False)
class POTFit(GPDFit):
    """Peaks-over-threshold model: GPD for excesses + Poisson/binomial exceedance rate."""

    rate: np.ndarray | None = None  # exceedances per year
    zeta: np.ndarray | None = None  # P(X > u)
    n_total: np.ndarray | None = None
    npy: float = 365.25

    def return_level(
        self,
        period,
        *,
        ci: str | None = None,
        alpha: float = 0.05,
        n_boot: int = 500,
        seed: int = 0,
        profile_points: int = 41,
    ) -> ReturnLevels:
        """Level exceeded on average once every ``period`` years (Coles 2001, §4.3.3).

        ``ci="delta"`` is a Wald interval from the GPD observed information and
        the binomial variance of the exceedance rate. ``ci="profile"`` profiles
        the joint likelihood of the excesses and that rate, with the return
        level as the interest parameter. ``ci="bootstrap"`` is a parametric
        percentile interval: resample the exceedance count and the excesses,
        then refit. A fit with ``fixed_shape`` keeps that shape in the profile
        interval; bootstrap intervals are not available for it.
        """
        if not self.stationary:
            raise ValueError("POT return levels are implemented for stationary models")
        T = np.atleast_1d(np.asarray(period, dtype=np.float64))
        B = int(np.prod(self.batch_shape, dtype=int))
        coef = self.coef.reshape(B, 2)
        zeta = np.asarray(self.zeta, float).reshape(B)
        u = np.broadcast_to(np.asarray(self.threshold, float), self.batch_shape).reshape(B)

        def rl(th, z, Tj):
            m = Tj * self.npy * z  # expected exceedances in T years
            return gpd_ppf(1.0 - 1.0 / m, jnp.exp(th[0]), th[1])

        est = np.asarray(jax.vmap(rl, in_axes=(0, 0, None))(jnp.asarray(coef), jnp.asarray(zeta), jnp.asarray(T)))
        est = est + u[:, None]
        if np.any(T * self.npy * zeta[:, None] <= 1):
            est = np.where(T * self.npy * zeta[:, None] <= 1, np.nan, est)
        lo = hi = None
        if ci == "delta":
            from scipy.stats import norm

            g = jax.vmap(jax.jacfwd(rl, argnums=(0, 1)), in_axes=(0, 0, None))(jnp.asarray(coef), jnp.asarray(zeta), jnp.asarray(T))
            gth, gz = np.asarray(g[0]), np.asarray(g[1])
            cov = self.cov.reshape(B, 2, 2)
            n = np.asarray(self.n_total, float).reshape(B)
            var = np.einsum("btp,bpq,btq->bt", gth, cov, gth) + gz**2 * (zeta * (1 - zeta) / n)[:, None]
            sd = np.sqrt(np.clip(var, 0, None))
            zq = norm.ppf(1 - alpha / 2)
            lo, hi = est - zq * sd, est + zq * sd
        elif ci == "profile":
            from .uncertainty import profile_pot_return_level

            lo, hi = profile_pot_return_level(self, T, alpha=alpha, n_points=profile_points)
        elif ci == "bootstrap":
            if self.free_params is not None:
                raise ValueError(
                    "bootstrap intervals refit a free shape; with fixed_shape use ci='delta' or 'profile'"
                )
            from .uncertainty import bootstrap_pot_return_level

            lo, hi = bootstrap_pot_return_level(self, T, n_boot=n_boot, seed=seed, alpha=alpha)
        elif ci is not None:
            raise ValueError("ci must be None, 'delta', 'profile' or 'bootstrap'")
        shp = (*self.batch_shape, len(T))
        rs = lambda a: None if a is None else np.asarray(a).reshape(shp)  # noqa: E731
        return ReturnLevels(T, rs(est), rs(lo), rs(hi), ci, alpha)


def fit_pot(
    data,
    threshold,
    *,
    npy: float = 365.25,
    decluster: int | None = None,
    method: str = "mle",
    prior: str | None = None,
    max_iter: int = 200,
    fixed_shape: float | None = None,
) -> POTFit:
    """Peaks-over-threshold analysis of (batched) series.

    Parameters
    ----------
    data
        ``(*batch, n)`` observations (e.g. daily values); NaN = missing.
    threshold
        Scalar or one value per series.
    npy
        Observations per year (365.25 for daily data).
    decluster
        If given, apply runs declustering with this run length before fitting
        (cluster maxima only).
    fixed_shape
        Hold the GPD shape ξ. ``0`` is an exponential excess model. The
        exceedance rate is unchanged. Stationary models only.
    """
    x = np.asarray(data, dtype=np.float64)
    if decluster is not None:
        from .blocks import decluster_runs

        x = decluster_runs(x, threshold, run_length=decluster)
    u = np.asarray(threshold, float)
    ub = u[..., None] if u.ndim else u
    exceed = np.isfinite(x) & (x > ub)
    n_total = np.isfinite(np.asarray(data, float)).sum(-1)
    k = exceed.sum(-1)
    zeta = k / n_total
    _reject_fixed_shape_extras(fixed_shape, method, ())
    if fixed_shape is not None:
        out = _fit_fixed_shape("gpd", _excesses(x, threshold), fixed_shape, None, max_iter)
    else:
        prior = _check_method(method, prior)
        out = _fit("gpd", _excesses(x, threshold), (None, None), None, prior, method, max_iter)
    return POTFit(**out, threshold=threshold, rate=zeta * npy, zeta=zeta, n_total=n_total, npy=npy)
