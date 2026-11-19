"""Generic maximum-likelihood fitting and the fit object every model returns.

``fit_gev`` (block maxima or r-largest), ``fit_gpd`` and ``fit_pot`` are thin
front ends for :func:`fit_model`. It takes prepared data, covariates for any parameter,
optional fixed parameters and a prior, and returns the fields of a
:class:`ModelFit`. Return levels, delta / profile / bootstrap intervals,
coefficient intervals and plots are methods of that one class, so they work
the same way for every model, stationary or not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._jax import jax, jnp
from ._engine import (
    LIKELIHOODS,
    LINKS,
    Design,
    Spec,
    back_transform_matrix,
    build_design,
    etas_of,
    full_nll,
    natural_params,
    objective,
    raw_design,
)
from .lmoments import gev_from_lmoments, gpd_from_lmoments, sample_lmoments
from .optim import minimize

__all__ = ["ModelFit", "ReturnLevels", "fit_model"]

_EULER = 0.5772156649015329


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

    def plot_map(self, lon, lat, *, period: float | None = None, **kwargs):
        """Scatter these return levels at longitude and latitude.

        ``lon`` and ``lat`` are the coordinates of the batch, in degrees.
        ``period`` selects one return period when this object holds several.
        A ``(latitude, longitude)`` grid is drawn as a mesh. The colour is
        the return level in the units of the fit. See
        :func:`tails.plot_return_level_map`.
        """
        from ..plotting import plot_return_level_map

        return plot_return_level_map(self, lon, lat, period=period, **kwargs)


# --------------------------------------------------------------- preparation
def prepare(data, mask=None):
    """``(*batch, n)`` → ``(B, n)`` values (0 where missing), mask, batch shape, n."""
    y = np.asarray(data, dtype=np.float64)
    if y.ndim == 0:
        raise ValueError("data must have at least one dimension")
    batch_shape, n = y.shape[:-1], y.shape[-1]
    y2 = y.reshape(-1, n)
    m2 = np.isfinite(y2)
    if mask is not None:
        m2 &= np.broadcast_to(np.asarray(mask, bool), y.shape).reshape(-1, n)
    return np.where(m2, y2, 0.0), m2, batch_shape, n


def inv_hessian(H):
    """Inverse of a batch of observed-information matrices; NaN where not positive definite."""
    H = np.asarray(H)
    H = 0.5 * (H + np.swapaxes(H, -1, -2))
    out = np.full_like(H, np.nan)
    w, V = np.linalg.eigh(np.where(np.isfinite(H), H, 0.0))
    ok = np.all(w > 0, axis=-1) & np.all(np.isfinite(H), axis=(-1, -2))
    inv = (V / np.where(w > 0, w, 1.0)[..., None, :]) @ np.swapaxes(V, -1, -2)
    out[ok] = inv[ok]
    return out


def check_method(method, prior):
    if method not in ("mle", "gmle", "lmom"):
        raise ValueError("method must be 'mle', 'gmle' or 'lmom'")
    if method == "gmle":
        prior = prior or "martins-stedinger"
    return prior


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(p) - np.log1p(-p)


# ----------------------------------------------------------- starting values
def _gev_start(yy, prior, fixed_shape):
    lm = np.asarray(sample_lmoments(jnp.asarray(yy)))
    l1, l2 = lm[:, 0], lm[:, 1]
    sc = l2 / np.log(2.0)
    fb = np.stack([l1 - _EULER * sc, np.log(sc), np.zeros_like(sc)], axis=-1)  # Gumbel: always feasible
    if fixed_shape is not None:
        sc = np.maximum(sc, 1e-8)
        x0 = np.stack([l1 - _EULER * sc, np.log(sc), np.full_like(sc, fixed_shape)], axis=-1)
        return x0, None
    p = np.array(gev_from_lmoments(jnp.asarray(lm)))
    if prior is not None:
        p[:, 2] = np.clip(p[:, 2], -0.45, 0.45)
    return np.stack([p[:, 0], np.log(p[:, 1]), p[:, 2]], axis=-1), fb


def _gpd_start(yy, prior, fixed_shape):
    mean = np.nanmean(yy, axis=-1)
    if fixed_shape is not None:
        mean = np.where(np.isfinite(mean) & (mean > 0), mean, 1.0)
        return np.stack([np.log(mean), np.full_like(mean, fixed_shape)], axis=-1), None
    p = np.array(gpd_from_lmoments(sample_lmoments(jnp.asarray(yy))))
    ymax = np.nanmax(yy, axis=-1)
    neg = p[:, 1] < 0
    p[:, 0] = np.where(neg, np.maximum(p[:, 0], 1.05 * np.abs(p[:, 1]) * ymax), p[:, 0])
    if prior is not None:
        p[:, 1] = np.clip(p[:, 1], -0.45, 0.45)
    return np.stack([np.log(p[:, 0]), p[:, 1]], axis=-1), np.stack([np.log(mean), np.zeros_like(mean)], axis=-1)


def _repair_fixed_start(family, x0, yy, xi):
    """Move an infeasible start inside the support of a fixed shape."""
    ymin, ymax = np.nanmin(yy, axis=1), np.nanmax(yy, axis=1)
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
        return np.stack([loc, np.log(sc), x0[:, 2]], axis=-1)
    x0[:, 0] = np.log(np.maximum(span, 1e-3))
    return x0


def _repair_gpd_start(x0, y, mask):
    """Push a negative-shape start off an endpoint that the excesses cross."""
    x0 = np.array(x0, dtype=np.float64, copy=True)
    ymax = np.max(np.where(mask, y, 0.0), axis=-1)
    scale = np.exp(x0[:, 0])
    xi = x0[:, 1]
    endpoint = np.where(xi < 0.0, -scale / np.minimum(xi, -1e-12), np.inf)
    need = (xi < 0.0) & (ymax >= 0.98 * endpoint)
    if np.any(need):
        x0[need, 0] = np.log(np.maximum(scale, (-xi) * np.maximum(ymax, 1e-3) * 1.25)[need])
    return x0


def _ones_designs(spec, n):
    return [jnp.ones((n, 1))] * len(spec.ks)


def _evaluate(spec, theta, y, m, Xs, axes):
    fun, _ = objective(Spec(spec.lik, spec.ks, spec.prior))
    f = jax.jit(jax.vmap(fun, in_axes=(0, 0, 0, None, *axes)))
    return np.asarray(f(jnp.asarray(theta), jnp.asarray(y), jnp.asarray(m), None, *Xs))


def _run(spec, x_full, y, m, Xs, axes, max_iter):
    """Newton fit of the free coefficients, starting from ``x_full``."""
    fun, vgh = objective(spec)
    free = list(range(spec.p)) if spec.free is None else list(spec.free)
    res = minimize(fun, jnp.asarray(x_full[:, free]), jnp.asarray(y), jnp.asarray(m), jnp.asarray(x_full), *Xs,
                   arg_axes=(0, 0, 0, *axes), max_iter=max_iter, vgh=vgh)
    theta = np.array(x_full, copy=True)
    theta[:, free] = np.asarray(res.x)
    return theta, res


# -------------------------------------------------------------- generic fit
def fit_model(lik_key, y, mask, batch_shape, covs: Mapping[str, Any], *, method="mle", prior=None,
              max_iter=200, fixed: Mapping[str, float] | None = None, start_values=None, n_obs=None):
    """Fit any registered likelihood. Returns the keyword fields of a :class:`ModelFit`.

    ``y`` / ``mask`` are ``(B, n)`` (or ``(B, n, r)`` for r-largest blocks).
    ``start_values`` is the ``(B, n)`` NaN-padded sample used for L-moment
    starts (block maxima or excesses). ``fixed`` holds parameters (intercept
    only, natural scale) at given values, e.g. ``{"shape": 0.0}``.
    """
    lik = LIKELIHOODS[lik_key]
    fixed = {k: float(v) for k, v in (fixed or {}).items() if v is not None}
    for name, v in fixed.items():
        if name not in lik.params:
            raise ValueError(f"unknown parameter {name!r}; the parameters are {lik.params}")
        if covs.get(name) is not None:
            raise ValueError(f"{name} is fixed, so it cannot also have covariates")
        if not np.isfinite(v):
            raise ValueError(f"fixed {name} must be finite")
    if fixed and method != "mle":
        raise ValueError("fixed parameters (e.g. fixed_shape) require method='mle'")
    B, n = y.shape[0], y.shape[1]
    designs = [build_design(covs.get(nm), n, batch_shape, nm) for nm in lik.params]
    raws = [raw_design(covs.get(nm), n) for nm in lik.params]
    ks = tuple(d.k for d in designs)
    p = int(sum(ks))
    offs = [int(v) for v in np.concatenate([[0], np.cumsum(ks)[:-1]])]
    fixed_idx = {offs[lik.params.index(nm)] for nm in fixed}
    free = None if not fixed else tuple(i for i in range(p) if i not in fixed_idx)
    spec = Spec(lik_key, ks, prior, free)
    yy = start_values
    nobs = np.isfinite(yy).sum(-1) if n_obs is None else n_obs
    if np.any(np.isfinite(yy).sum(-1) < 2):
        raise ValueError("every series needs at least 2 finite observations" + (" above the threshold" if lik.family == "gpd" else ""))

    # stationary start: L-moments, then a feasibility check on the actual likelihood
    fixed_shape = fixed.get("shape")
    x0, fb = (_gev_start if lik.family == "gev" else _gpd_start)(yy, prior, fixed_shape)
    if lik_key == "pot":
        rate0 = _logit(np.isfinite(yy).sum(-1) / mask.sum(-1))[:, None]
        x0 = np.concatenate([x0, rate0], axis=1)
        fb = None if fb is None else np.concatenate([fb, rate0], axis=1)
    spec1 = spec.with_ks((1,) * len(ks))
    X1 = _ones_designs(spec1, n)
    axes1 = (None,) * len(ks)
    bad = ~np.isfinite(_evaluate(spec1, x0, y, mask, X1, axes1))
    if np.any(bad):
        if fb is not None:
            x0 = np.where(bad[:, None], fb, x0)
        elif fixed_shape is not None and fixed_shape != 0.0:
            rep = _repair_fixed_start(lik.family, x0[:, :2] if lik.family == "gpd" else x0, yy, fixed_shape)
            x0[:, : rep.shape[1]] = np.where(bad[:, None], rep, x0[:, : rep.shape[1]])

    Xs = [jnp.asarray(d.X) for d in designs]
    axes = tuple(None if d.shared else 0 for d in designs)
    if method == "lmom":
        if any(k > 1 for k in ks):
            raise ValueError("method='lmom' only supports stationary models")
        theta_std = x0
        f = _evaluate(spec, theta_std, y, mask, Xs, axes)
        H = np.full((B, p, p), np.nan)
        conv, n_iter = np.isfinite(f), np.zeros(B, int)
    else:
        th1, r1 = _run(spec1, x0, y, mask, X1, axes1, max_iter)
        if ks == (1,) * len(ks):
            theta_std, res = th1, r1
        else:
            st = np.where(np.asarray(r1.converged)[:, None], th1, x0)
            xf = np.zeros((B, p))
            xf[:, offs] = st
            theta_std, res = _run(spec, xf, y, mask, Xs, axes, max_iter)
        f, H = np.asarray(res.fun), np.asarray(res.hess)
        conv, n_iter = np.asarray(res.converged), np.asarray(res.n_iter)

    cov_std = np.zeros((B, p, p))
    fr = list(range(p)) if free is None else list(free)
    cov_std[np.ix_(np.arange(B), fr, fr)] = inv_hessian(H)
    A = back_transform_matrix(designs, B)
    coef = np.einsum("bij,bj->bi", A, theta_std)
    cov = A @ cov_std @ np.swapaxes(A, -1, -2)
    if prior is not None and method != "lmom":
        f = _evaluate(Spec(lik_key, ks, None), theta_std, y, mask, Xs, axes)
    loglik = -np.asarray(f)
    rename = {"scale": "log_scale", "rate": "logit_rate"}
    names = []
    for d in designs:
        for nm in d.names:
            head, _, rest = nm.partition(":")
            names.append(rename.get(head, head) + (":" + rest if rest else ""))
    return dict(
        family=lik.family,
        coef=coef.reshape(*batch_shape, p),
        cov=cov.reshape(*batch_shape, p, p),
        loglik=loglik.reshape(batch_shape),
        converged=conv.reshape(batch_shape),
        n_iter=n_iter.reshape(batch_shape),
        nobs=np.asarray(nobs).reshape(batch_shape),
        param_names=names,
        batch_shape=batch_shape,
        method=method,
        prior=prior,
        _theta_std=theta_std.reshape(*batch_shape, p),
        _designs=designs,
        _raw=raws,
        free_params=(len(fr) if fixed else None),
        _spec=spec,
        _data=y,
        _dmask=mask,
        _coef_all=coef.reshape(*batch_shape, p),
        _cov_all=cov.reshape(*batch_shape, p, p),
    )


# ------------------------------------------------------------- fit objects
@dataclass
class ModelFit:
    """A fitted model from either family, batched and possibly non-stationary.

    ``coef`` / ``cov`` are user-scale coefficients (``log_scale`` for the
    scale) and their covariance from the observed information. ``params``
    maps parameter names to natural values. Every fit has the same surface:
    ``summary()``, ``confint()``, ``params_at()``, ``return_level()`` and
    ``plot_diagnostics()``.
    """

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
    # Number of free coefficients when something is held fixed (None = all free).
    free_params: int | None = None
    _spec: Spec | None = field(default=None, repr=False)
    _data: np.ndarray | None = field(default=None, repr=False)
    _dmask: np.ndarray | None = field(default=None, repr=False)
    # Coefficients of every component (POT adds the exceedance rate to coef).
    _coef_all: np.ndarray | None = field(default=None, repr=False)
    _cov_all: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------ basics
    @property
    def _B(self) -> int:
        return int(np.prod(self.batch_shape, dtype=int))

    @property
    def likelihood(self):
        return self._spec.likelihood

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
        return int(len(self.param_names))

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

    def _coef_full(self) -> np.ndarray:
        """User-scale coefficients of every component (``(B, p_full)``)."""
        c = self.coef if self._coef_all is None else self._coef_all
        return np.asarray(c).reshape(self._B, -1)

    def _cov_full(self) -> np.ndarray:
        c = self.cov if self._cov_all is None else self._cov_all
        return np.asarray(c).reshape(self._B, *np.shape(c)[-2:])

    def _nat(self, theta_std=None):
        lik = self.likelihood
        th = jnp.asarray((self._theta_std if theta_std is None else theta_std).reshape(self._B, -1))
        Xs = [jnp.asarray(d.X) for d in self._designs]
        axes = tuple(None if d.shared else 0 for d in self._designs)
        return jax.vmap(lambda t, *X: natural_params(lik, t, X, self.ks), in_axes=(0, *axes))(th, *Xs)

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Natural parameters. Stationary: shape ``batch``; else ``(*batch, n)``."""
        out = {}
        for name, v in zip(self.likelihood.params, self._nat()):
            if name == "rate":
                continue
            v = np.asarray(v)
            out[name] = v[:, 0].reshape(self.batch_shape) if self.stationary else v.reshape(*self.batch_shape, -1)
        return out

    def summary(self):
        """Tidy :class:`pandas.DataFrame` of coefficients and standard errors."""
        import pandas as pd

        coef = np.asarray(self.coef).reshape(self._B, -1)
        se = self.se.reshape(self._B, -1)
        rows = [(b, nm, coef[b, j], se[b, j]) for b in range(self._B) for j, nm in enumerate(self.param_names)]
        return pd.DataFrame(rows, columns=["series", "term", "estimate", "std_error"])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        head = f"{type(self).__name__}(family={self.family!r}, batch_shape={self.batch_shape}, terms={self.param_names}"
        if self._B == 1:
            c = ", ".join(f"{n}={v:.4g}±{s:.2g}" for n, v, s in zip(self.param_names, np.ravel(self.coef), self.se.ravel()))
            return head + f", {c}, loglik={float(np.ravel(self.loglik)[0]):.4f})"
        return head + f", converged={float(np.mean(self.converged)):.1%})"

    # ---------------------------------------------------- prediction rows
    def _covariate_blocks(self, covariates):
        """Normalise ``covariates`` to ``{param: rows}``. A DataFrame is split by column names."""
        if covariates is None:
            return None
        if hasattr(covariates, "columns") and not isinstance(covariates, Mapping):
            out = {}
            for name, d in zip(self.likelihood.params, self._designs):
                if d.k > 1:
                    cols = [nm.split(":", 1)[1] for nm in d.names[1:]]
                    missing = [c for c in cols if c not in covariates.columns]
                    if missing:
                        raise ValueError(
                            f"covariates is missing column(s) {missing} for {name}; "
                            "fit with pandas covariates to match by name, or pass {param: rows}"
                        )
                    out[name] = covariates[cols].to_numpy(dtype=float)
            return out
        if not isinstance(covariates, Mapping):
            raise TypeError("covariates must be a mapping {parameter: rows} or a pandas DataFrame")
        unknown = set(covariates) - set(self.likelihood.params)
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)} in covariates; use {self.likelihood.params}")
        return covariates

    def _rows(self, covariates):
        """Prediction rows per parameter: user scale and standardised, each ``(B, m, k)``."""
        B = self._B
        blocks = self._covariate_blocks(covariates)
        raw, std = [], []
        for j, (name, d) in enumerate(zip(self.likelihood.params, self._designs)):
            if d.k == 1:
                r = s = np.ones((B, 1, 1))
            elif blocks is not None and name in blocks:
                r0 = np.asarray(blocks[name], float)
                r0 = r0[:, None] if r0.ndim == 1 else r0
                s = d.rows(r0, B)
                r = np.concatenate([np.ones((B, r0.shape[0], 1)), np.broadcast_to(r0, (B, *r0.shape))], axis=-1)
            else:
                rw = np.asarray(self._raw[j], float)
                r = np.broadcast_to(rw, (B, *rw.shape)) if rw.ndim == 2 else rw
                s = d.batched(B)
            raw.append(r)
            std.append(s)
        m = max(X.shape[1] for X in raw)
        for lst in (raw, std):
            for i, X in enumerate(lst):
                if X.shape[1] == 1 and m > 1:
                    lst[i] = np.broadcast_to(X, (B, m, X.shape[-1]))
                elif X.shape[1] != m:
                    raise ValueError(
                        "covariate blocks disagree on the number of rows; "
                        "pass the same number of rows for each non-stationary parameter"
                    )
        return [np.ascontiguousarray(X) for X in raw], [np.ascontiguousarray(X) for X in std], m

    def params_at(self, covariates=None) -> dict[str, np.ndarray]:
        """Natural parameters at covariate rows, each ``(*batch, m)``.

        ``covariates`` is ``{param: rows}`` or a DataFrame with the fitted
        covariate columns. Default: the fitted rows.
        """
        raw, _, m = self._rows(covariates)
        coef = self._coef_full()
        lik = self.likelihood
        out, off = {}, 0
        for name, link, X, k in zip(lik.params, lik.links, raw, self.ks):
            eta = np.einsum("bmk,bk->bm", X, coef[:, off : off + k])
            out[name] = np.asarray(LINKS[link](jnp.asarray(eta))).reshape(*self.batch_shape, m)
            off += k
        return out

    # ------------------------------------------------------- return levels
    def _level_context(self, T, kind):
        """Per-series, per-period constants ``c`` (B, nT, 2) for the family's level map."""
        B = self._B
        c = np.zeros((B, T.size, 2))
        if self.family == "gev":
            if kind not in (None, "block"):
                raise ValueError("kind='interval' applies to POT fits; a GEV return level is per block")
            c[..., 0] = -np.log(-np.log(1.0 - 1.0 / T))
            return c
        rate = self._spec.lik == "pot"
        if kind not in (None, "interval", "block"):
            raise ValueError("kind must be 'interval' (mean recurrence) or 'block' (the block maximum, e.g. annual)")
        if kind == "block" and not rate:
            raise ValueError("kind='block' needs the exceedance rate: fit with fit_pot")
        if rate:
            h = np.log(T) if kind in (None, "interval") else -np.log(-np.log1p(-1.0 / T))
            c[..., 0] = h + np.log(float(self.npy))
        else:
            c[..., 0] = np.log(T)
        u = getattr(self, "threshold", None)
        u = np.asarray(0.0 if u is None else u, float)
        if u.ndim and u.shape not in (self.batch_shape, ()):
            raise ValueError("return levels need one threshold per series (a time-varying threshold has none)")
        c[..., 1] = np.broadcast_to(u, self.batch_shape).reshape(B)[:, None]
        return c

    def _level_fn(self):
        lik, ks = self.likelihood, self.ks

        def rl(theta, c, T, *Xr):  # theta user-scale, Xr (m, k) user-scale rows → (m, nT)
            etas, off = [], 0
            for X, k in zip(Xr, ks):
                etas.append(X @ theta[off : off + k])
                off += k
            f = lambda cc, t: lik.level(etas, t, cc)  # noqa: E731
            return jax.vmap(f, in_axes=(0, 0), out_axes=1)(c, T)

        return rl

    def return_level(
        self,
        period,
        *,
        covariates=None,
        ci: str | None = None,
        alpha: float = 0.05,
        n_boot: int = 500,
        seed: int = 0,
        profile_points: int = 41,
        kind: str | None = None,
    ) -> ReturnLevels:
        """T-period return levels, optionally with confidence intervals.

        Parameters
        ----------
        period
            Return period(s), each > 1: in blocks (years) for block maxima and
            POT, in exceedances for a GPD fitted without the rate.
        covariates
            For non-stationary models: ``{"loc": rows, "scale": rows, ...}``
            with new covariate rows (user scale), or a DataFrame holding the
            fitted covariate columns. Default: the fitted rows (the effective
            return level of each observation).
        ci
            ``None``, ``"delta"`` (Wald, observed information), ``"profile"``
            (profile likelihood: the return level replaces the location, or for
            POT the scale, at each covariate row) or ``"bootstrap"``
            (parametric, percentile). All three work for every model,
            stationary or not; fixed parameters stay fixed.
        kind
            POT only. ``"interval"`` (default): the level exceeded on average
            once every ``period`` years (Coles, 2001, §4.3.3). ``"block"``: the
            level the block maximum (an annual or seasonal maximum, with
            ``npy`` observations per block) exceeds with probability
            ``1/period``, from the Poisson rate of exceedances.
        """
        T = np.atleast_1d(np.asarray(period, dtype=np.float64))
        if np.any(T <= 1):
            raise ValueError("return periods must be > 1")
        raw, _, m = self._rows(covariates)
        c = self._level_context(T, kind)
        rl = self._level_fn()
        coef = jnp.asarray(self._coef_full())
        args = (jnp.asarray(c), jnp.asarray(T), *[jnp.asarray(X) for X in raw])
        in_axes = (0, 0, None, *([0] * len(raw)))
        est = np.asarray(jax.vmap(rl, in_axes=in_axes)(coef, *args))  # (B, m, nT)
        lo = hi = None
        if ci == "delta":
            from scipy.stats import norm

            J = np.asarray(jax.vmap(jax.jacfwd(rl), in_axes=in_axes)(coef, *args))  # (B, m, nT, p)
            var = np.einsum("bmtp,bpq,bmtq->bmt", J, self._cov_full(), J)
            sd = np.sqrt(np.clip(var, 0, None))
            zq = norm.ppf(1 - alpha / 2)
            lo, hi = est - zq * sd, est + zq * sd
        elif ci == "profile":
            from .uncertainty import profile_levels

            lo, hi = profile_levels(self, T, covariates=covariates, alpha=alpha, n_points=profile_points, kind=kind)
        elif ci == "bootstrap":
            from .uncertainty import _bootstrap_full

            boot = _bootstrap_full(self, n_boot=n_boot, seed=seed)  # (B, R, p_full)
            f = jax.vmap(jax.vmap(rl, in_axes=(0, None, None, *([None] * len(raw)))), in_axes=in_axes)
            draws = np.asarray(f(jnp.asarray(boot), *args))  # (B, R, m, nT)
            with np.errstate(invalid="ignore"):
                lo = np.nanquantile(draws, alpha / 2, axis=1)
                hi = np.nanquantile(draws, 1 - alpha / 2, axis=1)
        elif ci is not None:
            raise ValueError("ci must be None, 'delta', 'profile' or 'bootstrap'")
        squeeze = self.stationary and covariates is None
        shape_out = (*self.batch_shape, T.size) if squeeze else (*self.batch_shape, m, T.size)
        rs = lambda a: None if a is None else np.asarray(a).reshape(shape_out)  # noqa: E731
        return ReturnLevels(T, rs(est), rs(lo), rs(hi), ci, alpha)

    # ------------------------------------------------------- coefficients
    def confint(self, alpha: float = 0.05, method: str = "profile", n_points: int = 41):
        """Confidence intervals for every coefficient, as a DataFrame.

        ``method`` is ``"profile"`` (profile likelihood, the default) or
        ``"wald"``. Fixed coefficients are reported with equal bounds.
        """
        import pandas as pd
        from scipy.stats import norm

        coef = np.asarray(self.coef).reshape(self._B, -1)
        if method == "wald":
            se = self.se.reshape(self._B, -1)
            zq = norm.ppf(1 - alpha / 2)
            lo, hi = coef - zq * se, coef + zq * se
        elif method == "profile":
            from .uncertainty import profile_coefs

            lo, hi = profile_coefs(self, alpha=alpha, n_points=n_points)
        else:
            raise ValueError("method must be 'profile' or 'wald'")
        rows = [(b, nm, coef[b, j], lo[b, j], hi[b, j]) for b in range(self._B) for j, nm in enumerate(self.param_names)]
        return pd.DataFrame(rows, columns=["series", "term", "estimate", "lower", "upper"])

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


def etas_at_obs(fit, theta_std=None):
    """Linear predictors at the fitted observations, each ``(B, n)``."""
    th = jnp.asarray((fit._theta_std if theta_std is None else theta_std).reshape(fit._B, -1))
    Xs = [jnp.asarray(d.X) for d in fit._designs]
    axes = tuple(None if d.shared else 0 for d in fit._designs)
    return jax.vmap(lambda t, *X: etas_of(fit._spec, t, X), in_axes=(0, *axes))(th, *Xs)


def nll_at(fit, theta_std):
    """Full objective (with prior) at standardised coefficients ``(B, p)``."""
    Xs = [jnp.asarray(d.X) for d in fit._designs]
    axes = tuple(None if d.shared else 0 for d in fit._designs)
    f = jax.jit(jax.vmap(full_nll(Spec(fit._spec.lik, fit._spec.ks, fit.prior)), in_axes=(0, 0, 0, *axes)))
    return np.asarray(f(jnp.asarray(theta_std), jnp.asarray(fit._data), jnp.asarray(fit._dmask), *Xs))
