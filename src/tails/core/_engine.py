"""The likelihood engine behind every fit: two model families, options not models.

**GEV family.** One likelihood for the ``r`` largest values of each block
(Coles, 2001, §3.5). ``r = 1`` is exactly the GEV likelihood of block maxima,
so ``fit_gev(y)`` and ``fit_gev(top, r=k)`` are the same model with a
different amount of data per block.

**GPD family.** One likelihood for threshold excesses. Turning on the
exceedance-rate component adds a Bernoulli term for every observation (does
it exceed the threshold?), which is the peaks-over-threshold model. So
``fit_gpd`` and ``fit_pot`` are the same model, with and without the rate.

Every parameter has a linear predictor ``eta_j = X_j @ beta_j`` with a link:
identity for location and shape, log for scale, logit for the exceedance
probability. Stationary models are the special case ``X_j = 1``. Any
coefficient can be held fixed (a Gumbel fit fixes the shape at 0).

Covariates are centred and scaled internally (better conditioning,
intercepts that mean "parameter at the average covariate"). Coefficients and
covariances are mapped back to the user's scale by :func:`back_transform_matrix`.

Each :class:`Likelihood` supplies three things, and everything else (fitting,
profile likelihoods, the bootstrap, return levels) is generic:

* ``contrib(y, mask, *etas)`` — per-observation log-likelihood in eta space;
* ``derivs`` — optional closed-form per-observation derivatives (else autodiff);
* ``level`` / ``invert`` — return level from the parameters at one covariate
  row, and the inverse map that profile likelihoods use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np

from .._jax import jax, jnp
from ._derivs import gev_derivs, gpd_derivs
from ._special import expm1_over_x
from .distributions import gev_logcdf, gev_logpdf, gev_ppf, gpd_logpdf, gpd_ppf

LINKS = {"identity": (lambda e: e), "log": jnp.exp, "logit": jax.nn.sigmoid}


# --------------------------------------------------------------- likelihoods
def _gev_contrib(y, mask, mu, phi, xi):
    """Block log-likelihood. ``y`` is ``(n,)`` (maxima) or ``(n, r)`` (r largest, decreasing)."""
    sc = jnp.exp(phi)
    if y.ndim == 1:
        lp = gev_logpdf(jnp.where(mask, y, mu), mu, sc, xi)
        return jnp.where(mask, lp, 0.0)
    mu2, sc2, xi2 = mu[:, None], sc[:, None], xi[:, None]
    zs = jnp.where(mask, y, mu2)
    lp = gev_logpdf(zs, mu2, sc2, xi2)
    lc = gev_logcdf(zs, mu2, sc2, xi2)
    feasible = jnp.all(jnp.where(mask, jnp.isfinite(lp), True), axis=-1)
    ok = mask & feasible[:, None]
    lps = jnp.where(ok, lp, 0.0)
    lcs = jnp.where(ok, lc, 0.0)
    nxt = jnp.concatenate([mask[:, 1:], jnp.zeros_like(mask[:, :1])], axis=1)
    last = mask & ~nxt
    ll = jnp.sum(lps - lcs, axis=-1) + jnp.sum(jnp.where(last, lcs, 0.0), axis=-1)
    return jnp.where(feasible, ll, -jnp.inf)


def _masked(mask, lp, G, H):
    w = mask.astype(lp.dtype)
    G = [g * w for g in G]
    H = [[None if h is None else h * w for h in row] for row in H]
    return jnp.where(mask, lp, 0.0), G, H


def _gev_derivs(y, mask, mu, phi, xi):
    if y.ndim != 1:
        return None  # r > 1: autodiff of the block likelihood
    return _masked(mask, *gev_derivs(jnp.where(mask, y, mu), mu, phi, xi, stacked=False))


def _gpd_contrib(y, mask, phi, xi):
    lp = gpd_logpdf(jnp.where(mask, y, 0.0), jnp.exp(phi), xi)
    return jnp.where(mask, lp, 0.0)


def _gpd_derivs(y, mask, phi, xi):
    return _masked(mask, *gpd_derivs(jnp.where(mask, y, 0.0), phi, xi, stacked=False))


def _pot_contrib(y, mask, phi, xi, eta):
    """GPD on the excesses plus Bernoulli(exceed) on every valid observation.

    ``mask`` marks valid observations; an exceedance is a valid ``y > 0``
    (``y`` holds the excess, 0 below the threshold).
    """
    e = mask & (y > 0.0)
    lp = gpd_logpdf(jnp.where(e, y, 0.0), jnp.exp(phi), xi)
    return jnp.where(e, lp - jax.nn.softplus(-eta), jnp.where(mask, -jax.nn.softplus(eta), 0.0))


def _pot_derivs(y, mask, phi, xi, eta):
    e = mask & (y > 0.0)
    lp, G, H = gpd_derivs(jnp.where(e, y, 0.0), phi, xi, stacked=False)
    zeta = jax.nn.sigmoid(eta)
    ef = e.astype(y.dtype)
    w = mask.astype(y.dtype)
    lp = jnp.where(e, lp - jax.nn.softplus(-eta), jnp.where(mask, -jax.nn.softplus(eta), 0.0))
    z0 = jnp.zeros_like(lp)
    G = [jnp.where(e, G[0], z0), jnp.where(e, G[1], z0), (ef - zeta) * w]
    H = [
        [jnp.where(e, H[0][0], z0), jnp.where(e, H[0][1], z0), z0],
        [None, jnp.where(e, H[1][1], z0), z0],
        [None, None, -zeta * (1.0 - zeta) * w],
    ]
    return lp, G, H


def _gev_level(etas, T, c):
    """GEV quantile exceeded with probability ``1/T`` per block."""
    mu, phi, xi = etas
    return gev_ppf(1.0 - 1.0 / T, mu, jnp.exp(phi), xi)


def _gev_invert(z, etas, c):
    """Location (eta) that puts the level at ``z``; ``c[0] = -log(-log(1 - 1/T))``."""
    _, phi, xi = etas
    w = c[0]
    return z - jnp.exp(phi) * w * expm1_over_x(xi * w), jnp.asarray(True)


def _gpd_log_m(etas, c):
    """Log of the expected number of exceedances in the return period."""
    if len(etas) == 3:
        return c[0] - jax.nn.softplus(-etas[2])
    return c[0]


def _gpd_level(etas, T, c):
    """``c = (log(T-term * npy), u)``; NaN when fewer than one exceedance is expected."""
    log_m = _gpd_log_m(etas, c)
    q = gpd_ppf(-jnp.expm1(-log_m), jnp.exp(etas[0]), etas[1])
    return jnp.where(log_m > 0.0, c[1] + q, jnp.nan)


def _gpd_invert(z, etas, c):
    """Log scale that puts the level at ``z``."""
    xi = etas[1]
    log_m = _gpd_log_m(etas, c)
    w = jnp.where(log_m > 0.0, log_m, 1.0)
    denom = w * expm1_over_x(xi * w)
    span = z - c[1]
    ok = (log_m > 0.0) & (span > 0.0) & (denom > 0.0)
    return jnp.log(jnp.where(ok, span / jnp.where(ok, denom, 1.0), 1.0)), ok


@dataclass(frozen=True)
class Likelihood:
    """One registered likelihood: the only family-specific code in the package."""

    key: str
    family: str
    params: tuple[str, ...]
    links: tuple[str, ...]
    contrib: Callable
    derivs: Callable | None
    level: Callable
    invert: Callable
    target: str  # parameter that profile likelihoods solve for from the return level


GEV = Likelihood("gev", "gev", ("loc", "scale", "shape"), ("identity", "log", "identity"),
                 _gev_contrib, _gev_derivs, _gev_level, _gev_invert, "loc")
GPD = Likelihood("gpd", "gpd", ("scale", "shape"), ("log", "identity"),
                 _gpd_contrib, _gpd_derivs, _gpd_level, _gpd_invert, "scale")
POT = Likelihood("pot", "gpd", ("scale", "shape", "rate"), ("log", "identity", "logit"),
                 _pot_contrib, _pot_derivs, _gpd_level, _gpd_invert, "scale")
LIKELIHOODS = {"gev": GEV, "gpd": GPD, "pot": POT}


# ----------------------------------------------------------------- shape priors
def _martins_stedinger(xi):
    """Log Beta(6, 9) prior on -xi over (-0.5, 0.5) (Martins & Stedinger, 2000)."""
    inside = (xi > -0.5) & (xi < 0.5)
    xs = jnp.where(inside, xi, 0.0)
    val = 5.0 * jnp.log(0.5 - xs) + 8.0 * jnp.log(0.5 + xs) + 9.799126537211386
    return jnp.where(inside, val, -jnp.inf)


PRIORS = {None: None, "martins-stedinger": _martins_stedinger}


# ------------------------------------------------------------------- models
@dataclass(frozen=True)
class Spec:
    """A model: likelihood, coefficients per parameter, prior and fixed coefficients.

    ``free`` lists the optimised entries of the coefficient vector (``None`` =
    all). Fixed entries take their value from the ``base`` vector passed to the
    objective, so any coefficient can be held (a fixed shape, a zero slope).
    """

    lik: str
    ks: tuple[int, ...]
    prior: str | None = None
    free: tuple[int, ...] | None = None

    @property
    def likelihood(self) -> Likelihood:
        return LIKELIHOODS[self.lik]

    @property
    def p(self) -> int:
        return int(sum(self.ks))

    def offsets(self) -> list[int]:
        return [int(v) for v in np.concatenate([[0], np.cumsum(self.ks)[:-1]])]

    def index(self, param: str) -> int:
        """Position of a parameter's intercept in the coefficient vector."""
        return self.offsets()[self.likelihood.params.index(param)]

    def shape_coefs(self) -> list[int]:
        j = self.likelihood.params.index("shape")
        o = self.offsets()[j]
        return list(range(o, o + self.ks[j]))

    def with_ks(self, ks) -> Spec:
        """Same model with other design sizes (fixed entries must be intercepts)."""
        if self.free is None:
            return Spec(self.lik, tuple(ks), self.prior, None)
        old, new = self.offsets(), [int(v) for v in np.concatenate([[0], np.cumsum(ks)[:-1]])]
        fixed = set(range(self.p)) - set(self.free)
        moved = {new[old.index(i)] for i in fixed}
        return Spec(self.lik, tuple(ks), self.prior, tuple(i for i in range(int(sum(ks))) if i not in moved))


def etas_of(spec: Spec, theta, Xs):
    """Linear predictors; an intercept-only parameter is a broadcast scalar (no matmul)."""
    out, off = [], 0
    for X, k in zip(Xs, spec.ks):
        out.append(jnp.broadcast_to(theta[off], X.shape[:1]) if k == 1 else X @ theta[off : off + k])
        off += k
    return out


def natural_params(family, theta, Xs, ks):
    """Natural parameters from coefficients (``family`` is a :class:`Likelihood`)."""
    out, off = [], 0
    for X, k, link in zip(Xs, ks, family.links):
        out.append(LINKS[link](X @ theta[off : off + k]))
        off += k
    return out


def _full(spec, th, base):
    if spec.free is None:
        return th
    return base.at[jnp.asarray(spec.free)].set(th)


@cache
def full_nll(spec: Spec):
    """``nll(theta_full, y, mask, *Xs)`` including the shape prior."""
    lik, prior_fn = spec.likelihood, PRIORS[spec.prior]

    def nll(theta, y, mask, *Xs):
        ll = jnp.sum(lik.contrib(y, mask, *etas_of(spec, theta, Xs)))
        if prior_fn is not None:
            ll = ll + prior_fn(theta[spec.index("shape")])
        return -ll

    return nll


@cache
def objective(spec: Spec):
    """``(fun, vgh)`` over the free coefficients: ``fun(th, y, mask, base, *Xs)``.

    ``vgh`` uses the "linear predictor" trick: the log-density of observation
    ``i`` depends on ``theta`` only through its handful of ``eta_ij``, so
    per-observation derivatives in eta space (closed form where available,
    forward-mode autodiff otherwise) give ``g = sum_i X_i' d_i`` and
    ``H = sum_i X_i' D_i X_i``. This is 3-6x faster than ``jax.hessian`` of the
    summed likelihood.
    """
    lik, prior_fn, ks = spec.likelihood, PRIORS[spec.prior], spec.ks
    P = len(ks)
    nll = full_nll(spec)
    free = None if spec.free is None else jnp.asarray(spec.free)

    def fun(th, y, mask, base, *Xs):
        return nll(_full(spec, th, base), y, mask, *Xs)

    def eta_derivs(y, mask, etas):
        d = lik.derivs(y, mask, *etas) if lik.derivs is not None else None
        if d is not None:
            return d
        f = lambda *e: jnp.sum(lik.contrib(y, mask, *e))  # noqa: E731
        lp = lik.contrib(y, mask, *etas)
        G = list(jax.grad(f, argnums=tuple(range(P)))(*etas))
        H = [[None] * P for _ in range(P)]
        ones = [jnp.ones_like(e) for e in etas]
        for b in range(P):
            tang = [jnp.zeros_like(e) for e in etas]
            tang[b] = ones[b]
            _, col = jax.jvp(lambda *e: jax.grad(f, argnums=tuple(range(P)))(*e), tuple(etas), tuple(tang))
            for a in range(b + 1):
                H[a][b] = col[a]
        return lp, G, H

    def vgh(th, y, mask, base, *Xs):
        theta = _full(spec, th, base)
        etas = etas_of(spec, theta, Xs)
        lp, G, Hh = eta_derivs(y, mask, etas)
        f = -jnp.sum(lp)
        def xt(a, v):  # X_a' v, with intercept-only designs as plain sums
            return jnp.sum(v, axis=0, keepdims=True) if ks[a] == 1 else Xs[a].T @ v

        gs = [-xt(a, G[a]) for a in range(P)]
        Hb = [[None] * P for _ in range(P)]
        for a in range(P):
            for b in range(a, P):
                h = Hh[a][b]
                Hb[a][b] = -xt(a, h[:, None] if ks[b] == 1 else Xs[b] * h[:, None])
                if b > a:
                    Hb[b][a] = Hb[a][b].T
        g = jnp.concatenate(gs)
        H = jnp.block(Hb)
        if prior_fn is not None:
            o = spec.index("shape")
            x = theta[o]
            f = f - prior_fn(x)
            g = g.at[o].add(-jax.grad(prior_fn)(x))
            H = H.at[o, o].add(-jax.hessian(prior_fn)(x))
        if free is not None:
            g = g[free]
            H = H[free][:, free]
        return f, g, H

    return fun, vgh


# ------------------------------------------------------------------- designs
@dataclass
class Design:
    """Standardised design matrices for one parameter."""

    X: np.ndarray  # (n, k) shared or (B, n, k) per series, intercept first
    names: list[str]
    mean: np.ndarray  # (k-1,) or (B, k-1)
    sd: np.ndarray
    shared: bool

    @property
    def k(self) -> int:
        return int(self.X.shape[-1])

    def rows(self, raw, B: int) -> np.ndarray:
        """User-scale covariate rows ``(m, k-1)`` → standardised rows ``(B, m, k)``."""
        raw = np.asarray(raw, dtype=np.float64)
        raw = raw[:, None] if raw.ndim == 1 else raw
        if raw.ndim != 2:
            raise ValueError("covariate rows must have shape (m,) or (m, k)")
        mean, sd = np.asarray(self.mean, float), np.asarray(self.sd, float)
        if raw.shape[-1] != mean.shape[-1]:
            raise ValueError(f"expected {mean.shape[-1]} covariate column(s) {self.names[1:]}, got {raw.shape[-1]}")
        if mean.ndim == 1:
            Z = np.broadcast_to((raw - mean) / sd, (B, *raw.shape))
        else:
            Z = (raw[None] - mean[:, None, :]) / sd[:, None, :]
        return np.concatenate([np.ones((B, raw.shape[0], 1)), Z], axis=-1)

    def batched(self, B: int) -> np.ndarray:
        X = np.asarray(self.X, dtype=np.float64)
        return np.broadcast_to(X, (B, *X.shape)).copy() if X.ndim == 2 else X


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
        raise ValueError(f"covariate has {cov.shape[0]} rows but data has {n} observations (blocks)")
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


def build_design(cov, n: int, batch_shape: tuple, prefix: str) -> Design:
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


def raw_design(cov, n):
    """User-scale design (intercept + raw covariates), for prediction and plots."""
    if cov is None:
        return np.ones((n, 1))
    arr = cov.to_numpy() if hasattr(cov, "to_numpy") else cov
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim > 2:
        arr = arr.reshape(-1, n, arr.shape[-1])
    return np.concatenate([np.ones((*arr.shape[:-1], 1)), arr], axis=-1)


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
