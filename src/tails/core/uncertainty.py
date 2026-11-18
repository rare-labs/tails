"""Uncertainty quantification: profile likelihood and parametric bootstrap.

Both are batched: profile likelihoods for many series, covariate rows and
return periods are traced with a continuation ``lax.scan`` inside one
``vmap``; bootstrap replicates of all series are refitted in a single batched
optimisation. Stationary GEV intervals live in ``profile_return_level`` and
``bootstrap_coef``. ``profile_covariate_return_level`` is the same profile for
a GEV with covariates (the location intercept is the nuisance parameter that
the return level replaces). POT intervals profile, or bootstrap, the joint
model of the excesses and the binomial exceedance rate.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm

from .._jax import jax, jnp
from ._engine import FAMILIES, PRIORS, back_transform_matrix, make_nll, make_vgh, natural_params
from ._special import expm1_over_x
from .distributions import gev_logpdf, gev_ppf, gpd_logpdf, gpd_ppf
from .optim import _Settings, _single

__all__ = [
    "profile_return_level",
    "profile_fixed_shape_return_level",
    "profile_covariate_return_level",
    "profile_pot_return_level",
    "bootstrap_coef",
    "bootstrap_pot_return_level",
]


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


# ------------------------------------------------------- profile bookkeeping
def _se_from_bounds(lower, upper, alpha):
    zq = float(norm.ppf(1.0 - alpha / 2.0))
    return (np.asarray(upper, float) - np.asarray(lower, float)) / (2.0 * zq)


def _level_grids(est, se, n_points, up=12.0, down=6.0):
    """Grids of return levels centred on ``est``, in units of the Wald standard error."""
    se = np.where(np.isfinite(se) & (se > 0), se, 0.1 * np.abs(est) + 1e-8)
    centre = np.where(np.isfinite(est), est, 0.0)
    n_dn = max(int(n_points) // 3, 4)
    n_up = max(int(n_points) - n_dn, 4)
    zs_up = centre[..., None] + se[..., None] * np.linspace(0.0, up, n_up)
    zs_dn = centre[..., None] - se[..., None] * np.linspace(0.0, down, n_dn)
    return zs_dn, zs_up


def _search_side(eval_dev, nu0, zs, crit, max_extend=2):
    """Deviance path along ``zs``, extended while the χ² level is still uncrossed.

    ``eval_dev(nu, zs) -> (nu_final, deviance)`` with ``deviance`` shaped like
    ``zs``. The grid is ordered from the estimate outward. Returns the linearly
    interpolated crossing, NaN where the path never reaches ``crit``.
    """
    nu = np.asarray(nu0, dtype=np.float64)
    z_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    bound = np.full(zs.shape[0], np.nan)
    for extend in range(max_extend + 1):
        nu, dev = eval_dev(nu, zs)
        if extend:
            z_parts.append(np.asarray(zs[:, 1:], dtype=np.float64))
            d_parts.append(dev[:, 1:])
        else:
            z_parts.append(np.asarray(zs, dtype=np.float64))
            d_parts.append(dev)
        bound = _crossing(np.concatenate(z_parts, axis=1), np.concatenate(d_parts, axis=1), crit)
        above = d_parts[-1] > crit
        idx = np.argmax(above, axis=1)
        found = above[np.arange(len(idx)), idx] & (idx > 0)
        # A previous chunk may already have crossed; don't let a short last chunk hide it.
        if extend:
            found = found | np.isfinite(bound)
        last = dev[:, -1]
        need = (~found) & np.isfinite(last) & (last < crit)
        if not np.any(need) or extend == max_extend:
            return bound
        step = zs[:, -1] - zs[:, -2]
        n_ext = zs.shape[1] - 1
        extra = zs[:, -1:] + step[:, None] * np.arange(1, n_ext + 1)
        zs = np.concatenate([zs[:, -1:], extra], axis=1)
    return bound


def _tile_batch(a, repeats):
    """Repeat each batch row ``repeats`` times: ``(B, ...)`` → ``(B * repeats, ...)``."""
    a = np.asarray(a)
    return np.repeat(a, repeats, axis=0)


# ----------------------------------------------- non-stationary GEV profile
def _as_cov_matrix(cov):
    arr = np.asarray(cov, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("covariates must have shape (m,) or (m, k)")
    return arr


def _standardize_rows(design, cov, B):
    """User-scale covariate rows → standardised design ``(B, m, k)`` (intercept first)."""
    raw = _as_cov_matrix(cov)
    mean = np.asarray(design.mean, dtype=np.float64)
    sd = np.asarray(design.sd, dtype=np.float64)
    if raw.shape[-1] != mean.shape[-1]:
        raise ValueError(f"expected {mean.shape[-1]} covariate column(s), got {raw.shape[-1]}")
    if mean.ndim == 1:
        Z = (raw - mean) / sd
        X = np.concatenate([np.ones((raw.shape[0], 1)), Z], axis=-1)
        return np.broadcast_to(X, (B, X.shape[0], X.shape[1])).copy()
    Z = (raw[None, :, :] - mean[:, None, :]) / sd[:, None, :]
    ones = np.ones((B, raw.shape[0], 1))
    return np.concatenate([ones, Z], axis=-1)


def _design_batch(design, B):
    X = np.asarray(design.X, dtype=np.float64)
    if X.ndim == 2:
        X = np.broadcast_to(X, (B, X.shape[0], X.shape[1])).copy()
    return X


def _prediction_designs(fit, covariates, B):
    """Standardised rows ``(B, m, k)`` at which the return level is evaluated."""
    names = ("loc", "scale", "shape")
    cols = []
    for j, name in enumerate(names):
        design = fit._designs[j]
        k = design.X.shape[-1]
        if k == 1:
            cols.append(np.ones((B, 1, 1)))
        elif covariates is not None and name in covariates:
            cols.append(_standardize_rows(design, covariates[name], B))
        else:
            cols.append(_design_batch(design, B))
    m = max(X.shape[1] for X in cols)
    out = []
    for X in cols:
        if X.shape[1] == m:
            out.append(X)
        elif X.shape[1] == 1:
            out.append(np.broadcast_to(X, (B, m, X.shape[-1])).copy())
        else:
            raise ValueError(
                "covariate blocks disagree on the number of rows; "
                "pass the same number of rows for each non-stationary parameter"
            )
    return out


_NS_PATH: dict = {}


def _ns_profile_path(k0, k1, k2, prior):
    """Continuation profile for one GEV parameterisation ``(k_loc, k_scale, k_shape)``."""
    key = (k0, k1, k2, prior)
    if key in _NS_PATH:
        return _NS_PATH[key]
    prior_fn = PRIORS[prior]
    n_pre = (k0 - 1) + k1

    def nll(nu, y, mask, z, w, Xloc, Xsc, Xsh, xloc, xsc, xsh):
        loc_rest = nu[: k0 - 1]
        scale_th = nu[k0 - 1 : k0 - 1 + k1]
        shape_th = nu[k0 - 1 + k1 :]
        # Location intercept implied by the return level at the prediction row.
        # ``loc_free`` has a zero intercept so the dot is only the other location terms.
        loc_free = jnp.concatenate([jnp.zeros((1,), dtype=nu.dtype), loc_rest])
        sig_star = jnp.exp(jnp.dot(xsc, scale_th))
        xi_star = jnp.dot(xsh, shape_th)
        phi = w * expm1_over_x(xi_star * w)
        loc0 = z - jnp.dot(xloc, loc_free) - sig_star * phi
        loc_th = loc_free.at[0].set(loc0)
        mu = Xloc @ loc_th
        sig = jnp.exp(Xsc @ scale_th)
        xi = Xsh @ shape_th
        ysafe = jnp.where(mask, y, mu)
        lp = gev_logpdf(ysafe, mu, sig, xi)
        val = -jnp.sum(jnp.where(mask, lp, 0.0))
        if prior_fn is not None:
            val = val - prior_fn(shape_th[0])
        return val

    def path(nu0, y, mask, zs, w, Xloc, Xsc, Xsh, xloc, xsc, xsh):
        def step(prev, z):
            args = (y, mask, z, w, Xloc, Xsc, Xsh, xloc, xsc, xsh)
            f_prev = nll(prev, *args)
            alt = jnp.concatenate([prev[:n_pre], jnp.zeros((k2,), dtype=prev.dtype)])
            start = jnp.where(jnp.isfinite(f_prev), prev, alt)
            r = _single(nll, start, args, _SET)
            nxt = jnp.where(r.converged, r.x, prev)
            return nxt, jnp.where(r.converged, r.fun, jnp.nan)

        final, fs = jax.lax.scan(step, nu0, zs)
        return final, fs

    fn = jax.jit(jax.vmap(path, in_axes=(0,) * 11))
    _NS_PATH[key] = fn
    return fn


def profile_covariate_return_level(fit, period, covariates=None, alpha=0.05, n_points=41):
    """Profile-likelihood intervals for GEV return levels, with or without covariates.

    The return level at each requested covariate row replaces the location
    intercept; the remaining coefficients (location slopes, log-scale, shape)
    are the nuisance parameters. A Martins–Stedinger prior, when the fit used
    one, stays in the profiled objective. Returns ``(lower, upper)`` of shape
    ``(B, m, len(period))``, where ``m`` is the number of covariate rows
    (``m = 1`` for a stationary fit). Bounds are NaN when the deviance does
    not cross the χ² threshold inside the searched range.
    """
    if fit.family != "gev":
        raise ValueError("profile_covariate_return_level is for GEV fits")
    if fit.free_params is not None:
        raise ValueError("fixed-shape profiles use profile_fixed_shape_return_level")
    T = np.atleast_1d(np.asarray(period, dtype=np.float64))
    B = int(np.prod(fit.batch_shape, dtype=int))
    rl = fit.return_level(T, covariates=covariates, ci="delta", alpha=alpha)
    est = np.asarray(rl.estimate, dtype=np.float64).reshape(B, -1, T.size)
    se = _se_from_bounds(rl.lower, rl.upper, alpha).reshape(B, -1, T.size)
    m = est.shape[1]
    pred = _prediction_designs(fit, covariates, B)
    if any(X.shape[1] != m for X in pred):
        raise ValueError(
            f"return level has {m} covariate row(s) but the prediction design does not; "
            "pass covariates with one row per level, or omit them to use the fitted rows"
        )
    obs = [_design_batch(d, B) for d in fit._designs]
    k0, k1, k2 = (d.X.shape[-1] for d in fit._designs)
    theta = np.asarray(fit._theta_std, dtype=np.float64).reshape(B, -1)
    nu0 = theta[:, 1:]
    nll_hat = -np.asarray(fit.loglik, dtype=np.float64).reshape(B)
    if fit.prior is not None:
        xi0 = theta[:, k0 + k1]
        nll_hat = nll_hat - np.asarray(PRIORS[fit.prior](jnp.asarray(xi0)), dtype=np.float64)
    repeats = m * T.size
    N = B * repeats
    nu = _tile_batch(nu0, repeats)
    y = _tile_batch(fit._y, repeats)
    mask = _tile_batch(fit._mask, repeats)
    nll_hat_flat = _tile_batch(nll_hat, repeats)
    # Prediction rows are (B, m, k); repeat each row across periods, in the same
    # order as ``est`` reshaped C-contiguously (row, then period).
    def _rows_then_periods(X):
        # (B, m, k) → (B, m, nT, k) → (N, k)
        X = np.repeat(X[:, :, None, :], T.size, axis=2)
        return np.ascontiguousarray(X.reshape(N, X.shape[-1]))

    xloc, xsc, xsh = (_rows_then_periods(X) for X in pred)
    Xloc, Xsc, Xsh = (np.ascontiguousarray(_tile_batch(X, repeats)) for X in obs)
    w = -np.log(-np.log(1.0 - 1.0 / T))
    w = np.broadcast_to(w, (B, m, T.size)).reshape(N).copy()
    path = _ns_profile_path(k0, k1, k2, fit.prior)
    args = (
        jnp.asarray(y),
        jnp.asarray(mask),
        jnp.asarray(w),
        jnp.asarray(Xloc),
        jnp.asarray(Xsc),
        jnp.asarray(Xsh),
        jnp.asarray(xloc),
        jnp.asarray(xsc),
        jnp.asarray(xsh),
    )

    def eval_dev(nu_cur, zs):
        final, fs = path(jnp.asarray(nu_cur), args[0], args[1], jnp.asarray(zs), *args[2:])
        dev = 2.0 * (np.asarray(fs) - nll_hat_flat[:, None])
        dev = np.where(np.isfinite(dev), dev, np.inf)
        return np.asarray(final), dev

    crit = float(chi2.ppf(1.0 - alpha, 1))
    finite = np.isfinite(est)
    zs_dn, zs_up = _level_grids(est, se, n_points)
    lo = _search_side(eval_dev, nu, np.ascontiguousarray(zs_dn.reshape(N, -1)), crit)
    hi = _search_side(eval_dev, nu, np.ascontiguousarray(zs_up.reshape(N, -1)), crit)
    lo = np.where(finite.reshape(N), lo, np.nan).reshape(B, m, T.size)
    hi = np.where(finite.reshape(N), hi, np.nan).reshape(B, m, T.size)
    return lo, hi


# --------------------------------------------------------------- POT profile
def _pot_profile_nll(xi, phi, y, mask, z, log_base, u, k, n, prior_fn):
    """Joint GPD + binomial NLL with scale fixed by a return level ``z``.

    A shape prior, when the fit used one, is included so the profile is of the
    same objective the parameters were estimated under.
    """
    log_m = log_base - jax.nn.softplus(-phi)
    w = jnp.where(log_m > 0.0, log_m, 1.0)
    denom = w * expm1_over_x(xi * w)
    span = z - u
    scale = span / denom
    ok = (log_m > 0.0) & (span > 0.0) & (denom > 0.0) & (scale > 0.0) & jnp.isfinite(scale)
    scale_s = jnp.where(ok, scale, 1.0)
    xi_s = jnp.where(jnp.isfinite(xi), xi, 0.0)
    ysafe = jnp.where(mask, y, 0.0)
    lp = gpd_logpdf(ysafe, scale_s, xi_s)
    val = -jnp.sum(jnp.where(mask, lp, 0.0)) + k * jax.nn.softplus(-phi) + (n - k) * jax.nn.softplus(phi)
    if prior_fn is not None:
        val = val - prior_fn(xi_s)
    return jnp.where(ok, val, jnp.inf)


_POT_PATH: dict = {}


def _pot_profile_path(fixed_shape: bool, prior):
    key = (fixed_shape, prior)
    if key in _POT_PATH:
        return _POT_PATH[key]
    prior_fn = PRIORS[prior]
    if fixed_shape:

        def nll(th, y, mask, z, log_base, u, k, n, xi):
            return _pot_profile_nll(xi, th[0], y, mask, z, log_base, u, k, n, prior_fn)

        def path(th0, y, mask, zs, log_base, u, k, n, xi):
            def step(prev, z):
                args = (y, mask, z, log_base, u, k, n, xi)
                r = _single(nll, prev, args, _SET)
                nxt = jnp.where(r.converged, r.x, prev)
                return nxt, jnp.where(r.converged, r.fun, jnp.nan)

            final, fs = jax.lax.scan(step, th0, zs)
            return final, fs

        fn = jax.jit(jax.vmap(path, in_axes=(0,) * 9))
    else:

        def nll(th, y, mask, z, log_base, u, k, n):
            return _pot_profile_nll(th[0], th[1], y, mask, z, log_base, u, k, n, prior_fn)

        def path(th0, y, mask, zs, log_base, u, k, n):
            def step(prev, z):
                args = (y, mask, z, log_base, u, k, n)
                f_prev = nll(prev, *args)
                alt = jnp.stack([jnp.zeros_like(prev[0]), prev[1]])
                start = jnp.where(jnp.isfinite(f_prev), prev, alt)
                r = _single(nll, start, args, _SET)
                nxt = jnp.where(r.converged, r.x, prev)
                return nxt, jnp.where(r.converged, r.fun, jnp.nan)

            final, fs = jax.lax.scan(step, th0, zs)
            return final, fs

        fn = jax.jit(jax.vmap(path, in_axes=(0,) * 8))
    _POT_PATH[key] = fn
    return fn


def _logit(zeta):
    zc = np.clip(np.asarray(zeta, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(zc) - np.log1p(-zc)


def profile_pot_return_level(fit, period, alpha=0.05, n_points=41):
    """Profile-likelihood intervals for stationary POT return levels.

    The profiled objective is the GPD likelihood of the excesses plus the
    binomial likelihood of the exceedance rate. For each fixed level the scale
    is determined by the level, the shape and the rate; those two (or only the
    rate, when ξ is held fixed) are maximised. Returns ``(lower, upper)`` of
    shape ``(B, len(period))``.
    """
    if getattr(fit, "zeta", None) is None:
        raise ValueError("profile_pot_return_level is for POT fits")
    if not fit.stationary:
        raise ValueError("POT return levels are implemented for stationary models")
    T = np.atleast_1d(np.asarray(period, dtype=np.float64))
    B = int(np.prod(fit.batch_shape, dtype=int))
    rl = fit.return_level(T, ci="delta", alpha=alpha)
    est = np.asarray(rl.estimate, dtype=np.float64).reshape(B, T.size)
    se = _se_from_bounds(rl.lower, rl.upper, alpha).reshape(B, T.size)
    k = np.asarray(fit.nobs, dtype=np.float64).reshape(B)
    n = np.asarray(fit.n_total, dtype=np.float64).reshape(B)
    zeta = np.asarray(fit.zeta, dtype=np.float64).reshape(B)
    phi = _logit(zeta)
    # Same binomial term the profile objective uses at this rate (softplus of the logit).
    binom = k * np.asarray(jax.nn.softplus(-jnp.asarray(phi))) + (n - k) * np.asarray(jax.nn.softplus(jnp.asarray(phi)))
    nll_hat = -np.asarray(fit.loglik, dtype=np.float64).reshape(B) + binom
    coef = np.asarray(fit.coef, dtype=np.float64).reshape(B, -1)
    xi = coef[:, 1]
    if fit.prior is not None:
        nll_hat = nll_hat - np.asarray(PRIORS[fit.prior](jnp.asarray(xi)), dtype=np.float64)
    u = np.broadcast_to(np.asarray(fit.threshold, dtype=np.float64), fit.batch_shape).reshape(B)
    repeats = T.size
    N = B * repeats
    y = _tile_batch(fit._y, repeats)
    mask = _tile_batch(fit._mask, repeats)
    log_base = np.broadcast_to(np.log(T * float(fit.npy)), (B, T.size)).reshape(N).copy()
    u_flat = _tile_batch(u, repeats)
    k_flat = _tile_batch(k, repeats)
    n_flat = _tile_batch(n, repeats)
    nll_flat = _tile_batch(nll_hat, repeats)
    fixed = fit.free_params is not None
    path = _pot_profile_path(fixed, fit.prior)
    if fixed:
        th0 = phi.reshape(B, 1)
        xi_flat = _tile_batch(xi, repeats)
        head = (
            jnp.asarray(y),
            jnp.asarray(mask),
            jnp.asarray(log_base),
            jnp.asarray(u_flat),
            jnp.asarray(k_flat),
            jnp.asarray(n_flat),
            jnp.asarray(xi_flat),
        )
    else:
        th0 = np.column_stack([xi, phi])
        head = (
            jnp.asarray(y),
            jnp.asarray(mask),
            jnp.asarray(log_base),
            jnp.asarray(u_flat),
            jnp.asarray(k_flat),
            jnp.asarray(n_flat),
        )
    nu = _tile_batch(th0, repeats)

    def eval_dev(nu_cur, zs):
        final, fs = path(jnp.asarray(nu_cur), head[0], head[1], jnp.asarray(zs), *head[2:])
        dev = 2.0 * (np.asarray(fs) - nll_flat[:, None])
        dev = np.where(np.isfinite(dev), dev, np.inf)
        return np.asarray(final), dev

    crit = float(chi2.ppf(1.0 - alpha, 1))
    finite = np.isfinite(est)
    zs_dn, zs_up = _level_grids(est, se, n_points)
    lo = _search_side(eval_dev, nu, np.ascontiguousarray(zs_dn.reshape(N, -1)), crit)
    hi = _search_side(eval_dev, nu, np.ascontiguousarray(zs_up.reshape(N, -1)), crit)
    lo = np.where(finite.reshape(N), lo, np.nan).reshape(B, T.size)
    hi = np.where(finite.reshape(N), hi, np.nan).reshape(B, T.size)
    return lo, hi


def _repair_gpd_start(x0, y, mask):
    """Push a negative-shape start off an endpoint that the simulated excesses cross."""
    x0 = np.array(x0, dtype=np.float64, copy=True)
    ymax = np.max(np.where(mask, y, 0.0), axis=-1)
    scale = np.exp(x0[:, 0])
    xi = x0[:, 1]
    endpoint = np.where(xi < 0.0, -scale / np.minimum(xi, -1e-12), np.inf)
    need = (xi < 0.0) & (ymax >= 0.98 * endpoint)
    if np.any(need):
        scale_new = np.maximum(scale, (-xi) * np.maximum(ymax, 1e-3) * 1.25)
        x0[need, 0] = np.log(scale_new[need])
    return x0


def bootstrap_pot_return_level(fit, period, n_boot=500, seed=0, alpha=0.05, chunk=50_000):
    """Parametric percentile intervals for stationary POT return levels.

    Each replicate draws a new exceedance count from the fitted binomial rate
    and that many excesses from the fitted GPD, refits, and recomputes the
    return level. Replicates that do not converge, or that have fewer than two
    exceedances, are ignored by the percentile. Returns ``(lower, upper)`` of
    shape ``(B, len(period))``.
    """
    if getattr(fit, "zeta", None) is None:
        raise ValueError("bootstrap_pot_return_level is for POT fits")
    if not fit.stationary:
        raise ValueError("POT return levels are implemented for stationary models")
    if fit.free_params is not None:
        raise ValueError("bootstrap intervals refit a free shape; with fixed_shape use ci='delta' or 'profile'")
    from .optim import minimize

    T = np.atleast_1d(np.asarray(period, dtype=np.float64))
    B = int(np.prod(fit.batch_shape, dtype=int))
    n_total = np.asarray(fit.n_total, dtype=np.float64).reshape(B)
    n_max = int(np.max(n_total))
    zeta = np.asarray(fit.zeta, dtype=np.float64).reshape(B)
    coef = np.asarray(fit.coef, dtype=np.float64).reshape(B, -1)
    scale = np.exp(coef[:, 0])
    shape = coef[:, 1]
    u = np.broadcast_to(np.asarray(fit.threshold, dtype=np.float64), fit.batch_shape).reshape(B)
    key_e, key_g = jax.random.split(jax.random.PRNGKey(int(seed)))
    exceed_u = np.asarray(jax.random.uniform(key_e, (B, n_boot, n_max)))
    valid = np.arange(n_max)[None, :] < n_total[:, None]
    exc = (exceed_u < zeta[:, None, None]) & valid[:, None, :]
    gpd_u = jax.random.uniform(key_g, (B, n_boot, n_max), minval=1e-12, maxval=1.0 - 1e-12)
    excess = np.asarray(gpd_ppf(gpd_u, scale[:, None, None], shape[:, None, None]))
    y = np.where(exc, excess, 0.0).reshape(B * n_boot, n_max)
    mask = exc.reshape(B * n_boot, n_max)
    k_boot = exc.sum(axis=-1)  # (B, n_boot)
    x0 = _repair_gpd_start(np.repeat(coef, n_boot, axis=0), y, mask)
    nll = make_nll("gpd", (1, 1), fit.prior)
    vgh = make_vgh("gpd", (1, 1), fit.prior)
    X = jnp.ones((n_max, 1))
    res_x = np.empty_like(x0)
    for s in range(0, x0.shape[0], chunk):
        sl = slice(s, s + chunk)
        r = minimize(
            nll,
            jnp.asarray(x0[sl]),
            jnp.asarray(y[sl]),
            jnp.asarray(mask[sl]),
            X,
            X,
            arg_axes=(0, 0, None, None),
            vgh=vgh,
        )
        xs = np.array(r.x)
        xs[~np.asarray(r.converged)] = np.nan
        res_x[sl] = xs
    good = k_boot.reshape(-1) >= 2
    res_x[~good] = np.nan
    zeta_boot = (k_boot / n_total[:, None]).reshape(B * n_boot)
    zeta_boot[~good] = np.nan
    scale_b = np.exp(res_x[:, 0]).reshape(B, n_boot)
    shape_b = res_x[:, 1].reshape(B, n_boot)
    zeta_b = zeta_boot.reshape(B, n_boot)
    m = T[None, None, :] * float(fit.npy) * zeta_b[:, :, None]
    p = 1.0 - 1.0 / np.where(m > 1.0, m, np.nan)
    draws = np.asarray(gpd_ppf(jnp.asarray(p), jnp.asarray(scale_b[:, :, None]), jnp.asarray(shape_b[:, :, None])))
    draws = u[:, None, None] + draws
    draws = np.where((m > 1.0) & np.isfinite(draws), draws, np.nan)
    with np.errstate(invalid="ignore"):
        lo = np.nanquantile(draws, alpha / 2.0, axis=1)
        hi = np.nanquantile(draws, 1.0 - alpha / 2.0, axis=1)
    return lo, hi
