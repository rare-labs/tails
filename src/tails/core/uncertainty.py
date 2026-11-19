"""Uncertainty for any fitted model: profile likelihood and parametric bootstrap.

There is one profile-likelihood routine for return levels and one for
coefficients, and one bootstrap. Each works with any registered likelihood
(GEV / r-largest, GPD / POT), with covariates, fixed parameters and priors,
because each only talks to the engine (:mod:`tails.core._engine`):

* **Return levels.** At each covariate row the return level replaces the
  intercept of one parameter (the GEV location, the GPD scale), through the
  family's ``invert`` map. The other coefficients, including the exceedance
  rate of a POT fit, are the nuisance parameters. The deviance is traced
  along a grid of levels with a continuation ``lax.scan`` inside one ``vmap``
  over series, rows and periods.
* **Coefficients.** One coefficient (on the user's scale) is held at grid
  values and the others are maximised.
* **Bootstrap.** Simulate from the fitted model (keeping the missing-data
  pattern and covariates), refit every replicate of every series in one
  batched optimisation.

The older entry points (``profile_return_level``,
``profile_covariate_return_level``, ``profile_pot_return_level``, ...) are
kept as thin wrappers.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.stats import chi2, norm

from .._jax import jax, jnp
from ._engine import Spec, full_nll, objective
from .distributions import gev_ppf, gpd_ppf
from .optim import _Settings, _single, minimize

__all__ = [
    "profile_levels",
    "profile_coefs",
    "profile_return_level",
    "profile_fixed_shape_return_level",
    "profile_covariate_return_level",
    "profile_pot_return_level",
    "bootstrap_coef",
    "bootstrap_pot_return_level",
]

_SET = _Settings(max_iter=100)


# ----------------------------------------------------------- path search
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
    return np.where(found, z0 + frac * (z1 - z0), np.nan)


def _grids(est, se, n_points, up=12.0, down=6.0):
    """Grids centred on ``est`` in units of the Wald standard error, ordered outward."""
    se = np.where(np.isfinite(se) & (se > 0), se, 0.1 * np.abs(est) + 1e-8)
    centre = np.where(np.isfinite(est), est, 0.0)
    n_dn = max(int(n_points) // 3, 4) if up != down else max(int(n_points) // 2, 4)
    n_up = max(int(n_points) - n_dn, 4)
    zs_up = centre[..., None] + se[..., None] * np.linspace(0.0, up, n_up)
    zs_dn = centre[..., None] - se[..., None] * np.linspace(0.0, down, n_dn)
    return zs_dn, zs_up


def _search_side(eval_dev, nu0, zs, crit, max_extend=2):
    """Deviance path along ``zs``, extended while the χ² level is still uncrossed."""
    nu = np.asarray(nu0, dtype=np.float64)
    z_parts, d_parts = [], []
    bound = np.full(zs.shape[0], np.nan)
    for extend in range(max_extend + 1):
        nu, dev = eval_dev(nu, zs)
        z_parts.append(np.asarray(zs[:, 1:] if extend else zs, dtype=np.float64))
        d_parts.append(dev[:, 1:] if extend else dev)
        bound = _crossing(np.concatenate(z_parts, axis=1), np.concatenate(d_parts, axis=1), crit)
        above = d_parts[-1] > crit
        idx = np.argmax(above, axis=1)
        found = above[np.arange(len(idx)), idx] & (idx > 0)
        if extend:
            found = found | np.isfinite(bound)
        last = dev[:, -1]
        need = (~found) & np.isfinite(last) & (last < crit)
        if not np.any(need) or extend == max_extend:
            return bound
        step = zs[:, -1] - zs[:, -2]
        extra = zs[:, -1:] + step[:, None] * np.arange(1, zs.shape[1])
        zs = np.concatenate([zs[:, -1:], extra], axis=1)
    return bound


def _scan_path(obj, n_shape_pos):
    """Continuation along a grid; restart from shape = 0 when the previous point is infeasible."""
    shape_pos = jnp.asarray(n_shape_pos, dtype=int) if n_shape_pos else None

    def path(nu0, zs, *args):
        def step(prev, z):
            a = (z, *args)
            start = prev
            if shape_pos is not None:
                alt = prev.at[shape_pos].set(0.0)
                start = jnp.where(jnp.isfinite(obj(prev, *a)), prev, alt)
            r = _single(obj, start, a, _SET)
            return jnp.where(r.converged, r.x, prev), jnp.where(r.converged, r.fun, jnp.nan)

        return jax.lax.scan(step, nu0, zs)

    return path


def _tile(a, repeats):
    return np.repeat(np.asarray(a), repeats, axis=0)


def _data_axes(fit):
    return tuple(None if d.shared else 0 for d in fit._designs)


def _nll_hat(fit):
    from .model import nll_at

    return nll_at(fit, np.asarray(fit._theta_std).reshape(fit._B, -1))


# ------------------------------------------------------ return-level profile
@cache
def _level_path(spec: Spec, nuis: tuple, axes: tuple):
    lik, nll, ks = spec.likelihood, full_nll(spec), spec.ks
    jt = lik.params.index(lik.target)
    t0 = spec.offsets()[jt]
    nuis_j = jnp.asarray(nuis)
    shape_pos = [i for i, q in enumerate(nuis) if q in spec.shape_coefs()]

    def obj(nu, z, y, mask, base, c, xr, *Xs):
        theta = base.at[nuis_j].set(nu)
        etas, off = [], 0
        for k in ks:
            etas.append(jnp.dot(xr[off : off + k], theta[off : off + k]))
            off += k
        eta_t, ok = lik.invert(z, etas, c)
        theta = theta.at[t0].set(eta_t - etas[jt])  # base intercept is 0, so etas[jt] is the rest of the row
        val = nll(theta, y, mask, *Xs)
        return jnp.where(ok & jnp.isfinite(eta_t), val, jnp.inf)

    path = _scan_path(obj, shape_pos)
    return jax.jit(jax.vmap(path, in_axes=(0, 0, 0, 0, 0, 0, 0, *axes)))


def profile_levels(fit, period, covariates=None, alpha=0.05, n_points=41, kind=None):
    """Profile-likelihood intervals for return levels of any fit.

    Returns ``(lower, upper)`` of shape ``(B, m, len(period))`` with ``m`` the
    number of covariate rows (1 for a stationary fit without ``covariates``).
    Bounds are NaN when the deviance does not cross the χ² level inside the
    searched range.
    """
    spec = fit._spec
    lik = spec.likelihood
    T = np.atleast_1d(np.asarray(period, dtype=np.float64))
    B = fit._B
    rl = fit.return_level(T, covariates=covariates, ci="delta", alpha=alpha, kind=kind)
    est = np.asarray(rl.estimate, dtype=np.float64).reshape(B, -1, T.size)
    zq = float(norm.ppf(1.0 - alpha / 2.0))
    se = ((np.asarray(rl.upper) - np.asarray(rl.lower)) / (2.0 * zq)).reshape(B, -1, T.size)
    m = est.shape[1]
    _, std_rows, m2 = fit._rows(covariates)
    if m2 != m:
        raise ValueError(f"return level has {m} covariate row(s) but the prediction design has {m2}")
    t0 = spec.index(lik.target)
    free = list(range(spec.p)) if spec.free is None else list(spec.free)
    if t0 not in free:
        raise ValueError(f"profile intervals need a free {lik.target}")
    nuis = tuple(i for i in free if i != t0)
    theta = np.asarray(fit._theta_std, dtype=np.float64).reshape(B, -1)
    base = theta.copy()
    base[:, t0] = 0.0
    nll_hat = _nll_hat(fit)
    c = fit._level_context(T, kind)  # (B, nT, 2)
    repeats = m * T.size
    N = B * repeats
    xr = np.concatenate(std_rows, axis=-1)  # (B, m, p)
    xr = np.repeat(xr[:, :, None, :], T.size, axis=2).reshape(N, -1)
    cc = np.repeat(c[:, None, :, :], m, axis=1).reshape(N, 2)
    axes = _data_axes(fit)
    Xs = [jnp.asarray(d.X if d.shared else _tile(d.X, repeats)) for d in fit._designs]
    head = (jnp.asarray(_tile(fit._data, repeats)), jnp.asarray(_tile(fit._dmask, repeats)),
            jnp.asarray(_tile(base, repeats)), jnp.asarray(cc), jnp.asarray(xr))
    path = _level_path(Spec(spec.lik, spec.ks, fit.prior, spec.free), nuis, axes)
    nll_flat = _tile(nll_hat, repeats)

    def eval_dev(nu_cur, zs):
        final, fs = path(jnp.asarray(nu_cur), jnp.asarray(zs), *head, *Xs)
        dev = 2.0 * (np.asarray(fs) - nll_flat[:, None])
        return np.asarray(final), np.where(np.isfinite(dev), dev, np.inf)

    crit = float(chi2.ppf(1.0 - alpha, 1))
    nu = _tile(theta[:, list(nuis)], repeats)
    zs_dn, zs_up = _grids(est, se, n_points)
    lo = _search_side(eval_dev, nu, np.ascontiguousarray(zs_dn.reshape(N, -1)), crit)
    hi = _search_side(eval_dev, nu, np.ascontiguousarray(zs_up.reshape(N, -1)), crit)
    finite = np.isfinite(est).reshape(N)
    return (np.where(finite, lo, np.nan).reshape(B, m, T.size), np.where(finite, hi, np.nan).reshape(B, m, T.size))


# ------------------------------------------------------ coefficient profile
@cache
def _coef_path(spec: Spec, nuis: tuple, j: int, axes: tuple):
    nll = full_nll(spec)
    nuis_j = jnp.asarray(nuis)
    shape_pos = [i for i, q in enumerate(nuis) if q in spec.shape_coefs()]

    def obj(nu, value, y, mask, base, Ainv, *Xs):
        th_user = base.at[nuis_j].set(nu).at[j].set(value)
        return nll(Ainv @ th_user, y, mask, *Xs)

    path = _scan_path(obj, shape_pos)
    return jax.jit(jax.vmap(path, in_axes=(0, 0, 0, 0, 0, 0, *axes)))


def profile_coefs(fit, alpha=0.05, n_points=41):
    """Profile-likelihood intervals for every user-scale coefficient: ``(lower, upper)``, each ``(B, p)``.

    For a POT fit this covers the GPD coefficients (``coef``).
    """
    from ._engine import back_transform_matrix

    spec = fit._spec
    B = fit._B
    p_main = np.asarray(fit.coef).reshape(B, -1).shape[1]
    coef = fit._coef_full()
    se = np.sqrt(np.clip(np.diagonal(fit._cov_full(), axis1=-2, axis2=-1), 0, None))
    free = list(range(spec.p)) if spec.free is None else list(spec.free)
    A = back_transform_matrix(fit._designs, B)
    Ainv = np.linalg.inv(A)
    nll_hat = _nll_hat(fit)
    axes = _data_axes(fit)
    Xs = [jnp.asarray(d.X) for d in fit._designs]
    crit = float(chi2.ppf(1.0 - alpha, 1))
    lo = np.array(coef[:, :p_main], copy=True)
    hi = np.array(coef[:, :p_main], copy=True)
    pspec = Spec(spec.lik, spec.ks, fit.prior, spec.free)
    for j in range(p_main):
        if j not in free:
            continue
        nuis = tuple(i for i in free if i != j)
        path = _coef_path(pspec, nuis, j, axes)
        head = (jnp.asarray(fit._data), jnp.asarray(fit._dmask), jnp.asarray(coef), jnp.asarray(Ainv))

        def eval_dev(nu_cur, zs, path=path, head=head):
            final, fs = path(jnp.asarray(nu_cur), jnp.asarray(zs), *head, *Xs)
            dev = 2.0 * (np.asarray(fs) - nll_hat[:, None])
            return np.asarray(final), np.where(np.isfinite(dev), dev, np.inf)

        zs_dn, zs_up = _grids(coef[:, j], se[:, j], n_points, up=6.0, down=6.0)
        nu = coef[:, list(nuis)]
        lo[:, j] = _search_side(eval_dev, nu, np.ascontiguousarray(zs_dn), crit)
        hi[:, j] = _search_side(eval_dev, nu, np.ascontiguousarray(zs_up), crit)
    return lo, hi


# ----------------------------------------------------------------- bootstrap
def _simulate(fit, n_boot, seed):
    """Datasets drawn from the fitted model: ``(B * n_boot, n[, r])`` values and mask."""
    from .model import etas_at_obs

    B, R = fit._B, int(n_boot)
    lik = fit._spec.lik
    etas = [np.asarray(e) for e in etas_at_obs(fit)]
    n = fit._data.shape[1]
    etas = [np.broadcast_to(e, (B, n))[:, None, :] for e in etas]
    key = jax.random.PRNGKey(int(seed))
    mask = _tile(fit._dmask, R)
    if lik == "gev":
        u = jax.random.uniform(key, (B, R, n), minval=1e-12, maxval=1 - 1e-12)
        loc, sc, xi = etas[0], np.exp(etas[1]), etas[2]
        if fit._data.ndim == 2:
            y = np.asarray(gev_ppf(u, loc, sc, xi)).reshape(B * R, n)
        else:  # r largest: points of the limiting Poisson process, Γ_k = Γ_1 + E_2 + ... + E_k
            r = fit._data.shape[2]
            gam = -jnp.log(u)[..., None] + jnp.concatenate(
                [jnp.zeros((B, R, n, 1)), jnp.cumsum(jax.random.exponential(jax.random.fold_in(key, 1), (B, R, n, r - 1)), -1)], -1)
            y = np.asarray(gev_ppf(jnp.exp(-gam), loc[..., None], sc[..., None], xi[..., None])).reshape(B * R, n, r)
        return np.where(mask, y, 0.0), mask
    if lik == "gpd":
        u = jax.random.uniform(key, (B, R, n), minval=1e-12, maxval=1 - 1e-12)
        y = np.asarray(gpd_ppf(u, np.exp(etas[0]), etas[1])).reshape(B * R, n)
        return np.where(mask, y, 0.0), mask
    key_e, key_g = jax.random.split(key)  # POT: Bernoulli exceedances, then GPD excesses
    zeta = np.asarray(jax.nn.sigmoid(jnp.asarray(etas[2])))
    hit = (np.asarray(jax.random.uniform(key_e, (B, R, n))) < zeta).reshape(B * R, n) & mask
    g = jax.random.uniform(key_g, (B, R, n), minval=1e-12, maxval=1.0 - 1e-12)
    y = np.asarray(gpd_ppf(g, np.exp(etas[0]), etas[1])).reshape(B * R, n)
    return np.where(hit, y, 0.0), mask


def _bootstrap_full(fit, n_boot=500, seed=0, chunk=50_000):
    """Parametric bootstrap: user-scale coefficients of every component, ``(B, n_boot, p)``.

    Replicates that do not converge (or, for POT, have fewer than two
    exceedances) are NaN.
    """
    from ._engine import back_transform_matrix
    from .model import _repair_gpd_start

    spec = fit._spec
    B, R = fit._B, int(n_boot)
    ysim, msim = _simulate(fit, R, seed)
    th = np.asarray(fit._theta_std).reshape(B, -1)
    x_full = _tile(th, R)
    if spec.likelihood.family == "gpd":
        hit = msim & (ysim > 0) if spec.lik == "pot" else msim
        x_full[:, :2] = _repair_gpd_start(x_full[:, :2], ysim, hit)
    fun, vgh = objective(Spec(spec.lik, spec.ks, fit.prior, spec.free))
    free = list(range(spec.p)) if spec.free is None else list(spec.free)
    axes = _data_axes(fit)
    Xrep = [jnp.asarray(d.X if d.shared else _tile(d.X, R)) for d in fit._designs]
    res = np.array(x_full, copy=True)
    for s in range(0, x_full.shape[0], chunk):
        sl = slice(s, s + chunk)
        Xc = [X if ax is None else X[sl] for X, ax in zip(Xrep, axes)]
        r = minimize(fun, jnp.asarray(x_full[sl][:, free]), jnp.asarray(ysim[sl]), jnp.asarray(msim[sl]),
                     jnp.asarray(x_full[sl]), *Xc, arg_axes=(0, 0, 0, *axes), vgh=vgh)
        xs = np.asarray(r.x)
        out = res[sl]
        out[:, free] = xs
        out[~np.asarray(r.converged)] = np.nan
        res[sl] = out
    if spec.lik == "pot":
        res[(msim & (ysim > 0)).sum(-1) < 2] = np.nan
    A = _tile(back_transform_matrix(fit._designs, B), R)
    return np.einsum("bij,bj->bi", A, res).reshape(B, R, -1)


def bootstrap_coef(fit, n_boot=500, seed=0, chunk=50_000):
    """Parametric bootstrap of a fitted model's coefficients, ``(B, n_boot, p)``.

    Simulates ``n_boot`` datasets per series from the fitted model (keeping the
    missing-data pattern and covariates), refits them all in batched form and
    returns user-scale coefficients (``coef`` order; POT: the GPD part).
    """
    boot = _bootstrap_full(fit, n_boot=n_boot, seed=seed, chunk=chunk)
    return boot[..., : np.asarray(fit.coef).reshape(fit._B, -1).shape[1]]


# ------------------------------------------------- backward-compatible names
def _squeeze_rows(lo, hi):
    return lo[:, 0, :], hi[:, 0, :]


def profile_return_level(fit, period, alpha=0.05, n_points=41):
    """Profile-likelihood intervals for stationary return levels, ``(B, len(period))`` each."""
    return _squeeze_rows(*profile_levels(fit, period, alpha=alpha, n_points=n_points))


def profile_fixed_shape_return_level(fit, period, alpha=0.05, n_points=41):
    """Same as :func:`profile_return_level`; a fixed shape stays fixed."""
    return profile_return_level(fit, period, alpha=alpha, n_points=n_points)


def profile_covariate_return_level(fit, period, covariates=None, alpha=0.05, n_points=41):
    """Profile-likelihood intervals at covariate rows, ``(B, m, len(period))`` each (any family)."""
    return profile_levels(fit, period, covariates=covariates, alpha=alpha, n_points=n_points)


def profile_pot_return_level(fit, period, alpha=0.05, n_points=41):
    """Profile-likelihood intervals for stationary POT return levels, ``(B, len(period))`` each."""
    if getattr(fit, "zeta", None) is None:
        raise ValueError("profile_pot_return_level is for POT fits")
    return profile_return_level(fit, period, alpha=alpha, n_points=n_points)


def bootstrap_pot_return_level(fit, period, n_boot=500, seed=0, alpha=0.05):
    """Parametric percentile intervals for POT return levels, ``(B, len(period))`` each."""
    if getattr(fit, "zeta", None) is None:
        raise ValueError("bootstrap_pot_return_level is for POT fits")
    rl = fit.return_level(period, ci="bootstrap", n_boot=n_boot, seed=seed, alpha=alpha)
    B = fit._B
    return np.asarray(rl.lower).reshape(B, -1), np.asarray(rl.upper).reshape(B, -1)
