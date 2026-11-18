"""Goodness-of-fit test for scaling models.

Port of ``IDFCurves.scalingtest``: the model is fitted without one duration
(by default the shortest), the held-out annual maxima are compared with the
extrapolated distribution through a Cramér–von Mises statistic, and the
p-value of its asymptotic weighted-chi-square law is approximated with
Zolotarev's tail formula (valid for small p-values only).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .._jax import jax, jnp
from ..core.distributions import gev_cdf, gev_ppf
from .data import IDFData
from .models import IDFModel, _fit_arrays, marginal_params

__all__ = ["scaling_test", "cvm_statistic", "zolotarev_cdf"]


def cvm_statistic(u_sorted):
    n = len(u_sorted)
    i = np.arange(1, n + 1)
    return 1 / (12 * n) + np.sum(((2 * i - 1) / (2 * n) - u_sorted) ** 2)


def zolotarev_cdf(lams, x):
    """Zolotarev tail approximation of P(sum lam_i X_i^2 <= x), X_i ~ N(0,1)."""
    lams = np.asarray(lams, float)
    gam, mult = [], []
    for v in lams:  # exact-equality unique, preserving order (as Julia `unique`)
        if gam and v == gam[-1]:
            mult[-1] += 1
        elif v in gam:
            mult[gam.index(v)] += 1
        else:
            gam.append(v)
            mult.append(1)
    g1 = gam[0]
    t1 = -sum(0.5 * mult[i] * np.log(1 - gam[i] / g1) for i in range(1, len(gam)))
    t2 = -gammaln(0.5 * mult[0])
    t3 = (0.5 * mult[0] - 1) * np.log(x / (2 * g1))
    t4 = -x / (2 * g1)
    return max(1 - np.exp(t1 + t2 + t3 + t4), 0.0)


def scaling_test(data: IDFData, marginal: str = "general", d_out: float | None = None, q: int = 100) -> float:
    """p-value of the scaling-model test for one station."""
    model = IDFModel(marginal)
    if d_out is None:
        d_out = float(np.min(data.durations))
    j = int(np.argmin(np.abs(data.durations - d_out)))
    y = data.values[:, j]
    y = y[np.isfinite(y)]
    ell = len(y)
    train = data.exclude_duration(d_out)
    fit = _fit_arrays(model, train.values[None], train.durations[None], d_out)
    nat = jnp.asarray(fit.params[0])
    mu, sigma, xi = marginal_params(model, nat, jnp.asarray(d_out), d_out)
    u = np.sort(np.asarray(gev_cdf(jnp.asarray(y), mu, sigma, xi)))
    S = cvm_statistic(u)
    H = fit.hessian()[0]
    I = H / ell

    def cdf_at(theta, x):
        m, s, k = marginal_params(model, theta, jnp.asarray(d_out), d_out)
        return gev_cdf(x, m, s, k)

    grid = (2 * np.arange(1, q + 1) - 1) / (2 * q)
    xs = np.asarray(gev_ppf(jnp.asarray(grid), mu, sigma, xi))
    G = np.asarray(jax.vmap(lambda x: jax.grad(cdf_at)(nat, x))(jnp.asarray(xs)))  # (q, p)
    K = np.minimum.outer(grid, grid) - np.outer(grid, grid) + G @ np.linalg.solve(I, G.T)
    lams = np.linalg.eigvalsh(K / q)[::-1]
    return 1 - zolotarev_cdf(lams, S)
