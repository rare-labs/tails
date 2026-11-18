"""IDF scaling models with cross-duration dependence (port of IDFCurves.jl).

Marginal models (Koutsoyiannis et al., 1998) for the annual maximum
intensity of duration ``d``::

    simple  (SimpleScaling):  mu_d = mu0 (d/d0)^-alpha,               sigma_d = sigma0 (d/d0)^-alpha
    general (GeneralScaling, a.k.a. d-GEV):
                              mu_d = mu0 ((d+delta)/(d0+delta))^-alpha, idem for sigma_d
    xi_d = xi for all durations.

Dependence between durations of the same year:
an elliptical copula (Gaussian or Student-t with integer ``df``) whose
correlation between durations ``d_i`` and ``d_j`` is a function of
``h = |log d_i - log d_j|``: exponential ``exp(-h/rho)`` or Matérn(nu, rho).

Everything is batched across stations: one call fits the same model to
hundreds of stations. Missing values are handled exactly (marginalising the
copula over unobserved durations).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache

import numpy as np
from jax.scipy.special import gammaln, ndtri
from jax.scipy.stats import t as jt

from .._jax import jax, jnp
from ..core.distributions import gev_cdf, gev_logpdf, gev_ppf
from ..core.fit import fit_gev
from ..core.optim import minimize
from .data import IDFData, stack
from .special import exponential, matern, t_ppf

__all__ = ["IDFModel", "IDFFit", "fit_idf", "loglikelihood"]

_MARG = {"simple": ("mu0", "sigma0", "xi", "alpha"), "general": ("mu0", "sigma0", "xi", "alpha", "delta")}
_CORR = {None: (), "exponential": ("rho",), "matern": ("nu", "rho")}


@dataclass(frozen=True)
class IDFModel:
    """Model specification.

    Parameters
    ----------
    marginal : ``"simple"`` or ``"general"`` (d-GEV)
    correlation : ``None`` (independence), ``"exponential"`` or ``"matern"``
    copula : ``"gaussian"`` or ``"t"`` (ignored when ``correlation`` is None)
    df : degrees of freedom of the Student-t copula
    """

    marginal: str = "general"
    correlation: str | None = None
    copula: str = "gaussian"
    df: float = 5.0

    def __post_init__(self):
        if self.marginal not in _MARG:
            raise ValueError("marginal must be 'simple' or 'general'")
        if self.correlation not in _CORR:
            raise ValueError("correlation must be None, 'exponential' or 'matern'")
        if self.copula not in ("gaussian", "t"):
            raise ValueError("copula must be 'gaussian' or 't'")

    @property
    def names(self) -> tuple[str, ...]:
        return _MARG[self.marginal] + _CORR[self.correlation]

    @property
    def n_marginal(self) -> int:
        return len(_MARG[self.marginal])


# ------------------------------------------------------------------ transforms
def to_natural(model: IDFModel, th):
    out = [th[0], jnp.exp(th[1]), th[2], jax.nn.sigmoid(th[3])]
    if model.marginal == "general":
        out.append(jnp.exp(th[4]))
    k = model.n_marginal
    out += [jnp.exp(th[k + j]) for j in range(len(_CORR[model.correlation]))]
    return jnp.stack(out)


def to_real(model: IDFModel, nat):
    nat = np.asarray(nat, float)
    out = [nat[..., 0], np.log(nat[..., 1]), nat[..., 2], np.log(nat[..., 3] / (1 - nat[..., 3]))]
    if model.marginal == "general":
        out.append(np.log(nat[..., 4]))
    k = model.n_marginal
    out += [np.log(nat[..., k + j]) for j in range(len(_CORR[model.correlation]))]
    return np.stack(out, -1)


def marginal_params(model: IDFModel, nat, d, d0):
    mu0, sigma0, xi, alpha = nat[0], nat[1], nat[2], nat[3]
    delta = nat[4] if model.marginal == "general" else 0.0
    s = jnp.exp(-alpha * (jnp.log(d + delta) - jnp.log(d0 + delta)))
    return mu0 * s, sigma0 * s, xi


def correlation_matrix(model: IDFModel, nat, d):
    logd = jnp.log(d)
    h = jnp.abs(logd[:, None] - logd[None, :])
    k = model.n_marginal
    if model.correlation == "exponential":
        return exponential(h, nat[k])
    if model.correlation == "matern":
        return matern(h, nat[k], nat[k + 1])
    return jnp.eye(d.shape[0])


def _copula_ll(model: IDFModel, R, u, M):
    """Copula log-density summed over years, marginalising missing durations."""
    m = R.shape[0]
    Mf = M.astype(R.dtype)
    eye = jnp.eye(m)

    def one(ut, mt):
        Rt = R * (mt[:, None] * mt[None, :]) + jnp.diag(1.0 - mt)
        L = jnp.linalg.cholesky(Rt)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        if model.copula == "gaussian":
            x = ndtri(ut) * mt
            sol = jax.scipy.linalg.solve_triangular(L, x, lower=True)
            q = jnp.sum(sol * sol)
            return -0.5 * logdet - 0.5 * (q - jnp.sum(x * x))
        nu = model.df
        x = t_ppf(ut, nu) * mt
        sol = jax.scipy.linalg.solve_triangular(L, x, lower=True)
        q = jnp.sum(sol * sol)
        k = jnp.sum(mt)
        mvt = gammaln((nu + k) / 2) - gammaln(nu / 2) - 0.5 * k * jnp.log(nu * jnp.pi) - 0.5 * logdet - 0.5 * (nu + k) * jnp.log1p(q / nu)
        uni = jnp.sum(jnp.where(mt > 0, jt.logpdf(x, nu), 0.0))
        return mvt - uni

    del eye
    vals = jax.vmap(one)(u, Mf)
    return jnp.sum(jnp.where(jnp.sum(Mf, -1) > 1, vals, 0.0))


def loglikelihood_natural(model: IDFModel, nat, Y, M, d, d0):
    """Log-likelihood of natural parameters ``nat`` for one station."""
    mu, sigma, xi = marginal_params(model, nat, d, d0)
    mu_b = jnp.broadcast_to(mu, Y.shape)
    Ys = jnp.where(M, Y, mu_b)
    ll = jnp.sum(jnp.where(M, gev_logpdf(Ys, mu, sigma, xi), 0.0))
    if model.correlation is None:
        return ll
    u = jnp.clip(gev_cdf(Ys, mu, sigma, xi), 1e-15, 1 - 1e-15)
    u = jnp.where(M, u, 0.5)
    R = correlation_matrix(model, nat, d)
    return ll + _copula_ll(model, R, u, M)


def loglikelihood(model: IDFModel, params, data: IDFData, d0: float = 1.0) -> float:
    """Log-likelihood of natural parameters (``model.names`` order) for one station."""
    M = np.isfinite(data.values)
    return float(
        loglikelihood_natural(model, jnp.asarray(params, float), jnp.asarray(np.where(M, data.values, 0.0)), jnp.asarray(M), jnp.asarray(data.durations), d0)
    )


@cache
def _nll_real(model: IDFModel):
    def f(th, Y, M, d, d0):
        v = -loglikelihood_natural(model, to_natural(model, th), Y, M, d, d0)
        return jnp.where(jnp.isfinite(v), v, jnp.inf)

    return f


@cache
def _hess_fn(model: IDFModel):
    return jax.jit(jax.vmap(jax.hessian(_nll_nat(model)), in_axes=(0, 0, 0, 0, None)))


@cache
def _nll_nat(model: IDFModel):
    def f(nat, Y, M, d, d0):
        return -loglikelihood_natural(model, nat, Y, M, d, d0)

    return f


# -------------------------------------------------------------- initialisation
def _kendall_tau_b(Y, M):
    """Pairwise-complete Kendall tau-b between all duration pairs (one station)."""
    n, m = Y.shape
    sg = jnp.sign(Y[:, None, :] - Y[None, :, :])  # (n, n, m)
    upper = jnp.triu(jnp.ones((n, n), bool), 1)

    def pair(a, b):
        v = M[:, a][:, None] & M[:, a][None, :] & M[:, b][:, None] & M[:, b][None, :] & upper
        sa, sb = sg[:, :, a], sg[:, :, b]
        S = jnp.sum(jnp.where(v, sa * sb, 0.0))
        n0 = jnp.sum(v)
        n1 = jnp.sum(v & (sa == 0))
        n2 = jnp.sum(v & (sb == 0))
        return S / jnp.sqrt((n0 - n1) * (n0 - n2))

    ii, jj = np.triu_indices(m, 1)
    return jax.vmap(pair)(jnp.asarray(ii), jnp.asarray(jj)), ii, jj


def _init_marginal(model, Y, M, D, d0):
    """Per-duration GEV fits + log-linear regression (as IDFCurves.initialize)."""
    S, n, m = Y.shape
    flat = np.where(M, Y, np.nan).transpose(0, 2, 1).reshape(S * m, n)
    ok = np.isfinite(flat).sum(-1) >= 3
    mu = np.full(S * m, np.nan)
    sg = np.full(S * m, np.nan)
    if ok.any():
        f = fit_gev(flat[ok])
        c = f.coef
        mu[ok], sg[ok] = c[:, 0], np.exp(c[:, 1])
        bad = ~f.converged
        if bad.any():  # PWM fallback, like IDFCurves
            fl = fit_gev(flat[ok][bad], method="lmom").coef
            tmp_mu, tmp_sg = mu[ok], sg[ok]
            tmp_mu[bad], tmp_sg[bad] = fl[:, 0], np.exp(fl[:, 1])
            mu[ok], sg[ok] = tmp_mu, tmp_sg
    mu, sg = mu.reshape(S, m), sg.reshape(S, m)
    out = np.zeros((S, model.n_marginal))
    for s in range(S):
        v = np.isfinite(mu[s]) & (mu[s] > 0)
        ld = np.log(D[s, v] / d0)
        k = v.sum()
        X = np.zeros((2 * k, 3))
        y = np.zeros(2 * k)
        X[0::2, 0] = 1
        X[0::2, 2] = ld
        y[0::2] = np.log(mu[s, v])
        X[1::2, 1] = 1
        X[1::2, 2] = ld
        y[1::2] = np.log(sg[s, v])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        row = [np.exp(b[0]), max(0.001, np.exp(b[1])), 1e-4, min(0.999, max(0.001, -b[2]))]
        if model.marginal == "general":
            row.append(0.001)
        out[s] = row
    return out


def _corr_mse(kind):
    def f(th, tau, h):
        if kind == "exponential":
            c = jnp.exp(-h / jnp.exp(th[0]))
        else:
            c = matern(h, jnp.exp(th[0]), jnp.exp(th[1]))
        r = jnp.where(jnp.isfinite(tau), c - tau, 0.0)
        return jnp.sum(r * r)

    return f


_MSE = {k: _corr_mse(k) for k in ("exponential", "matern")}


def _init_correlation(model, Y, M, D):
    S = Y.shape[0]
    taus, hs = [], []
    for s in range(S):
        tau, ii, jj = _kendall_tau_b(jnp.asarray(np.where(M[s], Y[s], 0.0)), jnp.asarray(M[s]))
        ld = np.log(D[s])
        taus.append(np.sin(np.pi / 2 * np.asarray(tau)))
        hs.append(np.abs(ld[ii] - ld[jj]))
    tau, h = np.stack(taus), np.stack(hs)
    k = len(_CORR[model.correlation])
    r = minimize(_MSE[model.correlation], np.zeros((S, k)), jnp.asarray(tau), jnp.asarray(h))
    th = np.where(np.asarray(r.converged)[:, None], np.asarray(r.x), 0.0)
    return np.maximum(0.001, np.exp(th))


# --------------------------------------------------------------------- fitting
@dataclass
class IDFFit:
    """Fitted IDF model(s). Arrays carry a leading station axis ``S``."""

    model: IDFModel
    params: np.ndarray  # (S, p) natural parameters
    loglik: np.ndarray
    converged: np.ndarray
    d0: float
    names: tuple[str, ...]
    Y: np.ndarray = field(repr=False)
    M: np.ndarray = field(repr=False)
    D: np.ndarray = field(repr=False)
    stations: list[str] = field(default_factory=list)
    _H: np.ndarray | None = field(default=None, repr=False)

    # -------------------------------------------------------------- basics
    @property
    def n_stations(self) -> int:
        return self.params.shape[0]

    def param_table(self):
        import pandas as pd

        df = pd.DataFrame(self.params, columns=list(self.names))
        df.insert(0, "station", self.stations or list(range(self.n_stations)))
        df["loglik"] = self.loglik
        df["converged"] = self.converged
        return df

    def hessian(self) -> np.ndarray:
        """Hessian of the negative log-likelihood in natural parameters, (S, p, p)."""
        if self._H is None:
            f = _hess_fn(self.model)
            self._H = np.asarray(f(jnp.asarray(self.params), jnp.asarray(np.where(self.M, self.Y, 0.0)), jnp.asarray(self.M), jnp.asarray(self.D), self.d0))
        return self._H

    def variability(self) -> np.ndarray:
        """Variability matrix ``J = sum_years u_t u_t'`` of yearly scores (S, p, p)."""
        model = self.model

        def year_ll(nat, y, m, d):
            return loglikelihood_natural(model, nat, y[None, :], m[None, :], d, self.d0)

        score = jax.vmap(jax.grad(year_ll), in_axes=(None, 0, 0, None))

        def J(nat, Y, M, d):
            U = score(nat, Y, M, d)
            return U.T @ U

        f = jax.jit(jax.vmap(J))
        return np.asarray(f(jnp.asarray(self.params), jnp.asarray(np.where(self.M, self.Y, 0.0)), jnp.asarray(self.M), jnp.asarray(self.D)))

    def covariance(self, kind: str = "hessian") -> np.ndarray:
        """Parameter covariance: ``"hessian"`` (H⁻¹) or ``"sandwich"`` (Godambe, H⁻¹ J H⁻¹)."""
        H = self.hessian()
        Hi = np.linalg.inv(H)
        if kind == "hessian":
            return Hi
        if kind == "sandwich":
            return Hi @ self.variability() @ Hi
        raise ValueError("kind must be 'hessian' or 'sandwich'")

    def godambe(self) -> np.ndarray:
        """Godambe information ``G = H J⁻¹ H`` (as IDFCurves.godambe)."""
        H = self.hessian()
        return H @ np.linalg.solve(self.variability(), H)

    # ----------------------------------------------------------- quantiles
    def _q_fn(self):
        model, d0 = self.model, self.d0

        def q(nat, d, p):
            mu, sigma, xi = marginal_params(model, nat, d, d0)
            return gev_ppf(p, mu, sigma, xi)

        return q

    def quantile(self, duration, period) -> np.ndarray:
        """Return levels (intensity) of shape ``(S, len(duration), len(period))``."""
        d = jnp.asarray(np.atleast_1d(duration), float)
        p = 1.0 - 1.0 / jnp.asarray(np.atleast_1d(period), float)
        q = self._q_fn()
        f = jax.vmap(lambda nat: q(nat, d[:, None], p[None, :]))
        return np.asarray(f(jnp.asarray(self.params)))

    def quantile_ci(self, duration, period, *, method: str = "hessian", alpha: float = 0.05, n_boot: int = 500, seed: int = 0):
        """Confidence intervals for return levels.

        ``method``: ``"hessian"`` (Wald / delta method, as IDFCurves
        ``quantilecint``), ``"sandwich"`` (Godambe-robust Wald) or
        ``"bootstrap"`` (parametric bootstrap from the fitted *joint* model,
        percentile intervals — recommended when the two Wald versions disagree).
        Returns ``(estimate, lower, upper)`` each ``(S, len(duration), len(period))``.
        """
        from scipy.stats import norm

        est = self.quantile(duration, period)
        if method == "bootstrap":
            draws = self.bootstrap(n_boot=n_boot, seed=seed).quantile(duration, period)
            draws = draws.reshape(self.n_stations, n_boot, *draws.shape[1:])
            return est, np.nanquantile(draws, alpha / 2, 1), np.nanquantile(draws, 1 - alpha / 2, 1)
        cov = self.covariance(method)
        d = jnp.asarray(np.atleast_1d(duration), float)
        p = 1.0 - 1.0 / jnp.asarray(np.atleast_1d(period), float)
        q = self._q_fn()
        J = jax.vmap(jax.jacfwd(lambda nat: q(nat, d[:, None], p[None, :])))(jnp.asarray(self.params))
        J = np.asarray(J)  # (S, nd, nT, p)
        var = np.einsum("sdtp,spq,sdtq->sdt", J, cov, J)
        sd = np.sqrt(np.clip(var, 0, None))
        z = norm.ppf(1 - alpha / 2)
        return est, est - z * sd, est + z * sd

    def idf_table(self, periods=(2, 5, 10, 25, 50, 100), station: int = 0, *, ci: str | None = None, alpha: float = 0.05):
        """Tidy table of return levels for one station."""
        import pandas as pd

        d = self.D[station]
        if ci is None:
            est = self.quantile(d, periods)[station]
            lo = hi = np.full_like(est, np.nan)
        else:
            e, lo, hi = self.quantile_ci(d, periods, method=ci, alpha=alpha)
            est, lo, hi = e[station], lo[station], hi[station]
        rows = []
        for i, dd in enumerate(d):
            for j, T in enumerate(periods):
                rows.append((dd, T, est[i, j], lo[i, j], hi[i, j]))
        return pd.DataFrame(rows, columns=["duration_h", "period", "intensity", "lower", "upper"])

    # ---------------------------------------------------------- simulation
    def simulate(self, n_years: int | None = None, n_rep: int = 1, seed: int = 0) -> np.ndarray:
        """Simulate ``(S, n_rep, n_years, m)`` intensities from the fitted joint model."""
        S, n, m = self.Y.shape
        n_years = n_years or n
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        model = self.model

        def one(nat, d, k1, k2):
            R = correlation_matrix(model, nat, d)
            L = jnp.linalg.cholesky(R + 1e-12 * jnp.eye(m))
            z = jax.random.normal(k1, (n_rep, n_years, m)) @ L.T
            if model.correlation is None:
                u = jax.scipy.stats.norm.cdf(z)
            elif model.copula == "gaussian":
                u = jax.scipy.stats.norm.cdf(z)
            else:
                w = jax.random.gamma(k2, model.df / 2, (n_rep, n_years, 1)) * 2 / model.df
                u = jt.cdf(z / jnp.sqrt(w), model.df)
            u = jnp.clip(u, 1e-15, 1 - 1e-15)
            mu, sigma, xi = marginal_params(model, nat, d, self.d0)
            return gev_ppf(u, mu, sigma, xi)

        keys1 = jax.random.split(k1, S)
        keys2 = jax.random.split(k2, S)
        return np.asarray(jax.vmap(one)(jnp.asarray(self.params), jnp.asarray(self.D), keys1, keys2))

    def bootstrap(self, n_boot: int = 500, seed: int = 0) -> IDFFit:
        """Parametric bootstrap: simulate with the observed missingness, refit all
        replicates in one batched call. Returns an :class:`IDFFit` with
        ``S * n_boot`` rows (station-major)."""
        S, n, m = self.Y.shape
        sim = self.simulate(n, n_boot, seed)  # (S, B, n, m)
        M = np.repeat(self.M[:, None], n_boot, 1)
        sim = np.where(M, sim, np.nan).reshape(S * n_boot, n, m)
        D = np.repeat(self.D, n_boot, 0)
        x0 = np.repeat(to_real(self.model, self.params), n_boot, 0)
        return _fit_arrays(self.model, sim, D, self.d0, x0=x0)

    # ----------------------------------------------------------------- plot
    def plot(self, station: int = 0, periods=(2, 10, 25, 100), ax=None, data: bool = True, *, ci: str | None = "hessian", alpha: float = 0.05, title: str | None = None):
        """IDF curves on log–log axes, with CI ribbons and empirical return-period points.

        Parameters
        ----------
        station
            Index of the station to plot (default 0).
        periods
            Return periods (years) drawn as separate curves.
        data
            If True, overlay empirical T-year intensities at each observed
            duration (Weibull plotting positions), labelled in the legend.
        ci
            Ribbon method passed to :meth:`quantile_ci` (``"hessian"``,
            ``"sandwich"``, ``"bootstrap"``), or ``None`` for curves only.
        """
        from ..plotting import plot_idf

        return plot_idf(self, station=station, periods=periods, ax=ax, data=data, ci=ci, alpha=alpha, title=title)


def _fit_arrays(model: IDFModel, Y, D, d0, x0=None, max_iter=200) -> IDFFit:
    M = np.isfinite(Y)
    Y0 = np.where(M, Y, 0.0)
    if x0 is None:
        init = _init_marginal(model, Y, M, D, d0)
        if model.correlation is not None:
            init = np.concatenate([init, _init_correlation(model, Y, M, D)], axis=1)
        x0 = to_real(model, init)
    S = Y.shape[0]
    f = _nll_real(model)
    r = minimize(f, jnp.asarray(x0), jnp.asarray(Y0), jnp.asarray(M), jnp.asarray(D), jnp.full(S, float(d0)), max_iter=max_iter)
    th = np.asarray(r.x)
    nat = np.asarray(jax.vmap(lambda t: to_natural(model, t))(jnp.asarray(th)))
    return IDFFit(model, nat, -np.asarray(r.fun), np.asarray(r.converged), float(d0), model.names, Y, M, D)


def fit_idf(
    data: IDFData | Sequence[IDFData],
    marginal: str = "general",
    correlation: str | None = None,
    copula: str = "gaussian",
    df: float = 5.0,
    d0: float = 1.0,
    *,
    init=None,
    max_iter: int = 200,
) -> IDFFit:
    """Fit an IDF scaling model by maximum likelihood, for one or many stations.

    Parameters
    ----------
    data
        One :class:`IDFData` or a list of them (fitted jointly in one batched
        optimisation, independently per station).
    marginal
        ``"simple"`` or ``"general"`` (d-GEV).
    correlation
        ``None`` (independent durations — composite likelihood),
        ``"exponential"`` or ``"matern"``.
    copula
        ``"gaussian"`` or ``"t"`` (Student-t with ``df`` degrees of freedom).
    d0
        Reference duration (hours) of ``mu0`` and ``sigma0``.
    init
        Optional natural-parameter starting values ``(p,)`` or ``(S, p)``.

    Examples
    --------
    >>> from tails.idf import IDFData, fit_idf
    >>> data = IDFData.from_eccc("701S001")          # Québec City, 1961–2024
    >>> fit = fit_idf(data, "general", "matern")      # d-GEV + Matérn Gaussian copula
    >>> fit.quantile([1.0], [100])                      # 100-year 1-h intensity
    """
    model = IDFModel(marginal, correlation, copula, df)
    items = [data] if isinstance(data, IDFData) else list(data)
    Y, D = stack(items)
    x0 = None
    if init is not None:
        x0 = to_real(model, np.broadcast_to(np.asarray(init, float), (len(items), len(model.names))))
    fit = _fit_arrays(model, Y, D, d0, x0=x0, max_iter=max_iter)
    fit.stations = [it.name for it in items]
    return fit
