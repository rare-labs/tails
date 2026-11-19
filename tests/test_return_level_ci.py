"""Profile and bootstrap intervals for POT and for GEV models with covariates."""

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import chi2

import tails
from tails.core.distributions import gev_logpdf, gpd_logpdf


def _bracket(rl):
    est, lo, hi = np.asarray(rl.estimate), np.asarray(rl.lower), np.asarray(rl.upper)
    assert np.all(np.isfinite(est)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(lo < est) and np.all(est < hi)


def test_pot_profile_and_bootstrap_bracket_and_cover_shape(rng):
    n = 3000
    base = rng.normal(size=(2, n))
    excess = stats.genpareto.rvs(0.15, scale=1.8, size=(2, 250), random_state=rng)
    base[:, :250] = 5.0 + excess
    base[0, -20:] = np.nan
    pot = tails.fit_pot(base, 5.0, npy=365.25)
    prof = pot.return_level([10, 50], ci="profile", profile_points=25)
    assert prof.estimate.shape == (2, 2)
    _bracket(prof)
    delta = pot.return_level([10, 50], ci="delta")
    assert not np.allclose(prof.upper, delta.upper)
    assert np.all((prof.upper - prof.estimate) > (prof.estimate - prof.lower))
    boot = pot.return_level([20], ci="bootstrap", n_boot=40, seed=1)
    assert boot.estimate.shape == (2, 1)
    _bracket(boot)
    with pytest.raises(ValueError, match="profile"):
        pot.return_level([10], ci="nope")


def test_pot_profile_bound_matches_chi2_deviance(rng):
    npy = 100.0
    threshold = 4.0
    period = 25.0
    raw = rng.normal(size=2500)
    excess = stats.genpareto.rvs(0.1, scale=1.5, size=200, random_state=rng)
    raw[:200] = threshold + excess
    pot = tails.fit_pot(raw, threshold, npy=npy)
    rl = pot.return_level([period], ci="profile", profile_points=31)
    _bracket(rl)
    y = raw[raw > threshold] - threshold
    k = float(y.size)
    n = float(np.isfinite(raw).sum())
    zeta = float(pot.zeta)
    xi_hat = float(pot.params["shape"])
    phi_hat = np.log(zeta) - np.log1p(-zeta)
    nll_hat = -float(pot.loglik) + _binom_nll(k, n, zeta)
    crit = float(chi2.ppf(0.95, 1))

    def nll_at(th, z):
        xi, phi = th
        zeta_ = 1.0 / (1.0 + np.exp(-phi))
        m = period * npy * zeta_
        if m <= 1.0 or z <= threshold:
            return 1e6
        if abs(xi) < 1e-8:
            denom = np.log(m) * (1.0 + xi * np.log(m) / 2.0)
        else:
            denom = (m**xi - 1.0) / xi
        scale = (z - threshold) / denom
        if not np.isfinite(scale) or scale <= 0:
            return 1e6
        ll = gpd_logpdf(y, scale, xi)
        if not np.all(np.isfinite(ll)):
            return 1e6
        return float(-np.sum(ll)) + _binom_nll(k, n, zeta_)

    z_hat = float(np.asarray(rl.estimate).reshape(-1)[0])
    anchor = nll_at([xi_hat, phi_hat], z_hat)
    assert anchor == pytest.approx(nll_hat, abs=1e-4)
    start = np.array([xi_hat, phi_hat])
    for z in (float(np.asarray(rl.lower).reshape(-1)[0]), float(np.asarray(rl.upper).reshape(-1)[0])):
        res = minimize(lambda th, z=z: nll_at(th, z), start, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-12, "fatol": 1e-14})
        assert res.success or res.fun < nll_hat + crit
        assert 2.0 * (res.fun - nll_hat) == pytest.approx(crit, abs=0.35)


def test_exponential_pot_profile_keeps_shape_fixed(rng):
    raw = 10.0 + rng.exponential(2.0, size=1500)
    pot = tails.fit_pot(raw, 10.0, npy=50.0, fixed_shape=0.0)
    rl = pot.return_level([5, 20], ci="profile", profile_points=21)
    _bracket(rl)
    manual = 10.0 + float(pot.params["scale"]) * np.log(np.array([5.0, 20.0]) * pot.npy * float(pot.zeta))
    np.testing.assert_allclose(rl.estimate, manual, rtol=1e-6)
    boot = pot.return_level([5], ci="bootstrap", n_boot=40, seed=1)
    assert boot.lower[0] < boot.estimate[0] < boot.upper[0]


def test_nonstationary_profile_and_bootstrap(rng):
    t = np.linspace(0.0, 1.0, 70)
    y = stats.genextreme.rvs(-0.12, loc=8.0 + 5.0 * t, scale=2.0, size=70, random_state=rng)
    fit = tails.fit_gev(y, loc=t)
    rows = np.array([[0.0], [1.0]])
    prof = fit.return_level([15, 40], covariates={"loc": rows}, ci="profile", profile_points=25)
    assert prof.estimate.shape == (2, 2)
    _bracket(prof)
    assert prof.estimate[1, 0] > prof.estimate[0, 0]
    # Right tail of a Fréchet-type GEV (ξ > 0): the profile is skewed upward.
    assert np.all((prof.upper - prof.estimate) > (prof.estimate - prof.lower))
    boot = fit.return_level([20], covariates={"loc": np.array([[1.0]])}, ci="bootstrap", n_boot=60, seed=2)
    assert boot.estimate.shape == (1, 1)
    _bracket(boot)
    fitted_rows = fit.return_level([10], ci="profile", profile_points=15)
    assert fitted_rows.estimate.shape == (t.size, 1)
    _bracket(fitted_rows)


def test_nonstationary_profile_bound_matches_chi2_deviance(rng):
    t = np.linspace(0.0, 1.0, 80)
    y = stats.genextreme.rvs(-0.1, loc=10.0 + 4.0 * t, scale=2.2, size=80, random_state=rng)
    fit = tails.fit_gev(y, loc=t)
    period = 30.0
    t_star = 1.0
    rl = fit.return_level([period], covariates={"loc": np.array([[t_star]])}, ci="profile", profile_points=31)
    _bracket(rl)
    nll_hat = -float(np.asarray(fit.loglik))
    w = -np.log(-np.log(1.0 - 1.0 / period))
    coef = np.asarray(fit.coef, float).ravel()

    def nll_at(th, z):
        b1, log_s, xi = th
        scale = np.exp(log_s)
        if abs(xi) < 1e-8:
            phi = w * (1.0 + xi * w / 2.0 + (xi * w) ** 2 / 6.0)
        else:
            phi = (np.exp(xi * w) - 1.0) / xi
        b0 = z - b1 * t_star - scale * phi
        mu = b0 + b1 * t
        ll = np.asarray(gev_logpdf(y, mu, scale, xi))
        if not np.all(np.isfinite(ll)):
            return 1e6
        return float(-np.sum(ll))

    start = np.array([coef[1], coef[2], coef[3]])
    z_hat = float(np.asarray(rl.estimate).reshape(-1)[0])
    anchor = nll_at(start, z_hat)
    assert anchor == pytest.approx(nll_hat, abs=1e-3)
    crit = float(chi2.ppf(0.95, 1))
    bounds = (float(np.asarray(rl.lower).reshape(-1)[0]), float(np.asarray(rl.upper).reshape(-1)[0]))
    for z in bounds:
        res = minimize(
            lambda th, z=z: nll_at(th, z),
            start,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-12, "fatol": 1e-14},
        )
        assert 2.0 * (res.fun - nll_hat) == pytest.approx(crit, abs=0.4)


def test_scale_covariate_and_per_series_location_profile(rng):
    t = np.linspace(-1.0, 1.0, 60)
    scale = np.exp(0.8 + 0.35 * t)
    y = stats.genextreme.rvs(0.05, loc=15.0, scale=scale, size=60, random_state=rng)
    fit = tails.fit_gev(y, scale=t)
    rl = fit.return_level([20], covariates={"scale": np.array([[-1.0], [1.0]])}, ci="profile", profile_points=21)
    assert rl.estimate.shape == (2, 1)
    _bracket(rl)
    assert rl.estimate[1, 0] > rl.estimate[0, 0]

    t1 = np.arange(36.0)
    t2 = np.linspace(-4.0, 4.0, 36)
    cov = np.stack([t1, t2], axis=0)[..., None]
    yy = np.stack(
        [
            rng.gumbel(5.0 + 0.08 * t1, 1.5),
            rng.gumbel(12.0 + 0.4 * t2, 1.2),
        ]
    )
    batched = tails.fit_gev(yy, loc=cov)
    out = batched.return_level([12], covariates={"loc": np.array([[0.0], [3.0]])}, ci="profile", profile_points=17)
    assert out.estimate.shape == (2, 2, 1)
    _bracket(out)
    gmle = tails.fit_gev(yy[0], loc=t1, method="gmle")
    g = gmle.return_level([8], covariates={"loc": np.array([[0.0], [35.0]])}, ci="profile", profile_points=17)
    _bracket(g)


def test_covariate_profiler_matches_stationary_profile(rng):
    y = stats.genextreme.rvs(-0.1, loc=30.0, scale=8.0, size=90, random_state=rng)
    fit = tails.fit_gev(y)
    direct = fit.return_level([20, 100], ci="profile", profile_points=31)
    lo, hi = tails.profile_covariate_return_level(fit, [20, 100], n_points=31)
    np.testing.assert_allclose(lo[0, 0], np.asarray(direct.lower), rtol=2e-2, atol=0.2)
    np.testing.assert_allclose(hi[0, 0], np.asarray(direct.upper), rtol=2e-2, atol=0.2)


def test_batch_location_trend_bootstrap(rng):
    t = np.arange(45.0)
    y = np.stack(
        [
            stats.genextreme.rvs(0.0, loc=10 + 0.05 * t, scale=2, random_state=rng),
            stats.genextreme.rvs(0.05, loc=20 + 0.03 * t, scale=2.5, random_state=rng),
        ]
    )
    fit = tails.fit_gev(y, loc=t)
    rl = fit.return_level([25], covariates={"loc": np.array([[0.0], [44.0]])}, ci="bootstrap", n_boot=40, seed=3)
    assert rl.estimate.shape == (2, 2, 1)
    _bracket(rl)
    assert np.all(rl.estimate[:, 1, 0] > rl.estimate[:, 0, 0])


def _binom_nll(k, n, zeta):
    zc = float(np.clip(zeta, 1e-12, 1.0 - 1e-12))
    return -k * np.log(zc) - (n - k) * np.log1p(-zc)
