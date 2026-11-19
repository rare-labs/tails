import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize as sp_minimize
from scipy.stats import genextreme

import tails
from tails import block_top_r, fit_gev


def _top_r(rng, n_years=40, r=4, n_days=365):
    daily = rng.gamma(0.6, 6.0, size=(n_years, n_days))
    return -np.sort(-daily, axis=1)[:, :r]


def _scipy_nll(theta, z):
    """Independent r-largest NLL with scipy (c = -xi)."""
    loc, scale, xi = theta[0], np.exp(theta[1]), theta[2]
    lp = genextreme.logpdf(z, -xi, loc, scale)
    lc = genextreme.logcdf(z, -xi, loc, scale)
    if not np.all(np.isfinite(lp)):
        return np.inf
    return -(np.sum(lp - lc) + np.sum(lc[:, -1]))


def test_r1_is_the_gev_fit():
    rng = np.random.default_rng(0)
    z = _top_r(rng, r=3)
    f1 = fit_gev(z, r=1)
    g = tails.fit_gev(z[:, 0])
    np.testing.assert_allclose(f1.coef, g.coef, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(f1.loglik, g.loglik, rtol=1e-9)
    assert f1.r == 1


def test_matches_independent_scipy_likelihood():
    rng = np.random.default_rng(1)
    z = _top_r(rng, r=4)
    f = fit_gev(z, r=4)
    assert bool(f.converged)
    theta = np.asarray(f.coef)
    np.testing.assert_allclose(-_scipy_nll(theta, z), f.loglik, rtol=1e-9)
    res = sp_minimize(_scipy_nll, theta + np.array([0.5, 0.05, 0.02]), args=(z,), method="Nelder-Mead",
                      options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 20000})
    np.testing.assert_allclose(res.x, theta, atol=2e-4)
    assert f.loglik >= -res.fun - 1e-7


def test_ragged_blocks_and_order_within_block():
    rng = np.random.default_rng(2)
    z = _top_r(rng, r=3)
    z_ragged = z.copy()
    z_ragged[5, 2] = np.nan
    z_ragged[7, 1:] = np.nan
    shuffled = z_ragged[:, ::-1]
    a = fit_gev(z_ragged, r=3)
    b = fit_gev(shuffled, r=3)
    np.testing.assert_allclose(a.coef, b.coef, rtol=1e-8)
    keep = np.isfinite(z_ragged)
    theta = np.asarray(a.coef)
    loc, scale, xi = theta[0], np.exp(theta[1]), theta[2]
    lp = genextreme.logpdf(z, -xi, loc, scale)
    lc = genextreme.logcdf(z, -xi, loc, scale)
    last = keep.sum(1) - 1
    ll = np.sum(np.where(keep, lp - lc, 0.0)) + np.sum(lc[np.arange(len(z)), last])
    np.testing.assert_allclose(ll, a.loglik, rtol=1e-9)


def test_batch_equals_separate_fits():
    rng = np.random.default_rng(3)
    z1, z2 = _top_r(rng, r=3), _top_r(rng, r=3)
    both = fit_gev(np.stack([z1, z2]), r=3)
    assert both.coef.shape == (2, 3)
    np.testing.assert_allclose(both.coef[0], fit_gev(z1, r=3).coef, rtol=1e-6)
    np.testing.assert_allclose(both.coef[1], fit_gev(z2, r=3).coef, rtol=1e-6)


def test_return_levels_and_profile():
    rng = np.random.default_rng(4)
    z = _top_r(rng, n_years=50, r=3)
    f = fit_gev(z, r=3)
    rl = f.return_level([10, 100], ci="profile")
    assert rl.estimate.shape == (2,)
    assert np.all(rl.lower < rl.estimate) and np.all(rl.estimate < rl.upper)
    d = f.return_level([100], ci="delta")
    np.testing.assert_allclose(d.estimate, rl.estimate[1:])
    boot = f.return_level([100], ci="bootstrap", n_boot=60, seed=1)
    assert boot.lower[0] < boot.estimate[0] < boot.upper[0]


def test_block_top_r_separation():
    idx = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    s = pd.Series(0.0, index=idx)
    s["2000-03-01"], s["2000-03-02"], s["2000-07-01"], s["2000-09-01"] = 50.0, 45.0, 30.0, 20.0
    s["2001-05-01"] = 10.0
    out = block_top_r(s, 3, min_separation="3D")
    assert list(out.columns) == [1, 2, 3]
    assert out.loc["2000-01-01"].tolist() == [50.0, 30.0, 20.0]
    assert out.loc["2001-01-01"].tolist()[0] == 10.0
    no_gap = block_top_r(s, 2)
    assert no_gap.loc["2000-01-01"].tolist() == [50.0, 45.0]
    assert block_top_r(s, 2, min_count=400).empty


def test_input_checks():
    with pytest.raises(ValueError):
        fit_gev(np.arange(10.0), r=1)
    with pytest.raises(ValueError):
        fit_gev(np.ones((10, 3)), r=4)
