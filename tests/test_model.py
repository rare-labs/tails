"""The shared engine: two families, options not models.

Golden numbers were produced by the per-model code that the generic engine
replaced (tails 0.1.0 plus the first r-largest commit); the engine must
reproduce them to about 1e-6.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.optimize import minimize as sp_minimize
from scipy.stats import chi2, genextreme

import tails

GEV_COEF = [[29.6682370738, 2.1628841771, 0.0733873569], [29.183575298, 1.966316675, 0.1491341791],
            [32.248059318, 2.1979864372, -0.0116868699]]
GEV_PROF = [[[63.57044898], [59.77846667], [61.61315453]], [[121.52616942], [141.73170161], [107.51152759]]]
NS_PROF = [[[59.19164362], [67.56817886]], [[50.43414704], [58.72942735]], [[75.98477403], [84.66868709]]]
POT_PROF = [[11.33772445, 20.10715155], [8.45907204, 12.58044374], [19.16329904, 47.72765538]]
POTEXP_PROF = [[7.59095186], [6.87321253], [8.46383656]]
RL_COEF = [30.68983681, 1.76532974, -0.09817006]
RL_PROF = [[49.6318576], [44.748954], [63.8991338]]


@pytest.fixture(scope="module")
def golden_data():
    rng = np.random.default_rng(7)
    y = stats.genextreme.rvs(-0.1, loc=30, scale=8, size=(3, 50), random_state=rng)
    x = rng.normal(size=2000)
    x[:150] = 2.0 + stats.genpareto.rvs(0.1, scale=1.3, size=150, random_state=rng)
    d = rng.gamma(0.6, 6.0, size=(30, 365))
    return y, x, -np.sort(-d, axis=-1)[:, :3]


def test_engine_reproduces_the_per_model_code(golden_data):
    y, x, z = golden_data
    f = tails.fit_gev(y)
    np.testing.assert_allclose(f.coef, GEV_COEF, rtol=1e-8)
    r = f.return_level([100], ci="profile")
    np.testing.assert_allclose(np.stack([r.lower, r.upper]), GEV_PROF, rtol=1e-6)
    t = np.arange(50.0)
    ns = tails.fit_gev(y[0] + 0.2 * t, loc=t)
    r = ns.return_level([20], covariates={"loc": np.array([0.0, 49.0])}, ci="profile")
    np.testing.assert_allclose(np.stack([r.estimate, r.lower, r.upper]), NS_PROF, rtol=1e-6)
    pot = tails.fit_pot(x, 2.0, npy=100.0)
    r = pot.return_level([10, 50], ci="profile")
    np.testing.assert_allclose(np.stack([r.estimate, r.lower, r.upper]), POT_PROF, rtol=1e-6)
    exp = tails.fit_pot(x, 2.0, npy=100.0, fixed_shape=0.0)
    r = exp.return_level([10], ci="profile")
    np.testing.assert_allclose(np.stack([r.estimate, r.lower, r.upper]), POTEXP_PROF, rtol=1e-6)
    rl = tails.fit_gev(z, r=z.shape[-1])
    np.testing.assert_allclose(rl.coef, RL_COEF, rtol=1e-6)
    r = rl.return_level([50], ci="profile")
    np.testing.assert_allclose(np.stack([r.estimate, r.lower, r.upper]), RL_PROF, rtol=1e-6)


# ------------------------------------------------------------ GEV family
def _rlargest_with_trend(rng, n=60, r=3, slope=0.15):
    """Top-r of each block from a GEV point process whose location drifts."""
    t = np.arange(n, dtype=float)
    loc = 20.0 + slope * t
    gam = np.cumsum(rng.exponential(size=(n, r)), axis=1)
    xi, sc = 0.1, 4.0
    z = loc[:, None] + sc * (gam ** (-xi) - 1.0) / xi
    return z, t


def test_fit_gev_r_and_r1_is_block_maxima():
    rng = np.random.default_rng(11)
    z, _ = _rlargest_with_trend(rng)
    a = tails.fit_gev(z, r=3)
    assert a.r == 3 and isinstance(a, tails.GEVFit) and a.coef.shape == (3,)
    assert not hasattr(tails, "fit_rlargest") and not hasattr(tails, "RLargestFit")
    one = tails.fit_gev(z, r=1)
    np.testing.assert_allclose(one.coef, tails.fit_gev(z.max(axis=1)).coef, rtol=1e-8)


def test_covariate_rlargest_matches_independent_likelihood():
    rng = np.random.default_rng(12)
    z, t = _rlargest_with_trend(rng)
    f = tails.fit_gev(z, r=3, loc=t)
    assert bool(f.converged)
    assert f.param_names == ["loc", "loc:x0", "log_scale", "shape"]

    def nll(th):
        loc = th[0] + th[1] * t
        sc, xi = np.exp(th[2]), th[3]
        lp = genextreme.logpdf(z, -xi, loc[:, None], sc)
        lc = genextreme.logcdf(z, -xi, loc[:, None], sc)
        if not np.all(np.isfinite(lp)):
            return np.inf
        return -(np.sum(lp - lc) + np.sum(lc[:, -1]))

    th = np.asarray(f.coef)
    np.testing.assert_allclose(-nll(th), f.loglik, rtol=1e-9)
    res = sp_minimize(nll, th + [0.3, 0.01, 0.05, 0.02], method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000})
    np.testing.assert_allclose(res.x, th, atol=5e-4)
    # same slope sign and size as the simulation; more data per block → smaller SE than maxima alone
    assert 0.08 < th[1] < 0.22
    bm = tails.fit_gev(z[:, 0], loc=t)
    assert f.se[1] < bm.se[1]
    rl = f.return_level([50], covariates={"loc": [0.0, 59.0]}, ci="profile")
    assert rl.estimate.shape == (2, 1)
    assert np.all(rl.lower < rl.estimate) and np.all(rl.estimate < rl.upper)


def test_fixed_shape_is_the_generic_path():
    rng = np.random.default_rng(13)
    y = rng.gumbel(10.0, 2.0, size=120)
    g = tails.fit_gev(y, fixed_shape=0.0)
    loc, scale = stats.gumbel_r.fit(y)
    np.testing.assert_allclose([g.coef[0], np.exp(g.coef[1])], [loc, scale], rtol=1e-5)
    assert g.coef[2] == 0.0 and g.cov[2, 2] == 0.0 and g.n_params == 2
    assert g._spec.free == (0, 1)
    # a fixed shape on r-largest data and with a location trend also go through the same path
    z, t = _rlargest_with_trend(rng)
    fr = tails.fit_gev(z, r=3, loc=t, fixed_shape=0.1)
    assert fr.converged and np.allclose(fr.params["shape"], 0.1) and fr.n_params == 3
    lo, hi = fr.return_level([20], covariates={"loc": [10.0]}, ci="profile").lower, None
    assert np.isfinite(lo).all()
    del hi


# ------------------------------------------------------------ GPD family
def _seasonal_pot(rng, n_years=40, npy=100):
    n = n_years * npy
    temp = np.repeat(rng.normal(size=n_years), npy)
    zeta = 1.0 / (1.0 + np.exp(-(-3.0 + 0.6 * temp)))
    hit = rng.uniform(size=n) < zeta
    scale = np.exp(0.3 + 0.25 * temp)
    excess = stats.genpareto.rvs(0.1, scale=scale, random_state=rng)
    x = np.where(hit, 5.0 + excess, rng.uniform(0.0, 5.0, size=n))
    return x, temp


def test_nonstationary_pot_return_level():
    rng = np.random.default_rng(14)
    x, temp = _seasonal_pot(rng)
    pot = tails.fit_pot(x, 5.0, npy=100.0, scale=temp, rate=temp)
    assert pot.param_names == ["log_scale", "log_scale:x0", "shape"]
    assert pot.rate_coef.shape == (2,) and pot.zeta.shape == (x.size,)
    assert 0.3 < pot.rate_coef[1] < 0.9  # logit slope of the exceedance probability
    np.testing.assert_allclose(np.mean(pot.rate), np.mean(x > 5.0) * 100.0, rtol=0.02)
    rows = {"scale": np.array([-1.0, 1.0]), "rate": np.array([-1.0, 1.0])}
    rl = pot.return_level([10, 50], covariates=rows, ci="profile")
    assert rl.estimate.shape == (2, 2)
    assert np.all(rl.lower < rl.estimate) and np.all(rl.estimate < rl.upper)
    assert np.all(rl.estimate[1] > rl.estimate[0])  # hot years: higher levels
    p = pot.params_at(rows)
    sc, xi, zeta = p["scale"], p["shape"], p["rate"]
    manual = 5.0 + sc[:, None] / xi[:, None] * ((np.array([10.0, 50.0]) * 100.0 * zeta[:, None]) ** xi[:, None] - 1.0)
    np.testing.assert_allclose(rl.estimate, manual, rtol=1e-8)
    block = pot.return_level([10, 50], covariates=rows, kind="block")
    lam = 100.0 * zeta[:, None]
    manual_b = 5.0 + sc[:, None] / xi[:, None] * ((lam / -np.log1p(-1.0 / np.array([10.0, 50.0]))) ** xi[:, None] - 1.0)
    np.testing.assert_allclose(block.estimate, manual_b, rtol=1e-8)
    boot = pot.return_level([20], covariates=rows, ci="bootstrap", n_boot=40, seed=1)
    assert np.all(boot.lower < boot.upper)


def test_pot_profile_matches_direct_deviance():
    """The profile bound of a non-stationary POT level sits at the χ² crossing."""
    rng = np.random.default_rng(15)
    x, temp = _seasonal_pot(rng, n_years=30)
    pot = tails.fit_pot(x, 5.0, npy=100.0, rate=temp)
    rl = pot.return_level([25], covariates={"rate": [1.0]}, ci="profile", profile_points=31)
    z_hi = float(rl.upper.ravel()[0])
    e = x > 5.0
    y = x[e] - 5.0
    crit = chi2.ppf(0.95, 1)

    def nll(th, z=None):
        xi, a, b = th
        zeta = 1.0 / (1.0 + np.exp(-(a + b * temp)))
        zs = 1.0 / (1.0 + np.exp(-(a + b * 1.0)))
        m = 25.0 * 100.0 * zs
        sc = (z - 5.0) / ((m**xi - 1.0) / xi)
        if sc <= 0:
            return np.inf
        lp = stats.genpareto.logpdf(y, xi, scale=sc)
        if not np.all(np.isfinite(lp)):
            return np.inf
        return -(np.sum(lp) + np.sum(np.log(zeta[e])) + np.sum(np.log1p(-zeta[~e])))

    xi0, a0, b0 = float(pot.coef[1]), *map(float, pot.rate_coef)
    nll_hat = -(float(pot.loglik) + float(pot.loglik_rate))
    res = sp_minimize(nll, [xi0, a0, b0], args=(z_hi,), method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    assert 2.0 * (res.fun - nll_hat) == pytest.approx(crit, abs=0.05)


def test_gpd_return_level_in_exceedances_and_covariate_profile():
    rng = np.random.default_rng(16)
    e = stats.genpareto.rvs(0.2, scale=2.0, size=300, random_state=rng)
    g = tails.fit_gpd(e)
    rl = g.return_level([100], ci="profile")
    sc, xi = float(g.params["scale"]), float(g.params["shape"])
    np.testing.assert_allclose(rl.estimate, sc / xi * (100.0**xi - 1.0), rtol=1e-8)
    assert rl.lower[0] < rl.estimate[0] < rl.upper[0]
    grp = (np.arange(300) % 2).astype(float)
    gs = tails.fit_gpd(e * np.where(grp == 1, 3.0, 1.0), scale=grp)  # shared shape, two scales
    lo, hi = tails.profile_covariate_return_level(gs, [50], covariates={"scale": [0.0, 1.0]})
    assert lo.shape == (1, 2, 1) and np.all(lo < hi)
    with pytest.raises(ValueError, match="fit_pot"):
        g.return_level([10], kind="block")


# ------------------------------------------------------------ surface
def test_dataframe_covariates_and_params_at():
    rng = np.random.default_rng(17)
    df = pd.DataFrame({"year": np.arange(1960, 2020, dtype=float), "temp": rng.normal(size=60)})
    y = rng.gumbel(30.0, 5.0, size=60) + 0.1 * (df.year - 1960) + 2.0 * df.temp
    f = tails.fit_gev(y.to_numpy(), loc=df[["year", "temp"]])
    assert f.param_names == ["loc", "loc:year", "loc:temp", "log_scale", "shape"]
    new = pd.DataFrame({"year": [1970.0, 2010.0], "temp": [0.0, 1.0]})
    a = f.return_level([50], covariates=new)
    b = f.return_level([50], covariates={"loc": new[["year", "temp"]].to_numpy()})
    np.testing.assert_allclose(a.estimate, b.estimate)
    p = f.params_at(new)
    np.testing.assert_allclose(p["loc"], f.coef[0] + f.coef[1] * new.year + f.coef[2] * new.temp)
    with pytest.raises(ValueError, match="missing column"):
        f.return_level([50], covariates=new[["year"]])
    with pytest.raises(ValueError, match="unknown parameter"):
        f.return_level([50], covariates={"location": [1.0]})


def test_confint_profile_close_to_wald_for_large_samples():
    rng = np.random.default_rng(18)
    y = stats.genextreme.rvs(-0.05, loc=10, scale=2, size=(2, 2000), random_state=rng)
    f = tails.fit_gev(y)
    prof = f.confint()
    wald = f.confint(method="wald")
    assert list(prof.columns) == ["series", "term", "estimate", "lower", "upper"]
    np.testing.assert_allclose(prof.lower, wald.lower, rtol=0, atol=0.02)
    np.testing.assert_allclose(prof.upper, wald.upper, rtol=0, atol=0.02)
    gum = tails.fit_gev(y[0], fixed_shape=0.0).confint()
    assert gum.lower.iloc[2] == gum.upper.iloc[2] == 0.0
