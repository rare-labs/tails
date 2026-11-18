import numpy as np
import pytest
from scipy import stats

import tails


def test_fit_gev_matches_scipy_mle(rng):
    y = stats.genextreme.rvs(-0.1, loc=30, scale=8, size=(4, 80), random_state=rng)
    fit = tails.fit_gev(y)
    assert fit.converged.all()
    for i in range(4):
        c, loc, scale = stats.genextreme.fit(y[i])
        ll_scipy = stats.genextreme.logpdf(y[i], c, loc, scale).sum()
        # our optimum is at least as good as scipy's
        assert fit.loglik[i] >= ll_scipy - 1e-6
        assert fit.coef[i, 0] == pytest.approx(loc, rel=1e-3)


def test_batch_shape_and_missing(rng):
    y = rng.gumbel(10, 2, size=(3, 5, 40))
    y[0, 0, :10] = np.nan
    fit = tails.fit_gev(y)
    assert fit.coef.shape == (3, 5, 3)
    assert fit.nobs[0, 0] == 30
    one = tails.fit_gev(y[0, 0])
    np.testing.assert_allclose(one.coef, fit.coef[0, 0], rtol=1e-6)


def test_nonstationary_location_trend(rng):
    t = np.arange(100.0)
    y = stats.genextreme.rvs(0.0, loc=20 + 0.1 * t, scale=3, random_state=rng)
    fit = tails.fit_gev(y, loc=t)
    assert fit.param_names == ["loc", "loc:x0", "log_scale", "shape"]
    assert fit.coef[1] == pytest.approx(0.1, abs=0.04)
    st = tails.fit_gev(y)
    assert fit.loglik > st.loglik  # nested model fits better
    rl = fit.return_level([100], covariates={"loc": np.array([[0.0], [99.0]])}, ci="delta")
    assert rl.estimate.shape == (2, 1)
    assert rl.estimate[1, 0] > rl.estimate[0, 0]


def test_return_level_ci_methods_bracket_estimate(rng):
    y = stats.genextreme.rvs(-0.1, loc=30, scale=8, size=(2, 60), random_state=rng)
    fit = tails.fit_gev(y)
    for ci in ("delta", "profile", "bootstrap"):
        rl = fit.return_level([10, 100], ci=ci, n_boot=100)
        assert np.all(rl.lower < rl.estimate) and np.all(rl.estimate < rl.upper), ci
    prof = fit.return_level([100], ci="profile")
    delt = fit.return_level([100], ci="delta")
    # profile intervals are right-skewed for heavy tails
    assert np.all((prof.upper - prof.estimate) > (prof.estimate - prof.lower))
    assert delt.to_pandas().shape[0] == 2


def test_gmle_and_lmom(rng):
    y = stats.genextreme.rvs(-0.2, loc=30, scale=8, size=(20, 25), random_state=rng)
    g = tails.fit_gev(y, method="gmle")
    assert np.all(np.abs(g.coef[:, 2]) < 0.5)
    lm = tails.fit_gev(y, method="lmom")
    assert lm.coef.shape == (20, 3)


def test_gpd_and_pot(rng):
    x = stats.genpareto.rvs(0.2, scale=2, size=20000, random_state=rng)
    fit = tails.fit_gpd(x, threshold=1.0)
    assert fit.converged
    assert fit.coef[1] == pytest.approx(0.2, abs=0.05)
    pot = tails.fit_pot(x, 1.0, npy=365.25)
    rl = pot.return_level([10, 100], ci="delta")
    assert np.all(np.diff(rl.estimate) > 0)
    assert np.all(rl.lower < rl.estimate)


def test_lmoments_known_values():
    x = np.arange(1.0, 11.0)
    lm = np.asarray(tails.sample_lmoments(x))
    assert lm[0] == pytest.approx(5.5)
    assert lm[1] == pytest.approx(1.8333333333)
    assert lm[2] == pytest.approx(0.0, abs=1e-12)


def test_minimize_rosenbrock_batch():
    import jax.numpy as jnp

    def f(x, a):
        return (a - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    r = tails.minimize(f, np.zeros((3, 2)), jnp.array([1.0, 2.0, -1.0]))
    np.testing.assert_allclose(r.x[:, 0], [1, 2, -1], atol=1e-6)
    assert r.converged.all()
