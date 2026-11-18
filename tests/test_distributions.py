import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from tails.core._derivs import gev_derivs, gpd_derivs
from tails.core.distributions import gev_cdf, gev_logpdf, gev_ppf, gpd_cdf, gpd_logpdf, gpd_ppf

xis = st.floats(-0.6, 0.6)


@given(xi=xis, y=st.floats(-5, 50), loc=st.floats(-5, 5), scale=st.floats(0.2, 5))
@settings(max_examples=200, deadline=None)
def test_gev_logpdf_matches_scipy(xi, y, loc, scale):
    ours = float(gev_logpdf(y, loc, scale, xi))
    ref = stats.genextreme.logpdf(y, 0.0 if abs(xi) < 1e-12 else -xi, loc, scale)
    if np.isfinite(ref):
        assert ours == pytest.approx(ref, rel=1e-8, abs=1e-8)
    else:
        assert ours == -np.inf


@given(xi=xis, p=st.floats(1e-6, 1 - 1e-6))
@settings(max_examples=200, deadline=None)
def test_gev_ppf_cdf_roundtrip(xi, p):
    q = gev_ppf(p, 1.0, 2.0, xi)
    assert float(gev_cdf(q, 1.0, 2.0, xi)) == pytest.approx(p, rel=1e-7, abs=1e-10)


@given(xi=xis, y=st.floats(0, 40), scale=st.floats(0.2, 5))
@settings(max_examples=200, deadline=None)
def test_gpd_matches_scipy(xi, y, scale):
    # scipy overflows (-1/c) for subnormal c, so compare tiny shapes with its exact c=0 branch;
    # tails itself is evaluated at the generated xi.
    xs = 0.0 if abs(xi) < 1e-12 else xi
    ref = stats.genpareto.logpdf(y, xs, 0, scale)
    ours = float(gpd_logpdf(y, scale, xi))
    if np.isfinite(ref):
        assert ours == pytest.approx(ref, rel=1e-8, abs=1e-8)
        assert float(gpd_cdf(y, scale, xi)) == pytest.approx(stats.genpareto.cdf(y, xs, 0, scale), rel=1e-7, abs=1e-12)


def test_shape_zero_is_gumbel_and_smooth():
    y = jnp.linspace(-2, 10, 7)
    g = stats.gumbel_r.logpdf(np.asarray(y), 1.0, 2.0)
    for xi in (0.0, 1e-12, -1e-9, 1e-7):
        np.testing.assert_allclose(gev_logpdf(y, 1.0, 2.0, xi), g, rtol=1e-6)
    # gradient w.r.t. shape is finite at exactly zero
    gr = jax.grad(lambda x: jnp.sum(gev_logpdf(y, 1.0, 2.0, x)))(0.0)
    assert np.isfinite(gr)


def test_gpd_ppf_inverse():
    p = jnp.linspace(0.01, 0.99, 9)
    for xi in (-0.3, 0.0, 0.4):
        np.testing.assert_allclose(gpd_cdf(gpd_ppf(p, 2.0, xi), 2.0, xi), p, rtol=1e-9)


@pytest.mark.parametrize("xi", [0.3, -0.2, 0.0, 1e-4, -5e-4, 2e-3])
def test_closed_form_derivatives_match_autodiff(xi):
    f = lambda e, y: gev_logpdf(y, e[0], jnp.exp(e[1]), e[2])  # noqa: E731
    g = lambda e, y: gpd_logpdf(y, jnp.exp(e[0]), e[1])  # noqa: E731
    for y in (25.0, 33.0, 50.0):
        e = jnp.array([30.0, np.log(8.0), xi])
        lp, G, H = gev_derivs(y, *e)
        np.testing.assert_allclose(G, jax.grad(f)(e, y), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(H, jax.hessian(f)(e, y), rtol=1e-5, atol=1e-6)
        e2 = jnp.array([np.log(8.0), xi])
        lp, G, H = gpd_derivs(y - 20.0, *e2)
        np.testing.assert_allclose(G, jax.grad(g)(e2, y - 20.0), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(H, jax.hessian(g)(e2, y - 20.0), rtol=1e-5, atol=1e-6)
