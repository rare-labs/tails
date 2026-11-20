import numpy as np
import pytest


def test_empirical_body_preserves_atoms_and_tail_inversion():
    from tails.margins import TailMargin

    margin = TailMargin(np.array([0., 0., 1., 2., 4.]), 2., .8, 1., 0.)
    assert margin.cdf(0.) == pytest.approx(.4)
    assert margin.sf(0.) == pytest.approx(.6)
    np.testing.assert_array_equal(margin.ppf([.1, .3]), [0., 0.])
    p = np.array([.85, .95, .999999])
    np.testing.assert_allclose(margin.cdf(margin.ppf(p)), p)
    assert margin.cdf(-1.) == 0
    assert margin.sf(1000.) == 0


def test_grouped_margins_keep_distinct_distributions_and_require_context():
    from tails.margins import fit_marginal

    rng = np.random.default_rng(42)
    groups = np.repeat(["idle", "loaded"], 1000)
    x = rng.exponential(size=2000) * np.repeat([1., 4.], 1000)
    fit = fit_marginal(x, groups=groups, threshold_quantile=.9)
    with pytest.raises(ValueError, match="groups"):
        fit.sf(5.)
    assert fit.sf(5., groups="loaded") > fit.sf(5., groups="idle")
    np.testing.assert_allclose(fit.ppf(fit.cdf(x, groups=groups), groups=groups), x, rtol=1e-7, atol=1e-7)
    assert fit.labels == ("idle", "loaded")


def test_margin_tail_limit_and_missing_values():
    from tails.margins import TailMargin

    margin = TailMargin(np.arange(1., 6.), 4., .8, 1., -.2)
    assert margin.ppf(1.) == pytest.approx(9.)
    assert margin.sf(10.) == 0
    assert np.isnan(margin.cdf(np.nan))


def test_missing_group_labels_are_rejected():
    from tails.margins import fit_marginal

    with pytest.raises(ValueError, match="label"):
        fit_marginal(np.arange(100.), groups=np.full(100, np.nan))

