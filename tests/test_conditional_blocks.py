import numpy as np
import pytest

import tails


def test_conditional_block_refits_report_all_attempts():
    x = np.random.default_rng(42).exponential(size=(2, 1200))
    fit = tails.fit_conditional(x, threshold_quantile=.9, marginal_quantile=.8)
    result = fit.quantile([.5, .9], given=5., n_boot=3, bootstrap="block",
                          block_length=4, size=300, seed=8)
    assert result.bootstrap.diagnostics.n_attempted == 3
    assert result.bootstrap.draws.shape == (3, 2, 2)
    assert result.estimate.shape == (2, 2)
    with pytest.raises(ValueError, match="block_length"):
        fit.quantile(n_boot=2, bootstrap="block")


def test_conditional_context_is_explicit_and_preserves_gaps():
    x = np.random.default_rng(1).exponential(size=(2, 1200))
    groups = np.repeat(["low", "high"], 600)
    with pytest.raises(ValueError, match="context"):
        tails.fit_conditional(x, groups=groups)
    fit = tails.fit_conditional(x, groups=groups, context="low", threshold_quantile=.9,
                                marginal_quantile=.8)
    assert np.isnan(fit._data[:, 600:]).all()
    assert fit.context == "low"
    assert fit.sample(10).shape == (10, 2)


def test_conditional_default_bootstrap_holds_original_given_fixed():
    x = np.random.default_rng(13).exponential(size=(2, 1200))
    fit = tails.fit_conditional(x, threshold_quantile=.9, marginal_quantile=.8)
    result = fit.quantile(n_boot=4, size=200, seed=3)
    diagnostics = result.bootstrap.diagnostics
    assert diagnostics.n_attempted == 4
    if diagnostics.n_successful < 4:
        assert np.isnan(result.lower).all()
