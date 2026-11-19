import numpy as np
import pandas as pd
import pytest

import tails


def test_conditional_independence_simulation_and_bootstrap():
    x = np.random.default_rng(45).exponential(size=(3, 2200))
    fit = tails.fit_conditional(x, conditioning=0)
    samples = fit.sample(6000, given=6, seed=8)
    assert samples.shape == (6000, 3)
    assert np.all(samples[:, 0] > 6)
    assert np.quantile(samples[:, 1], .9) == pytest.approx(-np.log(.1), abs=.7)
    np.testing.assert_array_equal(samples[:100], fit.sample(6000, given=6, seed=8)[:100])
    q = fit.quantile(.9, given=6, n_boot=12, size=600, seed=4)
    assert q.estimate.shape == (3, 1)
    assert np.all(q.lower <= q.upper)
    assert q.lower[1, 0] < -np.log(.1) < q.upper[1, 0]


def test_conditional_batched_positive_dependence():
    rng = np.random.default_rng(91)
    x = rng.exponential(size=(2, 1800))
    y = .7 * x + rng.exponential(.3, size=x.shape)
    fit = tails.fit_conditional(np.stack([x, y], axis=1))
    assert fit.alpha.shape == (2, 1)
    assert np.all(fit.alpha > .3)
    samples = fit.sample(1000, given=6, seed=0)
    assert samples.shape == (2, 1000, 2)
    assert np.all(np.median(samples[..., 1], axis=-1) > 3)


def test_conditional_rejects_body_condition_and_short_records():
    x = np.random.default_rng(10).exponential(size=(2, 1000))
    fit = tails.fit_conditional(x)
    with pytest.raises(ValueError, match="threshold"):
        fit.sample(10, given=.1)
    with pytest.raises(ValueError):
        tails.fit_conditional(x[:, :10])


def test_conditional_multiple_conditions():
    x = np.random.default_rng(54).exponential(size=(3, 1500))
    fit = tails.fit_conditional(x)
    draw = fit.sample(500, given=5, conditions={1: 1.5}, seed=2)
    assert np.all(draw[:, 0] > 5)
    assert np.all(draw[:, 1] > 1.5)


def test_conditional_rejects_perfect_dependence():
    x = np.random.default_rng(4).exponential(size=1500)
    with pytest.raises(ValueError, match="perfect"):
        tails.fit_conditional(np.stack([x, 2 * x]))


def test_conditional_no_artificial_upper_tail_cap():
    x = np.random.default_rng(44).exponential(size=(2, 1800))
    fit = tails.fit_conditional(x)
    margin = fit._margins[0][0]
    # Exponential margin beyond the old 1e-12 survival clipping boundary.
    margin.shape, margin.scale = 0., 1.
    draw = fit.sample(100, given=40, seed=1)
    assert np.all(draw[:, 0] > 40)
    margin.shape = -.5
    with pytest.raises(ValueError, match="support"):
        fit.sample(10, given=40)


def test_conditional_tied_dry_margin_preserves_tail_mass():
    rng = np.random.default_rng(18)
    x = rng.exponential(size=(2, 2400))
    x[0, :2280] = 0  # 95% zero, empirical tail mass is 5%, not nominal 10%.
    fit = tails.fit_conditional(x)
    margin = fit._margins[0][0]
    assert margin.cdf(0) == pytest.approx(.95)
    assert fit.thresholds == pytest.approx(0)


def test_conditional_hourly_frame_and_default_bootstrap():
    x = np.random.default_rng(32).exponential(size=(2, 1500))
    frame = pd.DataFrame(x.T, index=pd.date_range("2000-01-01", periods=1500, freq="h"))
    fit = tails.fit_conditional(frame)
    array_fit = tails.fit_conditional(x)
    np.testing.assert_allclose(fit.alpha, array_fit.alpha)
    q = fit.quantile(.8, n_boot=4, size=400, seed=3)
    assert np.isfinite(q.lower).all() and np.isfinite(q.upper).all()


def test_conditional_dry_target_does_not_become_mostly_wet():
    x = np.random.default_rng(18).exponential(size=(2, 2400))
    x[1, :2280] = 0
    fit = tails.fit_conditional(x)
    draws = fit.sample(10000, given=5, seed=2)
    assert np.mean(draws[:, 1] > 0) == pytest.approx(.05, abs=.03)
