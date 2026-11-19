import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, gamma

import tails


def test_joint_reductions_and_missing():
    x = np.array([[2, 4, 6, 8], [6, 3, 12, np.nan]])
    np.testing.assert_allclose(tails.compound_series(x, [2, 3]), [1, 1, 3, np.nan])
    np.testing.assert_allclose(tails.compound_series(x, [2, 3], event="kofn", k=1), [2, 2, 4, np.nan])
    np.testing.assert_allclose(tails.compound_series(x, 10, event="sum"), [.8, .7, 1.8, np.nan])


@pytest.mark.parametrize("event,k,value", [("all", None, 2), ("kofn", 2, 3), ("sum", None, 10)])
def test_joint_known_probabilities(event, k, value):
    x = np.random.default_rng(26).exponential(size=(3, 40000))
    p = {"all": np.exp(-6), "kofn": binom.sf(1, 3, np.exp(-3)), "sum": gamma.sf(10, 3)}[event]
    out = tails.how_rare_joint(x, value, event=event, k=k, ci=None)
    assert out.probability == pytest.approx(p, rel=.25)
    assert out.expected_count == pytest.approx(40000 * out.probability)
    assert out.return_period == pytest.approx(1 / (365.25 * out.probability))


def test_run_boundaries_and_observed_count():
    x = np.array([[[2, 2, 2, 0, 2]], [[2, 0, 2, 2, 2]]])
    y = tails.compound_series(x, 1, run_length=2)
    np.testing.assert_allclose(y, [[2, 2, 0, 0], [0, 0, 2, 2]])


def test_joint_run_monte_carlo_and_pooling():
    x = np.random.default_rng(71).exponential(size=(4, 1, 10000))
    out = tails.how_rare_joint(x, 2, run_length=2, ci=None)
    p = np.exp(-2)
    exact = p**2 + (10000 - 2) * (1 - p) * p**2
    assert out.expected_count == pytest.approx(exact, rel=.25)
    assert out.observed_count == sum(np.count_nonzero((a[0, 1:] > 2) & (a[0, :-1] > 2) & np.r_[True, a[0, :-2] <= 2]) for a in x) / 4


def test_joint_profile_subannual_and_count_bounds():
    x = np.random.default_rng(9).exponential(size=(2, 1800))
    out = tails.how_rare_joint(x, 2.1, ci="profile", record_length=365)
    assert out.lower < out.return_period < out.upper < 1
    assert out.count_lower < out.expected_count < out.count_upper


def test_joint_dataframe_preserves_calendar_gaps():
    df = pd.DataFrame({"a": [2, 3, 4]}, index=pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04"]))
    np.testing.assert_allclose(tails.compound_series(df, 1, run_length=2), [2, np.nan, np.nan])


@pytest.mark.parametrize("kwargs", [{"thresholds": 0}, {"thresholds": -1}, {"thresholds": 1, "k": 0, "event": "kofn"}, {"thresholds": 1, "run_length": 1.5}])
def test_joint_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        tails.compound_series(np.ones((2, 20)), **kwargs)


def test_joint_unequal_thresholds_and_exposure_scaling():
    x = np.random.default_rng(8).exponential(size=(2, 15000))
    out = tails.how_rare_joint(x, [2, 3], ci=None, record_length=100)
    assert out.probability == pytest.approx(np.exp(-5), rel=.2)
    assert out.expected_count == pytest.approx(100 * out.probability)


def test_joint_rejects_body_event_and_all_missing():
    with pytest.raises(ValueError, match="fitting threshold"):
        tails.how_rare_joint(np.ones((2, 500)), .1)
    with pytest.raises(ValueError, match="complete"):
        tails.how_rare_joint(np.full((2, 500), np.nan), 1)


def test_joint_zero_tail_does_not_invent_profile_bound():
    x = np.random.default_rng(20).beta(1, 3, size=(1, 3500))
    result = tails.how_rare_joint(x, 10, ci="profile")
    assert result.probability == 0
    assert result.return_period == np.inf
    assert np.isnan(result.lower)
    assert result.upper == np.inf
    assert np.isnan(result.count_upper)
