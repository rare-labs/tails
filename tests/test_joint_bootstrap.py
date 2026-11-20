import numpy as np
import pandas as pd
import pytest

import tails


def test_joint_subdaily_and_observation_units():
    rng = np.random.default_rng(8)
    x = rng.exponential(size=(1800, 2))
    frame = pd.DataFrame(x, index=pd.date_range("2020", periods=len(x), freq="h"))
    a = tails.how_rare_joint(frame, 3., ci=None, npy=None)
    assert a.period_unit == "observations"
    assert a.return_period == pytest.approx(1/a.probability)
    b = tails.how_rare_joint(frame.drop(frame.index[10]), 3., ci=None, npy=None, freq="h")
    assert b.record_length == len(frame)-1


def test_joint_grouped_block_bootstrap_and_fitted_threshold_crossings():
    x = np.random.default_rng(42).exponential(size=(2, 2000))
    groups = np.tile(["off", "on"], 1000)
    r = tails.how_rare_joint(x, 3., groups=groups, ci="block_bootstrap", block_length=4,
                            n_boot=6, seed=9, npy=None)
    assert r.bootstrap.diagnostics.n_attempted == 6
    assert r.expected_count > 0
    assert r.bootstrap.draws.shape == (6,)
    with pytest.raises(ValueError, match="profile"):
        tails.how_rare_joint(x, 3., groups=groups, ci="profile")


def test_event_limits_can_be_negative_with_positive_scales():
    x = np.array([[-5., -2., 0., 3.], [-4., -1., 1., 4.]])
    score = tails.compound_series(x, [-3., -2.], scales=[1., 2.])
    np.testing.assert_array_equal(score > 1, [False, True, True, True])
    low = tails.compound_series(x, [-3., -2.], scales=[1., 2.], directions=[-1, -1])
    np.testing.assert_array_equal(low > 1, [True, False, False, False])


def test_grouped_bootstrap_pools_short_replicates_and_ignores_incomplete_labels():
    x = np.random.default_rng(32).exponential(size=(12, 2, 100))
    groups = np.broadcast_to(np.tile([0., 1.], 50), (12, 100)).copy()
    x[:, 0, 0] = np.nan
    groups[:, 0] = np.nan
    result = tails.how_rare_joint(x, 3., groups=groups, threshold_quantile=.9,
                                  ci="block_bootstrap", block_length=3, n_boot=2)
    assert result.bootstrap.diagnostics.n_attempted == 2
    assert result.record_length == 99
