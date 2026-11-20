import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")


def test_grid_hourly_grouped_blocks_and_diagnostics():
    import tails.xarray as tx

    da = xr.DataArray(np.random.default_rng(42).exponential(size=(2, 1200)), dims=("site", "time"),
                      coords={"site": ["a", "b"], "time": pd.date_range("2020", periods=1200, freq="h")})
    groups = xr.DataArray(np.tile(["off", "on"], 600), dims="time", coords={"time": da.time})
    ds = tx.how_rare_joint(da, 3., groups=groups, ci="block_bootstrap", block_length=4,
                           n_boot=2, seed=3, npy=None)
    assert ds.bootstrap_attempted.item() == 2
    assert ds.return_period.attrs["units"] == "observations"
    assert ds.record_length.item() == 1200
    with pytest.raises(ValueError, match="align|index"):
        tx.how_rare_joint(da, 3., groups=groups.isel(time=slice(1, None)), ci=None)


def test_grid_rejects_split_joint_dimension_and_keeps_batch_coordinates():
    import tails.xarray as tx

    da = xr.DataArray(np.random.default_rng(2).exponential(size=(2, 2, 1200)),
                      dims=("scenario", "site", "time"), coords={"scenario": ["a", "b"]})
    with pytest.raises(ValueError, match="chunk"):
        tx.how_rare_joint(da.chunk({"site": 1}), 3., spatial_dims=["site"], ci=None)
    ds = tx.how_rare_joint(da, 3., spatial_dims=["site"], ci=None)
    np.testing.assert_array_equal(ds.scenario, ["a", "b"])


def test_group_metadata_must_match_and_all_leap_rate_is_366():
    import tails.xarray as tx

    dates = xr.date_range("2000", periods=1200, calendar="all_leap", use_cftime=True)
    da = xr.DataArray(np.random.default_rng(20).exponential(size=(2, 1200)), dims=("site", "time"),
                      coords={"time": dates, "regime_source": ("time", np.zeros(1200))})
    groups = xr.DataArray(np.zeros(1200), dims="time",
                          coords={"time": dates, "regime_source": ("time", np.ones(1200))})
    with pytest.raises(ValueError, match="coordinate"):
        tx.how_rare_joint(da, 3., groups=groups, ci=None)
    result = tx.how_rare_joint(da, 3., ci=None)
    assert result.return_period.item() == pytest.approx(1/(366*result.probability.item()))


def test_quantile_thresholds_accept_their_extra_scalar_coordinate():
    import tails.xarray as tx

    da = xr.DataArray(np.random.default_rng(5).exponential(size=(2, 1200)), dims=("site", "time"),
                      coords={"site": ["a", "b"]})
    result = tx.how_rare_joint(da, da.quantile(.99, dim="time"), ci=None)
    assert result.expected_count.item() >= 0
