import numpy as np
import pytest

import tails

xr = pytest.importorskip("xarray")


def test_compound_arbitrary_grid_and_region():
    import tails.xarray as tx

    rng = np.random.default_rng(30)
    values = rng.exponential(size=(2, 3, 2500))
    da = xr.DataArray(values, dims=("lat", "lon", "time"), coords={"lat": [46, 47], "lon": [-72, -71, -70]})
    region = xr.DataArray([[True, True, False], [False, False, False]], dims=("lat", "lon"), coords=da.coords)
    ds = tx.how_rare_joint(da, 3, spatial_dims=["lat", "lon"], regions={"west": region}, ci=None)
    direct = tails.how_rare_joint(values[0, :2], 3, ci=None)
    assert ds.probability.sel(region="west").item() == pytest.approx(direct.probability)
    assert ds.expected_count.sel(region="west").item() == pytest.approx(direct.expected_count)


def test_compound_dask_and_replicates():
    import tails.xarray as tx

    da = xr.DataArray(np.random.default_rng(2).exponential(size=(3, 2, 1800)), dims=("member", "site", "time"))
    eager = tx.how_rare_joint(da, 3, spatial_dims=["site"], replicate_dim="member", ci=None)
    lazy = tx.how_rare_joint(da.chunk({"member": -1, "site": -1, "time": -1}), 3, spatial_dims=["site"], replicate_dim="member", ci=None)
    xr.testing.assert_allclose(eager, lazy.compute())


def test_compound_rejects_misaligned_thresholds():
    import tails.xarray as tx

    da = xr.DataArray(np.ones((2, 30)), dims=("site", "time"), coords={"site": ["a", "b"]})
    threshold = xr.DataArray([1, 2], dims="site", coords={"site": ["a", "c"]})
    with pytest.raises(ValueError):
        tx.how_rare_joint(da, threshold, ci=None)


def test_compound_cftime_gap_breaks_run():
    pytest.importorskip("cftime")
    import tails.xarray as tx

    rng = np.random.default_rng(3)
    dates = xr.date_range("2000-01-01", periods=1800, calendar="noleap", use_cftime=True)
    da = xr.DataArray(rng.exponential(size=(2, 1800)), dims=("site", "time"), coords={"time": dates})
    gap = da.drop_isel(time=600)
    expected = tails.how_rare_joint(da.where(da.time != dates[600]).values, 2, run_length=2, npy=365, ci=None)
    actual = tx.how_rare_joint(gap, 2, run_length=2, ci=None)
    assert actual.probability.item() == pytest.approx(expected.probability)
    assert actual.return_period.item() == pytest.approx(expected.return_period)


def test_compound_dask_calendar_reindex_and_auxiliary_coordinates():
    import pandas as pd

    import tails.xarray as tx

    da = xr.DataArray(np.random.default_rng(9).exponential(size=(2, 1000)), dims=("site", "time"),
                      coords={"time": pd.date_range("2000-01-01", periods=1000), "lat": ("site", [46., 47.])})
    gap = da.drop_isel(time=[10, 100])
    eager = tx.how_rare_joint(gap, 3, ci=None)
    lazy = tx.how_rare_joint(gap.chunk({"site": -1, "time": -1}), 3, ci=None)
    xr.testing.assert_allclose(eager, lazy.compute())
    thresholds = xr.DataArray([3., 3.], dims="site", coords={"lat": ("site", [48., 49.])})
    with pytest.raises(ValueError, match="coordinate"):
        tx.how_rare_joint(gap, thresholds, ci=None)
