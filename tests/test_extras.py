import numpy as np
import pytest
from scipy import stats

import tails


def test_threshold_tools(rng):
    x = stats.genpareto.rvs(0.1, scale=1, size=3000, random_state=rng)
    mrl = tails.mean_residual_life(x, [0.5, 1.0])
    assert mrl["mean_excess"].shape == (2,)
    ps = tails.parameter_stability(x, np.linspace(0.2, 2, 6))
    assert np.all(np.abs(ps["shape"] - 0.1) < 0.3)
    sel = tails.select_threshold(x, np.linspace(0.0, 2.0, 5), n_boot=49)
    assert sel.threshold in sel.thresholds


def test_declustering_and_extremal_index():
    x = np.array([0, 5, 6, 0, 0, 0, 7, 0, 8, 0], float)
    d = tails.decluster_runs(x, 4, run_length=1)
    assert np.nansum(d) == 6 + 7 + 8
    th = tails.extremal_index(np.random.default_rng(0).standard_normal(5000), 2.0)
    assert 0.6 < th <= 1.0


def test_block_maxima():
    import pandas as pd

    idx = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    bm = tails.block_maxima(s, "YS", min_count=300)
    assert len(bm) == 3 and bm.iloc[-1] == len(idx) - 1


def test_records():
    from tails.records import record_test

    r = record_test(np.arange(30.0))
    assert r["observed"] == 30 and r["pvalue"] < 1e-6
    r = record_test(np.random.default_rng(1).standard_normal((2000, 50)))
    assert abs(r["observed"].mean() - r["expected"][0]) < 0.2


def test_xarray_wrapper(rng):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")
    da = xr.DataArray(rng.gumbel(30, 8, size=(3, 4, 50)), dims=("lat", "lon", "year"))
    da[0, 0] = np.nan
    ds = tails.xarray.fit_gev(da.chunk({"lat": 1}), dim="year").compute()
    assert np.isnan(ds["loc"][0, 0])
    assert bool(ds["converged"][1, 1])
    rl = tails.xarray.return_level(ds, [10, 100])
    assert rl.sizes["period"] == 2


def _simple_scaling_intensities(rng, leading, hours, *, mu0=25.0, sigma0=6.0, xi=-0.1, alpha=0.6):
    """GEV draws with simple scaling. ``leading`` includes the year axis."""
    hours = np.asarray(hours, float)
    scaled = hours**-alpha
    out = np.empty(leading + (hours.size,), dtype=float)
    for j, (loc, scale) in enumerate(zip(mu0 * scaled, sigma0 * scaled)):
        out[..., j] = stats.genextreme.rvs(-xi, loc=loc, scale=scale, size=leading, random_state=rng)
    return out


def test_xarray_idf_grid_matches_fit_idf(rng):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")
    from tails.idf import IDFData, fit_idf

    tags = ["1h", "2h", "6h", "24h"]
    hours = np.array([1.0, 2.0, 6.0, 24.0])
    n_year = 40
    values = _simple_scaling_intensities(rng, (2, 2, n_year), hours)
    values[0, 0] = np.nan
    values[0, 1, :5, 0] = np.nan
    da = xr.DataArray(
        values,
        dims=("lat", "lon", "year", "duration"),
        coords={"lat": [45.0, 46.0], "lon": [-73.0, -72.0], "duration": tags, "year": np.arange(1961, 1961 + n_year)},
    )
    lazy = tails.xarray.fit_idf(da.chunk({"lat": 1}), marginal="general")
    assert lazy["mu0"].chunks is not None
    q_lazy = tails.xarray.idf_quantile(lazy, [1.0, 24.0], [20, 100])
    assert q_lazy.chunks is not None
    q = q_lazy.compute()
    ds = lazy.compute()
    assert ds["mu0"].dims == ("lat", "lon")
    assert "year" not in ds.dims and "duration" not in ds.dims
    assert ds.attrs["tails_model"] == "idf"
    assert ds.attrs["tails_marginal"] == "general"
    assert ds.attrs["tails_correlation"] == "independent"
    assert ds.attrs["tails_d0"] == 1.0
    assert np.isnan(ds["mu0"].isel(lat=0, lon=0))
    assert not bool(ds["converged"].isel(lat=0, lon=0))
    assert q.dims == ("lat", "lon", "duration", "period")
    assert np.isnan(q.isel(lat=0, lon=0)).all()

    for i, j in ((0, 1), (1, 0), (1, 1)):
        data = IDFData(np.arange(1961, 1961 + n_year), hours, values[i, j], tags, name=f"{i},{j}")
        ref = fit_idf(data, "general")
        got = np.array([ds[name].isel(lat=i, lon=j).item() for name in ref.names])
        np.testing.assert_allclose(got, ref.params[0], rtol=1e-5, atol=1e-6)
        assert ds["loglik"].isel(lat=i, lon=j).item() == pytest.approx(float(ref.loglik[0]), rel=1e-6, abs=1e-4)
        assert bool(ds["converged"].isel(lat=i, lon=j)) == bool(ref.converged[0])
        se = np.sqrt(np.clip(np.diag(ref.covariance("hessian")[0]), 0, None))
        got_se = np.array([ds[f"{name}_se"].isel(lat=i, lon=j).item() for name in ref.names])
        np.testing.assert_allclose(got_se, se, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(q.isel(lat=i, lon=j).values, ref.quantile([1.0, 24.0], [20, 100])[0], rtol=1e-5, atol=1e-5)

    cell = values[1, 1]
    one = xr.DataArray(cell, dims=("year", "duration"), coords={"duration": tags})
    depth = tails.xarray.fit_idf(
        xr.DataArray(cell * hours, dims=("year", "duration"), coords={"duration": tags}),
        depth=True,
    )
    shifted = tails.xarray.fit_idf(one, d0=2.0)
    ref_cell = fit_idf(IDFData(np.arange(n_year), hours, cell, tags), "general")
    ref_d0 = fit_idf(IDFData(np.arange(n_year), hours, cell, tags), "general", d0=2.0)
    np.testing.assert_allclose([depth[name].item() for name in ref_cell.names], ref_cell.params[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose([shifted[name].item() for name in ref_d0.names], ref_d0.params[0], rtol=1e-5, atol=1e-6)


def test_xarray_idf_exponential(rng):
    xr = pytest.importorskip("xarray")
    from tails.idf import IDFData, fit_idf

    tags = ["1h", "6h", "24h"]
    hours = np.array([1.0, 6.0, 24.0])
    n_year = 30
    values = _simple_scaling_intensities(rng, (2, n_year), hours)
    da = xr.DataArray(values, dims=("station", "year", "duration"), coords={"station": ["a", "b"], "duration": tags})
    ds = tails.xarray.fit_idf(da, correlation="exponential", copula="gaussian", df=5.0)
    assert ds.attrs["tails_correlation"] == "exponential"
    assert "rho" in ds and "delta" in ds
    stations = [IDFData(np.arange(n_year), hours, values[i], tags, name=n) for i, n in enumerate(("a", "b"))]
    ref = fit_idf(stations, "general", "exponential")
    np.testing.assert_allclose(np.stack([ds[name].values for name in ref.names], axis=-1), ref.params, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ds["loglik"].values, ref.loglik, rtol=1e-6, atol=1e-4)
    assert ds["converged"].values.tolist() == ref.converged.astype(bool).tolist()


def test_xarray_idf_input_checks():
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(np.ones((4, 2)), dims=("year", "duration"))
    with pytest.raises(ValueError, match="durations"):
        tails.xarray.fit_idf(da)
    tagged = da.assign_coords(duration=["1h", "24h"])
    np.testing.assert_allclose(tails.xarray._duration_hours(tagged, "duration", None), [1.0, 24.0])
    td = da.assign_coords(duration=np.array([3600, 86400], dtype="timedelta64[s]"))
    np.testing.assert_allclose(tails.xarray._duration_hours(td, "duration", None), [1.0, 24.0])
    np.testing.assert_allclose(tails.xarray._duration_hours(da, "duration", [0.5, 3.0]), [0.5, 3.0])
    with pytest.raises(ValueError, match="positive"):
        tails.xarray.fit_idf(da.assign_coords(duration=[0.0, 1.0]))
    with pytest.raises(ValueError, match="marginal"):
        tails.xarray.fit_idf(tagged, marginal="nope")
    with pytest.raises(TypeError, match="Dataset"):
        tails.xarray.fit_idf(xr.Dataset({"rain": tagged}))
    with pytest.raises(ValueError, match="fit_idf"):
        tails.xarray.idf_quantile(xr.Dataset({"loc": 1.0}, attrs={"tails_model": "gev"}), 1.0, 100)
    pytest.importorskip("dask")
    with pytest.raises(ValueError, match="year"):
        tails.xarray.fit_idf(tagged.chunk({"year": 2}))


def test_idf_quantile_simple_scaling():
    xr = pytest.importorskip("xarray")
    from tails.core.distributions import gev_ppf

    ds = xr.Dataset(
        {"mu0": 20.0, "sigma0": 5.0, "xi": 0.1, "alpha": 0.5},
        attrs={"tails_model": "idf", "tails_marginal": "simple", "tails_d0": 1.0},
    )
    q = tails.xarray.idf_quantile(ds, [1.0, 16.0], [10.0, 100.0])
    assert q.name == "intensity"
    assert q.sizes["duration"] == 2 and q.sizes["period"] == 2
    for i, d in enumerate((1.0, 16.0)):
        s = d**-0.5
        for j, T in enumerate((10.0, 100.0)):
            expected = float(gev_ppf(1.0 - 1.0 / T, 20.0 * s, 5.0 * s, 0.1))
            assert q.values[i, j] == pytest.approx(expected, rel=1e-10, abs=1e-10)
    with pytest.raises(ValueError, match="return period"):
        tails.xarray.idf_quantile(ds, 1.0, 1.0)


def test_bayes_optional(rng):
    pytest.importorskip("numpyro")
    from tails.bayes import fit_gev_bayes

    y = rng.gumbel(30, 8, size=50)
    post = fit_gev_bayes(y, num_warmup=200, num_samples=200)
    rl = post.return_level([100])
    assert rl["lower"][0] < rl["median"][0] < rl["upper"][0]
