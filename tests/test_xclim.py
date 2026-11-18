"""xclim-style frequency analysis: batched fits, ξ, and block extremes."""

import sys

import numpy as np
import pytest
import xarray as xr
from scipy import stats

import tails
from tails.core.distributions import gpd_ppf
from tails.xclim import (
    as_scipy_params,
    block_extremes,
    fa,
    fit,
    frequency_analysis,
    parametric_quantile,
)


def _gumbel(rng, *shape):
    return rng.gumbel(30, 8, size=shape)


def test_fit_matches_fit_gev_and_keeps_dimension_order(rng):
    y = _gumbel(rng, 3, 40, 2)
    y[1, :5, 0] = np.nan
    da = xr.DataArray(y, dims=("lat", "year", "lon"))
    ds = fit(da, dist="gev", method="MLE", dim="year")
    assert tuple(ds.params.dims) == ("lat", "dparams", "lon")
    assert ds.attrs["shape_convention"] == "xi"
    assert ds.attrs["tails_method"] == "mle"
    ref = tails.fit_gev(np.moveaxis(y, 1, -1))
    got = ds.params.transpose("lat", "lon", "dparams")
    np.testing.assert_allclose(got.sel(dparams="loc"), ref.params["loc"], rtol=1e-10)
    np.testing.assert_allclose(got.sel(dparams="scale"), ref.params["scale"], rtol=1e-10)
    np.testing.assert_allclose(got.sel(dparams="shape"), ref.params["shape"], rtol=1e-10)
    quantile = parametric_quantile(ds, [0.9, 0.99])
    assert tuple(quantile.dims) == ("lat", "quantile", "lon")
    np.testing.assert_allclose(ds.se.sel(dparams="loc").transpose("lat", "lon"), ref.se[..., 0], rtol=1e-8)
    np.testing.assert_allclose(
        ds.se.sel(dparams="scale").transpose("lat", "lon"),
        ref.params["scale"] * ref.se[..., 1],
        rtol=1e-8,
    )
    np.testing.assert_allclose(ds.nobs.transpose("lat", "lon"), ref.nobs)
    assert bool(ds.converged.all())


def test_nan_and_short_series_stay_nan(rng):
    y = _gumbel(rng, 2, 25)
    y[0] = np.nan
    da = xr.DataArray(y, dims=("station", "year"))
    ds = fit(da, dim="year")
    assert np.isnan(ds.params.sel(dparams="loc").values[0])
    assert not bool(ds.converged.values[0])
    solo = tails.fit_gev(y[1])
    np.testing.assert_allclose(ds.params.isel(station=1).sel(dparams="loc"), solo.params["loc"], rtol=1e-8)
    short = fit(xr.DataArray([1.0, 2.0, np.nan], dims="time"))
    assert np.isnan(float(short.params.sel(dparams="loc")))
    assert int(short.nobs) == 2
    assert not bool(short.converged)


def test_pwm_matches_lmom_and_gmle_runs(rng):
    y = _gumbel(rng, 4, 30)
    da = xr.DataArray(y, dims=("station", "year"))
    pwm = fit(da, method="PWM", dim="year")
    ref = tails.fit_gev(y, method="lmom")
    np.testing.assert_allclose(pwm.params.sel(dparams="shape"), ref.params["shape"], rtol=1e-10)
    assert np.isnan(pwm.se.sel(dparams="shape").values).all()
    gmle = fit(da, method="GMLE", dim="year")
    assert bool(gmle.converged.all())
    assert gmle.attrs["method"] == "GMLE"


def test_gpd_threshold_shape_matches_fit_gpd(rng):
    excess = stats.genpareto.rvs(0.2, scale=2, size=500, random_state=rng)
    excess = excess[excess > 0]
    raw = excess + 4.0
    da = xr.DataArray(raw, dims="time")
    ds = fit(da, dist="genpareto", threshold=4)
    ref = tails.fit_gpd(excess)
    assert float(ds.params.sel(dparams="loc")) == 4.0
    assert np.isnan(float(ds.se.sel(dparams="loc")))
    np.testing.assert_allclose(ds.params.sel(dparams="scale"), ref.params["scale"], rtol=1e-8)
    np.testing.assert_allclose(ds.params.sel(dparams="shape"), ref.params["shape"], rtol=1e-8)
    assert float(ds.params.sel(dparams="shape")) > 0.05
    assert int(ds.nobs) == excess.size
    level = fa(da, [20, 100], dist="gpd", threshold=4)
    q = 1.0 - 1.0 / np.array([20.0, 100.0])
    expected = np.asarray(gpd_ppf(q, ref.params["scale"], ref.params["shape"])) + 4.0
    np.testing.assert_allclose(np.asarray(level), expected, rtol=1e-6)
    plain = fit(xr.DataArray(excess, dims="time"), dist="genpareto")
    assert float(plain.params.sel(dparams="loc")) == 0.0


def test_as_scipy_params_flips_gev_shape_only():
    y = stats.genextreme.rvs(-0.2, loc=40, scale=8, size=200, random_state=0)
    ds = fit(xr.DataArray(y, dims="time"))
    shape = float(ds.params.sel(dparams="shape"))
    assert shape > 0.05
    sp = as_scipy_params(ds)
    assert tuple(str(v) for v in sp.dparams.values) == ("c", "loc", "scale")
    assert sp.attrs["scipy_dist"] == "genextreme"
    assert sp.attrs["shape_convention"] == "scipy_c"
    c = float(sp.sel(dparams="c"))
    assert c == pytest.approx(-shape)
    q = 0.9
    ours = float(parametric_quantile(ds, q).item())
    theirs = stats.genextreme.ppf(q, c, loc=float(sp.sel(dparams="loc")), scale=float(sp.sel(dparams="scale")))
    assert ours == pytest.approx(theirs, rel=1e-6, abs=1e-6)

    excess = stats.genpareto.rvs(0.2, scale=3, size=400, random_state=1)
    gpd = fit(xr.DataArray(excess, dims="time"), dist="genpareto")
    spg = as_scipy_params(gpd.params)
    assert spg.attrs["scipy_dist"] == "genpareto"
    assert float(spg.sel(dparams="c")) == pytest.approx(float(gpd.params.sel(dparams="shape")))
    q = 0.8
    ours = float(parametric_quantile(gpd, q).item())
    theirs = stats.genpareto.ppf(
        q,
        float(spg.sel(dparams="c")),
        loc=float(spg.sel(dparams="loc")),
        scale=float(spg.sel(dparams="scale")),
    )
    assert ours == pytest.approx(theirs, rel=1e-6, abs=1e-6)


def test_fa_quantiles_and_negated_minima(rng):
    y = _gumbel(rng, 50)
    da = xr.DataArray(y, dims="year", attrs={"units": "mm"})
    level = fa(da, [100, 20], dim="year")
    assert tuple(level.dims) == ("return_period",)
    np.testing.assert_array_equal(level.return_period.values, [100, 20])
    assert float(level.sel(return_period=100)) > float(level.sel(return_period=20))
    assert level.attrs["units"] == "mm"
    q = 1.0 - 1.0 / np.array([100.0, 20.0])
    via_fit = parametric_quantile(fit(da, dim="year"), q)
    np.testing.assert_allclose(via_fit.values, level.values, rtol=1e-10)
    hi = fa(da, [20, 100], mode="max", dim="year")
    lo = fa(-da, [20, 100], mode="min", dim="year")
    np.testing.assert_allclose(lo.values, -hi.values, rtol=1e-10)
    assert lo.attrs["mode"] == "min"


def test_gev_intervals_bracket_the_estimate(rng):
    y = stats.genextreme.rvs(-0.1, loc=30, scale=8, size=60, random_state=rng)
    da = xr.DataArray(y, dims="year", attrs={"units": "mm"})
    for method in ("delta", "profile", "bootstrap"):
        out = fa(da, [10, 100], ci=method, n_boot=100, dim="year")
        assert np.all(out.lower < out.return_level), method
        assert np.all(out.return_level < out.upper), method
        assert out.attrs["ci"] == method
        assert out.return_level.attrs["units"] == "mm"


def test_frequency_analysis_blocks_and_annual_path(rng):
    import pandas as pd

    time = pd.date_range("2000-12-01", "2002-02-28", freq="D")
    vals = np.zeros(len(time))
    idx = pd.DatetimeIndex(time)
    vals[idx == "2000-12-15"] = 10
    vals[idx == "2001-01-15"] = 30
    vals[idx == "2001-02-15"] = 20
    vals[idx == "2001-07-01"] = 100
    vals[idx == "2001-12-20"] = 5
    vals[idx == "2002-01-10"] = 4
    da = xr.DataArray(vals, coords={"time": time}, dims="time")
    winter = block_extremes(da, "max", season="DJF")
    np.testing.assert_allclose(winter.values, [30, 5])
    january = block_extremes(da, "max", month=1)
    np.testing.assert_allclose(january.values, [30, 4])
    summer = block_extremes(da, "max", season="JJA")
    months = block_extremes(da, "max", month=[6, 7, 8])
    np.testing.assert_allclose(summer.values, months.values)
    assert float(summer.max()) == 100

    annual = xr.DataArray(
        _gumbel(rng, 36),
        coords={"time": pd.date_range("1980-01-01", periods=36, freq="YS")},
        dims="time",
        attrs={"units": "mm"},
    )
    direct = fa(annual, [20, 100])
    wrapped = frequency_analysis(annual, "max", [20, 100])
    np.testing.assert_allclose(wrapped.values, direct.values, rtol=1e-10)
    assert wrapped.attrs["units"] == "mm"
    low = frequency_analysis(-annual, "min", [20, 100])
    np.testing.assert_allclose(low.values, -direct.values, rtol=1e-10)


def test_rolling_mean_before_the_block():
    import pandas as pd

    time = pd.date_range("2001-01-01", periods=10, freq="D")
    vals = np.zeros(10)
    vals[5] = 9.0
    da = xr.DataArray(vals, coords={"time": time}, dims="time")
    blocks = block_extremes(da, "max", window=3)
    assert float(blocks.max()) == pytest.approx(3.0)


def test_dask_fit_matches_numpy(rng):
    pytest.importorskip("dask.array")
    y = _gumbel(rng, 3, 4, 30)
    da = xr.DataArray(y, dims=("lat", "lon", "year"))
    eager = fit(da, dim="year")
    lazy = fit(da.chunk({"lat": 1, "year": 10}), dim="year").compute()
    np.testing.assert_allclose(lazy.params, eager.params, rtol=1e-8)
    assert bool(lazy.converged.all())


def test_rejects_unsupported_options(rng):
    da = xr.DataArray(_gumbel(rng, 20), dims="year")
    with pytest.raises(ValueError, match="PWM"):
        fit(da, method="MM", dim="year")
    with pytest.raises(ValueError, match="Gumbel"):
        fit(da, dist="gumbel_r", dim="year")
    with pytest.raises(ValueError, match="genextreme"):
        fit(da, dist="gamma", dim="year")
    with pytest.raises(ValueError, match="min"):
        fa(da, 20, dist="genpareto", mode="min", dim="year")
    with pytest.raises(ValueError, match="GEV"):
        fa(da, 20, dist="genpareto", ci="delta", dim="year")
    with pytest.raises(ValueError, match="datetime"):
        frequency_analysis(da, "max", 20, dim="year")
    dated = xr.DataArray(da.values, coords={"time": np.arange(20)}, dims="time")
    with pytest.raises(ValueError, match="datetime"):
        block_extremes(dated, "max")
    import pandas as pd

    daily = xr.DataArray(
        np.zeros(40),
        coords={"time": pd.date_range("2001-01-01", periods=40, freq="D")},
        dims="time",
    )
    with pytest.raises(ValueError, match="season"):
        block_extremes(daily, "max", season="DJFM")
    with pytest.raises(ValueError, match="month"):
        frequency_analysis(daily, "max", 10, doy=15)


def test_backend_does_not_import_xclim():
    import tails.xclim as backend

    assert backend.__name__ == "tails.xclim"
    assert not [name for name in sys.modules if name == "xclim" or name.startswith("xclim.")]
