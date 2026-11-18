"""tails as a backend of xclim frequency analysis."""

import inspect
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

import tails
from tails.xclim import _levels_block, _parameters_block


def _gumbel(rng, *shape):
    return rng.gumbel(30.0, 8.0, size=shape)


def _annual(values, **attrs):
    import pandas as pd

    values = np.asarray(values, dtype=np.float64)
    time = pd.date_range("1980", periods=values.shape[-1], freq="YS")
    dims = ("time",) if values.ndim == 1 else ("lat", "time")
    attrs.setdefault("units", "mm")
    return xr.DataArray(values, dims=dims, coords={"time": time}, attrs=attrs)


def _stats():
    pytest.importorskip("xclim")
    import tails.xclim as tx

    assert tx.register()
    try:
        import xclim.indices.stats as stats
    except ImportError:
        import xclim.compute.stats as stats
    return stats


def test_gev_parameter_vector_uses_scipy_c(rng):
    y = _gumbel(rng, 60)
    got = _parameters_block(y, "gev", "mle", None)
    ref = tails.fit_gev(y)
    np.testing.assert_allclose(got[0], -float(ref.params["shape"]))
    np.testing.assert_allclose(got[1], float(ref.params["loc"]))
    np.testing.assert_allclose(got[2], float(ref.params["scale"]))


def test_gpd_parameter_vector_uses_scipy_c(rng):
    raw = np.abs(rng.normal(size=200)) + 0.1 + 5.0
    got = _parameters_block(raw, "gpd", "mle", 5.0)
    ref = tails.fit_gpd(raw, threshold=5.0)
    np.testing.assert_allclose(got[0], float(ref.params["shape"]))
    np.testing.assert_allclose(got[1], 5.0)
    np.testing.assert_allclose(got[2], float(ref.params["scale"]))


def test_short_series_are_nan(rng):
    short = np.array([1.0, np.nan, 2.0])
    assert np.isnan(_parameters_block(short, "gev", "mle", None)).all()
    batch = _gumbel(rng, 2, 40)
    batch[1, 1:] = np.nan
    got = _parameters_block(batch, "gev", "mle", None)
    assert np.isfinite(got[0]).all()
    assert np.isnan(got[1]).all()


def test_min_level_is_the_reflected_maximum(rng):
    y = _gumbel(rng, 80)
    periods = np.array([10.0, 50.0])
    hi = _levels_block(y, "gev", "mle", periods, "max", "delta", 0.05, 10, 0, None, None)
    lo = _levels_block(-y, "gev", "mle", periods, "min", "delta", 0.05, 10, 0, None, None)
    np.testing.assert_allclose(lo[..., 0], -hi[..., 0])
    np.testing.assert_allclose(lo[..., 1], -hi[..., 2])
    np.testing.assert_allclose(lo[..., 2], -hi[..., 1])


def test_import_tails_does_not_register_the_backend():
    script = (
        "import sys, tails; "
        "assert 'tails.xclim' not in sys.modules; "
        "assert not [n for n in sys.modules if n == 'xclim' or n.startswith('xclim.')]"
    )
    subprocess.check_call([sys.executable, "-c", script])


def test_user_import_selects_the_backend():
    pytest.importorskip("xclim")
    script = """
import tails.xclim
try:
    from xclim.compute.stats import fa, fit, frequency_analysis
except ImportError:
    from xclim.indices.stats import fa, fit, frequency_analysis
assert getattr(frequency_analysis, "_tails_backend", False)
assert getattr(fit, "_tails_backend", False)
assert getattr(fa, "_tails_backend", False)
try:
    import xclim.indices.stats as legacy
except ModuleNotFoundError:
    legacy = None
if legacy is not None:
    assert getattr(legacy.frequency_analysis, "_tails_backend", False)
"""
    subprocess.check_call([sys.executable, "-c", script])


def test_fit_matches_fit_gev_and_keeps_dimension_order(rng):
    xs = _stats()
    y = _gumbel(rng, 2, 40, 3)
    y[0, :4, 1] = np.nan
    da = xr.DataArray(y, dims=("lat", "time", "lon"))
    params = xs.fit(da, dist="genextreme", method="ML", backend="tails")
    assert params.dims == ("lat", "dparams", "lon")
    assert list(params.coords["dparams"].values) == ["c", "loc", "scale"]
    assert params.attrs["scipy_dist"] == "genextreme"
    assert params.attrs["shape_convention"] == "c=-xi"
    assert params.attrs["tails_backend"] == "tails"
    ref = tails.fit_gev(np.moveaxis(y, 1, -1))
    got = params.transpose("lat", "lon", "dparams")
    np.testing.assert_allclose(got.sel(dparams="c"), -ref.params["shape"])
    np.testing.assert_allclose(got.sel(dparams="loc"), ref.params["loc"])
    np.testing.assert_allclose(got.sel(dparams="scale"), ref.params["scale"])


def test_parametric_quantile_matches_fa_and_scipy(rng):
    xs = _stats()
    from scipy.stats import genextreme

    da = _annual(_gumbel(rng, 70), units="mm")
    params = xs.fit(da, "genextreme", backend="tails")
    level = xs.fa(da, 100, dist="genextreme", backend="tails")
    quantile = xs.parametric_quantile(params, 1.0 - 1.0 / 100.0)
    np.testing.assert_allclose(np.asarray(level).ravel(), np.asarray(quantile).ravel(), rtol=1e-6, atol=1e-6)
    c, loc, scale = (float(params.sel(dparams=name)) for name in ("c", "loc", "scale"))
    scipy_level = genextreme.isf(1.0 / 100.0, c, loc=loc, scale=scale)
    np.testing.assert_allclose(float(np.asarray(level).ravel()[0]), scipy_level, rtol=1e-6, atol=1e-6)
    assert level.attrs["units"] == "mm"
    assert quantile.attrs.get("units") == "mm"
    ref = tails.fit_gev(np.asarray(da)).return_level(100)
    np.testing.assert_allclose(float(np.asarray(level).ravel()[0]), float(np.asarray(ref.estimate).ravel()[0]))


def test_profile_and_delta_intervals_bracket_the_estimate(rng):
    xs = _stats()
    from scipy.stats import genextreme

    y = genextreme.rvs(-0.1, loc=40.0, scale=8.0, size=80, random_state=12345)
    da = _annual(y)
    for method in ("delta", "profile"):
        out = xs.fa(da, [10, 100], dist="genextreme", backend="tails", ci=method)
        assert set(out.data_vars) == {"return_level", "lower", "upper"}
        assert np.all(out.lower < out.return_level)
        assert np.all(out.return_level < out.upper)
    out = xs.frequency_analysis(da, "max", [10, 100], "genextreme", backend="tails", ci="delta")
    assert np.all(out.lower < out.return_level)
    assert np.all(out.return_level < out.upper)


def test_bootstrap_interval_is_ordered(rng):
    xs = _stats()
    da = _annual(_gumbel(rng, 60))
    out = xs.fa(da, [20], dist="genextreme", backend="tails", ci="bootstrap", n_boot=40, seed=1)
    assert np.all(np.isfinite(out.lower))
    assert np.all(out.lower < out.upper)


def test_default_backend_calls_xclim(rng):
    xs = _stats()
    da = _annual(_gumbel(rng, 40))
    wrapped = xs.fit(da, "genextreme")
    original = xs.fit.__wrapped__(da, "genextreme")
    np.testing.assert_allclose(wrapped.values, original.values)
    again = xs.fit(da, "genextreme", backend="xclim")
    np.testing.assert_allclose(again.values, original.values)
    level = xs.frequency_analysis(da, "max", 20, "genextreme")
    raw = xs.frequency_analysis.__wrapped__(da, "max", 20, "genextreme")
    np.testing.assert_allclose(level.values, raw.values)


def test_frequency_analysis_uses_xclim_blocks(rng):
    xs = _stats()
    import pandas as pd

    try:
        from xclim.compute.generic import statistics
    except ImportError:
        from xclim.indices.generic import statistics

    time = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    da = xr.DataArray(rng.random(time.size), coords={"time": time}, dims="time", name="pr", attrs={"units": "mm"})
    blocks = statistics(da, "max", "YS", month=1)
    direct = xs.fa(blocks, [10, 20], dist="genextreme", backend="tails")
    wrapped = xs.frequency_analysis(da, "max", [10, 20], "genextreme", month=1, backend="tails")
    np.testing.assert_allclose(wrapped, direct, equal_nan=True)

    winter = statistics(da, "max", "YS-DEC", season="DJF")
    direct = xs.fa(winter, [10], dist="genextreme", backend="tails")
    wrapped = xs.frequency_analysis(da, "max", [10], "genextreme", season="DJF", backend="tails")
    np.testing.assert_allclose(wrapped, direct, equal_nan=True)

    rolled = da.rolling(time=3).mean(skipna=False)
    blocks = statistics(rolled, "max", "YS")
    direct = xs.fa(blocks, [10], dist="genextreme", backend="tails")
    wrapped = xs.frequency_analysis(da, "max", [10], "genextreme", window=3, backend="tails")
    np.testing.assert_allclose(wrapped, direct, equal_nan=True)


def test_gpd_threshold_round_trip(rng):
    xs = _stats()
    from scipy.stats import genpareto

    raw = np.abs(rng.normal(size=250)) + 0.05 + 4.0
    da = _annual(raw)
    params = xs.fit(da, "genpareto", method="PWM", backend="tails", threshold=4.0)
    ref = tails.fit_gpd(raw, threshold=4.0, method="lmom")
    np.testing.assert_allclose(float(params.sel(dparams="c")), float(ref.params["shape"]))
    np.testing.assert_allclose(float(params.sel(dparams="loc")), 4.0)
    np.testing.assert_allclose(float(params.sel(dparams="scale")), float(ref.params["scale"]))
    level = xs.fa(da, 20, dist="genpareto", backend="tails", threshold=4.0, method="PWM")
    c, loc, scale = (float(params.sel(dparams=name)) for name in ("c", "loc", "scale"))
    scipy_level = genpareto.isf(1.0 / 20.0, c, loc=loc, scale=scale)
    np.testing.assert_allclose(float(np.asarray(level).ravel()[0]), scipy_level, rtol=1e-6, atol=1e-6)


def test_pwm_and_gmle(rng):
    xs = _stats()
    y = _gumbel(rng, 50)
    da = _annual(y)
    pwm = xs.fit(da, "genextreme", method="PWM", backend="tails")
    ref = tails.fit_gev(y, method="lmom")
    np.testing.assert_allclose(float(pwm.sel(dparams="c")), -float(ref.params["shape"]))
    gmle = xs.fit(da, "genextreme", method="GMLE", backend="tails")
    ref = tails.fit_gev(y, method="gmle")
    np.testing.assert_allclose(float(gmle.sel(dparams="c")), -float(ref.params["shape"]))


def test_dask_matches_eager(rng):
    pytest.importorskip("dask")
    xs = _stats()
    y = _gumbel(rng, 2, 36)
    da = _annual(y)
    eager = xs.fit(da, "genextreme", backend="tails")
    lazy = xs.fit(da.chunk({"lat": 1, "time": 10}), "genextreme", backend="tails")
    np.testing.assert_allclose(lazy.values, eager.values)


def test_backend_rejects_unknown_requests(rng):
    xs = _stats()
    da = _annual(_gumbel(rng, 30))
    with pytest.raises(ValueError, match="backend"):
        xs.fit(da, "genextreme", backend="lmoments")
    with pytest.raises(ValueError, match="tails"):
        xs.fa(da, 20, "genextreme", ci="profile")
    with pytest.raises(ValueError, match="Gumbel"):
        xs.fit(da, "gumbel_r", backend="tails")
    with pytest.raises(ValueError, match="ML"):
        xs.fit(da, "genextreme", method="MM", backend="tails")
    with pytest.raises(ValueError, match="GEV"):
        xs.fa(da, 20, "genpareto", backend="tails", ci="delta")
    with pytest.raises(ValueError, match="floc"):
        xs.fit(da, "genextreme", backend="tails", floc=0)


def test_register_is_idempotent_and_unregister_restores():
    xs = _stats()
    import tails.xclim as tx

    first = xs.frequency_analysis
    assert "backend" in inspect.signature(first).parameters
    assert tx.register() is True
    assert xs.frequency_analysis is first
    tx.unregister()
    try:
        assert xs.frequency_analysis is first.__wrapped__
        assert not getattr(xs.fit, "_tails_backend", False)
        assert not getattr(xs.fa, "_tails_backend", False)
    finally:
        tx.register()
