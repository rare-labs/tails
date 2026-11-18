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


def test_bayes_optional(rng):
    pytest.importorskip("numpyro")
    from tails.bayes import fit_gev_bayes

    y = rng.gumbel(30, 8, size=50)
    post = fit_gev_bayes(y, num_warmup=200, num_samples=200)
    rl = post.return_level([100])
    assert rl["lower"][0] < rl["median"][0] < rl["upper"][0]
