"""Datetime series → block maxima or peaks → return-level table."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import tails


def _daily_with_known_maxima():
    index = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    series = pd.Series(np.linspace(0.0, 1.0, len(index)), index=index, name="gauge")
    expected = {}
    for year in range(2000, 2010):
        stamp = pd.Timestamp(year=year, month=6, day=15)
        expected[year] = float(year - 1990)
        series.loc[stamp] = expected[year]
    return series, expected


def test_block_maxima_match_manual_extraction_and_fit():
    series, expected = _daily_with_known_maxima()
    ext = tails.fit_block_maxima(series, freq="YS", model="gev")
    assert ext.kind == "blocks" and ext.model == "gev"
    assert ext.extremes.name == "gauge"
    got = {ts.year: float(v) for ts, v in ext.extremes.dropna().items()}
    assert got == expected
    manual = np.asarray(tails.block_maxima(series, freq="YS"), dtype=float)
    direct = tails.fit_gev(manual)
    np.testing.assert_allclose(ext.fit.coef, direct.coef, rtol=1e-6, atol=1e-6)
    table = ext.summary(periods=[5, 10], ci="delta")
    assert list(table.columns) == ["period", "estimate", "lower", "upper"]
    assert table["period"].tolist() == [5, 10]
    assert table["estimate"].is_monotonic_increasing
    assert np.all(table["lower"] < table["estimate"]) and np.all(table["estimate"] < table["upper"])
    assert table.attrs["model"] == "gev"
    assert table.attrs["shape"] == pytest.approx(ext.shape)


def test_min_count_drops_short_block():
    index = pd.date_range("2016-01-01", "2018-12-31", freq="D")
    series = pd.Series(1.0, index=index)
    series.loc["2018-01-11":] = np.nan
    series.loc["2016-07-01"] = 8.0
    series.loc["2017-07-01"] = 6.0
    series.loc["2018-01-05"] = 9.0
    ext = tails.fit_block_maxima(series, min_count=300, model="gev")
    kept = ext.extremes.dropna()
    assert kept.index.year.tolist() == [2016, 2017]
    assert float(kept.loc["2016-01-01"]) == 8.0


def test_aic_selects_the_lower_score_and_frechet_prefers_gev():
    rng = np.random.default_rng(0)
    index = pd.date_range("1900-01-01", periods=80, freq="YS")
    gumbel = pd.Series(rng.gumbel(5.0, 1.5, size=80), index=index)
    mild = tails.fit_block_maxima(gumbel, model="aic")
    assert set(mild.aic_scores) == {"gev", "gumbel"}
    winner = min(mild.aic_scores, key=mild.aic_scores.get)
    # Tie goes to the Gumbel model, matching the selector.
    if mild.aic_scores["gumbel"] <= mild.aic_scores["gev"]:
        winner = "gumbel"
    assert mild.model == winner
    assert mild.aic == pytest.approx(mild.aic_scores[winner])

    frechet_y = stats.genextreme.rvs(-0.35, loc=10, scale=2, size=180, random_state=rng)
    frechet = pd.Series(frechet_y, index=pd.date_range("1800-01-01", periods=180, freq="YS"))
    heavy = tails.fit_block_maxima(frechet, model="aic")
    assert heavy.model == "gev"
    assert heavy.aic_scores["gev"] < heavy.aic_scores["gumbel"]
    assert heavy.fit.n_params == 3

    forced = tails.fit_block_maxima(frechet, model="gumbel")
    assert forced.model == "gumbel"
    assert forced.shape == pytest.approx(0.0)
    assert forced.fit.n_params == 2
    assert forced.fit.free_params == 2
    assert forced.parameters().set_index("term").loc["shape", "std_error"] == pytest.approx(0.0)


def test_gumbel_profile_interval_brackets_estimate():
    rng = np.random.default_rng(1)
    index = pd.date_range("1950-01-01", periods=60, freq="YS")
    series = pd.Series(rng.gumbel(20.0, 4.0, size=60), index=index)
    ext = tails.fit_block_maxima(series, model="gumbel")
    table = ext.summary(periods=[10, 50], ci="profile", alpha=0.05)
    assert np.all(np.isfinite(table[["estimate", "lower", "upper"]]))
    assert np.all(table["lower"] < table["estimate"]) and np.all(table["estimate"] < table["upper"])
    assert table.attrs["ci"] == "profile"
    assert table.attrs["model"] == "gumbel"


def test_peaks_decluster_inference_and_exponential_level():
    rng = np.random.default_rng(2)
    index = pd.date_range("2010-01-01", "2014-12-31", freq="D")
    series = pd.Series(rng.normal(size=len(index)), index=index, name="x")
    # One three-day storm and a handful of isolated peaks. The neighbours of 30
    # are smaller, so runs declustering keeps 30 and drops 4 and 5.
    series.iloc[10] = 4.0
    series.iloc[11] = 30.0
    series.iloc[12] = 5.0
    series.iloc[[100, 200, 400, 800]] = [8.0, 9.0, 7.0, 6.5]
    ext = tails.fit_peaks(series, threshold=3.0, run_length=1, model="exponential")
    assert ext.model == "exponential"
    assert ext.shape == pytest.approx(0.0)
    assert ext.npy == pytest.approx(365.25, rel=0.02)
    kept = set(np.round(ext.extremes.to_numpy(), 6))
    assert 30.0 in kept
    assert 4.0 not in kept and 5.0 not in kept
    table = ext.summary(periods=[2, 10])
    assert table.attrs["ci"] == "delta"
    assert table["estimate"].is_monotonic_increasing


def test_peaks_summary_accepts_profile_and_bootstrap():
    rng = np.random.default_rng(4)
    index = pd.date_range("2000-01-01", periods=1800, freq="D")
    values = rng.normal(size=len(index))
    excess = stats.genpareto.rvs(0.2, scale=1.4, size=160, random_state=rng)
    values[:160] = 2.0 + excess
    series = pd.Series(values, index=index)
    ext = tails.fit_peaks(series, threshold=2.0, model="gpd", npy=365.25)
    table = ext.summary(periods=[10, 40], ci="profile", profile_points=21)
    assert table.attrs["ci"] == "profile"
    assert np.all(table["lower"] < table["estimate"]) and np.all(table["estimate"] < table["upper"])
    boot = ext.summary(periods=[15], ci="bootstrap", n_boot=30, seed=1)
    assert boot.attrs["ci"] == "bootstrap"
    assert float(boot["lower"].iloc[0]) < float(boot["estimate"].iloc[0]) < float(boot["upper"].iloc[0])


def test_peaks_aic_on_heavy_tail_prefers_gpd():
    rng = np.random.default_rng(3)
    index = pd.date_range("2000-01-01", periods=2000, freq="D")
    # GPD excesses with ξ = 0.3 above a threshold of 1, and a bulk of zeros below it.
    excess = stats.genpareto.rvs(0.3, scale=1.5, size=400, random_state=rng)
    values = np.zeros(len(index))
    values[:400] = 1.0 + excess
    series = pd.Series(values, index=index)
    ext = tails.fit_peaks(series, threshold=1.0, model="aic", npy=365.25)
    assert ext.model == "gpd"
    assert ext.aic_scores["gpd"] < ext.aic_scores["exponential"]
    assert len(ext.extremes) == 400


def test_array_plus_times_and_offset_run_length():
    index = pd.date_range("2021-01-01", periods=48, freq="h")
    values = np.zeros(len(index))
    values[[5, 6, 20, 21, 40]] = [10.0, 3.0, 8.0, 4.0, 7.0]
    ext = tails.fit_peaks(values, threshold=2.0, times=index, run_length="2h", model="gpd")
    # "2h" on hourly data clusters neighbours, so 3 and 4 are dropped.
    kept = set(np.round(ext.extremes.to_numpy(), 6))
    assert kept == {10.0, 8.0, 7.0}
    assert ext.npy == pytest.approx(24 * 365.25, rel=0.02)


def test_rejects_bad_input():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        tails.fit_block_maxima(np.arange(10.0))
    frame = pd.DataFrame(
        {"a": np.arange(5.0), "b": np.arange(5.0)},
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    with pytest.raises(ValueError, match="one series"):
        tails.fit_block_maxima(frame)
    with pytest.raises(ValueError, match="model"):
        tails.fit_block_maxima(pd.Series(np.arange(5.0), index=pd.date_range("2020-01-01", periods=5, freq="YS")), model="gpd")


def test_quebec_annual_series_matches_published_fit():
    from tails.datasets import load_eccc_quebec

    df = load_eccc_quebec()
    sub = df.loc[df.station_id == "701S001", ["year", "24h"]].dropna().sort_values("year")
    series = pd.Series(
        sub["24h"].to_numpy(),
        index=pd.to_datetime(sub["year"].astype(str), format="%Y"),
        name="701S001",
    )
    ext = tails.fit_block_maxima(series, model="gev")
    direct = tails.fit_gev(sub["24h"].to_numpy())
    np.testing.assert_allclose(ext.fit.coef, direct.coef, rtol=1e-6, atol=1e-6)
    table = ext.summary(periods=[20, 100], ci="delta")
    direct_rl = direct.return_level([20, 100], ci="delta")
    np.testing.assert_allclose(table["estimate"], np.asarray(direct_rl.estimate).reshape(-1), rtol=1e-6)
    assert ext.fit.nobs == 64
    # Documented AIC comparison: ξ is near 0, so the Gumbel model wins.
    aic = tails.fit_block_maxima(series, model="aic")
    assert aic.model == "gumbel"
    assert aic.aic_scores["gumbel"] < aic.aic_scores["gev"]
    gumbel_100 = aic.summary(periods=[100], ci="profile")
    assert float(gumbel_100["estimate"].iloc[0]) == pytest.approx(104.669, rel=1e-4)
    assert float(gumbel_100["lower"].iloc[0]) < 104.669 < float(gumbel_100["upper"].iloc[0])
