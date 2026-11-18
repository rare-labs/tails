import numpy as np
import pandas as pd
import pytest

from tails.records import foster_stuart, record_breaks, record_indicators, record_test


def test_increasing_series_is_all_upper_records():
    x = np.arange(30.0)
    upper = record_breaks(x, kind="upper")
    lower = record_breaks(x, kind="lower")
    assert upper.observed == 30
    assert upper.pvalue < 1e-6
    assert lower.observed == 1
    assert lower.events()["value"].tolist() == [0.0]
    assert upper.ratio == pytest.approx(30 / upper.expected)


def test_record_test_matches_record_breaks():
    x = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
    got = record_test(x)
    breaks = record_breaks(x)
    assert got["observed"] == breaks.observed == 3
    assert got["expected"] == pytest.approx(breaks.expected)
    assert got["pvalue"] == pytest.approx(breaks.pvalue)
    # H_5 = 1 + 1/2 + 1/3 + 1/4 + 1/5
    assert breaks.expected == pytest.approx(1 + 0.5 + 1 / 3 + 0.25 + 0.2)
    events = breaks.events()
    assert events["value"].tolist() == [3.0, 4.0, 5.0]
    assert events["trial"].tolist() == [1, 3, 5]
    assert events["records_so_far"].tolist() == [1, 2, 3]
    assert events["expected_so_far"].iloc[-1] == pytest.approx(breaks.expected)


def test_ties_and_missing_values_do_not_count():
    x = np.array([1.0, np.nan, 1.0, 0.0, 2.0])
    upper = record_indicators(x, "upper")
    lower = record_indicators(x, "lower")
    assert upper.tolist() == [True, False, False, False, True]
    assert lower.tolist() == [True, False, False, True, False]
    breaks = record_breaks(x)
    # The missing value does not advance the trial. The new maximum is trial 4.
    assert breaks.trial.tolist() == [1, 0, 2, 3, 4]
    assert breaks.observed == 2


def test_foster_stuart_counts_and_an_upward_drift():
    x = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
    stat = foster_stuart(x)
    # After the first point: upper records at 4 and 5, one lower record at 1.
    assert stat.d == 1
    assert stat.s == 3
    assert stat.s_expected == pytest.approx(1 + 2 / 3 + 0.5 + 0.4)
    assert stat.d_variance == pytest.approx(stat.s_expected)

    trend = foster_stuart(np.arange(30.0))
    assert trend.d == 29
    assert trend.s == 29
    assert trend.pvalue_up < 1e-6


def test_batch_matches_one_series_and_the_null_is_near_harmonic():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((2000, 50))
    batch = record_breaks(y)
    one = record_breaks(y[3])
    assert batch.observed[3] == one.observed
    assert batch.expected[3] == pytest.approx(one.expected)
    assert abs(batch.observed.mean() - batch.expected[0]) < 0.2
    stat = foster_stuart(y)
    assert abs(stat.d.mean()) < 0.15
    table = batch.summary()
    assert len(table) == 2000
    assert table.loc[3, "observed"] == batch.observed[3]


def test_quebec_24h_record_years_match_the_howto():
    from tails.datasets import load_eccc_quebec

    df = load_eccc_quebec()
    sub = df.loc[df.station_id == "701S001", ["year", "24h"]].dropna()
    series = pd.Series(
        sub["24h"].to_numpy(),
        index=pd.to_datetime(sub["year"].astype(str), format="%Y"),
    )
    breaks = record_breaks(series)
    assert breaks.observed == 7
    assert breaks.expected == pytest.approx(4.743890656)
    assert breaks.pvalue == pytest.approx(0.1598477, rel=1e-5)
    events = breaks.events()
    assert [t.year for t in events["time"]] == [1961, 1962, 1978, 1979, 1996, 1998, 2005]
    assert events["value"].tolist() == pytest.approx([61.7, 71.4, 78.9, 81.2, 83.0, 104.8, 105.4])
    stat = foster_stuart(series)
    assert stat.d == 2
    assert stat.s == 10
    assert stat.s_expected == pytest.approx(7.487781807)
    assert stat.pvalue_up == pytest.approx(0.2917879, rel=1e-5)


def test_series_index_is_kept_and_bad_inputs_fail():
    series = pd.Series([1.0, 3.0, 2.0, 4.0], index=pd.date_range("2020-01-01", periods=4))
    events = record_breaks(series).events()
    assert list(events["time"]) == list(series.index[[0, 1, 3]])
    assert events["value"].tolist() == [1.0, 3.0, 4.0]
    with pytest.raises(TypeError):
        record_breaks(pd.DataFrame({"a": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        record_breaks(series, kind="max")
    with pytest.raises(IndexError):
        record_breaks(series).events(series=1)
