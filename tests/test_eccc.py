import pandas as pd
import pytest

from tails.datasets import cache_dir, eccc_daily, eccc_hourly, eccc_stations
from tails.datasets.eccc import _years_span


def test_years_span():
    assert _years_span(2016) == (2016, 2016)
    assert _years_span((2010, 2015)) == (2010, 2015)
    assert _years_span(range(2010, 2016)) == (2010, 2015)


def test_cache_dir_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TAILS_CACHE", str(tmp_path / "mycache"))
    p = cache_dir()
    assert p == tmp_path / "mycache"
    assert p.is_dir()


def test_eccc_daily_reads_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("TAILS_CACHE", str(tmp_path))
    # A cache key for climate-daily / 7016294 / 2016-2016.
    from tails.datasets.eccc import _cache_key

    key = _cache_key("climate-daily", "7016294", 2016, 2016)
    idx = pd.to_datetime(["2016-01-01", "2016-01-02"])
    df = pd.DataFrame({"precip": [0.0, 1.2], "tmax": [-8.0, -5.0]}, index=idx)
    df.index.name = "time"
    df.to_csv(tmp_path / f"{key}.csv.gz", compression="gzip")
    (tmp_path / f"{key}.json").write_text('{"station_id": "7016294", "station_name": "CACHED"}')
    out = eccc_daily("7016294", 2016)
    assert list(out.index.normalize()) == list(idx.normalize())
    assert out.attrs["station_name"] == "CACHED"
    assert float(out["precip"].iloc[1]) == 1.2


def test_eccc_daily_live_station():
    """Hit api.weather.gc.ca for Québec / Jean Lesage (7016294), year 2016."""
    try:
        meta = eccc_stations("7016294")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"MSC GeoMet unavailable: {e}")
    assert len(meta) == 1
    assert meta.loc[0, "station_id"] == "7016294"
    assert "QUEBEC" in str(meta.loc[0, "station_name"]).upper()
    # Geometry should be in degrees around Jean Lesage, not packed integers.
    assert 46.5 < float(meta.loc[0, "lat"]) < 47.0
    assert -72.0 < float(meta.loc[0, "lon"]) < -71.0

    try:
        df = eccc_daily("7016294", 2016, cache=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"MSC GeoMet daily download failed: {e}")
    assert not df.empty
    assert df.index.min().year == 2016
    assert df.index.max().year == 2016
    assert "precip" in df.columns
    assert df.attrs["station_id"] == "7016294"
    assert "api.weather.gc.ca" in df.attrs["source"]
    # A year of daily data should be in the ballpark of a calendar year.
    assert 300 <= len(df) <= 366


def test_eccc_hourly_live_short_window(tmp_path, monkeypatch):
    monkeypatch.setenv("TAILS_CACHE", str(tmp_path))
    try:
        df = eccc_hourly("7016294", 2010, cache=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"MSC GeoMet hourly download failed: {e}")
    if df.empty:
        pytest.skip("no hourly rows returned for 7016294 in 2010")
    assert "temp" in df.columns or "precip" in df.columns
    assert df.index.min().year == 2010
    # Cache file written.
    assert any(tmp_path.glob("climate-hourly-7016294-2010-2010-*.csv.gz"))
