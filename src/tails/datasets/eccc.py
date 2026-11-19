"""ECCC / MSC GeoMet OGC API downloaders (daily and hourly climate observations).

Uses the official endpoint ``https://api.weather.gc.ca`` (no API key).
Responses are cached on disk under ``$TAILS_CACHE`` or ``$XDG_CACHE_HOME/tails``
or ``~/.cache/tails``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .. import __version__

__all__ = ["eccc_daily", "eccc_hourly", "eccc_stations", "cache_dir"]

BASE = "https://api.weather.gc.ca"
USER_AGENT = f"tails/{__version__} (+https://github.com/rare-labs/tails)"
_DEFAULT_TIMEOUT = 60
_MAX_RETRIES = 4

_DAILY_RENAME = {
    "LOCAL_DATE": "time",
    "CLIMATE_IDENTIFIER": "station_id",
    "STATION_NAME": "station_name",
    "PROVINCE_CODE": "province",
    "MAX_TEMPERATURE": "tmax",
    "MIN_TEMPERATURE": "tmin",
    "MEAN_TEMPERATURE": "tmean",
    "TOTAL_PRECIPITATION": "precip",
    "TOTAL_RAIN": "rain",
    "TOTAL_SNOW": "snow",
    "SNOW_ON_GROUND": "snow_on_ground",
    "SPEED_MAX_GUST": "gust",
    "DIRECTION_MAX_GUST": "gust_dir",
    "MAX_TEMPERATURE_FLAG": "tmax_flag",
    "MIN_TEMPERATURE_FLAG": "tmin_flag",
    "MEAN_TEMPERATURE_FLAG": "tmean_flag",
    "TOTAL_PRECIPITATION_FLAG": "precip_flag",
    "TOTAL_RAIN_FLAG": "rain_flag",
    "TOTAL_SNOW_FLAG": "snow_flag",
}

_HOURLY_RENAME = {
    "LOCAL_DATE": "time",
    "CLIMATE_IDENTIFIER": "station_id",
    "STATION_NAME": "station_name",
    "PROVINCE_CODE": "province",
    "TEMP": "temp",
    "DEW_POINT_TEMP": "dewpoint",
    "RELATIVE_HUMIDITY": "rh",
    "PRECIP_AMOUNT": "precip",
    "WIND_SPEED": "wind_speed",
    "WIND_DIRECTION": "wind_dir",
    "STATION_PRESSURE": "pressure",
    "VISIBILITY": "visibility",
    "HUMIDEX": "humidex",
    "WINDCHILL": "windchill",
    "WEATHER_ENG_DESC": "weather",
    "TEMP_FLAG": "temp_flag",
    "PRECIP_AMOUNT_FLAG": "precip_flag",
    "UTC_DATE": "time_utc",
}


def cache_dir() -> Path:
    """Directory used to cache GeoMet downloads."""
    if os.environ.get("TAILS_CACHE"):
        p = Path(os.environ["TAILS_CACHE"])
    elif os.environ.get("XDG_CACHE_HOME"):
        p = Path(os.environ["XDG_CACHE_HOME"]) / "tails"
    else:
        p = Path.home() / ".cache" / "tails"
    p.mkdir(parents=True, exist_ok=True)
    return p


def eccc_stations(station) -> pd.DataFrame:
    """Look up climate stations by climate ID, station name, or ``STN_ID``.

    Returns one row per match, with coordinates taken from the GeoJSON
    geometry (degrees), not the packed integer latitude/longitude fields.
    """
    features = _find_stations(station)
    return _stations_frame(features)


def eccc_daily(station, years, *, cache: bool = True, timeout: int = _DEFAULT_TIMEOUT) -> pd.DataFrame:
    """Daily climate observations from the MSC GeoMet OGC API.

    Parameters
    ----------
    station
        Climate identifier (e.g. ``"7016294"`` or ``"701S001"``), a
        ``STN_ID``, or a station name (matched against ``STATION_NAME``).
    years
        A single year, an inclusive ``(start, end)`` pair, or any iterable
        of years.
    cache
        If True (default), reuse a previous download of the same station
        and year range.

    Returns
    -------
    pandas.DataFrame
        Indexed by local date, with tidy columns such as ``tmax``, ``tmin``,
        ``tmean``, ``precip`` (mm), and the corresponding flags. Metadata
        (station id, name, lat, lon) is stored in ``attrs``.
    """
    return _observations("climate-daily", station, years, cache=cache, timeout=timeout)


def eccc_hourly(station, years, *, cache: bool = True, timeout: int = _DEFAULT_TIMEOUT) -> pd.DataFrame:
    """Hourly climate observations from the MSC GeoMet OGC API.

    Same station / years / cache arguments as :func:`eccc_daily`. The
    frame is indexed by local time and includes ``temp``, ``precip``,
    wind, pressure and weather columns when the station reports them.
    """
    return _observations("climate-hourly", station, years, cache=cache, timeout=timeout)


# ------------------------------------------------------------------ internals
def _years_span(years) -> tuple[int, int]:
    if isinstance(years, (int, float)):
        y = int(years)
        return y, y
    if isinstance(years, (tuple, list)) and len(years) == 2 and not isinstance(years[0], (list, tuple)):
        a, b = int(years[0]), int(years[1])
        # A 2-tuple is treated as an inclusive range (2010, 2015) → 2010–2015.
        if b >= a and b - a < 300:
            return a, b
    ys = sorted(int(y) for y in years)
    if not ys:
        raise ValueError("years must not be empty")
    return ys[0], ys[-1]


def _observations(collection: str, station, years, *, cache: bool, timeout: int) -> pd.DataFrame:
    start, end = _years_span(years)
    if end < start:
        raise ValueError("years end is before start")
    meta = _resolve_station(station)
    climate_id = meta["station_id"]
    start_d = f"{start}-01-01"
    end_d = f"{end}-12-31"
    key = _cache_key(collection, climate_id, start, end)
    path = cache_dir() / f"{key}.csv.gz"
    meta_path = cache_dir() / f"{key}.json"
    if cache and path.is_file():
        df = pd.read_csv(path, parse_dates=["time"], index_col="time")
        if meta_path.is_file():
            df.attrs.update(json.loads(meta_path.read_text()))
        return df

    params = {
        "CLIMATE_IDENTIFIER": climate_id,
        "datetime": f"{start_d}/{end_d}",
        "sortby": "LOCAL_DATE",
        "limit": "10000",
        "f": "json",
    }
    features = _paginate(f"{BASE}/collections/{collection}/items", params, timeout=timeout)
    rename = _DAILY_RENAME if collection == "climate-daily" else _HOURLY_RENAME
    rows = []
    for feat in features:
        props = feat.get("properties") or {}
        row = {}
        for src, dst in rename.items():
            if src in props:
                row[dst] = props[src]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=list(rename.values()))
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
        df = df.set_index("time")
    attrs = {
        "station_id": climate_id,
        "station_name": meta.get("station_name"),
        "lat": meta.get("lat"),
        "lon": meta.get("lon"),
        "province": meta.get("province"),
        "collection": collection,
        "source": f"{BASE}/collections/{collection}",
        "years": [start, end],
        "attribution": "Environment and Climate Change Canada (MSC GeoMet), https://api.weather.gc.ca",
    }
    df.attrs.update(attrs)
    if cache:
        df.to_csv(path, compression="gzip")
        meta_path.write_text(json.dumps(attrs, default=str))
    return df


def _cache_key(collection: str, climate_id: str, start: int, end: int) -> str:
    # "v2": downloads cached before the paging fix may be truncated.
    raw = f"v2:{collection}:{climate_id}:{start}:{end}"
    return collection + "-" + climate_id + "-" + f"{start}-{end}-" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def _resolve_station(station) -> dict:
    features = _find_stations(station)
    if not features:
        raise LookupError(
            f"no ECCC climate station matched {station!r}. "
            "Pass a climate identifier (e.g. '7016294') or a STATION_NAME."
        )
    if len(features) == 1:
        return _station_meta(features[0])
    # Multiple name hits: prefer an exact (case-insensitive) name match, then
    # raise with the candidate list so the caller can pass a climate id.
    needle = str(station).strip().upper()
    exact = [f for f in features if str((f.get("properties") or {}).get("STATION_NAME") or "").upper() == needle]
    if len(exact) == 1:
        return _station_meta(exact[0])
    opts = []
    for f in features:
        m = _station_meta(f)
        opts.append(f"{m['station_id']} ({m.get('station_name')}, {m.get('province')})")
    raise LookupError("several stations matched " + repr(station) + "; pass a climate identifier: " + ", ".join(opts))


def _find_stations(station) -> list[dict]:
    q = str(station).strip()
    # Climate identifiers are 7 characters (digits, sometimes a letter, e.g. 701S001).
    if re.fullmatch(r"[0-9A-Za-z]{7}", q):
        feats = _items("climate-stations", {"CLIMATE_IDENTIFIER": q, "limit": "10", "f": "json"})
        if feats:
            return feats
    if re.fullmatch(r"\d{1,6}", q):
        feats = _items("climate-stations", {"STN_ID": q, "limit": "10", "f": "json"})
        if feats:
            return feats
    # Exact name, then a case-insensitive contains match via CQL.
    feats = _items("climate-stations", {"STATION_NAME": q, "limit": "50", "f": "json"})
    if feats:
        return feats
    like = q.replace("'", "''").upper()
    filt = f"UPPER(STATION_NAME) LIKE '%{like}%'"
    feats = _items("climate-stations", {"filter": filt, "limit": "50", "f": "json"})
    return feats


def _station_meta(feature: dict) -> dict:
    p = feature.get("properties") or {}
    lat = lon = None
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates")
    if coords and len(coords) >= 2:
        lon, lat = float(coords[0]), float(coords[1])
    return {
        "station_id": str(p.get("CLIMATE_IDENTIFIER") or ""),
        "station_name": p.get("STATION_NAME"),
        "province": p.get("PROV_STATE_TERR_CODE") or p.get("PROVINCE_CODE"),
        "stn_id": p.get("STN_ID"),
        "lat": lat,
        "lon": lon,
        "elevation": _try_float(p.get("ELEVATION")),
        "first_date": p.get("FIRST_DATE"),
        "last_date": p.get("LAST_DATE"),
    }


def _stations_frame(features: list[dict]) -> pd.DataFrame:
    rows = [_station_meta(f) for f in features]
    return pd.DataFrame(rows)


def _try_float(v):
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _items(collection: str, params: dict, timeout: int = _DEFAULT_TIMEOUT) -> list[dict]:
    url = f"{BASE}/collections/{collection}/items"
    return _paginate(url, params, timeout=timeout)


def _paginate(url: str, params: dict | None, timeout: int) -> list[dict]:
    """Follow ``next`` links until the server has returned every feature.

    The total is read from the first page only. GeoMet reports a smaller
    ``numberMatched`` on later pages (``offset + numberReturned``), so trusting
    each page's count stopped long records after the second page (20,000
    days, about 55 years of daily data).
    """
    features: list[dict] = []
    next_url = url if params is None else url + "?" + urllib.parse.urlencode(params)
    seen = set()
    total = None
    while next_url and next_url not in seen:
        seen.add(next_url)
        payload = _get_json(next_url, timeout=timeout)
        batch = payload.get("features") or []
        features.extend(batch)
        if total is None and payload.get("numberMatched") is not None:
            total = int(payload["numberMatched"])
        nxt = None
        for link in payload.get("links") or []:
            if link.get("rel") == "next" and link.get("href"):
                nxt = link["href"]
                break
        if not batch:
            break
        if total is not None and len(features) >= total:
            break
        next_url = nxt
    return features


def _get_json(url: str, timeout: int) -> dict:
    delay = 1.0
    last_err: Exception | None = None
    for _attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            code = getattr(e, "code", None)
            if code and 400 <= int(code) < 500 and int(code) != 429:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 8)
    raise last_err  # type: ignore[misc]
