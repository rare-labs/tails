"""Real datasets: bundled ECCC IDF annual maxima and an ECCC file parser.

* ``load_idfcurves(station)`` – the three datasets shipped with IDFCurves.jl
  (intensities in mm/h): ``"702S006"`` Montréal–Trudeau, ``"6158731"``
  Toronto Pearson, ``"1108446"`` Vancouver Harbour.
* ``load_eccc_quebec()`` – annual maximum rainfall *depths* (mm) for every
  Québec station of ECCC's Engineering Climate Datasets IDF v3.40
  (2025-12-05), extracted with :func:`parse_eccc_idf`.
* ``parse_eccc_idf(path_or_text)`` – parse any ECCC ``idf_v3-*.txt`` file.

Source: Environment and Climate Change Canada,
https://collaboration.cmc.ec.gc.ca/cmc/climate/Engineer_Climate/IDF/
(Open Government Licence – Canada).
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pandas as pd

__all__ = ["parse_eccc_return_levels", "load_idfcurves", "load_eccc_quebec", "parse_eccc_idf", "DURATIONS_H", "DURATION_TAGS"]

DURATION_TAGS = ["5min", "10min", "15min", "30min", "1h", "2h", "6h", "12h", "24h"]
DURATIONS_H = [5 / 60, 10 / 60, 15 / 60, 30 / 60, 1.0, 2.0, 6.0, 12.0, 24.0]


def _data_path(name: str):
    return resources.files("tails.datasets").joinpath("data", name)


def load_idfcurves(station: str = "702S006") -> pd.DataFrame:
    """Load an IDFCurves.jl dataset (columns ``Year`` + 9 durations, mm/h)."""
    p = _data_path(f"{station}.csv")
    if not p.is_file():
        raise ValueError(f"unknown dataset {station!r}; choose 702S006, 6158731 or 1108446")
    with p.open() as fh:
        return pd.read_csv(fh)


def load_eccc_quebec() -> pd.DataFrame:
    """Annual maximum rainfall depths (mm) for all Québec ECCC IDF stations.

    Long table with columns ``station_id``, ``name``, ``lat``, ``lon``,
    ``elevation``, ``year`` and the 9 duration columns (``5min`` ... ``24h``).
    Missing values are NaN.
    """
    with _data_path("eccc_idf_v3-40_QC.csv.gz").open("rb") as fh:
        return pd.read_csv(fh, compression="gzip", dtype={"station_id": str})


def _dms(deg: str, minutes: str, hemi: str) -> float:
    v = float(deg) + float(minutes) / 60.0
    return -v if hemi in ("S", "W") else v


def parse_eccc_idf(source) -> tuple[dict, pd.DataFrame]:
    """Parse an ECCC short-duration IDF text file (``idf_v3-*.txt``).

    Returns ``(meta, table)`` where ``table`` holds Table 1 (annual maximum
    depths in mm, one row per year, NaN for ``-99.9``) and ``meta`` contains
    station name, id, province, latitude, longitude and elevation.
    """
    if isinstance(source, (str, Path)) and Path(source).exists():
        raw = Path(source).read_bytes().decode("latin-1")
    elif isinstance(source, bytes):
        raw = source.decode("latin-1")
    else:
        raw = str(source)
    lines = raw.splitlines()
    meta = {}
    for ln in lines:
        m = re.match(r"^\s(\S.*?)\s{2,}([A-Z]{2})\s+(\w{7})\s*$", ln)
        if m and "name" not in meta:
            meta.update(name=m.group(1).strip(), province=m.group(2), station_id=m.group(3))
        m = re.search(r"Latitude:\s*(\d+)\s+(\d+)'([NS])\s+Longitude:\s*(\d+)\s+(\d+)'([EW])\s+Elevation/Altitude:\s*(-?[\d.]+)", ln)
        if m:
            meta["lat"] = _dms(m.group(1), m.group(2), m.group(3))
            meta["lon"] = _dms(m.group(4), m.group(5), m.group(6))
            meta["elevation"] = float(m.group(7))
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Table 1"))
    rows = []
    for ln in lines[start + 1 :]:
        s = ln.strip()
        if s.startswith("---"):
            break
        parts = s.split()
        if len(parts) == 10 and re.fullmatch(r"\d{4}", parts[0]):
            rows.append([int(parts[0])] + [float(v) for v in parts[1:]])
    df = pd.DataFrame(rows, columns=["year", *DURATION_TAGS])
    df[DURATION_TAGS] = df[DURATION_TAGS].where(df[DURATION_TAGS] > -99)
    return meta, df


def to_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Convert depth columns (mm) to intensities (mm/h)."""
    out = df.copy()
    for tag, d in zip(DURATION_TAGS, DURATIONS_H):
        out[tag] = out[tag] / d
    return out


def parse_eccc_return_levels(source) -> pd.DataFrame:
    """Parse ECCC's Table 2a (Gumbel return-period amounts, mm) from an IDF text file.

    Returns a table indexed by duration tag with columns ``2, 5, 10, 25, 50, 100, 200``.
    """
    if isinstance(source, (str, Path)) and Path(source).exists():
        raw = Path(source).read_bytes().decode("latin-1")
    elif isinstance(source, bytes):
        raw = source.decode("latin-1")
    else:
        raw = str(source)
    lines = raw.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Table 2a"))
    rows = []
    for ln in lines[start + 1 :]:
        if ln.strip().startswith("Table 2b"):
            break
        m = re.match(r"^\s*(\d+)\s*(min|h)\s+([-\d.\s]+)$", ln)
        if m:
            vals = [float(v) for v in m.group(3).split()]
            rows.append([f"{m.group(1)}{m.group(2)}"] + vals[:7])
    return pd.DataFrame(rows, columns=["duration", 2, 5, 10, 25, 50, 100, 200]).set_index("duration")
