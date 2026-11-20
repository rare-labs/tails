"""Download measured application data; keep raw downloads outside the repository.

Run from the repository root: python examples/application_data.py --cache PATH
Sources, units, and interpretation are documented in docs/applications.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

import tails

OPSD = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
UCI = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
NAB = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/cpu_utilization_asg_misconfiguration.csv"


def download(url, cache, filename):
    """Cache a completed download atomically; do not keep a partial file as data."""
    path = Path(cache) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(path)
    return path


def rainfall(cache):
    """Daily total precipitation (mm), three stations near Québec City, 2000–2022."""
    from tails.datasets import eccc_daily

    path = Path(cache) / "quebec-precip-2000-2022.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    frame = pd.concat({station: eccc_daily(station, (2000, 2022))["precip"]
                       for station in ["701S001", "7010565", "7041330"]}, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


def electricity(cache):
    """Hourly measured national electricity loads (MW), 2015–2019, DE/FR/AT."""
    columns = [f"{country}_load_actual_entsoe_transparency" for country in ["DE", "FR", "AT"]]
    frame = pd.read_csv(download(OPSD, cache, "opsd-2020-10-06.csv"),
                        usecols=["utc_timestamp", *columns], index_col="utc_timestamp", parse_dates=True)
    frame = frame[columns]
    frame.columns = ["DE", "FR", "AT"]
    return frame.loc["2015":"2019"]


def hydraulic(cache):
    """One vector per 60-second experimental cycle: maximum PS1, EPS1 and VS1."""
    with ZipFile(download(UCI, cache, "uci-hydraulic.zip")) as archive:
        columns = {}
        for sensor in ["PS1", "EPS1", "VS1"]:
            with archive.open(f"{sensor}.txt") as stream:
                columns[sensor] = np.loadtxt(stream).max(axis=1)
        with archive.open("profile.txt") as stream:
            profile = np.loadtxt(stream, dtype=int)
    frame = pd.DataFrame(columns)
    # Explicit experiment labels, not states inferred from large observations.
    frame.attrs["operating_regime"] = np.array(["/".join(map(str, row[:4])) for row in profile])
    frame.attrs["stable"] = profile[:, 4] == 0
    return frame


def computing(cache):
    """Five-minute CPU utilisation (%) averaged across one AWS autoscaling group."""
    path = download(NAB, cache, "nab-cpu-asg.csv")
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp").rename(columns={"value": "CPU"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "tails-applications")
    parser.add_argument("--example", choices=["rainfall", "electricity", "hydraulic", "computing", "all"], default="all")
    args = parser.parse_args()
    loaders = {"rainfall": rainfall, "electricity": electricity, "hydraulic": hydraulic, "computing": computing}
    for name, load in loaders.items():
        if args.example not in (name, "all"):
            continue
        frame = load(args.cache)
        # Illustrative limits from this record, not externally defined safety limits.
        thresholds = np.array([90.]) if name == "computing" else frame.quantile(.99).to_numpy()
        event = "kofn" if frame.shape[1] > 1 else "all"
        k = 1 if name == "hydraulic" else (2 if event == "kofn" else None)
        result = tails.how_rare_joint(frame, thresholds, event=event, k=k,
                                      npy=None, ci=None)
        print(f"{name}: {len(frame)} rows; {frame.notna().all(axis=1).sum()} complete; "
              f"limits={thresholds}; observed={result.observed_count:g}; "
              f"expected={result.expected_count:.3g}; period={result.return_period:.3g} observations")


if __name__ == "__main__":
    main()
