"""Optional real-data validation; large inputs stay outside the source tree.

Run: python validation/compound_events.py --runs C:/Users/CM/Downloads --ci profile
     python validation/compound_events.py --eccc --ci profile
Requires rdata only for the local RData inputs. Downloads use tails' ECCC cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tails


def describe(result):
    return {key: getattr(result, key) for key in ("observed_count", "expected_count", "count_lower", "count_upper",
                                                "probability", "return_period", "lower", "upper", "fit_threshold")}


def challenge(directory, ci):
    import rdata

    runs = []
    for i in range(1, 5):
        values = np.asarray(rdata.read_rda(directory / f"run{i}.RData")[f"run{i}"])
        if values.shape != (5, 5, 60225) or not np.isfinite(values).all():
            raise ValueError(f"run{i} does not match the challenge grid/calendar")
        runs.append(values.reshape(25, 60225))
    data = np.stack(runs)
    cases = {"all25_above1.7": (1.7, {}), "6of25_above5.7": (5.7, {"event": "kofn", "k": 6}),
             "3of25_above5_run2": (5., {"event": "kofn", "k": 3, "run_length": 2}),
             "preliminary_sum_above85": (85., {"event": "sum"})}
    for label, (threshold, kwargs) in cases.items():
        result = tails.how_rare_joint(data, threshold, npy=365, ci=ci, **kwargs)
        print(json.dumps({"case": label, "record_days": 60225, "replicates": 4, **describe(result)}), flush=True)


def quebec(ci):
    from tails.datasets import eccc_daily

    # Airport, Beauport and Château-Richer; ECCC climate station identifiers.
    stations = ["701S001", "7010565", "7041330"]
    frame = pd.concat({s: eccc_daily(s, (2000, 2022))["precip"] for s in stations}, axis=1)
    print(json.dumps({"stations": stations, "complete_days": int(frame.notna().all(axis=1).sum()),
                      "start": str(frame.index.min()), "end": str(frame.index.max())}), flush=True)
    for label, threshold, kwargs in [("all_above20mm", 20., {}), ("2of3_above30mm", 30., {"event": "kofn", "k": 2})]:
        result = tails.how_rare_joint(frame, threshold, ci=ci, **kwargs)
        print(json.dumps({"case": label, **describe(result)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, help="directory containing run1.RData through run4.RData")
    parser.add_argument("--eccc", action="store_true", help="download and analyse nearby Québec City daily precipitation")
    parser.add_argument("--ci", choices=["profile", "none"], default="profile")
    args = parser.parse_args()
    ci = None if args.ci == "none" else args.ci
    if args.runs:
        challenge(args.runs, ci)
    if args.eccc:
        quebec(ci)
    if not args.runs and not args.eccc:
        parser.error("supply --runs or --eccc")
