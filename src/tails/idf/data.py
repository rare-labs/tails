"""IDF data containers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..datasets import DURATION_TAGS, DURATIONS_H

__all__ = ["IDFData", "stack"]


@dataclass
class IDFData:
    """Annual maximum intensities for several durations at one station.

    Attributes
    ----------
    years : (n,) int array
    durations : (m,) durations in hours
    values : (n, m) annual maximum *intensities* (e.g. mm/h); NaN = missing
    tags : duration labels
    name : optional station name/id
    """

    years: np.ndarray
    durations: np.ndarray
    values: np.ndarray
    tags: list[str]
    name: str = ""
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        year: str = "Year",
        durations: Mapping[str, float] | None = None,
        *,
        depth: bool = False,
        name: str = "",
    ) -> IDFData:
        """Build from a wide table (one column per duration).

        ``durations`` maps column names to hours (default: the 9 ECCC
        durations ``5min``…``24h``). With ``depth=True`` values are depths
        (mm) and are converted to intensities (mm/h).
        """
        if durations is None:
            durations = {t: d for t, d in zip(DURATION_TAGS, DURATIONS_H) if t in df.columns}
        tags = list(durations)
        d = np.array([durations[t] for t in tags], float)
        v = df[tags].to_numpy(float)
        if depth:
            v = v / d[None, :]
        yr_col = year if year in df.columns else ("year" if "year" in df.columns else df.columns[0])
        return cls(df[yr_col].to_numpy(int), d, v, tags, name)

    @classmethod
    def from_eccc(cls, station_id: str) -> IDFData:
        """Québec ECCC station from the bundled :func:`tails.datasets.load_eccc_quebec`."""
        from ..datasets import load_eccc_quebec

        df = load_eccc_quebec()
        s = df[df.station_id == station_id]
        if s.empty:
            raise ValueError(f"station {station_id!r} not found")
        out = cls.from_dataframe(s, year="year", depth=True, name=f"{s.name.iloc[0]} ({station_id})")
        out.meta = {k: s[k].iloc[0] for k in ("station_id", "name", "lat", "lon", "elevation")}
        return out

    @property
    def n_years(self) -> int:
        return len(self.years)

    def exclude_duration(self, d: float) -> IDFData:
        keep = ~np.isclose(self.durations, d)
        return IDFData(self.years, self.durations[keep], self.values[:, keep], [t for t, k in zip(self.tags, keep) if k], self.name)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.values, columns=self.tags)
        df.insert(0, "year", self.years)
        return df

    def __repr__(self) -> str:  # pragma: no cover
        return f"IDFData({self.name!r}, years={self.years.min()}–{self.years.max()} (n={self.n_years}), durations={self.tags})"


def stack(datasets: Sequence[IDFData]):
    """Stack stations into padded arrays ``Y (S, n_max, m)``, durations ``(S, m)``.

    All stations must share the same number of durations (columns may be all
    NaN for durations a station lacks).
    """
    m = {len(d.durations) for d in datasets}
    if len(m) != 1:
        raise ValueError("all stations must have the same number of duration columns")
    n = max(d.n_years for d in datasets)
    S = len(datasets)
    Y = np.full((S, n, m.pop()), np.nan)
    D = np.stack([d.durations for d in datasets])
    for i, d in enumerate(datasets):
        Y[i, : d.n_years] = d.values
    return Y, D
