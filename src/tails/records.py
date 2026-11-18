"""Record-breaking toolkit: are records happening more often than chance?

Under a continuous i.i.d. series the probability that observation ``t`` is a
record is ``1/t``, whatever the distribution, and the indicators at different
``t`` are independent. The expected number of records in ``n`` observations is
the harmonic number ``H_n`` (Glick, 1978). Too many upper records is a
distribution-free sign of an upward drift; the Foster–Stuart statistics split
that into a trend and a change in spread (Foster and Stuart, 1954).

The last axis is the series, as everywhere else in tails. Missing values are
skipped: they are never records, and they do not advance the trial count.
Ties are not records (the null above is continuous, so ties have probability
zero). The first finite observation is both an upper and a lower record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "FosterStuart",
    "RecordBreaks",
    "foster_stuart",
    "record_breaks",
    "record_indicators",
    "record_test",
]


def _as_float(x):
    if isinstance(x, pd.DataFrame):
        raise TypeError("pass a Series or an array whose last axis is the series, not a DataFrame")
    index = x.index if isinstance(x, pd.Series) else None
    raw = np.asarray(x)
    if raw.dtype.kind not in "iuf":
        raise TypeError("record statistics need a numeric series")
    return np.asarray(raw, dtype=np.float64), index


def _kind(kind: str) -> str:
    key = str(kind).lower()
    if key not in {"upper", "lower"}:
        raise ValueError("kind must be 'upper' or 'lower'")
    return key


def _sd(var):
    return np.sqrt(np.maximum(np.asarray(var, dtype=np.float64), 1e-12))


def _norm_sf(z):
    from scipy.stats import norm

    return norm.sf(z)


def record_indicators(x, kind: str = "upper"):
    """Boolean array marking records along the last axis.

    A finite value is an upper record when it is strictly larger than every
    finite value before it, and a lower record when it is strictly smaller.
    NaN is never a record and does not change the running extreme.
    """
    values, _ = _as_float(x)
    key = _kind(kind)
    signed = values if key == "upper" else -values
    filled = np.where(np.isfinite(signed), signed, -np.inf)
    previous = np.maximum.accumulate(filled, axis=-1)
    previous = np.concatenate(
        [np.full(previous.shape[:-1] + (1,), -np.inf), previous[..., :-1]],
        axis=-1,
    )
    return np.isfinite(signed) & (signed > previous)


def _trial_weights(values):
    """Trial number ``t`` at each finite entry, and ``1/t`` there (else 0)."""
    valid = np.isfinite(values)
    trial = np.cumsum(valid, axis=-1)
    weight = np.where(valid, 1.0 / np.maximum(trial, 1), 0.0)
    return valid, trial, weight


@dataclass
class RecordBreaks:
    """Upper or lower records in one series or a batch.

    ``indicators`` and ``values`` have the input shape. ``values`` keeps the
    record-breaking observations and is NaN everywhere else. ``trial`` is the
    count of finite observations up to and including that position (0 on a
    missing value). The remaining fields drop the time axis: one number per
    series.

    ``pvalue`` is the one-sided normal approximation, with a continuity
    correction, for more records than the harmonic law. ``pvalue_fewer`` is
    the other tail. ``pvalue_two_sided`` covers either departure.
    """

    kind: str
    indicators: np.ndarray
    values: np.ndarray
    trial: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    variance: np.ndarray
    ratio: np.ndarray
    pvalue: np.ndarray
    pvalue_fewer: np.ndarray
    pvalue_two_sided: np.ndarray
    index: pd.Index | None = None

    def summary(self) -> pd.DataFrame:
        """One row per series: counts, the harmonic expectation, and p-values."""
        observed = np.asarray(self.observed, dtype=np.float64).reshape(-1)
        columns = {
            "kind": self.kind,
            "observed": observed,
            "expected": np.asarray(self.expected, dtype=np.float64).reshape(-1),
            "variance": np.asarray(self.variance, dtype=np.float64).reshape(-1),
            "ratio": np.asarray(self.ratio, dtype=np.float64).reshape(-1),
            "pvalue": np.asarray(self.pvalue, dtype=np.float64).reshape(-1),
            "pvalue_fewer": np.asarray(self.pvalue_fewer, dtype=np.float64).reshape(-1),
            "pvalue_two_sided": np.asarray(self.pvalue_two_sided, dtype=np.float64).reshape(-1),
        }
        frame = pd.DataFrame(columns)
        shape = np.asarray(self.observed).shape
        if shape == ():
            return frame
        frame.index = pd.RangeIndex(len(frame), name="series")
        frame.attrs["batch_shape"] = shape
        return frame

    def events(self, series: int = 0) -> pd.DataFrame:
        """One row per record in a single series, in time order.

        ``time`` is the pandas index when the input was a Series, and the
        position along the series otherwise. ``trial`` counts only finite
        observations. ``records_so_far`` and ``expected_so_far`` are the
        cumulative count and the harmonic number at that trial.
        """
        indicators, values, trial = self._series(series)
        mask = np.asarray(indicators, dtype=bool)
        positions = np.flatnonzero(mask)
        harmonic = np.cumsum(np.where(trial > 0, 1.0 / np.maximum(trial, 1), 0.0))
        count = np.cumsum(mask)
        if self.index is not None:
            time = self.index[positions]
        else:
            time = positions
        return pd.DataFrame(
            {
                "time": time,
                "value": np.asarray(values, dtype=np.float64)[mask],
                "trial": np.asarray(trial)[mask],
                "records_so_far": count[mask],
                "expected_so_far": harmonic[mask],
            }
        )

    def _series(self, series: int):
        indicators = self.indicators
        if indicators.ndim == 1:
            if series != 0:
                raise IndexError("a single series has only series=0")
            return indicators, self.values, self.trial
        flat_i = indicators.reshape(-1, indicators.shape[-1])
        flat_v = self.values.reshape(-1, self.values.shape[-1])
        flat_t = self.trial.reshape(-1, self.trial.shape[-1])
        if not 0 <= series < flat_i.shape[0]:
            raise IndexError(f"series must be in 0..{flat_i.shape[0] - 1}")
        return flat_i[series], flat_v[series], flat_t[series]


def record_breaks(x, *, kind: str = "upper") -> RecordBreaks:
    """Records along the last axis, with the harmonic null.

    ``kind`` is ``"upper"`` (a new maximum) or ``"lower"`` (a new minimum).
    A batch has shape ``(*batch, n)``; each field of counts has shape ``batch``.
    """
    values, index = _as_float(x)
    key = _kind(kind)
    indicators = record_indicators(values, key)
    _, trial, weight = _trial_weights(values)
    # 1/t * (1 - 1/t) on finite trials.
    variance_term = weight * (1.0 - weight)
    observed = indicators.sum(-1)
    expected = weight.sum(-1)
    variance = variance_term.sum(-1)
    ratio = np.divide(observed, expected, out=np.full(np.shape(expected), np.nan), where=expected > 0)
    scale = _sd(variance)
    more = _norm_sf((observed - expected - 0.5) / scale)
    fewer = _norm_sf((expected - observed - 0.5) / scale)
    both = np.minimum(1.0, 2.0 * _norm_sf(np.maximum(np.abs(observed - expected) - 0.5, 0.0) / scale))
    kept = np.where(indicators, values, np.nan)
    return RecordBreaks(
        kind=key,
        indicators=indicators,
        values=kept,
        trial=np.where(np.isfinite(values), trial, 0),
        observed=observed,
        expected=expected,
        variance=variance,
        ratio=ratio,
        pvalue=more,
        pvalue_fewer=fewer,
        pvalue_two_sided=both,
        index=index,
    )


def record_test(x, kind: str = "upper"):
    """Observed vs expected record counts under exchangeability.

    Returns a dict with ``observed``, ``expected``, ``variance`` and a one-sided
    normal-approximation ``pvalue`` for more records than chance (per series).
    :func:`record_breaks` is the same count with the table and both tails.
    """
    result = record_breaks(x, kind=kind)
    return {
        "observed": result.observed,
        "expected": result.expected,
        "variance": result.variance,
        "pvalue": result.pvalue,
    }


@dataclass
class FosterStuart:
    """Foster–Stuart record statistics for a trend and a change in spread.

    From the second finite observation on, ``d`` is the number of upper records
    minus the number of lower records, and ``s`` is their sum. The first finite
    observation is both a maximum and a minimum, so it is left out of both.

    Under continuous i.i.d. data, ``d`` has mean 0 and ``s`` has mean
    ``sum_{t=2}^{n} 2/t``. A positive ``d`` means upper records dominate
    (upward drift). ``pvalue_up`` is the one-sided test of that drift.
    ``pvalue_trend`` is two-sided. ``pvalue_spread`` is two-sided for ``s``:
    more records of both kinds than the harmonic law means the spread is
    growing, and fewer means it is shrinking.
    """

    d: np.ndarray
    s: np.ndarray
    d_variance: np.ndarray
    s_expected: np.ndarray
    s_variance: np.ndarray
    pvalue_up: np.ndarray
    pvalue_trend: np.ndarray
    pvalue_spread: np.ndarray

    def summary(self) -> pd.DataFrame:
        """One row per series."""
        frame = pd.DataFrame(
            {
                "d": np.asarray(self.d, dtype=np.float64).reshape(-1),
                "s": np.asarray(self.s, dtype=np.float64).reshape(-1),
                "d_variance": np.asarray(self.d_variance, dtype=np.float64).reshape(-1),
                "s_expected": np.asarray(self.s_expected, dtype=np.float64).reshape(-1),
                "s_variance": np.asarray(self.s_variance, dtype=np.float64).reshape(-1),
                "pvalue_up": np.asarray(self.pvalue_up, dtype=np.float64).reshape(-1),
                "pvalue_trend": np.asarray(self.pvalue_trend, dtype=np.float64).reshape(-1),
                "pvalue_spread": np.asarray(self.pvalue_spread, dtype=np.float64).reshape(-1),
            }
        )
        shape = np.asarray(self.d).shape
        if shape != ():
            frame.index = pd.RangeIndex(len(frame), name="series")
            frame.attrs["batch_shape"] = shape
        return frame


def foster_stuart(x) -> FosterStuart:
    """Foster–Stuart ``D`` and ``S`` along the last axis.

    See :class:`FosterStuart`. A batch has shape ``(*batch, n)``.
    """
    values, _ = _as_float(x)
    upper = record_indicators(values, "upper")
    lower = record_indicators(values, "lower")
    _, trial, _ = _trial_weights(values)
    after = np.isfinite(values) & (trial >= 2)
    # P(upper or lower record at t) = 2/t for t >= 2, and the two cannot coincide.
    weight = np.where(after, 2.0 / trial, 0.0)
    d = (upper & after).sum(-1).astype(np.float64) - (lower & after).sum(-1)
    s = ((upper | lower) & after).sum(-1).astype(np.float64)
    d_variance = weight.sum(-1)
    s_expected = d_variance
    s_variance = (weight * (1.0 - weight)).sum(-1)
    scale_d = _sd(d_variance)
    scale_s = _sd(s_variance)
    pvalue_up = _norm_sf((d - 0.0 - 0.5) / scale_d)
    pvalue_trend = np.minimum(1.0, 2.0 * _norm_sf(np.maximum(np.abs(d) - 0.5, 0.0) / scale_d))
    pvalue_spread = np.minimum(
        1.0,
        2.0 * _norm_sf(np.maximum(np.abs(s - s_expected) - 0.5, 0.0) / scale_s),
    )
    return FosterStuart(
        d=d,
        s=s,
        d_variance=d_variance,
        s_expected=s_expected,
        s_variance=s_variance,
        pvalue_up=pvalue_up,
        pvalue_trend=pvalue_trend,
        pvalue_spread=pvalue_spread,
    )
