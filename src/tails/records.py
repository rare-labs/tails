"""Record statistics: are records happening more often than chance?

Under an i.i.d. (stationary, exchangeable) series the probability that
observation ``i`` is a record is ``1/i`` regardless of the distribution, so the
expected number of records in ``n`` observations is the harmonic number
``H_n`` (Glick, 1978). Excess records are a distribution-free signal of change
(e.g. heat records, sport records) that needs no extreme-value fit at all.
"""

from __future__ import annotations

import numpy as np

__all__ = ["record_indicators", "record_test"]


def record_indicators(x, kind: str = "upper"):
    """Boolean array marking records along the last axis (NaN never a record)."""
    x = np.asarray(x, float)
    s = x if kind == "upper" else -x
    filled = np.where(np.isfinite(s), s, -np.inf)
    prev = np.maximum.accumulate(filled, axis=-1)
    prev = np.concatenate([np.full(prev.shape[:-1] + (1,), -np.inf), prev[..., :-1]], axis=-1)
    return np.isfinite(s) & (s > prev)


def record_test(x, kind: str = "upper"):
    """Observed vs expected record counts under exchangeability.

    Returns a dict with ``observed``, ``expected``, ``variance`` and a one-sided
    normal-approximation ``pvalue`` for "more records than chance" (per series).
    """
    x = np.asarray(x, float)
    rec = record_indicators(x, kind)
    valid = np.isfinite(x)
    i = np.cumsum(valid, axis=-1)
    p = np.where(valid, 1.0 / np.maximum(i, 1), 0.0)
    expected = p.sum(-1)
    var = (p * (1 - p)).sum(-1)
    obs = rec.sum(-1)
    from scipy.stats import norm

    z = (obs - expected - 0.5) / np.sqrt(np.maximum(var, 1e-12))
    return {"observed": obs, "expected": expected, "variance": var, "pvalue": norm.sf(z)}
