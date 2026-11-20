"""Aligned, gap-aware time-block resampling and explicit bootstrap diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BootstrapDiagnostics", "BootstrapResult", "block_bootstrap_indices", "resample_aligned", "bootstrap_statistic"]


@dataclass
class BootstrapDiagnostics:
    n_attempted: int
    n_successful: int
    failures: dict[str, int]
    body_replicates: int = 0
    nonregular_replicates: int = 0


@dataclass
class BootstrapResult:
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    draws: np.ndarray
    diagnostics: BootstrapDiagnostics
    alpha: float


def block_bootstrap_indices(valid, *, mean_block_length, seed=None):
    """Draw geometric-length blocks within finite segments of one record.

    Indices at missing destination positions are -1. Every valid source position
    can start a block, including the last observation. A block restarts at record
    boundaries and missing gaps; it never wraps the last day onto the first.
    ``mean_block_length`` is in observations, not days. Boundary clipping makes
    the realised mean shorter than the geometric distribution's nominal mean.
    Nonwrapping blocks also overweight segment ends; this boundary bias matters
    for short/fragmented records and long blocks. Check sensitivity to block length.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 1 or not np.isfinite(mean_block_length) or mean_block_length < 1:
        raise ValueError("valid must be one-dimensional and mean_block_length >= 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    candidates = np.flatnonzero(valid)
    out = np.full(len(valid), -1, dtype=int)
    if not len(candidates):
        return out
    pos = 0
    while pos < len(valid):
        if not valid[pos]:
            pos += 1
            continue
        source = int(rng.choice(candidates))
        length = int(rng.geometric(1 / mean_block_length))
        for _ in range(length):
            if pos >= len(valid) or source >= len(valid) or not valid[pos] or not valid[source]:
                break
            out[pos] = source
            pos += 1
            source += 1
    return out


def resample_aligned(data, *, mean_block_length, seed=None):
    """Resample ``(*replicate, series, time)`` with one shared draw per record.

    Missing any series makes that time incomplete. Its missing position stays
    missing for every series. Independent records are never concatenated.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim < 2 or not x.shape[-1] or np.isinf(x).any():
        raise ValueError("data must be finite/NaN with shape (..., series, time)")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    flat = x.reshape(-1, *x.shape[-2:])
    out = np.full_like(flat, np.nan)
    for i, row in enumerate(flat):
        valid = np.isfinite(row).all(axis=0)
        indices = block_bootstrap_indices(valid, mean_block_length=mean_block_length, seed=rng)
        out[i][:, valid] = row[:, indices[valid]]
    return out.reshape(x.shape)


def bootstrap_statistic(data, statistic, *, n_boot=200, mean_block_length=1, alpha=.05, seed=0,
                        estimate=None, resample=None):
    """Refit a user statistic on aligned blocks; retain every unsuccessful draw.

    ``statistic`` may return a value or ``(value, info)`` where info contains
    ``body`` / ``nonregular`` flags. ``resample`` may provide a fitted-margin
    transform, with signature ``resample(data, rng)``. Limits are unavailable
    if any draw fails: failed refits are never silently deleted. This operation
    does not establish calibration; block length and residual stationarity need
    application-specific checking.
    """
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 2 or not 0 < alpha < 1:
        raise ValueError("n_boot must be at least two and alpha must lie between zero and one")
    if not np.isfinite(mean_block_length) or mean_block_length < 1:
        raise ValueError("mean_block_length must be at least one")

    def unpack(value):
        return value if isinstance(value, tuple) else (value, {})

    if estimate is None:
        estimate, _ = unpack(statistic(data))
    estimate = np.asarray(estimate, dtype=float)
    draws = np.full((n_boot, *estimate.shape), np.nan)
    rng = np.random.default_rng(seed)
    failures = {}
    good = bodies = nonregular = 0
    for i in range(n_boot):
        try:
            sample = resample(data, rng) if resample is not None else resample_aligned(
                data, mean_block_length=mean_block_length, seed=rng)
            value, info = unpack(statistic(sample))
            value = np.asarray(value, dtype=float)
            if value.shape != estimate.shape or not np.isfinite(value).all():
                raise ValueError("statistic returned an invalid or nonfinite estimate")
            draws[i] = value
            good += 1
            bodies += bool(info.get("body", False))
            nonregular += bool(info.get("nonregular", False))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            key = f"{type(exc).__name__}: {exc}"
            failures[key] = failures.get(key, 0) + 1
    lo = hi = np.full(estimate.shape, np.nan)
    if good == n_boot:
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    diagnostics = BootstrapDiagnostics(n_boot, good, failures, bodies, nonregular)
    return BootstrapResult(estimate, lo, hi, draws, diagnostics, alpha)
