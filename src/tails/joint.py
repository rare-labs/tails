"""Joint exceedances reduced to a univariate peaks-over-threshold problem."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import chi2, genpareto

from ._jax import jnp
from .core.blocks import decluster_runs
from .core.fit import POTFit, fit_pot
from .core.optim import minimize
from .core.uncertainty import _pot_profile_nll

__all__ = ["JointRarity", "compound_series", "how_rare_joint"]


def _aligned_array(data):
    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.DatetimeIndex):
            if not data.index.is_monotonic_increasing or data.index.has_duplicates:
                raise ValueError("dates must be unique and increasing")
            if not data.index.equals(data.index.normalize()):
                raise ValueError("joint events require daily observations at midnight")
            data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq="D"))
        x = data.to_numpy(dtype=float).T
    else:
        x = np.asarray(data, dtype=float)
    if x.ndim < 2 or x.shape[-2] == 0 or x.shape[-1] == 0:
        raise ValueError("data must have shape (..., site, time), or be a DataFrame")
    if np.isinf(x).any():
        raise ValueError("infinite observations are not supported; use NaN for missing data")
    return x


def compound_series(data, thresholds, *, event: str = "all", k: int | None = None, run_length: int = 1):
    """Reduce aligned series to a score whose exceedance of 1 defines the event.

    Arrays have shape ``(*replicate, site, time)``; DataFrame columns are sites.
    Positive scalar or per-site ``thresholds`` divide each series. ``event`` is
    ``"all"`` (minimum), ``"kofn"`` (k-th largest), or ``"sum"`` (sum of ratios).
    For a raw sum exceeding U, pass ``thresholds=U, event="sum"``. Consecutive
    days use the rolling minimum of these scores. Any missing site invalidates
    that day, and missing calendar days in a DataFrame invalidate windows.
    No window crosses a replicate boundary. The time axis shrinks by r-1.
    """
    x = _aligned_array(data)
    if not isinstance(run_length, (int, np.integer)) or not 1 <= run_length <= x.shape[-1]:
        raise ValueError("run_length must be an integer between 1 and the record length")
    u = np.asarray(thresholds, float)
    if u.ndim > 1 or (u.ndim == 1 and u.shape != (x.shape[-2],)) or not np.all(np.isfinite(u) & (u > 0)):
        raise ValueError("thresholds must be finite positive scalars or one value per site")
    scaled = x / (u[..., None] if u.ndim else u)
    if event == "all":
        score = scaled.min(axis=-2)
    elif event == "kofn":
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= x.shape[-2]:
            raise ValueError("k must be an integer between 1 and the number of sites")
        score = np.partition(scaled, -k, axis=-2)[..., -k, :]
    elif event == "sum":
        score = scaled.sum(axis=-2)
    else:
        raise ValueError("event must be 'all', 'kofn' or 'sum'")
    score = np.where(np.isfinite(x).all(axis=-2), score, np.nan)
    if run_length > 1:
        score = np.lib.stride_tricks.sliding_window_view(score, run_length, axis=-1).min(axis=-1)
    return score


@dataclass
class JointRarity:
    """Fitted event rate and expected count over one record (not all replicates).

    ``lower``/``upper`` bound the return period in years; ``count_lower``/
    ``count_upper`` bound the expected count, not a future realised count.
    For runs, ``probability`` is a cluster rate per original observation.
    ``observed_count`` is the mean exact event count per supplied replicate.
    """

    probability: float
    return_period: float
    lower: float | None
    upper: float | None
    expected_count: float
    count_lower: float | None
    count_upper: float | None
    observed_count: float
    record_length: float
    fit_threshold: float
    alpha: float
    fit: POTFit
    event: str
    run_length: int


def _period_nll(th, y, mask, value, log_base, u, k, n):
    return _pot_profile_nll(th[0], th[1], y, mask, value, log_base, u, k, n, None)


def _profile_period(fit, value, period, alpha):
    """Profile the existing joint GPD/rate likelihood on a log-period grid."""
    if not np.isfinite(period):
        # A zero-probability MLE is on a support boundary. A finite grid is
        # not a profile of that boundary; do not manufacture a finite bound.
        return np.nan, np.inf
    center = period
    grid = np.unique(np.r_[np.geomspace(max(1.00001 / fit.npy, center / 1e4), center * 1e4, 101), center])
    zeta = float(fit.zeta)
    start_rate = np.clip(np.maximum(zeta, 1.1 / (grid * fit.npy)), 1e-10, 1 - 1e-9)
    th = np.column_stack([np.full(grid.size, float(fit.params["shape"])), np.log(start_rate / (1 - start_rate))])
    tile = lambda a: jnp.asarray(np.broadcast_to(a, (grid.size, *np.shape(a))))  # noqa: E731
    result = minimize(_period_nll, th, tile(fit._y[0]), tile(fit._mask[0]), tile(value),
                      jnp.asarray(np.log(grid * fit.npy)), tile(fit.threshold), tile(fit.nobs), tile(fit.n_total))
    n, k = float(fit.n_total), float(fit.nobs)
    base = -float(fit.loglik) - k * np.log(zeta) - (n - k) * np.log1p(-zeta)
    dev = np.where(np.asarray(result.converged), 2 * (np.asarray(result.fun) - base), np.nan)
    crit = chi2.ppf(1 - alpha, 1)
    mid = np.searchsorted(grid, center)

    def bound(indices, fallback):
        previous = mid
        for j in indices:
            if not np.isfinite(dev[j]):
                return np.nan  # numerical failure is not an unbounded interval
            if dev[j] >= crit:
                weight = (crit - dev[previous]) / (dev[j] - dev[previous])
                return float(np.exp(np.log(grid[previous]) + weight * np.log(grid[j] / grid[previous])))
            previous = j
        return fallback

    if not np.isfinite(dev[mid]) or abs(dev[mid]) > 1e-3:
        return np.nan, np.nan
    return bound(range(mid - 1, -1, -1), np.nan), bound(range(mid + 1, len(grid)), np.nan)


def how_rare_joint(data, thresholds, *, event: str = "all", k: int | None = None, run_length: int = 1,
                   threshold_quantile: float = .95, record_length: float | None = None,
                   npy: float = 365.25, ci: str | None = "profile", alpha: float = .05) -> JointRarity:
    """Estimate joint-event return periods and expected counts by scalar reduction.

    Uses :func:`compound_series`, then the existing GPD/POT likelihood above
    ``threshold_quantile`` of the pooled score. The event score must exceed
    that fitting threshold. Replicates share one stationary tail model and
    must have the same observation spacing. ``npy`` is observations per year
    (use 365 for the challenge's no-leap calendar).

    Runs are declustered separately within each replicate at the fitting
    threshold. Their rates use valid-window/original-day exposure, not the
    number of retained peaks. As with ordinary POT declustering, extrapolated
    run counts approximate clusters defined at the fitting threshold; these
    may merge distinct target-level runs. Profile intervals assume independent
    peaks and a binomial rate and do not account for threshold selection,
    remaining temporal dependence, seasonality or changing climate.

    ``record_length`` sets the target number of original observations; by
    default use the mean complete-day exposure of the supplied replicates.
    Only ``ci="profile"`` or ``None`` is supported. Expected-count bounds
    transform the period bounds in reverse order.
    """
    if not 0 < threshold_quantile < 1 or not 0 < alpha < 1:
        raise ValueError("threshold_quantile and alpha must be between 0 and 1")
    if not np.isfinite(npy) or npy <= 0 or ci not in (None, "profile"):
        raise ValueError("npy must be positive and ci must be None or 'profile'")
    x = _aligned_array(data)
    score = compound_series(x, thresholds, event=event, k=k, run_length=run_length)
    days = np.isfinite(x).all(axis=-2).sum()
    windows = np.isfinite(score).sum()
    if not windows:
        raise ValueError("no complete observation windows")
    nrep = int(np.prod(score.shape[:-1]))
    length = days / nrep if record_length is None else float(record_length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("record_length must be finite and positive")
    u = float(np.nanquantile(score, threshold_quantile))
    if u >= 1:
        raise ValueError("event is below the fitting threshold; lower threshold_quantile or raise event thresholds")
    flat = score.reshape(-1, score.shape[-1])
    exceed = flat > 1
    if run_length > 1:
        observed = np.count_nonzero(exceed & ~np.pad(exceed[:, :-1], ((0, 0), (1, 0)))) / nrep
        peaks = decluster_runs(flat, u, run_length=1)
    else:
        observed = np.count_nonzero(exceed) / nrep
        peaks = flat
    peaks = peaks[np.isfinite(peaks) & (peaks > u)]
    if peaks.size < 10:
        raise ValueError("at least 10 threshold excesses are required")
    # Compact the likelihood data, but preserve the full exposure for rate inference.
    fit = fit_pot(peaks, u, npy=npy * windows / days)
    zeta = peaks.size / windows
    fit = replace(fit, zeta=np.asarray(zeta), n_total=np.asarray(windows), rate=np.asarray(zeta * fit.npy))
    if not bool(fit.converged):
        raise RuntimeError("joint-event GPD fit did not converge")
    tail = genpareto.sf(1 - u, float(fit.params["shape"]), scale=float(fit.params["scale"]))
    probability = float(zeta * tail * windows / days)
    period = 1 / (npy * probability) if probability else np.inf
    lower = upper = count_lower = count_upper = None
    if ci is not None:
        lower, upper = _profile_period(fit, 1., period, alpha)
        count_lower = length / (npy * upper)
        count_upper = length / (npy * lower)
    return JointRarity(probability, period, lower, upper, length * probability, count_lower, count_upper,
                       observed, length, u, alpha, fit, event, run_length)
