"""Joint exceedances reduced to a univariate peaks-over-threshold problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, genpareto

from ._jax import jax, jnp
from .core._special import expm1_over_x
from .core.blocks import decluster_runs
from .core.distributions import gpd_logpdf
from .core.fit import POTFit, fit_pot
from .core.optim import minimize
from .margins import fit_marginal
from .resampling import BootstrapResult, bootstrap_statistic, resample_aligned

__all__ = ["JointRarity", "compound_series", "how_rare_joint"]


def _aligned_array(data, freq=None):
    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.DatetimeIndex):
            if not data.index.is_monotonic_increasing or data.index.has_duplicates:
                raise ValueError("dates must be unique and increasing")
            cadence = freq or data.index.freqstr or (pd.infer_freq(data.index) if len(data) >= 3 else None)
            if cadence is None:
                if data.index.equals(data.index.normalize()):
                    cadence = "D"
                else:
                    raise ValueError("freq is required for an irregular subdaily index")
            target = pd.date_range(data.index.min(), data.index.max(), freq=cadence)
            if not data.index.isin(target).all():
                raise ValueError("freq would discard observed timestamps")
            data = data.reindex(target)
        x = data.to_numpy(dtype=float).T
    else:
        x = np.asarray(data, dtype=float)
    if x.ndim < 2 or x.shape[-2] == 0 or x.shape[-1] == 0:
        raise ValueError("data must have shape (..., site, time), or be a DataFrame")
    if np.isinf(x).any():
        raise ValueError("infinite observations are not supported; use NaN for missing data")
    return x


def compound_series(data, thresholds, *, event: str = "all", k: int | None = None, run_length: int = 1,
                    scales=None, directions=None, freq=None):
    """Reduce aligned series to a score whose exceedance of 1 defines the event.

    Arrays have shape ``(*replicate, site, time)``; DataFrame columns are sites.
    Positive scalar or per-site ``thresholds`` divide each series. ``event`` is
    ``"all"`` (minimum), ``"kofn"`` (k-th largest), or ``"sum"`` (sum of ratios).
    For a raw sum exceeding U, pass ``thresholds=U, event="sum"``. Consecutive
    days use the rolling minimum of these scores. Any missing site invalidates
    that day, and missing calendar days in a DataFrame invalidate windows.
    No window crosses a replicate boundary. The time axis shrinks by r-1.
    """
    x = _aligned_array(data, freq=freq)
    if not isinstance(run_length, (int, np.integer)) or not 1 <= run_length <= x.shape[-1]:
        raise ValueError("run_length must be an integer between 1 and the record length")
    u = np.asarray(thresholds, float)
    if u.ndim > 1 or (u.ndim == 1 and u.shape != (x.shape[-2],)) or not np.all(np.isfinite(u)):
        raise ValueError("thresholds must be finite scalars or one value per series")
    if scales is None and directions is None:
        if np.any(u <= 0):
            raise ValueError("positive thresholds are required unless positive scales are supplied")
        scaled = x / (u[..., None] if u.ndim else u)
    else:
        if event == "sum":
            raise ValueError("sum events require the original common-unit threshold formulation")
        scale = np.asarray(scales, float)
        direction = np.asarray(1 if directions is None else directions, float)
        if scale.ndim > 1 or scale.shape not in [(), (x.shape[-2],)] or not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError("scales must be finite positive scalars or one per series")
        if direction.shape not in [(), (x.shape[-2],)] or not np.all(np.isin(direction, [-1, 1])):
            raise ValueError("directions must be +1 or -1 per series")
        scaled = 1 + (direction[..., None] if direction.ndim else direction) * (x - (u[..., None] if u.ndim else u)) / (scale[..., None] if scale.ndim else scale)
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
    fit: POTFit | None
    event: str
    run_length: int
    period_unit: str = "years"
    bootstrap: BootstrapResult | None = None
    fits_by_group: dict | None = None


def _period_nll(th, y, mask, value, log_base, u, k, n):
    shape, eta = th
    log_rate = -jax.nn.softplus(-eta)
    a = log_base + log_rate
    safe_a = jnp.where(a > 0, a, 1.)
    scale = (value - u) / (safe_a * expm1_over_x(shape * safe_a))
    lp = gpd_logpdf(y, scale, shape)
    excess_nll = -jnp.sum(jnp.where(mask, lp, 0.))
    rate_nll = -k * log_rate + (n - k) * jax.nn.softplus(eta)
    return jnp.where((a > 0) & (scale > 0), excess_nll + rate_nll, jnp.inf)


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
    excesses = fit._y[0][fit._mask[0]]
    k = float(len(excesses))
    result = minimize(_period_nll, th, tile(excesses), tile(np.ones(len(excesses), bool)), tile(value),
                      jnp.asarray(np.log(grid * fit.npy)), tile(fit.threshold), tile(k), tile(fit.n_total))
    n = float(fit.n_total)
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


def _groups_array(data, x, groups, freq):
    shape = (*x.shape[:-2], x.shape[-1])
    if groups is None:
        return np.zeros(shape, dtype=int)
    if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex):
        cadence = freq or data.index.freqstr or (pd.infer_freq(data.index) if len(data) >= 3 else None) or "D"
        index = pd.date_range(data.index.min(), data.index.max(), freq=cadence)
        source = groups if isinstance(groups, pd.Series) else pd.Series(groups, index=data.index)
        groups = source.reindex(index).to_numpy()
    try:
        return np.broadcast_to(np.asarray(groups), shape)
    except ValueError as exc:
        raise ValueError("groups must align with the observation axis and optional replicate axes") from exc


def _observed(score, run_length):
    above = score > 1
    if run_length > 1:
        above &= ~np.pad(above[:, :-1], ((0, 0), (1, 0)))
    return int(above.sum())


def how_rare_joint(data, thresholds, *, event="all", k=None, run_length=1,
                   threshold_quantile=.95, record_length=None, npy=365.25,
                   ci="profile", alpha=.05, groups=None, block_length=None,
                   n_boot=200, seed=0, scales=None, directions=None, freq=None):
    """Joint-event risk from aligned series, with optional full-refit block intervals.

    Arrays are ``(*replicate, series, observation)``. Positive thresholds retain
    the original ratio score. Explicit positive ``scales`` also allow signed
    limits; ``directions=-1`` selects lower extremes. DataFrame cadence is
    inferred when regular, otherwise provide ``freq``; gaps remain missing.

    ``npy`` is observations per year. ``npy=None`` reports return periods in
    observations and expected counts over the supplied record, without inventing
    an annual sampling rate. Groups are supplied context labels, never inferred
    seasons; each gets a separate GPD shape and scale. Grouped models support
    ``ci=None`` or ``'block_bootstrap'`` and one-observation events.

    Block intervals resample aligned observations and refit the complete event
    model. Grouped bootstraps resample on fitted probability margins and map
    back at the destination groups: residual dependence is assumed stationary
    after transformation. A refit whose event enters the empirical body uses
    its empirical event count. Event limits never move. Every failed refit is
    retained in ``result.bootstrap.diagnostics``; bounds are NaN on any failure.
    Block length is explicit and in observations. This is not a calibration
    guarantee or a prediction interval for future random counts.

    For runs, the existing fitting-threshold declustering approximation remains:
    it can merge distinct target-level runs. Profile intervals assume stationary,
    independent tail excesses and do not account for threshold choice.
    """
    if not 0 < threshold_quantile < 1 or not 0 < alpha < 1:
        raise ValueError("threshold_quantile and alpha must be between zero and one")
    if ci not in (None, "profile", "block_bootstrap"):
        raise ValueError("ci must be None, 'profile' or 'block_bootstrap'")
    cadence = 1. if npy is None else float(npy)
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("npy must be positive or None")
    if groups is not None and ci == "profile":
        raise ValueError("grouped margins require block intervals; a stationary profile is not appropriate")
    if groups is not None and ci == "block_bootstrap" and threshold_quantile <= .5:
        raise ValueError("grouped block bootstrap requires threshold_quantile > .5")
    if groups is not None and run_length != 1:
        raise ValueError("grouped run models are not supported; use stationary runs or one-observation grouped events")
    if ci == "block_bootstrap" and (block_length is None or not np.isfinite(block_length) or block_length < 1):
        raise ValueError("block_length >= 1 is required for block bootstrap")
    x = _aligned_array(data, freq=freq)
    g = _groups_array(data, x, groups, freq)
    if pd.isna(g[np.isfinite(x).all(axis=-2)]).any():
        raise ValueError("groups must supply a label for every complete observation")
    return _analyse(x, thresholds, event, k, run_length, threshold_quantile, record_length,
                    cadence, ci, alpha, g, groups is not None, block_length, n_boot, seed,
                    scales, directions, "observations" if npy is None else "years", False)[0]


def _analyse(x, thresholds, event, k, run_length, quantile, record_length, cadence,
             ci, alpha, g, grouped, block_length, n_boot, seed, scales, directions,
             period_unit, allow_body):
    score = compound_series(x, thresholds, event=event, k=k, run_length=run_length,
                            scales=scales, directions=directions)
    days = np.isfinite(x).all(axis=-2).sum()
    flat = score.reshape(-1, score.shape[-1])
    nrep = len(flat)
    windows = np.isfinite(score).sum()
    if not windows:
        raise ValueError("no complete observation windows")
    length = days / nrep if record_length is None else float(record_length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("record_length must be finite and positive")
    gf = g[..., run_length-1:].reshape(flat.shape)
    labels = list(dict.fromkeys(gf[np.isfinite(flat)].tolist()))
    fits = {}
    cutoffs = []
    probability = 0.
    body = nonregular = False
    for label in labels:
        selected = np.where(gf == label, flat, np.nan)
        count = np.isfinite(selected).sum()
        u = float(np.nanquantile(selected, quantile))
        cutoffs.append(u)
        if u >= 1:
            if not (allow_body or ci == "block_bootstrap"):
                raise ValueError("event is below the fitting threshold; raise event thresholds or use block bootstrap")
            probability += _observed(selected, run_length) / days
            fits[label] = None
            body = True
            continue
        peaks = decluster_runs(selected, u, run_length=1) if run_length > 1 else selected
        hits = np.isfinite(peaks) & (peaks > u)
        if hits.sum() < 10:
            raise ValueError("at least 10 threshold excesses are required per group")
        # Non-peak days remain observed non-exceedances for rate inference.
        likelihood_data = np.where(np.isfinite(selected), np.where(hits, peaks, u), np.nan).ravel()
        fit = fit_pot(likelihood_data, u, npy=cadence * windows / days)
        if not bool(fit.converged):
            raise RuntimeError("joint-event GPD fit did not converge")
        tail = genpareto.sf(1-u, float(fit.params["shape"]), scale=float(fit.params["scale"]))
        probability += float(fit.zeta) * float(tail) * count / days
        nonregular |= bool(float(fit.params["shape"]) <= -.5)
        fits[label] = fit
    expected = length * probability
    period = 1 / (cadence * probability) if probability else np.inf
    fit = next(iter(fits.values())) if len(fits) == 1 else None
    lower = upper = count_lower = count_upper = None
    boot = None
    if ci == "profile":
        lower, upper = _profile_period(fit, 1., period, alpha)
        count_lower = length / (cadence * upper)
        count_upper = length / (cadence * lower)
    if ci == "block_bootstrap":
        transform = None
        if grouped:
            rows = x.reshape(-1, *x.shape[-2:])
            labels_array = g.reshape(-1, g.shape[-1])
            complete = np.isfinite(rows).all(axis=1)
            aligned = np.where(complete[:, None, :], rows, np.nan)
            probabilities = np.full_like(rows, np.nan)
            # Replicates share fitted distributions, but never bootstrap blocks.
            margins = [fit_marginal(aligned[:, j, :].ravel(), groups=labels_array.ravel(),
                                    threshold_quantile=quantile) for j in range(rows.shape[1])]
            for i, row in enumerate(aligned):
                for j, margin in enumerate(margins):
                    probabilities[i, j] = margin.cdf(row[j], groups=labels_array[i])

            def transform(_data, rng):
                sampled = resample_aligned(probabilities, mean_block_length=block_length, seed=rng)
                out = np.full_like(sampled, np.nan)
                for i in range(len(rows)):
                    for j, margin in enumerate(margins):
                        out[i, j] = margin.ppf(sampled[i, j], groups=labels_array[i])
                return out.reshape(x.shape)

        def statistic(sample):
            result, info = _analyse(sample, thresholds, event, k, run_length, quantile, length,
                                    cadence, None, alpha, g, grouped, None, n_boot, seed,
                                    scales, directions, period_unit, True)
            return result.expected_count, info

        boot = bootstrap_statistic(x, statistic, n_boot=n_boot, mean_block_length=block_length,
                                   alpha=alpha, seed=seed, estimate=expected, resample=transform)
        count_lower, count_upper = float(boot.lower), float(boot.upper)
        lower = length / (cadence * count_upper) if count_upper > 0 else (np.inf if count_upper == 0 else np.nan)
        upper = length / (cadence * count_lower) if count_lower > 0 else (np.inf if count_lower == 0 else np.nan)
    result = JointRarity(probability, period, lower, upper, expected, count_lower, count_upper,
                         _observed(flat, run_length)/nrep, length, max(cutoffs), alpha,
                         fit, event, run_length, period_unit, boot, fits)
    return result, {"body": body, "nonregular": nonregular}


