"""Empirical bodies joined to GPD tails, optionally fitted by observed group.

Groups are supplied labels (season, shift, operating state, ...), never inferred
from the application. Each group keeps its own scale and shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import genpareto, laplace

from .core.fit import fit_gpd

__all__ = ["TailMargin", "MarginalFit", "fit_marginal"]


@dataclass
class TailMargin:
    """One empirical body / GPD tail. ``q`` is the observed mass at/below u."""

    values: np.ndarray
    threshold: float
    q: float
    scale: float
    shape: float

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        if (self.values.ndim != 1 or not len(self.values) or not np.isfinite(self.values).all()
                or np.any(np.diff(self.values) < 0)):
            raise ValueError("values must be a nonempty sorted finite series")
        if (not np.isfinite([self.threshold, self.scale, self.shape]).all()
                or self.scale <= 0 or not 0 < self.q < 1
                or not np.isclose(self.q, np.mean(self.values <= self.threshold))):
            raise ValueError("require finite parameters, positive scale and q equal to the empirical body mass")

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        body = np.searchsorted(self.values, x, side="right") / len(self.values)
        tail = self.q + (1 - self.q) * genpareto.cdf(np.maximum(x - self.threshold, 0), self.shape, scale=self.scale)
        return np.where(np.isnan(x), np.nan, np.where(x <= self.threshold, body, tail))

    def sf(self, x):
        x = np.asarray(x, dtype=float)
        body = (len(self.values) - np.searchsorted(self.values, x, side="right")) / len(self.values)
        tail = (1 - self.q) * genpareto.sf(np.maximum(x - self.threshold, 0), self.shape, scale=self.scale)
        return np.where(np.isnan(x), np.nan, np.where(x <= self.threshold, body, tail))

    def ppf(self, p):
        p = np.asarray(p, dtype=float)
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must be between zero and one")
        body_values = self.values[self.values <= self.threshold]
        # Empirical inverse, including point masses: no interpolation across ties.
        ranks = np.arange(1, len(body_values) + 1) / len(self.values)
        index = np.clip(np.searchsorted(ranks, p, side="left"), 0, len(body_values) - 1)
        body = body_values[index]
        tail = self.threshold + genpareto.isf(np.clip((1 - p) / (1 - self.q), 0, 1), self.shape, scale=self.scale)
        return np.where(np.isnan(p), np.nan, np.where(p <= self.q, body, tail))

    def to_laplace(self, x, *, midranks=False):
        x = np.asarray(x, dtype=float)
        p = self.cdf(x)
        if midranks:
            left = np.searchsorted(self.values, x, side="left")
            right = np.searchsorted(self.values, x, side="right")
            p = np.where(x <= self.threshold, (left + right) / (2 * len(self.values)), p)
        body = laplace.ppf(np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps))
        log_sf = np.log1p(-self.q) + genpareto.logsf(np.maximum(x - self.threshold, 0), self.shape, scale=self.scale)
        return np.where(x > self.threshold, -np.log(2.) - log_sf, body)

    def from_laplace(self, z):
        z = np.asarray(z, dtype=float)
        cut = -np.log(2 * (1 - self.q))
        body = self.ppf(np.minimum(laplace.cdf(z), self.q))
        survival = np.exp(-np.log(2.) - np.maximum(z, cut)) / (1 - self.q)
        tail = self.threshold + genpareto.isf(survival, self.shape, scale=self.scale)
        return np.where(z > cut, tail, body)


@dataclass
class MarginalFit:
    """One series' fitted distributions, with explicit context for grouped fits.

    ``cdf``, ``sf`` and ``ppf`` accept scalar/array values and matching ``groups``.
    These transformations describe a fitted distribution, not its uncertainty.
    """

    margins: dict
    threshold_quantile: float

    @property
    def labels(self):
        return tuple(self.margins)

    def _evaluate(self, method, x, groups):
        x = np.asarray(x, dtype=float)
        if groups is None:
            if len(self.margins) != 1:
                raise ValueError("groups must identify the context of a grouped margin")
            return getattr(next(iter(self.margins.values())), method)(x)
        x, groups = np.broadcast_arrays(x, np.asarray(groups))
        out = np.full(x.shape, np.nan)
        matched = np.zeros(x.shape, bool)
        for label, margin in self.margins.items():
            use = groups == label
            out[use] = getattr(margin, method)(x[use])
            matched |= use
        if np.any(~matched & np.isfinite(x)):
            raise ValueError("groups contain an unfitted label")
        return out

    def cdf(self, x, *, groups=None):
        return self._evaluate("cdf", x, groups)

    def sf(self, x, *, groups=None):
        return self._evaluate("sf", x, groups)

    def ppf(self, p, *, groups=None):
        return self._evaluate("ppf", p, groups)


def fit_marginal(data, *, threshold_quantile=.95, groups=None):
    """Fit an empirical body and GPD tail to one series, separately by group.

    NaNs are missing. Groups are one label per observation; each needs at least
    ten strict threshold exceedances. Group labels are retained in encounter
    order. Integer/count data require a continuous-tail approximation; the
    empirical body itself preserves ties exactly.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or np.isinf(x).any() or not .5 < threshold_quantile < 1:
        raise ValueError("data must be a one-dimensional finite/NaN series and .5 < threshold_quantile < 1")
    g = np.zeros(len(x), dtype=int) if groups is None else np.asarray(groups)
    if g.shape != x.shape:
        raise ValueError("groups must have one label per observation")
    finite = np.isfinite(x)
    if pd.isna(g[finite]).any():
        raise ValueError("groups must supply a label for every finite observation")
    labels = list(dict.fromkeys(g[finite].tolist()))
    if not labels:
        raise ValueError("no finite observations")
    samples = [np.sort(x[finite & (g == label)]) for label in labels]
    thresholds = np.array([np.quantile(a, threshold_quantile) for a in samples])
    if any(np.sum(a > u) < 10 for a, u in zip(samples, thresholds)):
        raise ValueError("each group requires at least 10 strict threshold excesses")
    padded = np.full((len(samples), max(map(len, samples))), np.nan)
    for i, a in enumerate(samples):
        padded[i, :len(a)] = a
    fit = fit_gpd(padded, threshold=thresholds)
    if not np.all(fit.converged):
        raise RuntimeError("a marginal GPD fit did not converge")
    margins = {label: TailMargin(a, float(u), float(np.mean(a <= u)), float(fit.params["scale"][i]),
                                float(fit.params["shape"][i]))
               for i, (label, a, u) in enumerate(zip(labels, samples, thresholds))}
    return MarginalFit(margins, threshold_quantile)
