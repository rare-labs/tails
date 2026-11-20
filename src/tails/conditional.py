"""Heffernan--Tawn conditional extremes on semiparametric Laplace margins.

The working Gaussian likelihood estimates normalisations; simulation resamples
whole residual vectors, preserving dependence between the other sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._jax import jnp
from .core.fit import fit_gpd
from .core.optim import minimize
from .joint import _aligned_array, _groups_array
from .margins import TailMargin as _Margin
from .resampling import BootstrapResult, bootstrap_statistic

__all__ = ["ConditionalFit", "ConditionalQuantiles", "fit_conditional"]


def _ht_nll(th, x, y, mask):
    a = jnp.tanh(th[0])
    b = 1 - jnp.exp(th[1])
    log_power = b * jnp.log(x)
    z = (y - a * x) * jnp.exp(-log_power)
    terms = th[3] + log_power + .5 * ((z - th[2]) * jnp.exp(-th[3])) ** 2
    return jnp.sum(jnp.where(mask, terms, 0.))


@dataclass
class ConditionalQuantiles:
    """Conditional quantiles on original units, shape ``(*batch, site, quantile)``.

    Bounds, when present, are percentile pairs-bootstrap confidence intervals
    for the quantiles; they are not prediction intervals for new observations.
    """

    probability: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    alpha: float
    bootstrap: BootstrapResult | None = None


@dataclass
class ConditionalFit:
    """Conditional extremes fitted above a designated site's high threshold.

    On Laplace margins, ``Y_j = alpha_j X + X**beta_j Z_j`` for ``X > u``.
    Arrays have a final axis for the other sites, in their original order.
    ``thresholds`` are the conditioning site's thresholds in original units.
    """

    alpha: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    converged: np.ndarray
    conditioning: int
    thresholds: np.ndarray
    _margins: list[list[_Margin]]
    _residuals: list[np.ndarray]
    _data: np.ndarray
    _threshold_quantile: float
    _marginal_quantile: float
    context: object = None
    _max_iter: int = 200

    @property
    def batch_shape(self):
        return self._data.shape[:-2]

    def sample(self, size: int, *, given=None, conditions=None, seed: int = 0):
        """Draw ``(*batch, size, site)`` values given the conditioning site > given.

        ``given`` is a scalar or one threshold per batch, in original units.
        It must be at least the fitted conditioning threshold; None uses that
        threshold. Residual vectors are sampled jointly, not site by site.
        This conditions on an exceedance, not on an exact value.
        ``conditions={site_index: lower_bound}`` additionally conditions other
        sites by rejection sampling. Bounds are scalars in original units.
        A very rare combination can exhaust the sampling budget and raises.
        """
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer")
        levels = self.thresholds if given is None else np.broadcast_to(np.asarray(given, float), self.batch_shape)
        if np.any(~np.isfinite(levels)) or np.any(levels < self.thresholds):
            raise ValueError("given must be at or above the fitted conditioning threshold")
        rng = np.random.default_rng(seed)
        sites = self._data.shape[-2]
        conditions = {} if conditions is None else conditions
        if any(not isinstance(j, (int, np.integer)) or not 0 <= j < sites or not np.isscalar(v) or not np.isfinite(v)
               for j, v in conditions.items()):
            raise ValueError("conditions must map valid site indices to finite scalar bounds")
        others = [j for j in range(sites) if j != self.conditioning]
        a, b = self.alpha.reshape(-1, sites - 1), self.beta.reshape(-1, sites - 1)
        out = np.empty((len(self._margins), size, sites))
        for i, margins in enumerate(self._margins):
            v = float(margins[self.conditioning].to_laplace(np.asarray(levels).reshape(-1)[i]))
            if not np.isfinite(v):
                raise ValueError("given is outside the fitted marginal support")
            residuals = self._residuals[i]
            accepted, remaining, attempts = [], size, 0
            while remaining and attempts < 100:
                count = max(remaining, 100) if conditions else remaining
                xx = v + rng.exponential(size=count)
                z = residuals[rng.integers(0, len(residuals), count)]
                yy = a[i] * xx[:, None] + xx[:, None] ** b[i] * z
                draw = np.empty((count, sites))
                draw[:, self.conditioning] = margins[self.conditioning].from_laplace(xx)
                for j, site in enumerate(others):
                    draw[:, site] = margins[site].from_laplace(yy[:, j])
                keep = np.ones(count, bool)
                for site, bound in conditions.items():
                    keep &= draw[:, site] > bound
                draw = draw[keep][:remaining]
                accepted.append(draw)
                remaining -= len(draw)
                attempts += 1
            if remaining:
                raise RuntimeError("too few simulated events satisfy the additional conditions")
            out[i] = np.concatenate(accepted)
        return out.reshape(*self.batch_shape, size, sites)

    def quantile(self, probability=.95, *, given=None, conditions=None, size: int = 5000, n_boot: int = 0,
                 alpha: float = .05, seed: int = 0, bootstrap="iid", block_length=None) -> ConditionalQuantiles:
        """Conditional quantiles with optional full-refit iid or block intervals.

        ``bootstrap='block'`` requires a mean ``block_length`` in observations.
        Each draw refits the margins, conditional normalisations and residuals.
        The original conditioning limit stays fixed, including when given=None.
        A refit whose conditioning threshold exceeds this limit fails explicitly;
        all attempts are retained and any failure makes the bounds unavailable.
        Bounds are confidence intervals, not prediction intervals. Monte Carlo
        error depends on ``size``; increase it for tail quantiles. Neither scheme
        establishes coverage calibration for a particular application.
        """
        p = np.atleast_1d(np.asarray(probability, float))
        if p.ndim != 1 or np.any(~np.isfinite(p) | (p <= 0) | (p >= 1)) or not 0 < alpha < 1:
            raise ValueError("probabilities and alpha must lie between 0 and 1")
        if not isinstance(n_boot, (int, np.integer)) or n_boot < 0 or 0 < n_boot < 2:
            raise ValueError("n_boot must be zero or an integer at least 2")
        if bootstrap not in ("iid", "block"):
            raise ValueError("bootstrap must be 'iid' or 'block'")
        if bootstrap == "block" and (block_length is None or not np.isfinite(block_length) or block_length < 1):
            raise ValueError("block_length >= 1 is required for block bootstrap")
        estimate = np.moveaxis(np.quantile(self.sample(size, given=given, conditions=conditions, seed=seed), p, axis=-2), 0, -1)
        lower = upper = None
        boot = None
        if n_boot:
            levels = self.thresholds if given is None else np.broadcast_to(np.asarray(given, float), self.batch_shape)

            def statistic(sample):
                fitted = fit_conditional(sample, conditioning=self.conditioning,
                                         threshold_quantile=self._threshold_quantile,
                                         marginal_quantile=self._marginal_quantile, max_iter=self._max_iter)
                if np.any(levels < fitted.thresholds):
                    raise ValueError("given is below a bootstrap conditioning threshold")
                draws = fitted.sample(size, given=levels, conditions=conditions, seed=seed + 1)
                value = np.moveaxis(np.quantile(draws, p, axis=-2), 0, -1)
                nonregular = any(m.shape <= -.5 for margins in fitted._margins for m in margins)
                return value, {"nonregular": nonregular}

            boot = bootstrap_statistic(self._data, statistic, estimate=estimate, n_boot=n_boot,
                                       mean_block_length=1 if bootstrap == "iid" else block_length,
                                       alpha=alpha, seed=seed)
            lower, upper = boot.lower, boot.upper
        return ConditionalQuantiles(p, estimate, lower, upper, alpha, boot)


def fit_conditional(data, *, conditioning: int = 0, threshold_quantile: float = .95,
                    marginal_quantile: float = .9, max_iter: int = 200,
                    groups=None, context=None, freq=None) -> ConditionalFit:
    """Fit batched Heffernan--Tawn conditional extremes for aligned series.

    Arrays are ``(*batch, site, time)``; DataFrame columns are sites. Unlike
    :func:`tails.how_rare_joint`, leading axes are separate fits, not pooled
    replicates. GPD upper tails and empirical bodies transform each site to
    Laplace margins. Gaussian working likelihoods fit ``alpha in (-1,1)`` and
    ``beta < 1`` with JAX, using three starting values. Simulation uses the
    empirical joint residual distribution. At least 20 complete conditioning
    exceedances per fit are required. Marginal and conditional stationarity
    are assumed. Exact perfect dependence is a degenerate model, unsupported.

    To fit one observed operating regime, supply both ``groups`` and ``context``.
    Other contexts become missing vectors without compressing the time axis;
    bootstrap blocks cannot join observations across excluded contexts. Each
    context is a separate conditional model, not an assumption that dependence
    stays invariant between regimes. Datetime gaps are preserved; supply ``freq``
    when a subdaily cadence cannot be inferred.

    Reference: Heffernan & Tawn (2004), JRSS B 66, 497--546,
    https://doi.org/10.1111/j.1467-9868.2004.02050.x .
    """
    # Conditional likelihoods use aligned vectors, not daily windows.
    raw = data
    data = _aligned_array(data, freq=freq)
    if (groups is None) != (context is None):
        raise ValueError("groups and context must be supplied together")
    if groups is not None:
        labels = _groups_array(raw, data, groups, freq)
        if pd.isna(labels[np.isfinite(data).all(axis=-2)]).any():
            raise ValueError("groups must supply a label for every complete observation")
        data = np.where((labels == context)[..., None, :], data, np.nan)
    sites = data.shape[-2]
    if not isinstance(conditioning, (int, np.integer)) or not 0 <= conditioning < sites or sites < 2:
        raise ValueError("conditioning must index a site, and at least two sites are required")
    if not .5 < marginal_quantile <= threshold_quantile < 1:
        raise ValueError("require .5 < marginal_quantile <= threshold_quantile < 1")
    flat = data.reshape(-1, sites, data.shape[-1])
    margins, pairs, thresholds = [], [], []
    for row in flat:
        complete = row[:, np.isfinite(row).all(axis=0)]
        if complete.shape[-1] < 40:
            raise ValueError("too few complete observations")
        us = np.quantile(complete, marginal_quantile, axis=-1)
        if np.any((complete > us[:, None]).sum(axis=-1) < 10):
            raise ValueError("at least 10 marginal threshold excesses per site are required")
        gpd = fit_gpd(complete, threshold=us)
        if not np.all(gpd.converged):
            raise RuntimeError("a conditional-model marginal GPD fit did not converge")
        masses = (complete <= us[:, None]).mean(axis=-1)
        ms = [_Margin(np.sort(complete[j]), us[j], masses[j], float(gpd.params["scale"][j]),
                      float(gpd.params["shape"][j])) for j in range(sites)]
        yy = np.stack([ms[j].to_laplace(complete[j], midranks=True) for j in range(sites)])
        threshold = ms[conditioning].ppf(threshold_quantile)
        use = complete[conditioning] > threshold
        if use.sum() < 20:
            raise ValueError("at least 20 complete conditioning threshold exceedances are required")
        if any(np.allclose(yy[j, use], yy[conditioning, use], rtol=0, atol=1e-7) for j in range(sites) if j != conditioning):
            raise ValueError("perfect dependence has a degenerate conditional likelihood")
        margins.append(ms)
        pairs.append(yy[:, use])
        thresholds.append(threshold)
    others = [j for j in range(sites) if j != conditioning]
    # A small number of padded widths avoids recompiling JAX for every slightly
    # different exceedance count in full-refit bootstrap draws.
    width = 1 << (max(p.shape[-1] for p in pairs) - 1).bit_length()
    xx = np.ones((len(flat) * (sites - 1), width))
    yy = np.zeros_like(xx)
    mask = np.zeros_like(xx, dtype=bool)
    for i, pair in enumerate(pairs):
        for j, site in enumerate(others):
            idx = i * (sites - 1) + j
            xx[idx, :pair.shape[-1]] = pair[conditioning]
            yy[idx, :pair.shape[-1]] = pair[site]
            mask[idx, :pair.shape[-1]] = True
    starts = []
    for beta in (0., .5, -.5):
        initial = []
        for x, y, m in zip(xx, yy, mask):
            a = np.clip(np.cov(x[m], y[m])[0, 1] / np.var(x[m]), -.9, .9)
            z = (y[m] - a * x[m]) / x[m] ** beta
            initial.append([np.arctanh(a), np.log(1 - beta), z.mean(), np.log(max(z.std(), .05))])
        starts.append(initial)
    count = xx.shape[0]
    result = minimize(_ht_nll, np.concatenate(starts), jnp.asarray(np.tile(xx, (3, 1))),
                      jnp.asarray(np.tile(yy, (3, 1))), jnp.asarray(np.tile(mask, (3, 1))), max_iter=max_iter)
    loss = np.asarray(result.fun).reshape(3, count)
    conv = np.asarray(result.converged).reshape(3, count)
    best = np.argmin(np.where(conv, loss, np.inf), axis=0)
    if not np.all(conv[best, np.arange(count)]):
        raise RuntimeError("a conditional normalisation fit did not converge")
    params = np.asarray(result.x).reshape(3, count, 4)[best, np.arange(count)]
    a, b = np.tanh(params[:, 0]), 1 - np.exp(params[:, 1])
    residuals = []
    for i, pair in enumerate(pairs):
        sl = slice(i * (sites - 1), (i + 1) * (sites - 1))
        residuals.append(((pair[others] - a[sl, None] * pair[conditioning]) / pair[conditioning] ** b[sl, None]).T)
    shape = (*data.shape[:-2], sites - 1)
    return ConditionalFit(a.reshape(shape), b.reshape(shape), params[:, 2].reshape(shape), np.exp(params[:, 3]).reshape(shape),
                          np.ones(shape, bool), conditioning, np.asarray(thresholds).reshape(data.shape[:-2]),
                          margins, residuals, data, threshold_quantile, marginal_quantile, context, max_iter)
