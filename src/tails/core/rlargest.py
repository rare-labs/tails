"""r-largest order statistics: the GEV fitted to the top ``r`` values of each block.

Block maxima throw away everything but one value per year. When a record is
short, the second, third, ... largest values of each year carry real
information about the same tail. Under the usual extreme-value limit, the
``r`` largest values ``z_1 >= ... >= z_r`` of a block have the joint density
(Coles, 2001, section 3.5)

.. math::

    \\exp\\{-t(z_r)\\} \\prod_{k=1}^{r} \\sigma^{-1} t(z_k)^{\\xi + 1},
    \\qquad t(z) = \\{1 + \\xi (z - \\mu) / \\sigma\\}^{-1/\\xi},

with the same ``(mu, sigma, xi)`` as the GEV of the block maxima. ``r = 1`` is
exactly the GEV likelihood, so this is not a separate model: it is the GEV
family's likelihood (:mod:`tails.core._engine`) with more data per block, and
:func:`fit_rlargest` is ``fit_gev(data, r=r)``. Covariates, fixed shapes,
profile and bootstrap intervals all come with it.

Blocks may hold fewer than ``r`` values (NaN-padded): each block then uses the
values it has. The values within a block should be separate events
(declustered). :func:`block_top_r` extracts them from a datetime series with a
minimum separation between events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._jax import jnp
from ._engine import GEV
from .fit import GEVFit, _gev

__all__ = ["RLargestFit", "fit_rlargest", "rlargest_loglik", "block_top_r"]


def rlargest_loglik(z, mask, loc, scale, shape):
    """Log-likelihood of one series of r-largest blocks.

    ``z`` has shape ``(n_blocks, r)`` with each row sorted in decreasing order
    and ``mask`` marks the values present (a prefix of each row).
    """
    n = z.shape[0]
    full = lambda v: jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64), (n,))  # noqa: E731
    return jnp.sum(GEV.contrib(jnp.asarray(z), jnp.asarray(mask), full(loc), jnp.log(full(scale)), full(shape)))


@dataclass(repr=False)
class RLargestFit(GEVFit):
    """A :class:`~tails.core.fit.GEVFit` fitted with ``r`` values per block (same methods)."""


def fit_rlargest(data, r: int | None = None, *, loc=None, scale=None, shape=None, method: str = "mle",
                 prior: str | None = None, fixed_shape: float | None = None, max_iter: int = 200) -> RLargestFit:
    """Fit a GEV to the ``r`` largest values of each block: ``fit_gev(data, r=r)``.

    Parameters
    ----------
    data
        Array of shape ``(*batch, n_blocks, r_max)``: for each block (usually a
        year), its largest values. Order within a block does not matter; NaN
        pads blocks with fewer values. Values in a block should be separate
        events (see :func:`block_top_r`).
    r
        Number of order statistics to use per block (default: all ``r_max``).
        ``r = 1`` gives the same fit as :func:`tails.fit_gev` on the maxima.
    loc, scale, shape, method, prior, fixed_shape, max_iter
        As in :func:`tails.fit_gev`; covariates have one row per block.

    Returns
    -------
    RLargestFit
        A GEV fit: ``params``, ``summary()``, ``return_level(T, ci=...)`` with
        ``"delta"``, ``"profile"`` or ``"bootstrap"``, and ``plot_diagnostics()``.

    Notes
    -----
    A larger ``r`` gives narrower intervals only while the GEV limit still
    holds for the r-th largest value. Compare the estimates across ``r`` and
    keep the largest ``r`` at which they are stable (Coles, 2001, section 3.5).
    """
    if r is None:
        r = np.shape(data)[-1] if np.ndim(data) >= 2 else None
    return RLargestFit(**_gev(data, r, {"loc": loc, "scale": scale, "shape": shape}, None, method, prior, max_iter,
                              fixed_shape, rlargest_input=True))


def block_top_r(series, r: int, *, freq: str = "YS", min_separation=None, min_count: int | None = None):
    """The ``r`` largest separate events of each block of a datetime series.

    Parameters
    ----------
    series
        pandas Series with a DatetimeIndex (e.g. daily rainfall).
    r
        Number of values to keep per block.
    freq
        Block length as a pandas frequency (default ``"YS"``, calendar years).
    min_separation
        Minimum time between two kept values of the same block, as a pandas
        Timedelta or string (e.g. ``"3D"``). A value closer than this to a
        larger one already kept is skipped, so one storm counts once.
        Default: no separation.
    min_count
        Drop blocks with fewer non-missing observations than this.

    Returns
    -------
    pandas.DataFrame
        One row per block (indexed by block start) and columns ``1 .. r``
        holding the values in decreasing order, NaN where a block has fewer
        than ``r`` events. ``df.to_numpy()`` goes straight into
        :func:`fit_rlargest`.
    """
    import pandas as pd

    s = series.dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    if int(r) < 1:
        raise ValueError("r must be >= 1")
    gap = pd.Timedelta(min_separation) if min_separation is not None else None
    rows = {}
    for start, block in s.groupby(pd.Grouper(freq=freq)):
        if len(block) == 0 or (min_count is not None and len(block) < min_count):
            continue
        kept_t, kept_v = [], []
        for t, v in block.sort_values(ascending=False, kind="stable").items():
            if gap is not None and any(abs(t - k) < gap for k in kept_t):
                continue
            kept_t.append(t)
            kept_v.append(float(v))
            if len(kept_v) == r:
                break
        rows[start] = kept_v + [np.nan] * (r - len(kept_v))
    out = pd.DataFrame.from_dict(rows, orient="index", columns=list(range(1, r + 1)))
    out.index.name = "block"
    return out
