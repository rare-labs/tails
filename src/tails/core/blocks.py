"""Block maxima, r largest values per block, declustering and extremal-index estimation."""

from __future__ import annotations

import numpy as np

__all__ = ["block_maxima", "block_top_r", "decluster_runs", "extremal_index"]


def block_maxima(series, freq: str = "YS", min_count: int | None = None):
    """Block maxima of a pandas Series / DataFrame with a DatetimeIndex.

    Parameters
    ----------
    series
        :class:`pandas.Series` or :class:`pandas.DataFrame` (one column per
        station) indexed by time, or an :class:`xarray.DataArray` with a
        ``time`` dimension.
    freq
        Block length as a pandas offset alias (``"YS"`` annual, ``"QS-DEC"``
        seasons, ``"MS"`` monthly...).
    min_count
        Blocks with fewer valid observations are set to NaN (incomplete years
        bias maxima low).
    """
    if hasattr(series, "dims"):  # xarray
        r = series.resample(time=freq)
        out = r.max()
        if min_count is not None:
            out = out.where(r.count() >= min_count)
        return out
    r = series.resample(freq)
    out = r.max()
    if min_count is not None:
        out = out.where(r.count() >= min_count)
    return out


def block_top_r(series, r: int, *, freq: str = "YS", min_separation=None, min_count: int | None = None):
    """The ``r`` largest separate events of each block of a datetime series.

    This feeds the r-largest GEV fit: ``tails.fit_gev(top.to_numpy(), r=r)``.

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
        :func:`tails.fit_gev` with ``r=r``.
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


def decluster_runs(x, threshold, run_length: int = 1):
    """Runs declustering: keep only the maximum of each cluster of exceedances.

    Exceedances separated by fewer than ``run_length`` non-exceedances belong
    to the same cluster (Smith & Weissman, 1994). Works on ``(*batch, n)``
    arrays; non-maxima are replaced by NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(threshold, float)
    ub = np.broadcast_to(u[..., None] if u.ndim else u, x.shape)
    flat, uflat = x.reshape(-1, x.shape[-1]), ub.reshape(-1, x.shape[-1])
    out = np.full_like(flat, np.nan)
    for b in range(flat.shape[0]):
        xb, exc = flat[b], np.isfinite(flat[b]) & (flat[b] > uflat[b])
        idx = np.flatnonzero(exc)
        if idx.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) > run_length) + 1
        for cl in np.split(idx, breaks):
            j = cl[np.argmax(xb[cl])]
            out[b, j] = xb[j]
    return out.reshape(x.shape)


def extremal_index(x, threshold):
    """Intervals estimator of the extremal index (Ferro & Segers, 2003).

    ``theta`` in (0, 1]; ``1/theta`` is the mean cluster size of extremes.
    Returns one value per series for ``(*batch, n)`` input.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(threshold, float)
    ub = np.broadcast_to(u[..., None] if u.ndim else u, x.shape)
    flat, uflat = x.reshape(-1, x.shape[-1]), ub.reshape(-1, x.shape[-1])
    res = np.full(flat.shape[0], np.nan)
    for b in range(flat.shape[0]):
        idx = np.flatnonzero(np.isfinite(flat[b]) & (flat[b] > uflat[b]))
        if idx.size < 3:
            continue
        T = np.diff(idx).astype(float)
        if T.max() <= 2:
            th = 2 * T.sum() ** 2 / ((T.size) * (T**2).sum())
        else:
            th = 2 * (T - 1).sum() ** 2 / (T.size * ((T - 1) * (T - 2)).sum())
        res[b] = min(1.0, th)
    return res.reshape(x.shape[:-1])
