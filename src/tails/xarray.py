"""xarray / Dask integration: fit every grid cell or station in one call.

Wraps :func:`tails.fit_gev` with :func:`xarray.apply_ufunc` so that labelled
dimensions are preserved and Dask-backed arrays are processed chunk by chunk
(each chunk is itself a single batched JAX fit). Coefficients here include
``log_scale`` and, when requested, covariates.

For a stationary frequency analysis that follows the usual block-maximum
workflow (annual or seasonal extremes, then a return level), use
:mod:`tails.xclim`. That module reports natural parameters and the shape ξ.
"""

from __future__ import annotations

import numpy as np

from .core.distributions import gev_ppf
from .core.fit import fit_gev as _fit_gev

__all__ = ["fit_gev", "return_level"]


def _fit_block(arr, covs, method, n_out):
    shp = arr.shape[:-1]
    flat = arr.reshape(-1, arr.shape[-1])
    valid = np.isfinite(flat).sum(-1) >= 3
    out = np.full((flat.shape[0], n_out), np.nan)
    if valid.any():
        fit = _fit_gev(flat[valid], loc=covs.get("loc"), scale=covs.get("scale"), shape=covs.get("shape"), method=method)
        p = fit.coef.shape[-1]
        out[valid, :p] = fit.coef
        out[valid, p : 2 * p] = fit.se
        out[valid, 2 * p] = fit.loglik
        out[valid, 2 * p + 1] = fit.converged
    return out.reshape(*shp, n_out)


def fit_gev(da, dim: str = "time", *, loc=None, scale=None, shape=None, method: str = "mle"):
    """Fit a GEV along ``dim`` for every other coordinate of ``da``.

    Parameters
    ----------
    da
        :class:`xarray.DataArray` of block maxima (NaN = missing), possibly
        Dask-backed (chunks must not split ``dim``).
    dim
        Dimension holding the block maxima (e.g. ``"time"`` or ``"year"``).
    loc, scale, shape
        Optional covariates shared by all cells: 1-D arrays/DataArrays along
        ``dim`` or 2-D ``(n, k)``.

    Returns
    -------
    xarray.Dataset
        Variables per coefficient (``loc``, ``log_scale``, ``shape`` and slopes
        such as ``loc:x0``) plus ``<term>_se``, ``loglik`` and ``converged``.
    """
    import xarray as xr

    covs = {}
    for name, c in (("loc", loc), ("scale", scale), ("shape", shape)):
        if c is not None:
            covs[name] = np.asarray(getattr(c, "values", c), float)
    # term names from a tiny probe fit
    n = da.sizes[dim]
    probe = np.random.default_rng(0).gumbel(size=(1, n))
    names = _fit_gev(probe, **{k: v for k, v in covs.items()}, method=method).param_names
    p = len(names)
    n_out = 2 * p + 2
    res = xr.apply_ufunc(
        _fit_block,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["_out"]],
        kwargs={"covs": covs, "method": method, "n_out": n_out},
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"_out": n_out}},
    )
    ds = xr.Dataset()
    for j, nm in enumerate(names):
        ds[nm] = res.isel(_out=j)
        ds[f"{nm}_se"] = res.isel(_out=p + j)
    ds["loglik"] = res.isel(_out=2 * p)
    ds["converged"] = res.isel(_out=2 * p + 1) > 0.5
    ds.attrs["tails_model"] = "gev"
    ds.attrs["tails_method"] = method
    return ds


def return_level(ds, period):
    """Stationary GEV return levels from a :func:`fit_gev` Dataset.

    Adds a ``period`` dimension.
    """
    import xarray as xr

    T = xr.DataArray(np.atleast_1d(np.asarray(period, float)), dims="period", name="period")
    T = T.assign_coords(period=T.values)
    p = 1.0 - 1.0 / T
    return xr.apply_ufunc(
        lambda pp, m, ls, s: np.asarray(gev_ppf(pp, m, np.exp(ls), s)),
        p,
        ds["loc"],
        ds["log_scale"],
        ds["shape"],
        dask="parallelized",
        output_dtypes=[float],
    ).rename("return_level")
