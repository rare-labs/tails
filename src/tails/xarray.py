"""xarray / Dask integration: fit every grid cell or station in one call.

Wraps :func:`tails.fit_gev` and :func:`tails.idf.fit_idf` with
:func:`xarray.apply_ufunc` so that labelled dimensions are preserved and
Dask-backed arrays are processed chunk by chunk (each chunk is itself a
single batched JAX fit). GEV coefficients here include ``log_scale`` and,
when requested, covariates. IDF coefficients are the natural parameters
(``mu0``, ``sigma0``, ``xi``, ``alpha``, plus ``delta`` and correlation
terms when the model has them).

For a stationary frequency analysis inside xclim, ``import tails.xclim`` and
call ``frequency_analysis(..., backend="tails")``.
"""

from __future__ import annotations

import numpy as np

from .core.distributions import gev_ppf
from .core.fit import fit_gev as _fit_gev

__all__ = ["fit_gev", "return_level", "fit_idf", "idf_quantile"]


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


def _chunk_sizes(da, dim):
    chunks = getattr(da, "chunks", None)
    if not chunks:
        return None
    if isinstance(chunks, tuple):
        try:
            return chunks[tuple(da.dims).index(dim)]
        except (ValueError, IndexError):
            return None
    get = getattr(chunks, "get", None)
    return None if get is None else get(dim)


def _reject_split_chunks(da, dims):
    for dim in dims:
        parts = _chunk_sizes(da, dim)
        if parts is not None and len(tuple(parts)) > 1:
            raise ValueError(f"chunks must not split {dim!r}; rechunk so that dimension is a single chunk")


def _duration_hours(da, duration, durations):
    """Durations in hours, aligned with the ``duration`` axis."""
    from .datasets import DURATION_TAGS, DURATIONS_H

    if durations is not None:
        hours = np.asarray(getattr(durations, "values", durations), dtype=float)
    else:
        if duration not in getattr(da, "coords", {}):
            raise ValueError(
                f"pass durations= in hours, or set a {duration!r} coordinate to hours, "
                "a timedelta, or ECCC tags (5min, 10min, 15min, 30min, 1h, 2h, 6h, 12h, 24h)"
            )
        coord = np.asarray(da.coords[duration].values)
        # timedelta64 is a NumPy number, so it has to be handled before that branch.
        if np.issubdtype(coord.dtype, np.timedelta64):
            hours = np.asarray(coord / np.timedelta64(1, "h"), dtype=float)
        elif np.issubdtype(coord.dtype, np.number):
            hours = np.asarray(coord, dtype=float)
        else:
            tags = [str(t) for t in coord.tolist()]
            known = dict(zip(DURATION_TAGS, DURATIONS_H, strict=True))
            unknown = [t for t in tags if t not in known]
            if unknown:
                shown = ", ".join(repr(t) for t in unknown[:4])
                raise ValueError(f"duration coordinate {shown} is not hours or an ECCC tag; pass durations= in hours")
            hours = np.array([known[t] for t in tags], dtype=float)
    hours = np.asarray(hours, dtype=float).reshape(-1)
    n = int(da.sizes[duration])
    if hours.size != n:
        raise ValueError(f"durations has length {hours.size}, but dimension {duration!r} has length {n}")
    if not np.all(np.isfinite(hours) & (hours > 0)):
        raise ValueError("durations must be finite and positive, in hours")
    return hours


def _se_from_hessian(H) -> np.ndarray:
    """Standard errors from observed information. NaN where ``H`` is not usable."""
    H = np.asarray(H, dtype=float)
    se = np.full(H.shape[:2], np.nan)
    for i, Hi in enumerate(H):
        if not np.isfinite(Hi).all():
            continue
        try:
            var = np.diag(np.linalg.inv(Hi))
        except np.linalg.LinAlgError:
            continue
        scale = float(np.max(np.abs(var))) if var.size else 0.0
        if np.isfinite(var).all() and scale > 0 and float(np.min(var)) >= -1e-8 * scale:
            se[i] = np.sqrt(np.clip(var, 0, None))
    return se


def _fit_idf_block(arr, hours, marginal, correlation, copula, df, d0, max_iter, depth, n_out):
    """Fit one chunk. ``arr`` has shape ``(..., n_year, n_duration)``."""
    from .idf.models import IDFModel, _fit_arrays

    arr = np.ascontiguousarray(arr, dtype=float)
    hours = np.asarray(hours, dtype=float)
    if depth:
        arr = arr / hours
    shp = arr.shape[:-2]
    n_year, n_dur = arr.shape[-2:]
    flat = arr.reshape(-1, n_year, n_dur)
    # Two durations with a GEV-sized record: the scaling regression needs both.
    valid = (np.isfinite(flat).sum(axis=1) >= 3).sum(axis=1) >= 2
    out = np.full((flat.shape[0], n_out), np.nan)
    if valid.any():
        model = IDFModel(marginal, correlation, copula, float(df))
        Y = flat[valid]
        D = np.repeat(hours.reshape(1, -1), Y.shape[0], axis=0)
        fit = _fit_arrays(model, Y, D, float(d0), max_iter=int(max_iter))
        p = fit.params.shape[-1]
        if 2 * p + 2 != n_out:
            raise RuntimeError("IDF parameter count does not match the xarray output block")
        out[valid, :p] = fit.params
        out[valid, 2 * p] = fit.loglik
        out[valid, 2 * p + 1] = np.asarray(fit.converged, dtype=float)
        if np.any(fit.converged):
            try:
                se = _se_from_hessian(fit.hessian())
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
                se = None
            if se is not None:
                se = np.array(se, dtype=float, copy=True)
                se[~np.asarray(fit.converged, dtype=bool)] = np.nan
                out[valid, p : 2 * p] = se
    return out.reshape(*shp, n_out)


def fit_idf(
    da,
    year: str = "year",
    duration: str = "duration",
    *,
    durations=None,
    marginal: str = "general",
    correlation: str | None = None,
    copula: str = "gaussian",
    df: float = 5.0,
    d0: float = 1.0,
    depth: bool = False,
    max_iter: int = 200,
):
    """Fit an IDF model along ``year`` and ``duration`` at every other coordinate.

    This is :func:`tails.idf.fit_idf` on a labelled grid. Each chunk is one
    batched fit; cells are independent. A cell with fewer than two durations
    that each have three finite years is left as NaN (``converged`` False).

    Parameters
    ----------
    da
        :class:`xarray.DataArray` of annual-maximum intensities (NaN = missing),
        possibly Dask-backed. Chunks must not split ``year`` or ``duration``.
        With ``depth=True`` the values are depths (for example mm) and are
        divided by the duration in hours before the fit.
    year, duration
        Dimensions of the annual record and of the durations.
    durations
        Durations in hours, aligned with ``duration``. The default is the
        ``duration`` coordinate when it is numeric hours, a timedelta, or
        ECCC tags (``5min`` … ``24h``).
    marginal
        ``"simple"`` or ``"general"`` (d-GEV). Same as :func:`tails.idf.fit_idf`.
    correlation
        ``None`` (independent durations), ``"exponential"`` or ``"matern"``.
    copula
        ``"gaussian"`` or ``"t"`` (Student-t with ``df`` degrees of freedom).
        Ignored when ``correlation`` is None.
    df
        Degrees of freedom of the Student-t copula.
    d0
        Reference duration (hours) of ``mu0`` and ``sigma0``.
    depth
        If True, convert depths to intensities by dividing by duration in hours.
    max_iter
        Iteration cap for the batched optimiser.

    Returns
    -------
    xarray.Dataset
        Natural parameters (``mu0``, ``sigma0``, ``xi``, ``alpha``, and
        ``delta`` / ``nu`` / ``rho`` when the model has them), ``<name>_se``
        from the observed Hessian, ``loglik`` and ``converged``. Intensities
        at other durations and return periods are :func:`idf_quantile`.

    Examples
    --------
    >>> import tails.xarray as tx
    >>> ds = tx.fit_idf(intensity, year="year", duration="duration")
    >>> tx.idf_quantile(ds, [1.0, 24.0], [20, 100])
    """
    import xarray as xr

    from .idf.models import IDFModel

    if hasattr(da, "data_vars"):
        raise TypeError("da must be a DataArray of intensities, not a Dataset")
    if not hasattr(da, "sizes") or not hasattr(da, "dims"):
        raise TypeError("da must be an xarray.DataArray of intensities")
    if year == duration:
        raise ValueError("year and duration must be different dimensions")
    missing = [dim for dim in (year, duration) if dim not in da.dims]
    if missing:
        raise ValueError(f"missing dimension(s) {missing}; dims are {tuple(da.dims)}")
    if int(da.sizes[year]) < 3 or int(da.sizes[duration]) < 2:
        raise ValueError("an IDF fit needs at least 2 durations and a year dimension of length at least 3")
    if not np.isfinite(d0) or float(d0) <= 0:
        raise ValueError("d0 must be finite and positive, in hours")
    hours = _duration_hours(da, duration, durations)
    _reject_split_chunks(da, (year, duration))
    model = IDFModel(marginal, correlation, copula, float(df))
    names = model.names
    p = len(names)
    n_out = 2 * p + 2
    res = xr.apply_ufunc(
        _fit_idf_block,
        da,
        input_core_dims=[[year, duration]],
        output_core_dims=[["_out"]],
        kwargs={
            "hours": hours,
            "marginal": marginal,
            "correlation": correlation,
            "copula": copula,
            "df": float(df),
            "d0": float(d0),
            "max_iter": int(max_iter),
            "depth": bool(depth),
            "n_out": n_out,
        },
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
    ds.attrs["tails_model"] = "idf"
    ds.attrs["tails_marginal"] = marginal
    ds.attrs["tails_correlation"] = correlation or "independent"
    ds.attrs["tails_copula"] = copula
    ds.attrs["tails_df"] = float(df)
    ds.attrs["tails_d0"] = float(d0)
    return ds


def idf_quantile(ds, duration, period):
    """Intensities from a :func:`fit_idf` Dataset.

    The marginal scaling is the one used by :meth:`tails.idf.IDFFit.quantile`:
    location and scale at duration ``d`` are the reference values multiplied
    by ``((d + δ) / (d0 + δ)) ** -α`` (``δ = 0`` for simple scaling).
    Correlation parameters do not enter the marginal quantile.

    Parameters
    ----------
    ds
        Dataset returned by :func:`fit_idf`.
    duration
        Durations in hours. A scalar or 1-D array. Always adds a ``duration``
        dimension.
    period
        Return periods in years, each greater than 1. Always adds a ``period``
        dimension.

    Returns
    -------
    xarray.DataArray
        Named ``intensity``, with the grid dimensions of ``ds`` plus
        ``duration`` and ``period``.
    """
    import xarray as xr

    if ds.attrs.get("tails_model") != "idf" or "tails_d0" not in ds.attrs:
        raise ValueError("ds must be the Dataset returned by tails.xarray.fit_idf")
    marginal = ds.attrs.get("tails_marginal")
    if marginal not in ("simple", "general"):
        raise ValueError("ds must be the Dataset returned by tails.xarray.fit_idf")
    for name in ("mu0", "sigma0", "xi", "alpha"):
        if name not in ds:
            raise ValueError("ds must be the Dataset returned by tails.xarray.fit_idf")
    if marginal == "general" and "delta" not in ds:
        raise ValueError("ds must be the Dataset returned by tails.xarray.fit_idf")
    hours = np.atleast_1d(np.asarray(duration, dtype=float))
    T = np.atleast_1d(np.asarray(period, dtype=float))
    if hours.ndim != 1 or T.ndim != 1:
        raise ValueError("duration and period must be scalars or 1-D")
    if not np.all(np.isfinite(hours) & (hours > 0)):
        raise ValueError("duration must be finite and positive, in hours")
    if not np.all(np.isfinite(T) & (T > 1)):
        raise ValueError("return periods must be finite and greater than 1")
    d0 = float(ds.attrs["tails_d0"])
    hour = xr.DataArray(hours, dims="duration", name="duration").assign_coords(duration=hours)
    prob = xr.DataArray(1.0 - 1.0 / T, dims="period", name="period").assign_coords(period=T)
    delta = ds["delta"] if marginal == "general" else 0.0
    scale = np.exp(-ds["alpha"] * (np.log(hour + delta) - np.log(d0 + delta)))
    mu = ds["mu0"] * scale
    sigma = ds["sigma0"] * scale
    out = xr.apply_ufunc(
        lambda pp, m, sg, sh: np.asarray(gev_ppf(pp, m, sg, sh)),
        prob,
        mu,
        sigma,
        ds["xi"],
        dask="parallelized",
        output_dtypes=[float],
    )
    grid = tuple(ds["mu0"].dims)
    return out.transpose(*grid, "duration", "period").rename("intensity")
