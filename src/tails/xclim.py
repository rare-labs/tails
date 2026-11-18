"""Frequency analysis for xarray data, in the shape of an xclim workflow.

``fit``, ``fa`` and ``frequency_analysis`` do the same jobs as the frequency
analysis in ``xclim`` (fit a distribution along one dimension, then read off
a return level). The fits are tails' batched JAX maximum likelihood, one
chunk at a time, rather than a Python loop of scipy fits.

The shape parameter is tails' ξ (Coles / extRemes): ``ξ > 0`` is a heavy
tail. For the GEV that is the opposite of ``scipy.stats.genextreme``'s ``c``.
These parameters are not a drop-in for ``xclim``'s ``parametric_quantile``.
Use :func:`parametric_quantile` here, or :func:`as_scipy_params` when a
scipy ``c`` vector is required.

Limits, compared with ``xclim.indices.stats`` and with :mod:`tails.xarray`:

- Stationary GEV (``genextreme``) and GPD (``genpareto``) only. Covariates
  stay on :func:`tails.fit_gev` and :func:`tails.xarray.fit_gev`.
- Methods are maximum likelihood (``ML``), L-moments (``PWM``) and GMLE.
  Method of moments and maximum spacing are not implemented.
- A Gumbel model is the GEV with ξ fixed at 0, which this backend does not
  constrain.
- Minima are fit as a GEV on the negated sample, and the quantile is negated
  back. That is the block-minimum model. It is not a GEV fit directly to the
  minima.
- Confidence intervals (``delta``, ``profile``, ``bootstrap``) are GEV only.
  A GPD quantile here is a quantile of the fitted sample, not a
  peaks-over-threshold annual return level; use :func:`tails.fit_pot` for
  that. ``threshold`` must be a scalar.
- ``frequency_analysis`` indexes ``month`` and meteorological ``season``
  (DJF, MAM, JJA, SON). The record dimension needs a datetime coordinate.
- Series with fewer than 3 finite values (GEV) or 2 (GPD) come back as NaN.
  A short series that does fit can still have a wild shape; check ``nobs``
  and ``shape`` before quoting a level.
- ``fit`` and ``fa`` keep dask chunks (the record dimension is rechunked
  into one piece). ``parametric_quantile`` and ``as_scipy_params`` return
  in-memory arrays.
"""

from __future__ import annotations

import numpy as np

from .core.distributions import gev_ppf, gpd_ppf
from .core.fit import fit_gev, fit_gpd

__all__ = [
    "as_scipy_params",
    "block_extremes",
    "fa",
    "fit",
    "frequency_analysis",
    "parametric_quantile",
]

_SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
_METHODS = {"ML": "mle", "MLE": "mle", "PWM": "lmom", "LM": "lmom", "LMOM": "lmom", "GMLE": "gmle"}
_METHOD_INFO = {
    "mle": ("ML", "maximum likelihood"),
    "lmom": ("PWM", "L-moments"),
    "gmle": ("GMLE", "generalised maximum likelihood"),
}
_DPARAMS = ("loc", "scale", "shape")
_MIN_N = {"gev": 3, "gpd": 2}


def _xr():
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "tails.xclim needs xarray. Install it with pip install 'rare-tails[xarray]'."
        ) from exc
    return xr


def _family(dist) -> str:
    name = getattr(dist, "name", dist)
    if not isinstance(name, str):
        raise TypeError(f"distribution must be a name or a scipy distribution, got {type(dist).__name__}")
    key = name.strip().lower()
    if key in {"gev", "genextreme"}:
        return "gev"
    if key in {"gpd", "genpareto"}:
        return "gpd"
    if key in {"gumbel", "gumbel_r", "gumbel_l"}:
        raise ValueError(
            "A Gumbel model is the GEV with shape ξ fixed at 0, which tails.xclim does not constrain. "
            "Fit dist='genextreme' and read the shape estimate."
        )
    raise ValueError(f"unsupported distribution {dist!r}; use 'genextreme' (GEV) or 'genpareto' (GPD)")


def _method(method) -> str:
    key = str(method).strip().upper()
    if key not in _METHODS:
        raise ValueError(
            f"fitting method {method!r} is not available. "
            "Use 'ML' (maximum likelihood), 'PWM' (L-moments) or 'GMLE'. "
            "Method of moments and maximum spacing are not implemented."
        )
    return _METHODS[key]


def _mode(mode) -> str:
    key = str(mode).strip().lower()
    if key in {"max", "high"}:
        return "max"
    if key in {"min", "low"}:
        return "min"
    raise ValueError("mode must be 'max' or 'min'")


def _periods(t) -> np.ndarray:
    if isinstance(t, str):
        raise TypeError("return period must be a number or a sequence of numbers")
    periods = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if periods.ndim != 1 or periods.size == 0 or not np.all(np.isfinite(periods)) or np.any(periods <= 1):
        raise ValueError("return periods must be finite numbers greater than 1")
    return periods


def _quantiles(q) -> np.ndarray:
    if isinstance(q, str):
        raise TypeError("quantile must be a number or a sequence of numbers")
    quantiles = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if quantiles.ndim != 1 or quantiles.size == 0 or np.any((quantiles <= 0) | (quantiles >= 1)):
        raise ValueError("quantiles must be a finite sequence in the open interval (0, 1)")
    return quantiles


def _ci(ci, family):
    if ci is None:
        return None
    if ci not in {"delta", "profile", "bootstrap"}:
        raise ValueError("ci must be None, 'delta', 'profile' or 'bootstrap'")
    if family != "gev":
        raise ValueError("confidence intervals in tails.xclim are implemented for the GEV only")
    return ci


def _scalar_threshold(threshold):
    if threshold is None:
        return None
    arr = np.asarray(threshold, dtype=np.float64)
    if arr.ndim != 0:
        raise ValueError("threshold must be a scalar; subtract a varying threshold before fitting")
    if not np.isfinite(arr):
        raise ValueError("threshold must be finite")
    return float(arr)


def _check_da(da, dim):
    xr = _xr()
    if not isinstance(da, xr.DataArray):
        raise TypeError("expected an xarray.DataArray")
    if dim not in da.dims:
        raise ValueError(f"dimension {dim!r} is not on the array (dims: {tuple(da.dims)})")
    if da.sizes[dim] < 1:
        raise ValueError(f"dimension {dim!r} is empty")
    return da


def _n_chunks(part) -> int:
    if isinstance(part, int):
        return 1
    return len(tuple(part))


def _along_full_chunks(da, dim):
    chunks = getattr(da, "chunksizes", None) or da.chunks
    if not chunks or dim not in chunks:
        return da
    if _n_chunks(chunks[dim]) > 1:
        return da.chunk({dim: -1})
    return da


def _prepare_sample(flat, family, threshold, mode):
    """Return ``(sample, location)`` ready for a stationary fit.

    Minima are negated, so the fit is a GEV for maxima of ``-X``. A scalar
    GPD threshold is subtracted and non-exceedances become missing.
    """
    data = np.array(flat, dtype=np.float64, copy=True)
    if mode == "min":
        data = -data
    loc = 0.0
    if family == "gpd" and threshold is not None:
        loc = float(threshold)
        data = np.where(np.isfinite(data) & (data > threshold), data - threshold, np.nan)
    return data, loc


def _fit_rows(sample, family, method):
    if family == "gev":
        return fit_gev(sample, method=method)
    return fit_gpd(sample, method=method)


def _fit_block(arr, *, family, method, threshold):
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, np.shape(arr)[-1])
    sample, loc = _prepare_sample(flat, family, threshold, "max")
    counts = np.isfinite(sample).sum(-1)
    valid = counts >= _MIN_N[family]
    out = np.full((flat.shape[0], 9), np.nan, dtype=np.float64)
    out[:, 7] = 0.0
    out[:, 8] = counts
    if np.any(valid):
        fitted = _fit_rows(sample[valid], family, method)
        scale = np.asarray(fitted.params["scale"], dtype=np.float64).reshape(-1)
        shape = np.asarray(fitted.params["shape"], dtype=np.float64).reshape(-1)
        se = np.asarray(fitted.se, dtype=np.float64)
        if family == "gev":
            out[valid, 0] = np.asarray(fitted.params["loc"], dtype=np.float64).reshape(-1)
            se = se.reshape(-1, 3)
            out[valid, 3] = se[:, 0]
            out[valid, 4] = scale * se[:, 1]
            out[valid, 5] = se[:, 2]
        else:
            out[valid, 0] = loc
            se = se.reshape(-1, 2)
            out[valid, 4] = scale * se[:, 0]
            out[valid, 5] = se[:, 1]
        out[valid, 1] = scale
        out[valid, 2] = shape
        out[valid, 6] = np.asarray(fitted.loglik, dtype=np.float64).reshape(-1)
        out[valid, 7] = np.asarray(fitted.converged, dtype=np.float64).reshape(-1)
    return out.reshape(*np.shape(arr)[:-1], 9)


def _fa_block(arr, *, periods, family, method, mode, ci, alpha, n_boot, seed, threshold):
    periods = np.atleast_1d(np.asarray(periods, dtype=np.float64))
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, np.shape(arr)[-1])
    sample, loc = _prepare_sample(flat, family, threshold, mode)
    valid = np.isfinite(sample).sum(-1) >= _MIN_N[family]
    n_t = periods.size
    est = np.full((flat.shape[0], n_t), np.nan, dtype=np.float64)
    lo = hi = None
    if ci is not None:
        lo = np.full_like(est, np.nan)
        hi = np.full_like(est, np.nan)
    if np.any(valid):
        rows = sample[valid]
        n_valid = rows.shape[0]
        sign = -1.0 if mode == "min" else 1.0
        if family == "gev":
            fitted = fit_gev(rows, method=method)
            rl = fitted.return_level(periods, ci=ci, alpha=alpha, n_boot=n_boot, seed=seed)
            point = sign * np.asarray(rl.estimate, dtype=np.float64).reshape(n_valid, n_t)
            est[valid] = point
            if ci is not None:
                lower = np.asarray(rl.lower, dtype=np.float64).reshape(n_valid, n_t)
                upper = np.asarray(rl.upper, dtype=np.float64).reshape(n_valid, n_t)
                if sign < 0:
                    lower, upper = -upper, -lower
                lo[valid] = lower
                hi[valid] = upper
        else:
            fitted = fit_gpd(rows, method=method)
            scale = np.asarray(fitted.params["scale"], dtype=np.float64).reshape(-1)
            shape = np.asarray(fitted.params["shape"], dtype=np.float64).reshape(-1)
            q = 1.0 - 1.0 / periods
            est[valid] = np.asarray(gpd_ppf(q, scale[:, None], shape[:, None]), dtype=np.float64) + loc
    shp = np.shape(arr)[:-1]
    est = est.reshape(*shp, n_t)
    if ci is None:
        return est
    return np.stack([est, lo.reshape(*shp, n_t), hi.reshape(*shp, n_t)], axis=-1)


def _restore(res, dims, dim, core_name, extra=()):
    order = [core_name if d == dim else d for d in dims]
    return res.transpose(*order, *extra)


def _require_datetime(da, dim):
    try:
        months = da[dim].dt.month
    except AttributeError as exc:
        raise ValueError(
            f"dimension {dim!r} has no datetime coordinate. "
            "frequency_analysis needs datetimes along that dimension; pass block maxima to fa()."
        ) from exc
    if dim not in months.dims or months.sizes[dim] != da.sizes[dim]:
        raise ValueError(f"dimension {dim!r} has no datetime coordinate")


def _months(indexer) -> tuple[int, ...] | None:
    if not indexer:
        return None
    unknown = sorted(set(indexer) - {"month", "season"})
    if unknown:
        raise ValueError(
            f"unsupported indexer {unknown}; tails.xclim accepts 'month' and 'season' (DJF, MAM, JJA, SON)"
        )
    if "month" in indexer and "season" in indexer:
        raise ValueError("pass either month or season, not both")
    if "season" in indexer:
        key = str(indexer["season"]).strip().upper()
        if key not in _SEASONS:
            raise ValueError(f"season must be one of {', '.join(_SEASONS)}")
        return _SEASONS[key]
    month = np.atleast_1d(np.asarray(indexer["month"], dtype=int))
    if month.ndim != 1 or month.size == 0 or np.any((month < 1) | (month > 12)):
        raise ValueError("month must be in 1..12")
    return tuple(int(m) for m in month)


def _as_params(p):
    xr = _xr()
    if isinstance(p, xr.Dataset):
        if "params" not in p:
            raise ValueError("dataset has no 'params' variable; pass a tails.xclim.fit result")
        p = p["params"]
    if not isinstance(p, xr.DataArray) or "dparams" not in p.dims:
        raise TypeError("expected the dataset returned by tails.xclim.fit, or its 'params' array")
    names = [str(v) for v in np.asarray(p.coords["dparams"].values)]
    missing = [name for name in _DPARAMS if name not in names]
    if missing:
        raise ValueError(f"parameter array is missing {missing}; expected loc, scale, shape")
    return p


def _family_of(params, dist) -> str:
    if dist is not None:
        return _family(dist)
    family = params.attrs.get("tails_family")
    if family not in {"gev", "gpd"}:
        raise ValueError("pass dist, or a parameters array produced by tails.xclim.fit")
    return family


def _numpy(da) -> np.ndarray:
    data = da.data
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data, dtype=np.float64)


def _level_attrs(src, family, method, mode):
    label, _estimator = _METHOD_INFO[method]
    attrs = {
        "long_name": "return level",
        "tails_family": family,
        "method": label,
        "tails_method": method,
        "shape_convention": "xi",
        "mode": mode,
    }
    if "units" in getattr(src, "attrs", {}):
        attrs["units"] = src.attrs["units"]
    return attrs


def _finish_level(src, res, dim, periods, family, method, mode):
    res = _restore(res, src.dims, dim, "return_period")
    res = res.assign_coords(return_period=("return_period", np.asarray(periods, dtype=np.float64)))
    res.name = "return_level"
    res.attrs.update(_level_attrs(src, family, method, mode))
    return res


def fit(da, dist="genextreme", method="ML", dim="time", *, threshold=None):
    """Fit a stationary GEV or GPD along ``dim``.

    Parameters
    ----------
    da
        :class:`xarray.DataArray`. ``dim`` is the record (block maxima, or
        excesses). Every other dimension is a batch. NaN is a missing value.
        Dask chunks along ``dim`` are joined before the fit.
    dist
        ``"genextreme"`` / ``"gev"`` or ``"genpareto"`` / ``"gpd"``. A scipy
        distribution with one of those names is accepted too.
    method
        ``"ML"`` (maximum likelihood), ``"PWM"`` (L-moments, tails'
        ``method="lmom"``) or ``"GMLE"``.
    dim
        Record dimension. It does not have to be named ``"time"`` and it does
        not have to be last.
    threshold
        Scalar threshold, GPD only. Observations at or below it are dropped
        and the reported location is this threshold. Without it, GPD ``loc``
        is 0 and ``da`` is treated as excesses.

    Returns
    -------
    xarray.Dataset
        ``params`` and ``se`` with a ``dparams`` coordinate ``loc``, ``scale``,
        ``shape`` (shape is ξ), plus ``loglik``, ``converged`` and ``nobs``.
        The scale standard error is the delta-method error of the natural
        scale. L-moment fits leave the Hessian, and therefore ``se``, as NaN.
        The GPD location is the threshold you supplied, not an estimated
        parameter, so its standard error is NaN.

    Notes
    -----
    A cell with fewer than 3 finite GEV observations, or fewer than 2 GPD
    excesses, is returned as NaN rather than fit. Cells that do fit can still
    be too short to quote: read ``nobs`` and ``shape`` first.
    """
    da = _check_da(da, dim)
    family = _family(dist)
    tails_method = _method(method)
    threshold = _scalar_threshold(threshold)
    if threshold is not None and family != "gpd":
        raise ValueError("threshold is only used with dist='genpareto'")
    xr = _xr()
    label, estimator = _METHOD_INFO[tails_method]
    res = xr.apply_ufunc(
        _fit_block,
        _along_full_chunks(da, dim),
        input_core_dims=[[dim]],
        output_core_dims=[["_out"]],
        kwargs={"family": family, "method": tails_method, "threshold": threshold},
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"_out": 9}},
    )
    res = _restore(res, da.dims, dim, "_out")
    params = res.isel(_out=slice(0, 3)).rename({"_out": "dparams"})
    se = res.isel(_out=slice(3, 6)).rename({"_out": "dparams"})
    params = params.assign_coords(dparams=list(_DPARAMS))
    se = se.assign_coords(dparams=list(_DPARAMS))
    attrs = {
        "tails_family": family,
        "method": label,
        "tails_method": tails_method,
        "estimator": estimator,
        "shape_convention": "xi",
        "dim": dim,
    }
    params.attrs.update(attrs)
    se.attrs["description"] = (
        "Standard error of location and shape. The scale entry is the delta-method "
        "standard error of the natural scale. NaN for L-moments, and for a GPD location."
    )
    ds = xr.Dataset(
        {
            "params": params,
            "se": se,
            "loglik": res.isel(_out=6, drop=True),
            "converged": res.isel(_out=7, drop=True) > 0.5,
            "nobs": res.isel(_out=8, drop=True),
        }
    )
    ds.attrs.update(attrs)
    if threshold is not None:
        ds.attrs["threshold"] = threshold
    return ds


def parametric_quantile(p, q, dist=None):
    """Quantile of a :func:`fit` at non-exceedance probabilities ``q``.

    Parameters
    ----------
    p
        Dataset from :func:`fit`, or its ``params`` array. ``shape`` is ξ.
    q
        Probability or probabilities, each strictly between 0 and 1.
    dist
        Optional ``"genextreme"`` or ``"genpareto"`` when ``p`` does not carry
        ``tails_family``.

    Returns
    -------
    xarray.DataArray
        In-memory array with a ``quantile`` dimension. GPD quantiles are on
        the original scale (excess quantile plus the location stored on ``p``).
    """
    xr = _xr()
    params = _as_params(p)
    family = _family_of(params, dist)
    quantiles = _quantiles(q)
    loc = _numpy(params.sel(dparams="loc"))
    scale = _numpy(params.sel(dparams="scale"))
    shape = _numpy(params.sel(dparams="shape"))
    qq = quantiles.reshape(*([1] * loc.ndim), -1)
    if family == "gev":
        vals = gev_ppf(qq, loc[..., None], scale[..., None], shape[..., None])
    else:
        vals = gpd_ppf(qq, scale[..., None], shape[..., None]) + loc[..., None]
    vals = np.asarray(vals, dtype=np.float64)
    base = params.sel(dparams="loc")
    coords = {name: coord for name, coord in base.coords.items() if "dparams" not in coord.dims}
    out = xr.DataArray(vals, dims=[*base.dims, "quantile"], coords=coords)
    out = out.assign_coords(quantile=("quantile", quantiles))
    order = ["quantile" if d == "dparams" else d for d in params.dims]
    out = out.transpose(*order)
    out.name = "quantile"
    out.attrs["tails_family"] = family
    out.attrs["shape_convention"] = "xi"
    return out


def as_scipy_params(p):
    """Copy fitted parameters into scipy's ``(c, loc, scale)`` vector.

    GEV (``genextreme``): ``c = -ξ``. GPD (``genpareto``): ``c = ξ``. The
    result is what scipy and ``xclim``'s ``parametric_quantile`` expect. It
    is not a parameter vector for :func:`tails.fit_gev`.

    Returns an in-memory :class:`xarray.DataArray` with ``dparams`` ordered
    ``c``, ``loc``, ``scale`` and attribute ``scipy_dist``.
    """
    xr = _xr()
    params = _as_params(p)
    family = _family_of(params, None)
    loc = _numpy(params.sel(dparams="loc"))
    scale = _numpy(params.sel(dparams="scale"))
    shape = _numpy(params.sel(dparams="shape"))
    c = -shape if family == "gev" else shape
    vals = np.stack([c, loc, scale], axis=-1)
    base = params.sel(dparams="loc")
    coords = {name: coord for name, coord in base.coords.items() if "dparams" not in coord.dims}
    coords["dparams"] = ["c", "loc", "scale"]
    out = xr.DataArray(vals, dims=[*base.dims, "dparams"], coords=coords)
    order = ["dparams" if d == "dparams" else d for d in params.dims]
    out = out.transpose(*order)
    out.attrs["tails_family"] = family
    out.attrs["scipy_dist"] = "genextreme" if family == "gev" else "genpareto"
    out.attrs["shape_convention"] = "scipy_c"
    out.attrs["description"] = "c = -ξ for genextreme; c = ξ for genpareto"
    return out


def fa(
    da,
    t,
    dist="genextreme",
    mode="max",
    method="ML",
    *,
    dim="time",
    ci=None,
    alpha=0.05,
    n_boot=500,
    seed=0,
    threshold=None,
):
    """Return level of a sample already reduced to one value per block.

    Parameters
    ----------
    da
        Block maxima or minima, or GPD excesses, as an :class:`xarray.DataArray`.
    t
        Return period or periods, each greater than 1. The unit is the block
        of ``da`` (years, when ``da`` is annual).
    dist, method, dim, threshold
        As for :func:`fit`.
    mode
        ``"max"`` (exceedance, quantile ``1 - 1/T``) or ``"min"``. Minima are
        fit on the negated sample. ``"min"`` is GEV only.
    ci
        ``None``, or ``"delta"``, ``"profile"`` or ``"bootstrap"`` for a GEV
        interval at level ``1 - alpha``. Bootstrap uses ``n_boot`` replicates
        and ``seed``.
    alpha
        Tail probability of the interval. Ignored when ``ci`` is ``None``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        A ``return_level`` array with a ``return_period`` dimension. With
        ``ci`` set, a dataset of ``return_level``, ``lower`` and ``upper``.
    """
    da = _check_da(da, dim)
    family = _family(dist)
    tails_method = _method(method)
    mode = _mode(mode)
    periods = _periods(t)
    ci = _ci(ci, family)
    threshold = _scalar_threshold(threshold)
    if threshold is not None and family != "gpd":
        raise ValueError("threshold is only used with dist='genpareto'")
    if mode == "min" and family != "gev":
        raise ValueError("mode 'min' is implemented for the GEV only")
    if ci == "bootstrap" and int(n_boot) < 2:
        raise ValueError("n_boot must be at least 2")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    xr = _xr()
    kwargs = {
        "periods": periods,
        "family": family,
        "method": tails_method,
        "mode": mode,
        "ci": ci,
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "threshold": threshold,
    }
    da_fit = _along_full_chunks(da, dim)
    if ci is None:
        res = xr.apply_ufunc(
            _fa_block,
            da_fit,
            input_core_dims=[[dim]],
            output_core_dims=[["return_period"]],
            kwargs=kwargs,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {"return_period": periods.size}},
        )
        return _finish_level(da, res, dim, periods, family, tails_method, mode)
    res = xr.apply_ufunc(
        _fa_block,
        da_fit,
        input_core_dims=[[dim]],
        output_core_dims=[["return_period", "bound"]],
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"return_period": periods.size, "bound": 3}},
    )
    res = _restore(res, da.dims, dim, "return_period", extra=("bound",))
    res = res.assign_coords(return_period=("return_period", periods))
    attrs = _level_attrs(da, family, tails_method, mode)
    attrs["ci"] = ci
    attrs["alpha"] = float(alpha)
    names = ("return_level", "lower", "upper")
    data = {name: res.isel(bound=i, drop=True) for i, name in enumerate(names)}
    for name in names:
        data[name].name = name
        data[name].attrs.update(attrs)
    ds = xr.Dataset(data)
    ds.attrs.update(attrs)
    return ds


def block_extremes(da, mode, *, window=1, freq=None, dim="time", **indexer):
    """Block maxima or minima along a datetime dimension.

    This is the reduction :func:`frequency_analysis` does before fitting.
    ``window > 1`` is a rolling mean (a window that contains a NaN becomes
    NaN), then the series is subset by ``month`` or ``season`` and reduced
    with ``max`` or ``min``.

    ``freq`` defaults to ``"YS"``, or ``"YS-DEC"`` when ``season="DJF"``, so
    December stays with the following January and February. Years with no
    finite values are NaN and are ignored by the fit.

    Parameters
    ----------
    da
        DataArray with a datetime coordinate on ``dim``.
    mode
        ``"max"`` or ``"min"``.
    window
        Rolling-mean length in steps of ``dim``, before the block reduction.
    freq
        Resample offset. ``None`` selects the default above.
    dim
        Datetime dimension.
    **indexer
        ``month`` (an integer or a sequence in 1..12) or ``season``
        (``DJF``, ``MAM``, ``JJA``, ``SON``).
    """
    da = _check_da(da, dim)
    mode = _mode(mode)
    _require_datetime(da, dim)
    if int(window) != window or int(window) < 1:
        raise ValueError("window must be an integer >= 1")
    window = int(window)
    months = _months(indexer)
    if freq is None:
        season = str(indexer.get("season", "")).strip().upper() if indexer else ""
        freq = "YS-DEC" if season == "DJF" else "YS"
    if window > 1:
        da = da.rolling({dim: window}).mean(skipna=False)
    if months is not None:
        da = da.sel({dim: da[dim].dt.month.isin(list(months))})
        if da.sizes[dim] < 1:
            raise ValueError(f"no observations left along {dim!r} after applying the indexer")
    reducer = "max" if mode == "max" else "min"
    grouped = da.resample({dim: freq})
    fn = getattr(grouped, reducer)
    try:
        out = fn(skipna=True)
    except TypeError:
        out = fn()
    out.attrs.update(da.attrs)
    out.attrs["tails_block"] = mode
    out.attrs["tails_freq"] = str(freq)
    return out


def frequency_analysis(
    da,
    mode,
    t,
    dist="genextreme",
    window=1,
    freq=None,
    method="ML",
    *,
    dim="time",
    ci=None,
    alpha=0.05,
    n_boot=500,
    seed=0,
    threshold=None,
    **indexer,
):
    """Block extremes of ``da``, then the return level of that sample.

    Positional arguments match the usual ``xclim`` call:
    ``frequency_analysis(da, "max", [20, 100], "genextreme")``. ``dim`` is
    the datetime record. ``month`` and ``season`` are the only indexers.
    See :func:`block_extremes` and :func:`fa` for the reduction and the fit.

    The GEV shape on the result is ξ, not scipy's ``c``. Intervals are the
    ``ci`` argument of :func:`fa`.
    """
    blocks = block_extremes(da, mode, window=window, freq=freq, dim=dim, **indexer)
    return fa(
        blocks,
        t,
        dist=dist,
        mode=mode,
        method=method,
        dim=dim,
        ci=ci,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        threshold=threshold,
    )
