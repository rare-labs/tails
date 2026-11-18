"""Register tails as a backend for xclim frequency analysis.

Import this module before the xclim functions you want to call. On xclim 1.0
and later those functions live in ``xclim.compute.stats``. On xclim 0.x they
live in ``xclim.indices.stats``. Both modules are patched when they exist.
A name that was imported earlier keeps the function it was bound to, so
import ``tails.xclim`` first:

>>> import tails.xclim
>>> from xclim.compute.stats import frequency_analysis, fit
>>> frequency_analysis(da, "max", [20, 100], "genextreme", backend="tails", ci="profile")
>>> fit(da, dist="genextreme", method="ML", backend="tails")

``backend="tails"`` fits with tails. Leaving ``backend`` out, or passing
``"scipy"`` or ``"xclim"``, calls the original function.

``fit`` returns the parameter array ``parametric_quantile`` already accepts:
dimension ``dparams`` ordered ``c``, ``loc``, ``scale``, and attribute
``scipy_dist``. For ``genextreme``, ``c = -ξ``. For ``genpareto``, ``c = ξ``.

``fa`` and ``frequency_analysis`` return a DataArray of return levels. With
``ci="delta"``, ``"profile"`` or ``"bootstrap"`` on a GEV they return a
Dataset with variables ``return_level``, ``lower`` and ``upper``.

An entry point ``xclim.stats.backends`` → ``tails.xclim:register`` is declared
for an xclim release that discovers backends itself. ``register`` is
idempotent. If that future function already takes ``backend``, this module
leaves it alone.

Limits. Stationary models. Methods are ML, PWM (L-moments) and GMLE. Method
of moments, maximum spacing, the approximate method and a Gumbel with ξ fixed
at 0 are not available on this backend. A GPD level is a quantile of the
fitted sample; a scalar ``threshold`` fits excesses and is added back.
Annual peaks-over-threshold levels stay on :func:`tails.fit_pot`. Intervals
are implemented for the GEV. ``mode="min"`` reflects the sample, fits a GEV
to the maxima of the negated values, and reflects the level and the interval.
``fit`` fits the sample it is given. Series with fewer than 3 finite GEV
values, or fewer than 2 GPD excesses, come back as NaN. Block reduction
(rolling mean, season, month, resample) is xclim's ``statistics``.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np

from .core.distributions import gpd_ppf
from .core.fit import fit_gev, fit_gpd

__all__ = ["register", "unregister"]

_MODULES = ("xclim.indices.stats", "xclim.compute.stats")
_FUNCTIONS = ("fit", "fa", "frequency_analysis")
_OPTION_KEYS = ("ci", "alpha", "n_boot", "seed", "threshold", "profile_points")
_PATCHED: list[tuple] = []

_FAMILIES = {
    "genextreme": "gev",
    "gev": "gev",
    "genpareto": "gpd",
    "gpd": "gpd",
    "gumbelr": "gumbel",
    "gumbel": "gumbel",
}
_SCIPY_NAME = {"gev": "genextreme", "gpd": "genpareto"}
_METHODS = {"ML": "mle", "MLE": "mle", "PWM": "lmom", "LM": "lmom", "LMOM": "lmom", "GMLE": "gmle"}
_DOC = """

tails backend
-------------
After ``import tails.xclim``, pass ``backend="tails"``. ``ci`` of
``"delta"``, ``"profile"`` or ``"bootstrap"`` on a GEV returns a Dataset
with ``return_level``, ``lower`` and ``upper``. The parameter array stays
in scipy order: ``c = -ξ`` for genextreme and ``c = ξ`` for genpareto.
"""


def register() -> bool:
    """Accept ``backend="tails"`` on xclim's ``fit``, ``fa`` and ``frequency_analysis``.

    Patches ``xclim.indices.stats`` (xclim 0.x) and ``xclim.compute.stats``
    (xclim 1.0 and later) when the module imports. Returns True when an
    xclim stats module was found. Safe to call more than once.
    """
    found = False
    for modname in _MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        found = True
        for fname in _FUNCTIONS:
            current = getattr(mod, fname, None)
            if current is None or getattr(current, "_tails_backend", False):
                continue
            try:
                params = inspect.signature(current).parameters
            except (TypeError, ValueError):
                params = {}
            if "backend" in params:
                continue
            wrapped = _wrap(fname, current)
            setattr(mod, fname, wrapped)
            _PATCHED.append((mod, fname, current))
    return found


def unregister() -> None:
    """Restore the xclim functions :func:`register` replaced."""
    while _PATCHED:
        mod, fname, original = _PATCHED.pop()
        setattr(mod, fname, original)


def _wrap(name, original):
    impl = {"fit": _dispatch_fit, "fa": _dispatch_fa, "frequency_analysis": _dispatch_frequency}[name]

    def wrapper(*args, **kwargs):
        return impl(original, *args, **kwargs)

    wrapper.__wrapped__ = original
    wrapper._tails_backend = True
    wrapper.__name__ = getattr(original, "__name__", name)
    wrapper.__qualname__ = getattr(original, "__qualname__", name)
    wrapper.__module__ = getattr(original, "__module__", "xclim")
    wrapper.__doc__ = (getattr(original, "__doc__", None) or "") + _DOC
    try:
        wrapper.__signature__ = _with_backend(original)
    except (TypeError, ValueError):
        pass
    return wrapper


def _with_backend(original):
    sig = inspect.signature(original)
    extras = [
        inspect.Parameter("backend", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("ci", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("alpha", inspect.Parameter.KEYWORD_ONLY, default=0.05),
        inspect.Parameter("n_boot", inspect.Parameter.KEYWORD_ONLY, default=500),
        inspect.Parameter("seed", inspect.Parameter.KEYWORD_ONLY, default=0),
        inspect.Parameter("threshold", inspect.Parameter.KEYWORD_ONLY, default=None),
    ]
    params = []
    var_kw = None
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            var_kw = param
        else:
            params.append(param)
    params.extend(extras)
    if var_kw is not None:
        params.append(var_kw)
    return sig.replace(parameters=params)


def _split_options(kwargs):
    raw = kwargs.pop("backend", None)
    if raw is None or str(raw).lower() in {"scipy", "xclim"}:
        backend = "scipy"
    elif str(raw).lower() == "tails":
        backend = "tails"
    else:
        raise ValueError(f"backend must be 'tails', 'scipy' or 'xclim', got {raw!r}")
    options = {key: kwargs.pop(key) for key in _OPTION_KEYS if key in kwargs}
    if backend != "tails" and options:
        names = ", ".join(sorted(options))
        raise ValueError(f"{names} require backend='tails'")
    return backend, options


def _bind(original, args, kwargs):
    sig = inspect.signature(original)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    extra = {}
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            extra = dict(arguments.pop(name) or {})
    return arguments, extra


def _dispatch_fit(original, *args, **kwargs):
    backend, options = _split_options(kwargs)
    if backend != "tails":
        return original(*args, **kwargs)
    unused = sorted(set(options) - {"threshold"})
    if unused:
        raise ValueError(f"{', '.join(unused)} apply to fa and frequency_analysis")
    arguments, fitkwargs = _bind(original, args, kwargs)
    threshold = options.get("threshold", fitkwargs.pop("threshold", None))
    if fitkwargs:
        names = ", ".join(sorted(fitkwargs))
        raise ValueError(f"backend='tails' does not take scipy fit keywords ({names})")
    return _fit_dataarray(
        arguments["da"],
        dist=arguments.get("dist", "norm"),
        method=arguments.get("method", "ML"),
        dim=arguments.get("dim", "time"),
        threshold=threshold,
    )


def _dispatch_fa(original, *args, **kwargs):
    backend, options = _split_options(kwargs)
    if backend != "tails":
        return original(*args, **kwargs)
    arguments, fitkwargs = _bind(original, args, kwargs)
    if fitkwargs:
        names = ", ".join(sorted(fitkwargs))
        raise ValueError(f"backend='tails' does not take scipy fit keywords ({names})")
    return _levels_dataarray(
        arguments["da"],
        t=arguments["t"],
        dist=arguments.get("dist", "norm"),
        mode=arguments.get("mode", "max"),
        method=arguments.get("method", "ML"),
        dim="time",
        **options,
    )


def _dispatch_frequency(original, *args, **kwargs):
    backend, options = _split_options(kwargs)
    if backend != "tails":
        return original(*args, **kwargs)
    arguments, indexer = _bind(original, args, kwargs)
    blocks = _block_reduce(
        arguments["da"],
        mode=arguments["mode"],
        window=arguments.get("window", 1),
        freq=arguments.get("freq"),
        indexer=indexer,
        original=original,
    )
    return _levels_dataarray(
        blocks,
        t=arguments["t"],
        dist=arguments["dist"],
        mode=arguments["mode"],
        method=arguments.get("method", "ML"),
        dim="time",
        **options,
    )


def _block_reduce(da, mode, window, freq, indexer, original):
    """Mirror ``xclim``'s frequency_analysis prelude and call its ``statistics``."""
    statistics = original.__globals__.get("statistics")
    uses_dask = original.__globals__.get("uses_dask")
    if statistics is None:
        raise RuntimeError("This xclim frequency_analysis has no statistics() helper for tails to call.")
    mode = _mode(mode)
    attrs = da.attrs.copy()
    if window > 1:
        da = da.rolling(time=window).mean(skipna=False)
        da.attrs.update(attrs)
    indexer = dict(indexer)
    out_units = indexer.pop("out_units") if isinstance(indexer.get("out_units"), str) else None
    freq = freq or ("YS-DEC" if indexer.get("season") == "DJF" else "YS")
    sel = statistics(da, statistic=mode, freq=freq, out_units=out_units, **indexer)
    if uses_dask is not None and uses_dask(sel):
        sel = sel.chunk({"time": -1})
    else:
        sel = _full_core_chunk(sel, "time")
    return sel


def _xr():
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("The xclim backend needs xarray. Install it with pip install 'rare-tails[xarray]'.") from exc
    return xr


def _require_da(da, dim):
    xr = _xr()
    if not isinstance(da, xr.DataArray):
        raise TypeError("expected an xarray.DataArray")
    if dim not in getattr(da, "dims", ()):
        raise ValueError(f"expected a {dim!r} dimension on the DataArray")
    return da


def _full_core_chunk(da, dim):
    """One chunk along ``dim``. ``DataArray.chunks`` is a tuple; ``chunksizes`` is a map."""
    sizes = getattr(da, "chunksizes", None)
    parts = sizes.get(dim) if sizes is not None and hasattr(sizes, "get") else None
    if parts is None:
        chunks = getattr(da, "chunks", None)
        if isinstance(chunks, dict):
            parts = chunks.get(dim)
    if parts is not None and len(tuple(parts)) > 1:
        return da.chunk({dim: -1})
    return da


def _family(dist):
    name = dist if isinstance(dist, str) else getattr(dist, "name", None)
    if name is None:
        raise ValueError("dist must be 'genextreme', 'genpareto', or a scipy distribution with a name")
    key = str(name).lower().replace("-", "").replace("_", "")
    family = _FAMILIES.get(key)
    if family == "gumbel":
        raise ValueError("A Gumbel model fixes ξ at 0. backend='tails' fits a free GEV shape; use dist='genextreme'.")
    if family is None:
        raise ValueError("backend='tails' fits genextreme (GEV) and genpareto (GPD)")
    return family


def _method(method):
    key = str(method).upper()
    if key not in _METHODS:
        raise ValueError(f"backend='tails' accepts methods ML, PWM and GMLE. {method!r} is not implemented.")
    return _METHODS[key]


def _mode(mode):
    key = str(mode).lower()
    if key in {"max", "high"}:
        return "max"
    if key in {"min", "low"}:
        return "min"
    raise ValueError("mode must be 'max' or 'min'")


def _ci_name(ci, family):
    if ci is None:
        return None
    key = str(ci).lower()
    if key not in {"delta", "profile", "bootstrap"}:
        raise ValueError("ci must be 'delta', 'profile' or 'bootstrap'")
    if family != "gev":
        raise ValueError("confidence intervals with backend='tails' are implemented for the GEV")
    return key


def _periods(t):
    arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("return periods must be a scalar or a 1-d sequence")
    if np.any(~np.isfinite(arr) | (arr <= 1)):
        raise ValueError("return periods must be greater than 1")
    return arr


def _threshold(value):
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        raise ValueError("threshold must be a single number")
    arr = np.asarray(getattr(value, "values", value), dtype=np.float64)
    if arr.shape != ():
        raise ValueError("threshold must be a single number")
    return float(arr)


def _parameters_block(arr, family, method, threshold):
    """Scipy ``(c, loc, scale)`` for each series. Last axis of ``arr`` is the record."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, np.asarray(arr).shape[-1])
    out = np.full((flat.shape[0], 3), np.nan)
    if family == "gev":
        prepared = flat
        min_n = 3
        loc = None
    else:
        prepared, loc = _excesses(flat, threshold)
        min_n = 2
    valid = np.isfinite(prepared).sum(-1) >= min_n
    if valid.any():
        if family == "gev":
            fit = fit_gev(prepared[valid], method=method)
            natural = fit.params
            out[valid, 0] = -np.asarray(natural["shape"], dtype=np.float64)
            out[valid, 1] = np.asarray(natural["loc"], dtype=np.float64)
            out[valid, 2] = np.asarray(natural["scale"], dtype=np.float64)
        else:
            fit = fit_gpd(prepared[valid], method=method)
            natural = fit.params
            out[valid, 0] = np.asarray(natural["shape"], dtype=np.float64)
            out[valid, 1] = loc
            out[valid, 2] = np.asarray(natural["scale"], dtype=np.float64)
    return out.reshape(*np.asarray(arr).shape[:-1], 3)


def _excesses(flat, threshold):
    if threshold is None:
        return flat, 0.0
    excess = flat - threshold
    return np.where(np.isfinite(flat) & (flat > threshold), excess, np.nan), float(threshold)


def _levels_block(arr, family, method, periods, mode, ci, alpha, n_boot, seed, threshold, profile_points):
    """Return levels with shape ``(*batch, n_period, n_col)``. Columns are estimate, then bounds."""
    values = np.asarray(arr, dtype=np.float64)
    if mode == "min":
        values = -values
    flat = values.reshape(-1, values.shape[-1])
    n_period = len(periods)
    n_col = 1 if ci is None else 3
    out = np.full((flat.shape[0], n_period, n_col), np.nan)
    if family == "gev":
        prepared = flat
        min_n = 3
        loc = None
    else:
        prepared, loc = _excesses(flat, threshold)
        min_n = 2
    valid = np.isfinite(prepared).sum(-1) >= min_n
    if valid.any():
        if family == "gev":
            fit = fit_gev(prepared[valid], method=method)
            kw = {"ci": ci, "alpha": alpha, "n_boot": n_boot, "seed": seed}
            if profile_points is not None:
                kw["profile_points"] = profile_points
            levels = fit.return_level(periods, **kw)
            columns = [np.asarray(levels.estimate, dtype=np.float64)]
            if ci is not None:
                columns.append(np.asarray(levels.lower, dtype=np.float64))
                columns.append(np.asarray(levels.upper, dtype=np.float64))
            stacked = np.stack(columns, axis=-1)
        else:
            natural = fit_gpd(prepared[valid], method=method).params
            probability = 1.0 - 1.0 / np.asarray(periods, dtype=np.float64)
            scale = np.asarray(natural["scale"], dtype=np.float64)[:, None]
            shape = np.asarray(natural["shape"], dtype=np.float64)[:, None]
            estimate = np.asarray(gpd_ppf(probability, scale, shape), dtype=np.float64)
            stacked = (estimate + loc)[..., None]
        if mode == "min":
            estimate = -stacked[..., 0]
            if stacked.shape[-1] == 3:
                lower = -stacked[..., 2]
                upper = -stacked[..., 1]
                stacked = np.stack([estimate, lower, upper], axis=-1)
            else:
                stacked = estimate[..., None]
        out[valid] = stacked
    return out.reshape(*values.shape[:-1], n_period, n_col)


def _fit_dataarray(da, dist, method, dim, threshold):
    xr = _xr()
    da = _require_da(da, dim)
    family = _family(dist)
    method_key = _method(method)
    threshold = _threshold(threshold)
    if family == "gev" and threshold is not None:
        raise ValueError("threshold applies to a GPD. A GEV is fit to the sample as given.")
    da = _full_core_chunk(da, dim)
    data = xr.apply_ufunc(
        _parameters_block,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["dparams"]],
        kwargs={"family": family, "method": method_key, "threshold": threshold},
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"dparams": 3}},
    )
    order = [d if d != dim else "dparams" for d in da.dims]
    out = data.assign_coords(dparams=["c", "loc", "scale"]).transpose(*order)
    out.attrs = {
        "long_name": f"{_SCIPY_NAME[family]} parameters",
        "description": f"Parameters of the {_SCIPY_NAME[family]} distribution",
        "method": str(method).upper(),
        "scipy_dist": _SCIPY_NAME[family],
        "units": "",
        "tails_backend": "tails",
        "shape_convention": "c=-xi" if family == "gev" else "c=xi",
    }
    if "units" in da.attrs:
        out.attrs["original_units"] = da.attrs["units"]
    return out


def _levels_dataarray(
    da, t, dist, mode, method, dim, ci=None, alpha=0.05, n_boot=500, seed=0, threshold=None, profile_points=None
):
    xr = _xr()
    da = _require_da(da, dim)
    family = _family(dist)
    mode = _mode(mode)
    if family == "gpd" and mode == "min":
        raise ValueError("A GPD backend models exceedances. mode='min' is available for the GEV.")
    ci_key = _ci_name(ci, family)
    periods = _periods(t)
    method_key = _method(method)
    threshold = _threshold(threshold)
    if family == "gev" and threshold is not None:
        raise ValueError("threshold applies to a GPD. A GEV is fit to the sample as given.")
    alpha = float(alpha)
    da = _full_core_chunk(da, dim)
    data = xr.apply_ufunc(
        _levels_block,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["return_period", "bound"]],
        kwargs={
            "family": family,
            "method": method_key,
            "periods": periods,
            "mode": mode,
            "ci": ci_key,
            "alpha": alpha,
            "n_boot": int(n_boot),
            "seed": int(seed),
            "threshold": threshold,
            "profile_points": profile_points,
        },
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"return_period": periods.size, "bound": 1 if ci_key is None else 3}},
    )
    order = [d if d != dim else "return_period" for d in da.dims]
    data = data.assign_coords(return_period=periods).transpose(*order, "bound")
    attrs = {
        "tails_backend": "tails",
        "mode": mode,
        "long_name": "return level",
        "scipy_dist": _SCIPY_NAME[family],
        "method": str(method).upper(),
    }
    if "units" in da.attrs:
        attrs["units"] = da.attrs["units"]
    if ci_key is None:
        out = data.isel(bound=0, drop=True)
        out.attrs = attrs
        out.name = "return_level"
        return out
    data = data.assign_coords(bound=["return_level", "lower", "upper"])
    ds = data.to_dataset(dim="bound")
    attrs["ci"] = ci_key
    attrs["alpha"] = alpha
    ds.attrs = attrs
    for name in ("return_level", "lower", "upper"):
        ds[name].attrs = {"units": attrs.get("units", ""), "long_name": name.replace("_", " ")}
    return ds


register()
