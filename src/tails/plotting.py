"""Optional matplotlib helpers for diagnostic and IDF plots.

Install the extra with ``pip install 'rare-tails[plots]'``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["require_matplotlib", "plot_fit_diagnostics", "plot_return_level_map"]

# Okabe–Ito, colour-blind friendly. First four are the IDF return-period defaults.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000")


def require_matplotlib():
    """Return ``matplotlib.pyplot``, or raise a helpful ImportError."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib. Install it with: pip install 'rare-tails[plots]'") from e
    return plt


def apply_axes_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.3, color="0.5")
    ax.tick_params(direction="out", length=3)
    return ax


def color(i: int) -> str:
    return PALETTE[int(i) % len(PALETTE)]


def _series_y(fit, index: int = 0) -> np.ndarray:
    B = int(np.prod(fit.batch_shape, dtype=int))
    y = np.asarray(fit._y).reshape(B, -1)[index]
    m = np.asarray(fit._mask).reshape(B, -1)[index]
    return np.where(m, y, np.nan)


def _stationary_params(fit, index: int = 0) -> dict[str, float]:
    if not fit.stationary:
        raise ValueError("plot_diagnostics currently supports stationary models")
    out = {}
    for k, v in fit.params.items():
        out[k] = float(np.asarray(v).reshape(-1)[index])
    return out


def empirical_return_levels(y, periods, method: str = "weibull") -> np.ndarray:
    """Empirical T-year levels from a single sample via plotting positions."""
    x = np.sort(np.asarray(y, float)[np.isfinite(y)])
    n = x.size
    periods = np.atleast_1d(np.asarray(periods, float))
    if n == 0:
        return np.full(periods.shape, np.nan)
    i = np.arange(1, n + 1, dtype=float)
    if method == "gringorten":
        p = (i - 0.44) / (n + 0.12)
    else:
        p = i / (n + 1.0)
    T = 1.0 / np.clip(1.0 - p, 1e-15, None)
    # For T beyond the largest plotting position (n+1 for Weibull), use the sample max.
    return np.interp(periods, T, x, left=np.nan, right=x[-1])


def plot_fit_diagnostics(
    fit,
    index: int = 0,
    *,
    ci: str | None = "delta",
    periods=None,
    n_band: int = 200,
    seed: int = 0,
    figsize=(8.2, 7.4),
    fig=None,
):
    """2×2 diagnostic figure for one series of a fitted GEV, GPD or POT model.

    Panels: QQ and PP plots with a simulated pointwise 95% envelope, a
    return-level plot with a CI ribbon, and a histogram plus fitted density.
    """
    plt = require_matplotlib()
    if not fit.stationary:
        raise ValueError("plot_diagnostics currently supports stationary models")
    y = _series_y(fit, index)
    x = y[np.isfinite(y)]
    if x.size < 2:
        raise ValueError("need at least 2 finite observations to plot diagnostics")
    params = _stationary_params(fit, index)
    family = fit.family
    is_pot = getattr(fit, "rate", None) is not None and family == "gpd"
    # For a GPD fit of excesses, density/QQ/PP are on the excess scale.
    ppf, cdf, pdf = _dist_funs(family, params, fit, index, is_pot)

    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    else:
        axes = np.asarray(fig.axes).reshape(2, 2)
    ax_qq, ax_pp, ax_rl, ax_den = axes.ravel()

    n = x.size
    p = (np.arange(1, n + 1) - 0.5) / n
    x_sorted = np.sort(x)
    theo = ppf(p)
    lo_q, hi_q, lo_p, hi_p = _sim_envelopes(ppf, n, n_band, seed)
    _band(ax_qq, theo, lo_q, hi_q)
    ax_qq.plot(theo, theo, color="0.35", lw=1.0, zorder=2)
    ax_qq.scatter(theo, x_sorted, s=18, color=color(0), zorder=3, edgecolors="white", linewidths=0.4)
    ax_qq.set_xlabel("Theoretical quantile")
    ax_qq.set_ylabel("Empirical quantile")
    ax_qq.set_title("QQ plot")
    apply_axes_style(ax_qq)

    emp_p = p
    model_p = np.clip(cdf(x_sorted), 0, 1)
    _band(ax_pp, emp_p, lo_p, hi_p)
    ax_pp.plot([0, 1], [0, 1], color="0.35", lw=1.0, zorder=2)
    ax_pp.scatter(emp_p, model_p, s=18, color=color(0), zorder=3, edgecolors="white", linewidths=0.4)
    ax_pp.set_xlabel("Empirical probability")
    ax_pp.set_ylabel("Model probability")
    ax_pp.set_title("PP plot")
    ax_pp.set_xlim(0, 1)
    ax_pp.set_ylim(0, 1)
    apply_axes_style(ax_pp)

    if periods is None:
        tmax = max(2 * n, 10)
        periods = np.unique(np.concatenate(([1.1, 1.5, 2, 5, 10, 20, 50, 100], [float(tmax)])))
        periods = periods[periods > 1]
    if hasattr(fit, "return_level") and (family == "gev" or is_pot):
        if ci is not None and not _return_level_supports(fit, ci):
            ci = "delta" if is_pot else ci
        rl = fit.return_level(periods, ci=ci)
        est, lo, hi = _rl_vectors(rl, index)
        ax_rl.plot(periods, est, color=color(0), lw=1.6, label="Fitted", zorder=3)
        if lo is not None:
            label = "95% CI" if rl.alpha == 0.05 else f"{100 * (1 - rl.alpha):.0f}% CI"
            ax_rl.fill_between(periods, lo, hi, color=color(0), alpha=0.22, linewidth=0, zorder=2, label=label)
        if family == "gev":
            emp = empirical_return_levels(x, periods)
            ax_rl.scatter(periods, emp, s=22, color="0.15", zorder=4, label="Observed (empirical return period)")
        ax_rl.set_xlabel("Return period (years)")
        ax_rl.set_ylabel("Return level")
        ax_rl.set_title("Return level")
    else:
        # GPD on excesses: quantile as a function of 1 / (1-p).
        est = ppf(1.0 - 1.0 / periods)
        ax_rl.plot(periods, est, color=color(0), lw=1.6, label="Fitted", zorder=3)
        ax_rl.set_xlabel(r"$1 / (1-p)$")
        ax_rl.set_ylabel("Excess quantile")
        ax_rl.set_title("Quantile function")
    ax_rl.set_xscale("log")
    ax_rl.legend(frameon=False, fontsize=8)
    apply_axes_style(ax_rl)

    lo_x, hi_x = np.nanmin(x), np.nanmax(x)
    pad = 0.1 * (hi_x - lo_x if hi_x > lo_x else max(abs(hi_x), 1.0))
    grid = np.linspace(lo_x - pad, hi_x + pad, 256)
    # Stay inside support when the endpoint is finite.
    dens = pdf(grid)
    dens = np.where(np.isfinite(dens) & (dens >= 0), dens, np.nan)
    ax_den.hist(x, bins="fd", density=True, color=color(0), alpha=0.35, edgecolor="white", linewidth=0.4, label="Data")
    ax_den.plot(grid, dens, color=color(0), lw=1.6, label="Fitted density")
    ax_den.set_xlabel("Value")
    ax_den.set_ylabel("Density")
    ax_den.set_title("Density")
    ax_den.legend(frameon=False, fontsize=8)
    apply_axes_style(ax_den)

    fig.tight_layout()
    return fig, axes


def _return_level_supports(fit, ci) -> bool:
    if ci is None:
        return True
    if ci == "bootstrap" and getattr(fit, "free_params", None) is not None:
        return False
    if fit.__class__.__name__ == "POTFit":
        return ci in ("delta", "profile", "bootstrap")
    return True


def _rl_vectors(rl, index: int):
    est = np.asarray(rl.estimate)
    # Stationary: (*batch, T)
    est = est.reshape(-1, est.shape[-1])[index]
    lo = hi = None
    if rl.lower is not None:
        lo = np.asarray(rl.lower).reshape(-1, est.size)[index]
        hi = np.asarray(rl.upper).reshape(-1, est.size)[index]
    return est, lo, hi


def _band(ax, x, lo, hi):
    order = np.argsort(x)
    ax.fill_between(x[order], lo[order], hi[order], color=color(0), alpha=0.18, linewidth=0, zorder=1, label="95% envelope")


def _sim_envelopes(ppf, n: int, n_band: int, seed: int):
    rng = np.random.default_rng(seed)
    u = rng.random((n_band, n))
    sim = np.sort(ppf(u), axis=1)
    lo_q, hi_q = np.quantile(sim, [0.025, 0.975], axis=0)
    # PP envelope: model CDF of simulated order stats is just the order stats' true probabilities.
    # Using the same uniforms, F(X_(i)) = U_(i).
    u_sorted = np.sort(u, axis=1)
    lo_p, hi_p = np.quantile(u_sorted, [0.025, 0.975], axis=0)
    return lo_q, hi_q, lo_p, hi_p


def _dist_funs(family, params, fit, index, is_pot):
    from .core.distributions import gev_cdf, gev_logpdf, gev_ppf, gev_support, gpd_cdf, gpd_logpdf, gpd_ppf

    if family == "gev":
        loc, scale, shape = params["loc"], params["scale"], params["shape"]

        def ppf(p):
            return np.asarray(gev_ppf(np.asarray(p, float), loc, scale, shape))

        def cdf(x):
            return np.asarray(gev_cdf(np.asarray(x, float), loc, scale, shape))

        def pdf(x):
            x = np.asarray(x, float)
            lo, hi = gev_support(loc, scale, shape)
            out = np.exp(np.asarray(gev_logpdf(x, loc, scale, shape)))
            return np.where((x >= lo) & (x <= hi), out, np.nan)

        return ppf, cdf, pdf

    scale, shape = params["scale"], params["shape"]
    # POT return levels are on the original scale; QQ/PP/density stay on excesses
    # (that's what was fitted). The return-level panel uses fit.return_level.

    def ppf(p):
        return np.asarray(gpd_ppf(np.asarray(p, float), scale, shape))

    def cdf(x):
        return np.asarray(gpd_cdf(np.asarray(x, float), scale, shape))

    def pdf(x):
        x = np.asarray(x, float)
        out = np.exp(np.asarray(gpd_logpdf(x, scale, shape)))
        return np.where(x >= 0, out, np.nan)

    return ppf, cdf, pdf


def plot_return_levels(rl, ax=None, index: int = 0, **kwargs):
    """Plot a :class:`~tails.core.fit.ReturnLevels` curve with a CI ribbon."""
    plt = require_matplotlib()
    ax = ax or plt.subplots(figsize=(6.2, 4.0))[1]
    est, lo, hi = _rl_vectors(rl, index)
    T = np.asarray(rl.period, float)
    c = kwargs.pop("color", color(0))
    ax.plot(T, est, color=c, lw=1.6, zorder=3, **kwargs)
    if lo is not None:
        ax.fill_between(T, lo, hi, color=c, alpha=0.22, linewidth=0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Return period")
    ax.set_ylabel("Return level")
    apply_axes_style(ax)
    return ax


def plot_idf(
    fit,
    station: int = 0,
    periods: Sequence[float] = (2, 10, 25, 100),
    ax=None,
    data: bool = True,
    *,
    ci: str | None = "hessian",
    alpha: float = 0.05,
    title: str | None = None,
):
    """IDF curves on log–log axes, with CI ribbons and empirical return-period points."""
    plt = require_matplotlib()
    ax = ax or plt.subplots(figsize=(7.0, 4.6))[1]
    d_obs = np.asarray(fit.D[station], float)
    d_grid = np.geomspace(d_obs.min(), d_obs.max(), 160)
    periods = tuple(float(T) for T in periods)
    q = fit.quantile(d_grid, periods)[station]
    lo = hi = None
    if ci is not None:
        _, lo, hi = fit.quantile_ci(d_grid, periods, method=ci, alpha=alpha)
        lo, hi = lo[station], hi[station]
    for j, T in enumerate(periods):
        c = color(j)
        ax.plot(d_grid, q[:, j], color=c, lw=1.8, label=f"{T:.0f}-year", zorder=3)
        if lo is not None:
            ax.fill_between(d_grid, lo[:, j], hi[:, j], color=c, alpha=0.18, linewidth=0, zorder=2)
    if data:
        Y = np.where(fit.M[station], fit.Y[station], np.nan)
        # One empirical T-year intensity per duration, coloured like the matching curve.
        for j, T in enumerate(periods):
            emp = np.array([empirical_return_levels(Y[:, k], T)[0] for k in range(Y.shape[1])])
            ax.scatter(d_obs, emp, s=28, color=color(j), zorder=4, edgecolors="white", linewidths=0.5)
        ax.scatter(
            [],
            [],
            s=28,
            color="0.2",
            edgecolors="white",
            linewidths=0.5,
            label="Observed (empirical return period)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Duration (h)")
    ax.set_ylabel("Intensity (mm/h)")
    if title is not None:
        ax.set_title(title)
    elif fit.stations:
        ax.set_title(str(fit.stations[station]))
    # Lower left is the empty corner of a log–log IDF plot (short duration, low intensity).
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    apply_axes_style(ax)
    ax.figure.tight_layout()
    return ax


def plot_return_level_map(
    levels,
    lon=None,
    lat=None,
    *,
    period: float | None = None,
    ax=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    s: float = 36,
    label: str | None = None,
    title: str | None = None,
    figsize=(6.4, 6.0),
):
    """Scatter or mesh of one return level at longitude and latitude.

    ``levels`` is a batch of return levels already computed by a fit: a
    :class:`~tails.core.fit.ReturnLevels`, a NumPy array of one value per
    location, or an xarray DataArray or Dataset (a Dataset must contain
    ``return_level``). For a DataArray with ``lat``/``lon`` or
    ``latitude``/``longitude`` coordinates, ``lon`` and ``lat`` can be
    omitted. A ``period`` or ``return_period`` dimension is indexed by
    ``period`` when it has more than one value.

    Stations (one coordinate pair per series) are a scatter. A rectangular
    ``(lat, lon)`` grid is a mesh. Axes are longitude and latitude in
    degrees, with an aspect of ``1 / cos(mean latitude)`` and no tile layer.
    The colour is the return level in the units of the fit. Pass ``label``
    to name those units; a DataArray ``units`` attribute is used when
    ``label`` is omitted.

    Parameters
    ----------
    levels
        Return levels, one per location after ``period`` is selected.
    lon, lat
        Longitude and latitude in degrees. Required unless ``levels``
        already carries those coordinates.
    period
        Return period to draw, in the same units as the fit (years for an
        annual maximum). Required when ``levels`` has more than one period.
        On a plain array it is not used to index; it only sets the default title.
    cmap
        Matplotlib colormap name. Default ``"viridis"``.
    vmin, vmax
        Colour limits. Default is the range of the finite levels.
    s
        Scatter marker area. Ignored for a grid.
    label
        Colour-bar text. Default ``"Return level"``, plus the DataArray
        units when those are set.
    title
        Axes title. Default is ``"{period}-year return level"`` when the
        period is known.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = require_matplotlib()
    values, used, coords, units = _map_values(levels, period)
    if lon is None or lat is None:
        if coords is None:
            raise ValueError(
                "pass lon and lat in degrees, or an xarray object with "
                "lon/longitude and lat/latitude coordinates"
            )
        if lon is None:
            lon = coords[0]
        if lat is None:
            lat = coords[1]
    kind, values, lon, lat = _align_map(values, lon, lat)
    if label is None:
        label = "Return level" if not units else f"Return level ({units})"
    if title is None and used is not None:
        title = f"{used:g}-year return level"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite return level to plot")
    if vmin is None and vmax is None and float(np.min(finite)) == float(np.max(finite)):
        vmin = float(finite[0]) - 0.5
        vmax = float(finite[0]) + 0.5
    artist_kw = {"cmap": cmap, "vmin": vmin, "vmax": vmax}
    if kind == "mesh":
        artist = ax.pcolormesh(lon, lat, values, shading="auto", **artist_kw)
        aspect_lat = lat
    else:
        mask = np.isfinite(values) & np.isfinite(lon) & np.isfinite(lat)
        if not mask.any():
            raise ValueError("no finite return level at a finite longitude and latitude")
        artist = ax.scatter(
            lon[mask],
            lat[mask],
            c=values[mask],
            s=s,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            **artist_kw,
        )
        aspect_lat = lat[mask]
    cbar = ax.figure.colorbar(artist, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label(label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    aspect = _geo_aspect(aspect_lat)
    if aspect is not None:
        ax.set_aspect(aspect, adjustable="box")
    _plain_deg(ax)
    apply_axes_style(ax)
    ax.figure.tight_layout()
    return ax


def _map_values(levels, period):
    """Return ``(values, period_or_None, (lon, lat) or None, units or None)``."""
    if hasattr(levels, "data_vars"):
        if "return_level" not in levels:
            found = ", ".join(str(name) for name in list(levels.data_vars)[:6])
            raise ValueError(f"dataset has no 'return_level' variable (found {found})")
        levels = levels["return_level"]
    estimate = getattr(levels, "estimate", None)
    periods = getattr(levels, "period", None)
    if estimate is not None and periods is not None and not hasattr(levels, "dims"):
        grid = np.asarray(periods, dtype=float).reshape(-1)
        idx, used = _select_period(grid, period)
        est = np.asarray(estimate, dtype=float)
        if est.ndim == 0 or est.shape[-1] != grid.size:
            length = 0 if est.ndim == 0 else est.shape[-1]
            raise ValueError(f"return-level array ends with length {length}, but there are {grid.size} periods")
        return est[..., idx], used, None, None
    if hasattr(levels, "dims") and hasattr(levels, "coords") and hasattr(levels, "values"):
        used = None
        dim = _period_dim(levels)
        if dim is not None:
            grid = np.atleast_1d(np.asarray(levels.coords[dim].values, dtype=float))
            idx, used = _select_period(grid, period)
            levels = levels.isel({dim: idx})
        elif period is not None:
            used = float(period)
        values, coords = _dataarray_field(levels)
        return values, used, coords, _units_attr(levels)
    if period is not None and not np.isscalar(period):
        raise ValueError("period must be a single return period")
    used = None if period is None else float(period)
    return np.asarray(levels, dtype=float), used, None, _units_attr(levels)


def _period_dim(da) -> str | None:
    found = [name for name in ("period", "return_period") if name in da.dims]
    if len(found) > 1:
        raise ValueError("levels has both 'period' and 'return_period' dimensions; select one")
    return found[0] if found else None


def _select_period(periods: np.ndarray, period) -> tuple[int, float]:
    periods = np.asarray(periods, dtype=float).reshape(-1)
    if periods.size == 0:
        raise ValueError("levels has no return period")
    if period is None:
        if periods.size != 1:
            shown = ", ".join(f"{p:g}" for p in periods)
            raise ValueError(f"levels has return periods ({shown}); pass period= to choose one")
        return 0, float(periods[0])
    if not np.isscalar(period):
        raise ValueError("period must be a single return period")
    chosen = float(period)
    hits = np.flatnonzero(np.isclose(periods, chosen))
    if hits.size == 0:
        shown = ", ".join(f"{p:g}" for p in periods)
        raise ValueError(f"period {chosen:g} is not among ({shown})")
    if hits.size > 1:
        raise ValueError(f"period {chosen:g} matches more than one entry")
    return int(hits[0]), float(periods[int(hits[0])])


def _named_coord(da, names: tuple[str, ...]):
    for name in names:
        if name in da.coords:
            return name, np.asarray(da.coords[name].values, dtype=float)
    return None, None


def _dataarray_field(da):
    lon_name, lon = _named_coord(da, ("lon", "longitude"))
    lat_name, lat = _named_coord(da, ("lat", "latitude"))
    if lon is None or lat is None:
        return np.asarray(da.values, dtype=float), None
    if lon_name in da.dims and lat_name in da.dims:
        others = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
        if others:
            shown = ", ".join(repr(dim) for dim in others)
            raise ValueError(
                f"levels still has dimension {shown} besides latitude and longitude; "
                "select one return level per location"
            )
        ordered = da.transpose(lat_name, lon_name)
        return (
            np.asarray(ordered.values, dtype=float),
            (
                np.asarray(ordered.coords[lon_name].values, dtype=float),
                np.asarray(ordered.coords[lat_name].values, dtype=float),
            ),
        )
    return np.asarray(da.values, dtype=float), (lon, lat)


def _units_attr(obj) -> str | None:
    attrs = getattr(obj, "attrs", None)
    if not isinstance(attrs, dict):
        return None
    units = attrs.get("units")
    if units is None:
        return None
    text = str(units).strip()
    return text or None


def _align_map(values, lon, lat):
    values = np.asarray(values, dtype=float)
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    candidates = [np.atleast_1d(values)]
    squeezed = np.atleast_1d(np.squeeze(values))
    if squeezed.shape != candidates[0].shape:
        candidates.append(squeezed)
    for candidate in candidates:
        kind = _map_kind(candidate, lon, lat)
        if kind is not None:
            if kind == "scatter" and candidate.shape != lon.shape:
                return kind, candidate.ravel(), lon.ravel(), lat.ravel()
            return kind, candidate, lon, lat
    raise ValueError(
        f"levels shape {tuple(np.asarray(values).shape)} does not match "
        f"lon {tuple(np.asarray(lon).shape)} and lat {tuple(np.asarray(lat).shape)}. "
        "Pass one return level per location (select a return period, and one covariate "
        "row when the fit is non-stationary)."
    )


def _map_kind(values, lon, lat) -> str | None:
    if values.ndim == 2 and lat.ndim == 1 and lon.ndim == 1 and values.shape == (lat.size, lon.size):
        return "mesh"
    if values.ndim == 2 and lon.shape == values.shape and lat.shape == values.shape:
        return "mesh"
    if values.shape == lon.shape == lat.shape and values.ndim == 1:
        return "scatter"
    if values.size == lon.size == lat.size and values.ndim <= 2 and lon.ndim == 1 and lat.ndim == 1:
        return "scatter"
    return None


def _geo_aspect(lat) -> float | None:
    finite = np.asarray(lat, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    mean = float(np.mean(finite))
    if abs(mean) > 90.0:
        return None
    cosine = float(np.cos(np.deg2rad(mean)))
    if cosine < 0.2:
        return None
    return 1.0 / cosine


def _plain_deg(ax):
    import matplotlib.ticker as mticker

    for axis in (ax.xaxis, ax.yaxis):
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)
