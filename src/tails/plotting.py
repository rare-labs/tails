"""Optional matplotlib helpers for diagnostic, threshold and IDF plots.

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


def _default_covariate_name(name: str) -> bool:
    return len(name) > 1 and name[0] == "x" and name[1:].isdigit()


def _covariate_axis_label(names: list[str]) -> str:
    """Axis title from covariate names. Unnamed columns (``x0``, ``x1``, …) are not shown."""
    pretty: list[str] = []
    for name in names:
        if _default_covariate_name(name) or name in {"", "None"}:
            continue
        if name not in pretty:
            pretty.append(name)
    if len(pretty) == 1:
        return pretty[0]
    return "Covariate"


def _covariate_axis(fit, index: int) -> tuple[np.ndarray, str]:
    """Horizontal axis for one series, in observation order.

    One covariate column — including the same column on more than one
    parameter — is drawn on that column's scale. A year, a temperature
    index, or any other series the user passed in is treated the same way.
    Several distinct columns stay in observation order: the curve is the
    fitted level at each row.
    """
    B = int(np.prod(fit.batch_shape, dtype=int))
    columns: list[np.ndarray] = []
    names: list[str] = []
    n_obs = None
    for design, raw in zip(fit._designs, fit._raw):
        if int(design.X.shape[-1]) <= 1:
            continue
        raw = np.asarray(raw, float)
        if raw.ndim == 2:
            cols = raw[:, 1:]
        else:
            cols = raw.reshape(B, -1, raw.shape[-1])[index, :, 1:]
        if n_obs is None:
            n_obs = int(cols.shape[0])
        labels = [nm.split(":", 1)[1] if ":" in nm else nm for nm in design.names[1:]]
        if len(labels) != cols.shape[1]:
            labels = [f"x{j}" for j in range(cols.shape[1])]
        for j in range(cols.shape[1]):
            columns.append(np.asarray(cols[:, j], float))
            names.append(labels[j])
    if not columns or n_obs is None:
        raise ValueError("plot_diagnostics found no covariate on this non-stationary fit")
    same = all(np.allclose(columns[0], col) for col in columns[1:])
    if same and np.unique(columns[0]).size >= 2:
        return columns[0], _covariate_axis_label(names)
    return np.arange(n_obs, dtype=float), "Observation"


def _trend_curve(rl, index: int):
    """Return level and bounds for one series, shape ``(n,)`` each."""
    est = np.asarray(rl.estimate, float)
    if est.ndim < 2:
        raise ValueError("a non-stationary return level has shape (*batch, n, period)")
    curve = est.reshape(-1, est.shape[-2], est.shape[-1])
    if index < 0 or index >= curve.shape[0]:
        raise IndexError(f"index {index} is outside the batch of {curve.shape[0]} series")
    est_i = curve[index, :, 0]
    lo = hi = None
    if rl.lower is not None:
        lo = np.asarray(rl.lower, float).reshape(curve.shape)[index, :, 0]
        hi = np.asarray(rl.upper, float).reshape(curve.shape)[index, :, 0]
    return est_i, lo, hi


def plot_trend_diagnostics(
    fit,
    index: int = 0,
    *,
    ci: str | None = "delta",
    period: float = 2.0,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: int = 0,
    profile_points: int = 41,
    figsize=(7.2, 4.4),
    fig=None,
):
    """Observations, fitted return level, and an interval along the covariate.

    The curve is the ``period``-year level at each fitted row. The default
    period is 2, the median of the GEV. ``ci`` is ``"delta"``, ``"profile"``,
    ``"bootstrap"`` or ``None``, the same intervals as
    :meth:`~tails.core.fit.GEVFit.return_level`. The horizontal axis is the
    covariate passed to ``fit_gev``. A pandas column name is used when the
    fit has one; an unnamed array is labelled ``Covariate``.
    """
    plt = require_matplotlib()
    period = float(period)
    y = _series_y(fit, index)
    x, xlabel = _covariate_axis(fit, index)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"covariate length {x.shape[0]} does not match the series length {y.shape[0]}")
    rl = fit.return_level(
        [period],
        ci=ci,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        profile_points=profile_points,
    )
    est, lo, hi = _trend_curve(rl, index)
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = fig.axes[0] if fig.axes else fig.add_subplot(111)

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ax.plot(xs, est[order], color=color(0), lw=1.6, zorder=3, label=f"Fitted {period:g}-year level")
    if lo is not None:
        band = "95% CI" if rl.alpha == 0.05 else f"{100 * (1.0 - rl.alpha):.0f}% CI"
        ax.fill_between(xs, lo[order], hi[order], color=color(0), alpha=0.22, linewidth=0, zorder=2, label=band)
    finite = np.isfinite(y)
    ax.scatter(
        x[finite],
        y[finite],
        s=22,
        color="0.15",
        zorder=4,
        edgecolors="white",
        linewidths=0.4,
        label="Observed",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Block maximum")
    ax.set_title("Fitted trend")
    ax.legend(frameon=False, fontsize=8)
    apply_axes_style(ax)
    fig.tight_layout()
    return fig, ax


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
    figsize=None,
    fig=None,
    period: float = 2.0,
    alpha: float = 0.05,
    n_boot: int = 500,
    profile_points: int = 41,
):
    """Diagnostic figure for one series of a fitted GEV, GPD or POT model.

    A stationary fit draws a 2×2 figure: QQ and PP plots with a simulated
    pointwise 95% envelope, a return-level plot with a CI ribbon, and a
    histogram plus the fitted density.

    A non-stationary GEV draws the observations, the fitted return level
    along the covariate, and a confidence ribbon. ``period`` (default 2,
    the median) selects that level. ``ci`` is ``"delta"``, ``"profile"``,
    ``"bootstrap"`` or ``None``.
    """
    plt = require_matplotlib()
    if not fit.stationary:
        if fit.family != "gev":
            raise ValueError(
                "plot_diagnostics supports stationary models; "
                "trend diagnostics are implemented for a non-stationary GEV"
            )
        if figsize is None:
            figsize = (7.2, 4.4)
        return plot_trend_diagnostics(
            fit,
            index=index,
            ci=ci,
            period=period,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            profile_points=profile_points,
            figsize=figsize,
            fig=fig,
        )
    if figsize is None:
        figsize = (8.2, 7.4)
    y = _series_y(fit, index)
    x = y[np.isfinite(y)]
    if x.size < 2:
        raise ValueError("need at least 2 finite observations to plot diagnostics")
    params = _stationary_params(fit, index)
    family = fit.family
    is_pot = getattr(fit, "rate", None) is not None and family == "gpd"
    # For a GPD fit of excesses, density/QQ/PP are on the excess scale.
    ppf, cdf, pdf = _dist_funs(family, params)

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


def _dist_funs(family, params):
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


def _one_series(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = np.ravel(arr)
    if arr.ndim != 1:
        raise ValueError("plot_threshold takes one series; pass a 1-d array")
    if int(np.isfinite(arr).sum()) < 10:
        raise ValueError("need at least 10 finite observations")
    return arr


def _with_unit(label: str, unit: str) -> str:
    if not unit:
        return label
    return f"{label} ({unit})"


def _default_thresholds(finite: np.ndarray) -> np.ndarray:
    """Quantile grid with a block of exceedances left above the top candidate."""
    n = int(finite.size)
    order = np.sort(finite)
    n_tail = int(min(40, max(8, n // 20)))
    hi = float(order[max(0, n - n_tail - 1)])
    lo = float(np.quantile(order, 0.50 if n >= 400 else 0.70))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(order[max(0, n // 5)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("could not build a default threshold grid from this series; pass thresholds")
    return np.linspace(lo, hi, 15)


def _candidate_thresholds(x: np.ndarray, thresholds) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if thresholds is None:
        u = _default_thresholds(finite)
    else:
        u = np.asarray(thresholds, dtype=np.float64)
        if u.ndim != 1:
            raise ValueError("thresholds must be a 1-d array of candidate values")
        if u.size == 0 or not np.all(np.isfinite(u)):
            raise ValueError("thresholds must be finite")
        u = np.unique(u)
    n_exc = (finite[:, None] > u[None, :]).sum(axis=0)
    u = u[n_exc >= 2]
    if u.size < 2:
        raise ValueError("need at least two thresholds with two or more exceedances")
    return u


def _check_plot_args(unit, alpha, n_boot) -> tuple[str, float, int]:
    if unit is None:
        unit = ""
    elif not isinstance(unit, str):
        raise TypeError("unit must be a string")
    else:
        unit = unit.strip()
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)):
        raise TypeError("n_boot must be an integer")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    return unit, alpha, n_boot


def _ci_label(alpha: float) -> str:
    pct = 100.0 * (1.0 - alpha)
    text = f"{pct:.0f}" if abs(pct - round(pct)) < 1e-8 else f"{pct:g}"
    return f"{text}% CI"


def _format_mark(value: float, unit: str) -> str:
    text = f"{value:.4g}"
    if unit:
        text = f"{text} {unit}"
    return f"ForwardStop ({text})"


def _series_with_ribbon(ax, x, y, lo, hi, ci_label):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(
        x,
        np.where(ok, lo, np.nan),
        np.where(ok, hi, np.nan),
        color=color(0),
        alpha=0.22,
        linewidth=0,
        zorder=2,
        label=ci_label,
    )
    ax.plot(
        x,
        np.where(np.isfinite(y), y, np.nan),
        color=color(0),
        lw=1.6,
        marker="o",
        ms=4,
        markerfacecolor=color(0),
        markeredgecolor="white",
        markeredgewidth=0.4,
        zorder=3,
    )


def plot_threshold_figure(
    x,
    thresholds=None,
    *,
    unit: str = "",
    alpha: float = 0.05,
    n_boot: int = 199,
    seed: int = 0,
    figsize=(10.0, 3.7),
    fig=None,
):
    """Draw mean excess, modified scale, shape, and the ForwardStop threshold."""
    plt = require_matplotlib()
    unit, alpha, n_boot = _check_plot_args(unit, alpha, n_boot)
    series = _one_series(x)
    u = _candidate_thresholds(series, thresholds)

    from .core.threshold import mean_residual_life, parameter_stability, select_threshold

    mrl = mean_residual_life(series, u, alpha=alpha)
    stability = parameter_stability(series, u, alpha=alpha)
    chosen = float(select_threshold(series, u, alpha=alpha, n_boot=n_boot, seed=seed).threshold)

    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    else:
        axes = np.asarray(fig.axes, dtype=object)
        if axes.size < 3:
            raise ValueError("fig needs at least three axes")
        axes = axes[:3]
    axes = np.asarray(axes, dtype=object).reshape(3)

    xlabel = _with_unit("Threshold", unit)
    mark = _format_mark(chosen, unit)
    ribbon = _ci_label(alpha)
    panels = (
        ("Mean excess", _with_unit("Mean excess", unit), mrl["threshold"], mrl["mean_excess"], mrl["lower"], mrl["upper"]),
        (
            "Modified scale",
            _with_unit("Modified scale", unit),
            stability["threshold"],
            stability["modified_scale"],
            stability["modified_scale_lower"],
            stability["modified_scale_upper"],
        ),
        (
            "Shape",
            r"Shape $\xi$",
            stability["threshold"],
            stability["shape"],
            stability["shape_lower"],
            stability["shape_upper"],
        ),
    )
    for i, (ax, (title, ylabel, px, py, lo, hi)) in enumerate(zip(axes, panels)):
        _series_with_ribbon(ax, px, py, lo, hi, ribbon if i == 0 else None)
        ax.axvline(chosen, color=color(1), lw=1.25, ls="--", zorder=4, label=mark if i == 0 else None)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        apply_axes_style(ax)
    fig.legend(frameon=False, fontsize=8, loc="outside upper center", ncol=2)
    fig.tight_layout()
    return fig, axes


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
