"""Optional matplotlib helpers for diagnostic, threshold and IDF plots.

Install the extra with ``pip install 'rare-tails[plots]'``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["require_matplotlib", "plot_fit_diagnostics"]

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
