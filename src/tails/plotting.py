"""Optional matplotlib helpers for diagnostic and IDF plots.

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
            ax_rl.scatter(periods, emp, s=22, color="0.15", zorder=4, label="Observed annual maxima (empirical return period)")
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
    if fit.__class__.__name__ == "POTFit":
        return ci in (None, "delta")
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
            label="Observed annual maxima (empirical return period)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Duration (h)")
    ax.set_ylabel("Intensity (mm/h)")
    if title is not None:
        ax.set_title(title)
    elif fit.stations:
        ax.set_title(str(fit.stations[station]))
    ax.legend(frameon=False, fontsize=8)
    apply_axes_style(ax)
    return ax
