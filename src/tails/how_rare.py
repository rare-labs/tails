"""One-liner: how rare is an observed value under a fitted (or just-fitted) model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from .core.blocks import block_maxima
from .core.distributions import gev_cdf
from .core.fit import GEVFit, POTFit, fit_gev, fit_pot

__all__ = ["HowRare", "how_rare"]


@dataclass
class HowRare:
    """Return period of an observed value, with an optional confidence interval."""

    value: float
    return_period: float
    lower: float | None
    upper: float | None
    alpha: float
    sentence: str
    model: str
    fit: Any
    date: pd.Timestamp | None = None
    name: str | None = None
    unit: str | None = None
    ax: Any = None

    def __float__(self) -> float:
        return float(self.return_period)

    def __repr__(self) -> str:  # pragma: no cover
        return f"HowRare({self.sentence})"


def how_rare(
    series,
    value_or_date,
    *,
    times=None,
    model: str = "gev",
    threshold=None,
    npy: float = 365.25,
    ci: str | None = "profile",
    alpha: float = 0.05,
    plot: bool = False,
    ax=None,
    unit: str | None = None,
    name: str | None = None,
    freq: str = "YS",
    min_count: int | None = None,
):
    """Return period of an observed value, as a 1-in-T-year event.

    Parameters
    ----------
    series
        Block maxima (for ``model="gev"``) or a raw series (daily values, a
        pandas Series with a DatetimeIndex, or an array plus ``times``).
        A DatetimeIndex daily/hourly series is reduced to annual maxima
        before a GEV is fitted.
    value_or_date
        The observed magnitude, or a date whose value is looked up in
        ``series`` (requires a DatetimeIndex or ``times``).
    model
        ``"gev"`` (annual / block maxima) or ``"pot"`` (peaks over
        ``threshold``).
    ci
        ``"profile"`` (default), ``"delta"``, ``"bootstrap"``, or ``None``.
        Profile and bootstrap intervals are available for the GEV and for
        peaks-over-threshold.
    plot
        If True, draw a return-level curve, a CI ribbon, and a marker for
        the event. Requires matplotlib (``rare-tails[plots]``).
    unit, name
        Optional wording for the plain-language sentence, e.g. ``unit="mm"``,
        ``"basis points"``, ``"USD per million Btu"``, or ``"Mw"``, and
        ``name="Québec City"``.

    Returns
    -------
    HowRare
        ``return_period``, optional ``lower`` / ``upper``, a ``sentence``,
        the fitted model, and ``ax`` if ``plot=True``.
    """
    s, dates = _as_series(series, times)
    if name is None:
        name = _infer_name(series, s)
    value, event_date = _resolve_value(s, dates, value_or_date)
    annual_index = _is_annual_index(dates, s)

    model = model.lower()
    if model == "gev":
        y = _annual_maxima(s, dates, freq=freq, min_count=min_count)
        fit = fit_gev(np.asarray(y, float))
        if ci == "profile" or (ci is not None and ci not in ("delta", "bootstrap", "profile")):
            if ci not in (None, "delta", "bootstrap", "profile"):
                raise ValueError("ci must be None, 'delta', 'profile' or 'bootstrap'")
        T, lo, hi = _period_from_gev(fit, value, ci=ci, alpha=alpha)
    elif model == "pot":
        if threshold is None:
            raise ValueError("model='pot' requires a threshold")
        if ci not in (None, "delta", "bootstrap", "profile"):
            raise ValueError("ci must be None, 'delta', 'profile' or 'bootstrap'")
        x = np.asarray(s, float)
        fit = fit_pot(x, threshold, npy=npy)
        T, lo, hi = _period_from_pot(fit, value, ci=ci, alpha=alpha)
    else:
        raise ValueError("model must be 'gev' or 'pot'")

    sentence = _sentence(value, T, lo, hi, alpha, unit, name, event_date, annual=annual_index)
    out_ax = None
    if plot or ax is not None:
        out_ax = _plot_event(fit, value, T, lo, hi, alpha, unit, name, event_date, ax, ci)
    return HowRare(
        value=float(value),
        return_period=float(T),
        lower=None if lo is None else float(lo),
        upper=None if hi is None else float(hi),
        alpha=float(alpha),
        sentence=sentence,
        model=model,
        fit=fit,
        date=event_date,
        name=name,
        unit=unit,
        ax=out_ax,
    )


def _as_series(series, times) -> tuple[pd.Series, pd.DatetimeIndex | None]:
    if isinstance(series, pd.Series):
        s = series
        if times is not None:
            s = pd.Series(np.asarray(s, float), index=pd.to_datetime(times))
        idx = s.index if isinstance(s.index, pd.DatetimeIndex) else None
        return s.astype(float), idx
    x = np.asarray(series, float).reshape(-1)
    if times is not None:
        idx = pd.to_datetime(times)
        return pd.Series(x, index=idx), idx
    return pd.Series(x), None


def _infer_name(original, s: pd.Series) -> str | None:
    if isinstance(original, pd.Series) and original.name:
        return str(original.name)
    if s.name:
        return str(s.name)
    return None


def _resolve_value(s: pd.Series, dates, value_or_date) -> tuple[float, pd.Timestamp | None]:
    if _is_date_like(value_or_date):
        if dates is None:
            raise ValueError("a date was given but the series has no DatetimeIndex (pass times=...)")
        ts = pd.Timestamp(value_or_date)
        # Exact timestamp, then same calendar day, then the block that contains the date.
        if ts in s.index:
            v = s.loc[ts]
            return float(np.asarray(v).ravel()[0]), ts.normalize()
        day = s.loc[s.index.normalize() == ts.normalize()]
        if len(day):
            v = day.max()
            return float(np.asarray(v).ravel()[0]), ts.normalize()
        # Yearly / coarse index: pick the period containing the date.
        try:
            pos = dates.get_indexer([ts], method="pad")[0]
        except Exception:
            pos = -1
        if pos >= 0:
            return float(s.iloc[pos]), ts.normalize()
        raise KeyError(f"no observation found for {ts.date()}")
    return float(value_or_date), None


def _is_date_like(v) -> bool:
    if isinstance(v, (datetime, date, pd.Timestamp, np.datetime64)):
        return True
    if isinstance(v, str):
        try:
            pd.Timestamp(v)
        except (ValueError, TypeError):
            return False
        # Don't treat plain numbers-as-strings as dates; require a separator or month name.
        return any(c in v for c in "-/T ") or v[:3].isalpha()
    return False


def _is_annual_index(dates, s) -> bool:
    if dates is None or len(s) < 2:
        return False
    span_days = (dates.max() - dates.min()) / pd.Timedelta(days=1)
    return span_days > 0 and len(s) <= span_days / 180.0 + 2


def _annual_maxima(s: pd.Series, dates, freq: str, min_count: int | None):
    if dates is None:
        return np.asarray(s, float)
    # Already annual (or coarser than ~2 per year): use as-is.
    if _is_annual_index(dates, s):
        return np.asarray(s, float)
    mc = min_count
    if mc is None and freq.upper().startswith("Y"):
        # Daily-ish series: require ~85% of a year.
        inferred = pd.Series(1, index=dates).index.inferred_freq
        if inferred and inferred[0] in "DHT":
            mc = 300 if inferred[0] == "D" else None
    return np.asarray(block_maxima(s, freq=freq, min_count=mc), float)


def _period_from_gev(fit: GEVFit, x: float, ci: str | None, alpha: float):
    loc, scale, shape = (float(fit.params[k]) for k in ("loc", "scale", "shape"))
    p = float(np.asarray(gev_cdf(x, loc, scale, shape)).ravel()[0])
    p = min(max(p, 0.0), 1.0 - 1e-15)
    T = 1.0 / (1.0 - p) if p < 1 else np.inf
    if ci is None or not np.isfinite(T):
        return T, None, None
    lo, hi = _invert_return_level_ci(fit, x, ci=ci, alpha=alpha)
    return T, lo, hi


def _invert_return_level_ci(fit: GEVFit, x: float, ci: str, alpha: float):
    """Bounds on T such that the T-year level equals x, using RL confidence bands."""

    def z(periods, with_ci=True):
        rl = fit.return_level(periods, ci=ci if with_ci else None, alpha=alpha)
        est = np.asarray(rl.estimate).reshape(-1)
        if with_ci and rl.lower is not None:
            return est, np.asarray(rl.lower).reshape(-1), np.asarray(rl.upper).reshape(-1)
        return est, None, None

    def solve(kind: str) -> float:
        # kind: "est", "lower" (z_hi(T)=x → T_low), "upper" (z_lo(T)=x → T_high)
        def f(T):
            est, lo, hi = z([T], with_ci=(kind != "est"))
            if kind == "est":
                return float(est[0]) - x
            if kind == "lower":
                return float(hi[0]) - x
            return float(lo[0]) - x

        # Bracket: T in (1, 1e6)
        lo_t, hi_t = 1.001, 1.0e6
        flo, fhi = f(lo_t), f(hi_t)
        if not np.isfinite(flo) or not np.isfinite(fhi):
            return np.nan
        if flo > 0 and fhi > 0:
            return 1.0  # x is below even the 1-year (upper/lower) level
        if flo < 0 and fhi < 0:
            return np.inf
        for _ in range(60):
            mid = (lo_t * hi_t) ** 0.5
            fm = f(mid)
            if not np.isfinite(fm):
                return np.nan
            if fm > 0:
                hi_t = mid
            else:
                lo_t = mid
        return (lo_t * hi_t) ** 0.5

    T_lo = solve("lower")
    T_hi = solve("upper")
    if T_lo > T_hi and np.isfinite(T_lo) and np.isfinite(T_hi):
        T_lo, T_hi = T_hi, T_lo
    return T_lo, T_hi


def _period_from_pot(fit: POTFit, x: float, ci: str | None, alpha: float):
    # Invert the POT return level: x = u + GPD quantile of 1 - 1/(T npy zeta)
    u = float(np.broadcast_to(np.asarray(fit.threshold, float), fit.batch_shape).ravel()[0])
    scale = float(np.asarray(fit.params["scale"]).ravel()[0])
    shape = float(np.asarray(fit.params["shape"]).ravel()[0])
    zeta = float(np.asarray(fit.zeta).ravel()[0])
    npy = float(fit.npy)
    excess = x - u
    if excess <= 0:
        # Below the threshold: not an exceedance. Return period ≲ 1 / (npy * zeta) * something
        # Using the GPD below 0 is undefined; report "more frequent than the threshold rate".
        T = 1.0 / max(npy * zeta, 1e-15)
        return T, None, None
    # F_GPD(excess) = p_exc; P(X > x) = zeta * (1 - F_GPD)
    from .core.distributions import gpd_cdf

    f_exc = float(np.asarray(gpd_cdf(excess, scale, shape)).ravel()[0])
    p_ex = zeta * (1.0 - min(max(f_exc, 0.0), 1.0 - 1e-15))
    T = 1.0 / max(npy * p_ex, 1e-15)
    if ci is None:
        return T, None, None
    lo, hi = _invert_pot_ci(fit, x, ci=ci or "delta", alpha=alpha)
    return T, lo, hi


def _invert_pot_ci(fit: POTFit, x: float, ci: str, alpha: float):
    def level(periods):
        rl = fit.return_level(periods, ci=ci, alpha=alpha)
        est = np.asarray(rl.estimate).reshape(-1)
        lo = None if rl.lower is None else np.asarray(rl.lower).reshape(-1)
        hi = None if rl.upper is None else np.asarray(rl.upper).reshape(-1)
        return est, lo, hi

    def solve(kind: str) -> float:
        def f(T):
            est, lo, hi = level([T])
            if kind == "lower":
                return float(hi[0]) - x
            return float(lo[0]) - x

        lo_t, hi_t = 1.001, 1.0e6
        flo, fhi = f(lo_t), f(hi_t)
        if not np.isfinite(flo) or not np.isfinite(fhi):
            return np.nan
        if flo > 0 and fhi > 0:
            return 1.0
        if flo < 0 and fhi < 0:
            return np.inf
        for _ in range(60):
            mid = (lo_t * hi_t) ** 0.5
            fm = f(mid)
            if not np.isfinite(fm):
                return np.nan
            if fm > 0:
                hi_t = mid
            else:
                lo_t = mid
        return (lo_t * hi_t) ** 0.5

    T_lo, T_hi = solve("lower"), solve("upper")
    if T_lo > T_hi and np.isfinite(T_lo) and np.isfinite(T_hi):
        T_lo, T_hi = T_hi, T_lo
    return T_lo, T_hi


def _fmt_years(T: float) -> str:
    if not np.isfinite(T):
        return "more than a million years"
    if T < 1.05:
        return f"{T:.2f}"
    if T < 10:
        return f"{T:.1f}"
    return f"{T:.0f}"


def _sentence(value, T, lo, hi, alpha, unit, name, event_date, annual: bool = False) -> str:
    unit_s = f" {unit}" if unit else ""
    where = f" at {name}" if name else ""
    if event_date is None:
        when = ""
    elif annual:
        when = f" in {pd.Timestamp(event_date).year}"
    else:
        when = f" on {pd.Timestamp(event_date).date().isoformat()}"
    head = f"A value of {value:g}{unit_s}{when}{where}"
    if not np.isfinite(T):
        return f"{head} is at or beyond the upper end of the fitted distribution."
    ev = f"{head} is a 1-in-{_fmt_years(T)}-year event"
    if lo is not None and hi is not None and (np.isfinite(lo) or np.isfinite(hi)):
        pct = 100 * (1 - alpha)
        return f"{ev} ({pct:.0f}% CI: {_fmt_years(lo)}–{_fmt_years(hi)} years)."
    return ev + "."


def _plot_event(fit, value, T, lo, hi, alpha, unit, name, event_date, ax, ci):
    from .plotting import apply_axes_style, color, require_matplotlib

    plt = require_matplotlib()
    ax = ax or plt.subplots(figsize=(6.4, 4.0))[1]
    tmax = 100.0 if not np.isfinite(T) else max(100.0, 2 * T, 10.0)
    periods = np.geomspace(1.1, tmax, 80)
    plot_ci = ci if ci in ("delta", "profile", "bootstrap") else None
    try:
        rl = fit.return_level(periods, ci=plot_ci, alpha=alpha)
    except Exception:
        rl = fit.return_level(periods, ci="delta", alpha=alpha)
    est = np.asarray(rl.estimate).reshape(-1)
    ax.plot(periods, est, color=color(0), lw=1.8, zorder=3, label="Fitted return level")
    if rl.lower is not None:
        ax.fill_between(
            periods,
            np.asarray(rl.lower).reshape(-1),
            np.asarray(rl.upper).reshape(-1),
            color=color(0),
            alpha=0.22,
            linewidth=0,
            zorder=2,
            label=f"{100 * (1 - alpha):.0f}% CI",
        )
    if np.isfinite(T):
        ax.axhline(value, color="0.35", lw=0.8, ls=":")
        ax.axvline(T, color="0.35", lw=0.8, ls=":")
        ax.scatter([T], [value], s=40, color=color(1), zorder=5, label="This event")
        if lo is not None and np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(value, lo, hi, color=color(1), lw=2.0, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel(f"Return level ({unit})" if unit else "Return level")
    title = "How rare was this?"
    if name:
        title += f" — {name}"
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    apply_axes_style(ax)
    return ax
