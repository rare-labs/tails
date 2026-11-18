"""One datetime series: block maxima or peaks, a fitted model, a return-level table.

The fits themselves are :func:`tails.fit_gev` and :func:`tails.fit_pot`. This
module only extracts the sample (with :func:`tails.block_maxima` or
:func:`tails.decluster_runs`) and, if asked, chooses between the full model
and the nested model with shape ξ fixed at 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .core.blocks import block_maxima, decluster_runs
from .core.fit import POTFit, fit_gev, fit_pot

__all__ = ["SeriesExtremes", "fit_block_maxima", "fit_peaks"]

_BLOCK_MODELS = {"gev", "gumbel", "aic"}
_PEAK_MODELS = {"gpd", "exponential", "aic"}


@dataclass
class SeriesExtremes:
    """Extremes taken from one datetime series, and the model fitted to them.

    ``model`` is ``"gev"`` or ``"gumbel"`` for block maxima, and ``"gpd"`` or
    ``"exponential"`` for peaks. Gumbel and exponential are the nested models
    with shape ξ fixed at 0. ``aic_scores`` is set when ``model="aic"`` was
    requested: a map from candidate name to AIC, and ``model`` is the winner
    (the smaller AIC; a tie goes to the model with ξ fixed).

    ``fit`` is the underlying :class:`~tails.core.fit.GEVFit` or
    :class:`~tails.core.fit.POTFit`. ``extremes`` is a pandas Series of the
    extracted values, indexed by block start (block maxima, NaN where a block
    was dropped) or by the time of each peak.
    """

    kind: str
    model: str
    fit: Any
    extremes: pd.Series
    aic_scores: dict[str, float] | None = None
    threshold: float | None = None
    npy: float | None = None
    freq: str | None = None

    @property
    def aic(self) -> float:
        return float(np.asarray(self.fit.aic).ravel()[0])

    @property
    def shape(self) -> float:
        """Fitted shape ξ. Zero when the Gumbel or exponential model was chosen."""
        return float(np.asarray(self.fit.params["shape"]).ravel()[0])

    def parameters(self) -> pd.DataFrame:
        """Natural parameters and standard errors. ``shape`` is ξ.

        A standard error of 0 means that parameter was held fixed. The scale
        row is the scale itself; its standard error is the delta-method
        transform of the log-scale coefficient.
        """
        params = self.fit.params
        se = np.asarray(self.fit.se).ravel()
        rows: list[tuple[str, float, float]] = []
        if "loc" in params:
            rows.append(("loc", float(np.asarray(params["loc"]).ravel()[0]), float(se[0])))
            log_se, shape_se = float(se[1]), float(se[2])
        else:
            log_se, shape_se = float(se[0]), float(se[1])
        scale = float(np.asarray(params["scale"]).ravel()[0])
        rows.append(("scale", scale, abs(scale) * log_se))
        rows.append(("shape", self.shape, shape_se))
        if isinstance(self.fit, POTFit):
            rows.append(("rate", float(np.asarray(self.fit.rate).ravel()[0]), np.nan))
        df = pd.DataFrame(rows, columns=["term", "estimate", "std_error"])
        df.attrs["model"] = self.model
        df.attrs["n"] = int(np.asarray(self.fit.nobs).ravel()[0])
        df.attrs["aic"] = self.aic
        return df

    def summary(
        self,
        periods: Any = (2, 5, 10, 20, 50, 100),
        *,
        ci: str | None = None,
        alpha: float = 0.05,
        n_boot: int = 500,
        seed: int = 0,
        profile_points: int = 41,
    ) -> pd.DataFrame:
        """Return levels at ``periods``, one row per period.

        Columns are ``period``, ``estimate``, ``lower`` and ``upper``. For
        block maxima the default interval is the profile likelihood (shape
        stays fixed when the model is Gumbel). For peaks the default is the
        delta-method interval. ``ci="profile"`` and ``ci="bootstrap"`` are
        available for both; an exponential peaks model keeps ξ fixed in the
        profile interval.

        The frame's ``attrs`` record the model, ξ, AIC, the interval and, when
        a comparison was run, the candidate AICs.
        """
        if ci is None:
            ci = "delta" if isinstance(self.fit, POTFit) else "profile"
        rl = self.fit.return_level(
            np.atleast_1d(np.asarray(periods, dtype=float)),
            ci=ci,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            profile_points=profile_points,
        )
        table = rl.to_pandas()
        # One series: the batch columns are empty. Keep the return-level columns only.
        keep = [c for c in ("period", "estimate", "lower", "upper") if c in table.columns]
        table = table[keep].reset_index(drop=True)
        table.attrs["kind"] = self.kind
        table.attrs["model"] = self.model
        table.attrs["shape"] = self.shape
        table.attrs["aic"] = self.aic
        table.attrs["ci"] = ci
        table.attrs["alpha"] = alpha
        table.attrs["n"] = int(np.asarray(self.fit.nobs).ravel()[0])
        if self.aic_scores is not None:
            table.attrs["aic_scores"] = dict(self.aic_scores)
        return table

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        p = self.fit.params
        bits = [f"kind={self.kind!r}", f"model={self.model!r}", f"n={int(np.asarray(self.fit.nobs).ravel()[0])}"]
        if "loc" in p:
            bits.append(f"loc={float(np.asarray(p['loc']).ravel()[0]):.4g}")
        bits.append(f"scale={float(np.asarray(p['scale']).ravel()[0]):.4g}")
        fixed = " fixed" if self.fit.free_params is not None else ""
        bits.append(f"ξ={self.shape:.4g}{fixed}")
        if self.aic_scores:
            shown = ", ".join(f"{name}={score:.2f}" for name, score in self.aic_scores.items())
            bits.append(f"AIC {shown}")
        else:
            bits.append(f"AIC={self.aic:.2f}")
        return "SeriesExtremes(" + ", ".join(bits) + ")"


def fit_block_maxima(
    series,
    *,
    freq: str = "YS",
    min_count: int | None = None,
    model: str = "gev",
    times=None,
) -> SeriesExtremes:
    """Block maxima of one datetime series, then a GEV or Gumbel fit.

    Parameters
    ----------
    series
        A pandas Series with a :class:`~pandas.DatetimeIndex`, or a 1-d array
        with ``times``. One column of a DataFrame is accepted. Several
        stations still go through :func:`tails.block_maxima` and
        :func:`tails.fit_gev`.
    freq
        Block length as a pandas offset alias. ``"YS"`` is one block per year.
    min_count
        Blocks with fewer finite observations become missing. Incomplete blocks
        pull maxima down; set this when a year should not count (300 is a
        typical daily-data rule).
    model
        ``"gev"`` (default), ``"gumbel"`` (ξ fixed at 0), or ``"aic"`` to fit
        both and keep the smaller AIC. A tie keeps the Gumbel model.
    times
        Timestamps, one per observation, when ``series`` is not already indexed
        by time.

    Returns
    -------
    SeriesExtremes
        The block-maxima series, the fitted model, and
        :meth:`SeriesExtremes.summary` for a return-level table. Shape is ξ.
    """
    model = str(model).lower()
    if model not in _BLOCK_MODELS:
        raise ValueError("model must be 'gev', 'gumbel' or 'aic'")
    s = _coerce_series(series, times, what="fit_block_maxima")
    blocks = block_maxima(s, freq=freq, min_count=min_count)
    if not isinstance(blocks, pd.Series):
        raise TypeError("fit_block_maxima expected block_maxima to return a Series for one series")
    blocks = pd.Series(np.asarray(blocks, dtype=float), index=blocks.index, name=s.name)
    y = np.asarray(blocks, dtype=float)
    fit, chosen, scores = _choose_block_model(y, model)
    return SeriesExtremes(
        kind="blocks",
        model=chosen,
        fit=fit,
        extremes=blocks,
        aic_scores=scores,
        freq=freq,
    )


def fit_peaks(
    series,
    threshold: float,
    *,
    npy: float | None = None,
    run_length: int | str | None = None,
    model: str = "gpd",
    times=None,
) -> SeriesExtremes:
    """Peaks over a threshold from one datetime series, then a POT fit.

    Parameters
    ----------
    series
        A pandas Series with a DatetimeIndex, or a 1-d array plus ``times``.
    threshold
        Values strictly above this level can be peaks.
    npy
        Observations per year, used to turn the exceedance rate into a yearly
        rate. The default infers it from the median spacing of the index
        (about 365.25 for daily data, about 1 for a series that is already annual).
    run_length
        Runs declustering before the fit. An integer is the run length in
        observations, passed to :func:`tails.decluster_runs`: exceedances
        separated by fewer than that many non-exceedances stay in one cluster,
        and only the cluster maximum is kept. A pandas offset such as
        ``"24h"`` is converted to a number of observations with the median
        spacing of the index. ``None`` (default) does not decluster.
    model
        ``"gpd"`` (default), ``"exponential"`` (ξ fixed at 0), or ``"aic"`` to
        fit both excess models and keep the smaller AIC. The exceedance rate
        is the same for both, so the comparison is the excess likelihood only.
        A tie keeps the exponential model.

    Returns
    -------
    SeriesExtremes
        Cluster peaks (or every exceedance, if not declustered), the fitted
        POT model, and :meth:`SeriesExtremes.summary`. Shape is ξ. Intervals
        from ``summary`` use the delta method.
    """
    model = str(model).lower()
    if model not in _PEAK_MODELS:
        raise ValueError("model must be 'gpd', 'exponential' or 'aic'")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    s = _coerce_series(series, times, what="fit_peaks")
    per_year = float(npy) if npy is not None else _observations_per_year(s.index)
    if not np.isfinite(per_year) or per_year <= 0:
        raise ValueError("npy must be a positive finite number")
    length = _run_length_observations(run_length, s.index)
    values = np.asarray(s, dtype=float)
    if length is None:
        chosen_fit, chosen, scores = _choose_peak_model(values, float(threshold), per_year, None, model)
        peaks = values.astype(float).copy()
        peaks[~np.isfinite(peaks) | (peaks <= threshold)] = np.nan
    else:
        declustered = decluster_runs(values, float(threshold), run_length=length)
        peaks = np.asarray(declustered, dtype=float).reshape(-1)
        chosen_fit, chosen, scores = _choose_peak_model(values, float(threshold), per_year, length, model)
    extremes = pd.Series(peaks, index=s.index, name=s.name).dropna()
    return SeriesExtremes(
        kind="peaks",
        model=chosen,
        fit=chosen_fit,
        extremes=extremes,
        aic_scores=scores,
        threshold=float(threshold),
        npy=per_year,
    )


def _choose_block_model(y, model: str):
    if model == "gumbel":
        return fit_gev(y, fixed_shape=0.0), "gumbel", None
    gev = fit_gev(y)
    if model == "gev":
        return gev, "gev", None
    gum = fit_gev(y, fixed_shape=0.0)
    scores = {"gev": float(np.asarray(gev.aic).ravel()[0]), "gumbel": float(np.asarray(gum.aic).ravel()[0])}
    if scores["gumbel"] <= scores["gev"]:
        return gum, "gumbel", scores
    return gev, "gev", scores


def _choose_peak_model(values, threshold: float, npy: float, run_length: int | None, model: str):
    kwargs = {"npy": npy, "decluster": run_length}
    if model == "exponential":
        return fit_pot(values, threshold, fixed_shape=0.0, **kwargs), "exponential", None
    gpd = fit_pot(values, threshold, **kwargs)
    if model == "gpd":
        return gpd, "gpd", None
    expo = fit_pot(values, threshold, fixed_shape=0.0, **kwargs)
    scores = {
        "gpd": float(np.asarray(gpd.aic).ravel()[0]),
        "exponential": float(np.asarray(expo.aic).ravel()[0]),
    }
    if scores["exponential"] <= scores["gpd"]:
        return expo, "exponential", scores
    return gpd, "gpd", scores


def _coerce_series(series, times, what: str) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        numeric = [c for c in series.columns if pd.api.types.is_numeric_dtype(series[c])]
        if len(numeric) != 1:
            raise ValueError(
                f"{what} takes one series. For many stations call tails.block_maxima "
                "or tails.decluster_runs, then tails.fit_gev or tails.fit_pot."
            )
        series = series[numeric[0]]
    if isinstance(series, pd.Series):
        name = series.name
        values = np.asarray(series, dtype=float).reshape(-1)
        if times is not None:
            index = pd.DatetimeIndex(pd.to_datetime(times))
        elif isinstance(series.index, pd.DatetimeIndex):
            index = series.index
        else:
            raise TypeError(f"{what} needs a DatetimeIndex, or pass times= with one timestamp per observation")
    else:
        values = np.asarray(series, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"{what} takes one series; got shape {values.shape}")
        if times is None:
            raise TypeError(f"{what} needs a DatetimeIndex, or pass times= with one timestamp per observation")
        index = pd.DatetimeIndex(pd.to_datetime(times))
        name = None
    if len(index) != len(values):
        raise ValueError(f"times has length {len(index)} but the series has length {len(values)}")
    out = pd.Series(values, index=index, name=name).sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError(f"{what} could not build a DatetimeIndex from times")
    return out


def _observations_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 365.25
    step = pd.Series(index).diff().dt.total_seconds().median()
    if step is None or not np.isfinite(step) or step <= 0:
        return 365.25
    return float(365.25 * 86400.0 / float(step))


def _run_length_observations(run_length, index: pd.DatetimeIndex) -> int | None:
    if run_length is None:
        return None
    if isinstance(run_length, (int, np.integer)):
        if int(run_length) < 1:
            raise ValueError("run_length must be at least 1 observation")
        return int(run_length)
    if isinstance(run_length, (float, np.floating)):
        raise TypeError("run_length must be an integer number of observations or a pandas offset such as '24h'")
    window = pd.Timedelta(pd.tseries.frequencies.to_offset(run_length))
    if len(index) < 3:
        raise ValueError("a time-span run_length needs at least three timestamps to infer the spacing")
    step = pd.Series(index).diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("could not infer a positive spacing from the index")
    return max(int(np.ceil(window / step)), 1)
