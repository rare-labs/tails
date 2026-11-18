import numpy as np
import pytest

import tails

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _gev_fit(n=80, seed=0):
    rng = np.random.default_rng(seed)
    # Gumbel(30, 8) is GEV with shape 0
    y = rng.gumbel(30, 8, size=n)
    return tails.fit_gev(y), y


def test_plot_diagnostics_returns_2x2_and_matches_return_levels():
    fit, y = _gev_fit()
    fig, axes = fit.plot_diagnostics(ci="delta", n_band=40, seed=1)
    assert axes.shape == (2, 2)
    rl_ax = axes[1, 0]
    # Fitted line x-data are return periods > 1, y-data close to return_level estimates.
    line = rl_ax.get_lines()[0]
    T = np.asarray(line.get_xdata())
    est = np.asarray(line.get_ydata())
    rl = fit.return_level(T, ci=None)
    np.testing.assert_allclose(est, np.asarray(rl.estimate).reshape(-1), rtol=1e-6, atol=1e-6)
    # A CI ribbon should be present.
    assert any("PolyCollection" in c.__class__.__name__ and c.get_alpha() for c in rl_ax.collections)
    matplotlib.pyplot.close(fig)


def test_return_levels_plot_ribbon():
    fit, _ = _gev_fit()
    rl = fit.return_level([2, 5, 10, 20, 50, 100], ci="delta")
    ax = rl.plot()
    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), np.asarray(rl.estimate).reshape(-1), rtol=1e-6)
    assert any("PolyCollection" in c.__class__.__name__ for c in ax.collections)
    matplotlib.pyplot.close(ax.figure)


def test_idf_plot_units_ribbons_and_empirical_legend():
    from tails.idf import IDFData, fit_idf
    from tails.plotting import empirical_return_levels

    data = IDFData.from_eccc("701S001")
    fit = fit_idf(data, "general")
    periods = (2, 10, 25, 100)
    ax = fit.plot(periods=periods, ci="hessian")
    # Duration axis is in hours (5 min ≈ 0.083 h, 24 h), intensity in mm/h.
    # A 24-h 100-year intensity should be a few mm/h (ECCC Gumbel depth is ~104 mm).
    q24 = float(fit.quantile([24.0], [100])[0, 0, 0])
    assert 2.0 < q24 < 15.0
    q1h = float(fit.quantile([1.0], [100])[0, 0, 0])
    assert 20.0 < q1h < 80.0
    # Each return-period line evaluated at 1 h matches quantile().
    for line, T in zip(ax.get_lines(), periods):
        xd, yd = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        y_at_1 = np.interp(1.0, xd, yd)
        q = float(fit.quantile([1.0], [T])[0, 0, 0])
        np.testing.assert_allclose(y_at_1, q, rtol=5e-3)
    # Ribbons present; legend names the empirical points (not unexplained grey dots).
    assert any("PolyCollection" in c.__class__.__name__ for c in ax.collections)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Observed (empirical return period)" in labels
    assert any("2-year" in t or "2-yr" in t for t in labels)
    # Empirical 2-year point at 1 h is the Weibull quantile of the 1-h column.
    col = list(data.durations).index(1.0)
    emp2 = float(empirical_return_levels(data.values[:, col], 2)[0])
    # Find the 2-year scatter (first PathCollection after ribbons, y near emp2)
    scatter_y = []
    for c in ax.collections:
        if c.__class__.__name__ == "PathCollection":
            offs = c.get_offsets()
            if offs.size:
                scatter_y.append(offs)
    # At least one scatter x includes 1 h and y close to the empirical 2-yr value
    found = False
    for offs in scatter_y:
        arr = np.asarray(offs)
        close = np.isclose(arr[:, 0], 1.0, rtol=0, atol=1e-9)
        if close.any() and np.isclose(arr[close][0, 1], emp2, rtol=1e-6):
            found = True
    assert found, "empirical 2-year 1-h point not plotted at the Weibull quantile"
    matplotlib.pyplot.close(ax.figure)


def _trend_line(ax):
    lines = ax.get_lines()
    assert len(lines) == 1
    return lines[0]


def _ribbon(ax):
    polys = [c for c in ax.collections if "Poly" in type(c).__name__]
    assert len(polys) == 1
    path = polys[0].get_paths()[0]
    verts = np.asarray(path.vertices, float)
    if path.codes is not None:
        verts = verts[np.asarray(path.codes) != matplotlib.path.Path.CLOSEPOLY]
    return verts


def test_trend_diagnostics_follows_the_covariate_and_delta_interval():
    rng = np.random.default_rng(0)
    n = 48
    # Values sit in a calendar range. The axis must not turn into "Year" on its own.
    covariate = rng.permutation(np.arange(1961.0, 1961.0 + n))
    y = rng.gumbel(20.0 + 0.12 * (covariate - 1990.0), 1.6)
    fit = tails.fit_gev(y, loc=covariate)
    fig, ax = fit.plot_diagnostics(ci="delta", period=2)
    assert ax.get_xlabel() == "Covariate"
    assert ax.get_ylabel() == "Block maximum"
    assert ax.get_title() == "Fitted trend"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Observed" in labels
    assert "Fitted 2-year level" in labels
    assert "95% CI" in labels
    text = " ".join([ax.get_xlabel(), ax.get_ylabel(), ax.get_title(), *labels]).lower()
    assert "rain" not in text and "mm" not in text

    rl = fit.return_level([2.0], ci="delta")
    est = np.asarray(rl.estimate, float).reshape(-1)
    lo = np.asarray(rl.lower, float).reshape(-1)
    hi = np.asarray(rl.upper, float).reshape(-1)
    order = np.argsort(covariate, kind="mergesort")
    line = _trend_line(ax)
    np.testing.assert_allclose(line.get_xdata(), covariate[order])
    np.testing.assert_allclose(line.get_ydata(), est[order])
    assert np.all(np.diff(np.asarray(line.get_xdata())) >= -1e-12)
    verts = _ribbon(ax)
    np.testing.assert_allclose(verts[:, 1].min(), lo.min())
    np.testing.assert_allclose(verts[:, 1].max(), hi.max())
    np.testing.assert_allclose(verts[:, 0].min(), covariate.min())
    np.testing.assert_allclose(verts[:, 0].max(), covariate.max())
    scatter = next(c for c in ax.collections if type(c).__name__ == "PathCollection")
    got = np.asarray(scatter.get_offsets())
    got = got[np.argsort(got[:, 0], kind="mergesort")]
    expect = np.column_stack([covariate, y])
    expect = expect[np.argsort(expect[:, 0], kind="mergesort")]
    np.testing.assert_allclose(got, expect)
    matplotlib.pyplot.close(fig)

    fig10, ax10 = fit.plot_diagnostics(period=10, ci="delta")
    rl10 = fit.return_level([10.0], ci="delta")
    est10 = np.asarray(rl10.estimate, float).reshape(-1)
    np.testing.assert_allclose(_trend_line(ax10).get_ydata(), est10[order])
    assert "Fitted 10-year level" in [t.get_text() for t in ax10.get_legend().get_texts()]
    matplotlib.pyplot.close(fig10)

    fig_a, ax_a = fit.plot_diagnostics(ci="delta", alpha=0.2)
    assert "80% CI" in [t.get_text() for t in ax_a.get_legend().get_texts()]
    matplotlib.pyplot.close(fig_a)


def test_trend_diagnostics_uses_a_named_covariate_and_profile_interval():
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 28
    temperature = pd.Series(np.linspace(-1.5, 2.0, n), name="temperature")
    y = rng.gumbel(8.0 + 1.1 * temperature.to_numpy(), 0.9)
    y[4] = np.nan
    fit = tails.fit_gev(y, loc=temperature)
    fig, ax = fit.plot_diagnostics(ci="profile", period=2, profile_points=9)
    assert ax.get_xlabel() == "temperature"
    rl = fit.return_level([2.0], ci="profile", profile_points=9)
    est = np.asarray(rl.estimate, float).reshape(-1)
    lo = np.asarray(rl.lower, float).reshape(-1)
    hi = np.asarray(rl.upper, float).reshape(-1)
    order = np.argsort(temperature.to_numpy(), kind="mergesort")
    line = _trend_line(ax)
    np.testing.assert_allclose(line.get_ydata(), est[order], rtol=1e-6, atol=1e-6)
    verts = _ribbon(ax)
    np.testing.assert_allclose(verts[:, 1].min(), np.nanmin(lo))
    np.testing.assert_allclose(verts[:, 1].max(), np.nanmax(hi))
    assert np.all(lo < est) and np.all(est < hi)
    scatter = next(c for c in ax.collections if type(c).__name__ == "PathCollection")
    assert len(scatter.get_offsets()) == n - 1
    assert len(line.get_xdata()) == n
    matplotlib.pyplot.close(fig)


def test_trend_diagnostics_scale_batch_and_several_columns():
    rng = np.random.default_rng(2)
    n = 40
    t = np.linspace(-1.0, 1.0, n)
    scale = np.exp(0.4 + 0.5 * t)
    y = 12.0 - scale * np.log(-np.log(rng.random(n)))
    scale_fit = tails.fit_gev(y, scale=t)
    fig, ax = scale_fit.plot_diagnostics(ci=None, period=2)
    assert ax.get_xlabel() == "Covariate"
    assert not any("Poly" in type(c).__name__ for c in ax.collections)
    est = np.asarray(scale_fit.return_level([2.0], ci=None).estimate, float).reshape(-1)
    order = np.argsort(t, kind="mergesort")
    np.testing.assert_allclose(_trend_line(ax).get_ydata(), est[order])
    # A scale trend moves the median; the curve is not a horizontal line.
    assert _trend_line(ax).get_ydata()[-1] != pytest.approx(_trend_line(ax).get_ydata()[0], rel=0.01)
    matplotlib.pyplot.close(fig)

    # The same column on location and scale is still that one covariate.
    both = tails.fit_gev(rng.gumbel(4.0 + 0.8 * t, np.exp(0.1 + 0.2 * t)), loc=t, scale=t)
    fig, ax = both.plot_diagnostics(ci="delta")
    assert ax.get_xlabel() == "Covariate"
    est = np.asarray(both.return_level([2.0], ci="delta").estimate, float).reshape(-1)
    np.testing.assert_allclose(_trend_line(ax).get_ydata(), est[order])
    matplotlib.pyplot.close(fig)

    a = np.linspace(0.0, 1.0, n)
    b = np.linspace(1.0, -1.0, n)
    y_ab = rng.gumbel(3.0 + 0.7 * a - 0.4 * b, 0.8)
    wide = tails.fit_gev(y_ab, loc=np.column_stack([a, b]))
    fig, ax = wide.plot_diagnostics(ci="delta")
    assert ax.get_xlabel() == "Observation"
    est = np.asarray(wide.return_level([2.0], ci="delta").estimate, float).reshape(-1)
    np.testing.assert_allclose(_trend_line(ax).get_xdata(), np.arange(n))
    np.testing.assert_allclose(_trend_line(ax).get_ydata(), est)
    matplotlib.pyplot.close(fig)

    t1 = np.linspace(0.0, 3.0, n)
    t2 = np.linspace(-2.0, 2.0, n)
    cov = np.stack([t1, t2], axis=0)[..., None]
    yy = np.stack([rng.gumbel(1.0 + 0.5 * t1, 0.7), rng.gumbel(6.0 + 1.2 * t2, 0.5)])
    batched = tails.fit_gev(yy, loc=cov)
    fig, ax = batched.plot_diagnostics(index=1, ci="delta", period=2)
    est = np.asarray(batched.return_level([2.0], ci="delta").estimate, float)[1, :, 0]
    order2 = np.argsort(t2, kind="mergesort")
    np.testing.assert_allclose(_trend_line(ax).get_xdata(), t2[order2])
    np.testing.assert_allclose(_trend_line(ax).get_ydata(), est[order2])
    scatter = next(c for c in ax.collections if type(c).__name__ == "PathCollection")
    got = np.asarray(scatter.get_offsets())
    np.testing.assert_allclose(np.sort(got[:, 0]), np.sort(t2))
    matplotlib.pyplot.close(fig)


def test_trend_diagnostics_rejects_nonstationary_gpd():
    rng = np.random.default_rng(3)
    n = 40
    excess = rng.exponential(1.5, size=n)
    fit = tails.fit_gpd(excess, scale=np.linspace(0.0, 1.0, n))
    assert not fit.stationary
    with pytest.raises(ValueError, match="non-stationary GEV"):
        fit.plot_diagnostics()


def test_plot_requires_matplotlib_message(monkeypatch):
    import tails.plotting as p

    def boom():
        raise ImportError("no matplotlib")

    monkeypatch.setattr("matplotlib.pyplot.subplots", boom)
    # require_matplotlib itself: simulate missing module
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("No module named matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="rare-tails\\[plots\\]"):
        p.require_matplotlib()
