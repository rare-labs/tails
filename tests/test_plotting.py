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
    assert "Observed annual maxima (empirical return period)" in labels
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
