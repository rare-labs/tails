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

def _scatter(ax):
    return next(c for c in ax.collections if c.__class__.__name__ == "PathCollection")

def _mesh(ax):
    return next(c for c in ax.collections if "QuadMesh" in c.__class__.__name__)

def test_plot_return_level_map_scatters_finite_coordinates():
    lon = np.array([-73.0, -71.4, -68.2])
    lat = np.array([45.5, 46.8, 48.4])
    values = np.array([40.0, 55.0, np.nan])
    ax = tails.plot_return_level_map(values, lon, lat, period=100, label="level (m)")
    coll = _scatter(ax)
    offs = np.asarray(coll.get_offsets())
    np.testing.assert_allclose(offs, np.column_stack([lon[:2], lat[:2]]))
    np.testing.assert_allclose(np.asarray(coll.get_array(), float), values[:2])
    assert ax.get_xlabel() == "Longitude"
    assert ax.get_ylabel() == "Latitude"
    assert ax.get_title() == "100-year return level"
    assert coll.colorbar.ax.get_ylabel() == "level (m)"
    aspect = 1.0 / np.cos(np.deg2rad(np.mean(lat[:2])))
    assert ax.get_aspect() == pytest.approx(aspect, rel=1e-6)
    matplotlib.pyplot.close(ax.figure)

def test_plot_return_level_map_from_return_levels_selects_period():
    rng = np.random.default_rng(1)
    fit = tails.fit_gev(rng.gumbel(30, 8, size=(4, 40)))
    rl = fit.return_level([20, 100])
    lon = np.linspace(-72.0, -70.0, 4)
    lat = np.linspace(45.0, 47.0, 4)
    with pytest.raises(ValueError, match="period"):
        tails.plot_return_level_map(rl, lon, lat)
    with pytest.raises(ValueError, match="not among"):
        rl.plot_map(lon, lat, period=3)
    ax = rl.plot_map(lon, lat, period=100)
    np.testing.assert_allclose(_scatter(ax).get_array(), np.asarray(rl.estimate)[:, 1], rtol=1e-6)
    assert _scatter(ax).colorbar.ax.get_ylabel() == "Return level"
    matplotlib.pyplot.close(ax.figure)
    one = fit.return_level([100])
    ax = tails.plot_return_level_map(one, lon, lat)
    assert ax.get_title() == "100-year return level"
    np.testing.assert_allclose(_scatter(ax).get_array(), np.asarray(one.estimate)[:, 0], rtol=1e-6)
    # Same function as the package attribute.
    import tails.plotting as plotting

    assert tails.plot_return_level_map is plotting.plot_return_level_map
    matplotlib.pyplot.close(ax.figure)

def test_plot_return_level_map_rejects_a_covariate_axis():
    class _Levels:
        # batch, covariate rows, one period — a non-stationary ReturnLevels layout
        period = np.array([100.0])
        estimate = np.ones((4, 3, 1))

    with pytest.raises(ValueError, match="covariate"):
        tails.plot_return_level_map(_Levels(), np.arange(4.0), np.arange(4.0) + 45.0)

def test_plot_return_level_map_meshes_a_gridded_batch():
    rng = np.random.default_rng(2)
    fit = tails.fit_gev(rng.gumbel(30, 8, size=(3, 4, 25)))
    rl = fit.return_level([100])
    lon = np.linspace(-80.0, -70.0, 4)
    lat = np.linspace(45.0, 48.0, 3)
    ax = rl.plot_map(lon, lat)
    painted = np.asarray(_mesh(ax).get_array(), float).reshape(3, 4)
    np.testing.assert_allclose(painted, np.asarray(rl.estimate)[..., 0], rtol=1e-6)
    matplotlib.pyplot.close(ax.figure)

def test_plot_return_level_map_mesh_uses_lat_lon_order():
    lon = np.linspace(-80.0, -60.0, 5)
    lat = np.linspace(45.0, 55.0, 4)
    values = np.arange(20.0).reshape(4, 5)
    values[0, 0] = np.nan
    ax = tails.plot_return_level_map(values, lon, lat)
    painted = np.asarray(_mesh(ax).get_array(), float).ravel()
    np.testing.assert_allclose(painted, values.ravel(), equal_nan=True)
    assert ax.get_title() == ""
    matplotlib.pyplot.close(ax.figure)

def test_plot_return_level_map_reads_xarray_coordinates_and_units():
    xr = pytest.importorskip("xarray")
    lon = np.array([-73.0, -71.0, -69.0])
    lat = np.array([45.0, 46.5, 48.0])
    grid = np.arange(12.0).reshape(3, 4)
    da = xr.DataArray(
        np.stack([grid, grid + 5.0], axis=-1),
        dims=("latitude", "longitude", "period"),
        coords={
            "latitude": np.linspace(45.0, 47.0, 3),
            "longitude": np.linspace(-74.0, -71.0, 4),
            "period": [20.0, 100.0],
        },
        name="return_level",
        attrs={"units": "m3 s-1"},
    )
    with pytest.raises(ValueError, match="period"):
        tails.plot_return_level_map(da)
    ax = tails.plot_return_level_map(da, period=100)
    painted = np.asarray(_mesh(ax).get_array(), float).ravel()
    np.testing.assert_allclose(painted, (grid + 5.0).ravel())
    assert _mesh(ax).colorbar.ax.get_ylabel() == "Return level (m3 s-1)"
    matplotlib.pyplot.close(ax.figure)
    # Dimensions stored as (longitude, latitude) are drawn as (latitude, longitude).
    swapped = xr.DataArray(
        (grid + 5.0).T,
        dims=("longitude", "latitude"),
        coords={
            "longitude": np.linspace(-74.0, -71.0, 4),
            "latitude": np.linspace(45.0, 47.0, 3),
        },
    )
    ax_swap = tails.plot_return_level_map(swapped)
    painted_swap = np.asarray(_mesh(ax_swap).get_array(), float).reshape(grid.shape)
    np.testing.assert_allclose(painted_swap, grid + 5.0)
    matplotlib.pyplot.close(ax_swap.figure)

    stations = xr.DataArray(
        np.array([10.0, 20.0, np.nan]),
        dims="station",
        coords={"station": ["a", "b", "c"], "lon": ("station", lon), "lat": ("station", lat)},
        name="return_level",
    )
    ds = xr.Dataset({"return_level": stations})
    ax = tails.plot_return_level_map(ds)
    offs = np.asarray(_scatter(ax).get_offsets())
    np.testing.assert_allclose(offs, np.column_stack([lon[:2], lat[:2]]))
    # Explicit coordinates replace the ones on the array.
    shift = lon + 10.0
    ax2 = tails.plot_return_level_map(stations, shift, lat)
    np.testing.assert_allclose(np.asarray(_scatter(ax2).get_offsets()), np.column_stack([shift[:2], lat[:2]]))
    matplotlib.pyplot.close(ax.figure)
    matplotlib.pyplot.close(ax2.figure)

def test_quebec_station_map_matches_the_howto():
    """Published subset in the how-to: 98 stations, colour scale 73.6–158.6 mm."""
    from tails.datasets import load_eccc_quebec

    df = load_eccc_quebec()
    wide = df.pivot_table(index="station_id", columns="year", values="24h")
    meta = df.drop_duplicates("station_id").set_index("station_id").loc[wide.index]
    fit = tails.fit_gev(wide.to_numpy())
    level = np.asarray(fit.return_level([100]).estimate)[:, 0].copy()
    keep = (fit.nobs >= 20) & (np.abs(fit.params["shape"]) < 0.5)
    level[~keep] = np.nan
    shown = level[np.isfinite(level)]
    assert shown.size == 98
    assert shown.min() == pytest.approx(62.98829296, rel=1e-6)
    assert np.median(shown) == pytest.approx(101.75779139, rel=1e-6)
    assert shown.max() == pytest.approx(247.94313973, rel=1e-6)
    qc = list(wide.index).index("701S001")
    assert level[qc] == pytest.approx(102.05684613, rel=1e-6)
    vmin, vmax = np.quantile(shown, [0.05, 0.95])
    assert vmin == pytest.approx(73.64778067, rel=1e-6)
    assert vmax == pytest.approx(158.62192566, rel=1e-6)
    ax = tails.plot_return_level_map(
        level,
        meta.lon.to_numpy(),
        meta.lat.to_numpy(),
        period=100,
        vmin=vmin,
        vmax=vmax,
        label="100-year 24-h level (mm)",
    )
    offs = np.asarray(_scatter(ax).get_offsets())
    assert offs.shape == (98, 2)
    np.testing.assert_allclose(offs[0], [meta.lon.to_numpy()[keep][0], meta.lat.to_numpy()[keep][0]])
    matplotlib.pyplot.close(ax.figure)

def test_plot_return_level_map_requires_coordinates():
    with pytest.raises(ValueError, match="lon and lat"):
        tails.plot_return_level_map(np.array([1.0, 2.0, 3.0]))


def _gpd_sample(n=700, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.exponential(scale=4.0, size=n)
    x[::11] = np.nan
    return x

def _line_with_x(ax, x):
    x = np.asarray(x, float)
    for line in ax.get_lines():
        xd = np.asarray(line.get_xdata(), float)
        if xd.shape == x.shape and np.allclose(xd, x, equal_nan=True):
            return line
    raise AssertionError("estimate line not found")

def _vline(ax, value):
    for line in ax.get_lines():
        xd = np.asarray(line.get_xdata(), float)
        if xd.size == 2 and np.allclose(xd, value):
            return line
    raise AssertionError("ForwardStop line not found")

def test_plot_threshold_draws_diagnostics_and_forwardstop():
    import pandas as pd

    x = _gpd_sample()
    finite = x[np.isfinite(x)]
    thresholds = np.quantile(finite, [0.85, 0.5, 0.7, 0.6, 0.75])
    series = pd.Series(x, name="gauge")
    fig, axes = tails.plot_threshold(series, thresholds, unit="mm", alpha=0.05, n_boot=29, seed=2)
    assert axes.shape == (3,)
    u = np.sort(np.unique(thresholds))
    mrl = tails.mean_residual_life(x, u)
    stability = tails.parameter_stability(x, u)
    chosen = tails.select_threshold(x, u, n_boot=29, seed=2).threshold

    mean_line = _line_with_x(axes[0], mrl["threshold"])
    scale_line = _line_with_x(axes[1], stability["threshold"])
    shape_line = _line_with_x(axes[2], stability["threshold"])
    np.testing.assert_allclose(mean_line.get_ydata(), mrl["mean_excess"], equal_nan=True)
    np.testing.assert_allclose(scale_line.get_ydata(), stability["modified_scale"], equal_nan=True)
    np.testing.assert_allclose(shape_line.get_ydata(), stability["shape"], equal_nan=True)
    for ax in axes:
        _vline(ax, chosen)
        assert any("PolyCollection" in c.__class__.__name__ for c in ax.collections)
        assert ax.get_xlabel() == "Threshold (mm)"
    assert axes[0].get_ylabel() == "Mean excess (mm)"
    assert axes[1].get_ylabel() == "Modified scale (mm)"
    assert axes[2].get_ylabel() == r"Shape $\xi$"
    assert "mm" not in axes[2].get_ylabel()
    legend = fig.legends[0]
    labels = [t.get_text() for t in legend.get_texts()]
    assert "95% CI" in labels
    assert any(label.startswith("ForwardStop (") and label.endswith("mm)") for label in labels)
    matplotlib.pyplot.close(fig)

def test_plot_threshold_default_grid_and_blank_unit():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    fig, axes = tails.plot_threshold(x, unit="", n_boot=9, seed=0)
    assert axes.shape == (3,)
    xd = np.asarray(_line_with_x(axes[0], axes[0].get_lines()[0].get_xdata()).get_xdata(), float)
    assert 3 <= xd.size <= 15
    assert np.all(np.diff(xd) > 0)
    assert xd[0] >= np.nanmin(x) - 1e-8
    assert xd[-1] < np.nanmax(x)
    assert axes[0].get_xlabel() == "Threshold"
    assert axes[0].get_ylabel() == "Mean excess"
    assert "(" not in axes[0].get_ylabel()
    _vline(axes[2], tails.select_threshold(x, xd, n_boot=9, seed=0).threshold)
    matplotlib.pyplot.close(fig)

def test_plot_threshold_docs_example_marks_10_85_mm():
    """The how-to rainfall figure quotes ForwardStop at 10.85 mm."""
    from scipy import stats

    rng = np.random.default_rng(0)
    body = rng.uniform(0.0, 10.0, size=5000)
    excess = stats.genpareto.rvs(0.15, scale=6.0, size=700, random_state=rng)
    rain = np.concatenate([body, 10.0 + excess])
    thresholds = np.linspace(2.0, 25.0, 14)
    fig, _axes = tails.plot_threshold(rain, thresholds, unit="mm")
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "ForwardStop (10.85 mm)" in labels
    shape = tails.parameter_stability(rain, thresholds)["shape"]
    i = int(np.argmin(np.abs(thresholds - 10.846153846153847)))
    np.testing.assert_allclose(shape[i], 0.12323037, rtol=1e-5, atol=1e-6)
    matplotlib.pyplot.close(fig)

def test_plot_threshold_rejects_bad_input():
    x = np.arange(40.0)
    with pytest.raises(ValueError, match="one series"):
        tails.plot_threshold(np.ones((2, 40)))
    with pytest.raises(ValueError, match="10 finite"):
        tails.plot_threshold([1.0, np.nan, 2.0])
    with pytest.raises(TypeError, match="unit"):
        tails.plot_threshold(x, unit=3)
    with pytest.raises(ValueError, match="two thresholds"):
        tails.plot_threshold(x, [1e9, 1e10])

def test_plot_threshold_reuses_figure():
    x = np.random.default_rng(0).exponential(2.0, size=200)
    fig, _ = matplotlib.pyplot.subplots(1, 3, figsize=(8, 3))
    out, axes = tails.plot_threshold(x, np.quantile(x, [0.5, 0.7, 0.85]), unit="kN", n_boot=9, seed=1, fig=fig)
    assert out is fig
    assert axes[1].get_ylabel() == "Modified scale (kN)"
    matplotlib.pyplot.close(fig)
    bare, _ = matplotlib.pyplot.subplots()
    with pytest.raises(ValueError, match="three axes"):
        tails.plot_threshold(x, fig=bare, n_boot=9)
    matplotlib.pyplot.close(bare)

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
    with pytest.raises(ImportError, match="rare-tails\\[plots\\]"):
        tails.plot_threshold(np.arange(40.0), unit="mm")
