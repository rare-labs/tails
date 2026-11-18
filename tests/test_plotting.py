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
