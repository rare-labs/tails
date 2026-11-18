import numpy as np
import pandas as pd
import pytest

import tails


def test_how_rare_gumbel_known_period():
    # Gumbel(30, 8): F(x) = exp(-exp(-(x-30)/8)); the 20-year level has p=0.95
    from tails.core.distributions import gev_ppf

    x20 = float(gev_ppf(0.95, 30.0, 8.0, 0.0))
    rng = np.random.default_rng(0)
    y = rng.gumbel(30, 8, size=400)
    result = tails.how_rare(y, x20, ci="delta", unit="mm")
    assert result.unit == "mm"
    # With 400 years of Gumbel data, T should be near 20.
    assert 10 < result.return_period < 40
    assert result.lower < result.return_period < result.upper
    assert "1-in-" in result.sentence
    assert "mm" in result.sentence
    assert "95% CI" in result.sentence


def test_how_rare_bundled_quebec_2005():
    # Worked number from the docs: 24-h annual max 105.4 mm in 2005 at 701S001.
    from tails.idf import IDFData

    data = IDFData.from_eccc("701S001")
    y = pd.Series(data.values[:, -1], index=pd.to_datetime(data.years.astype(str), format="%Y"), name="701S001")
    # The 24-h depths are stored as intensities (mm/h); convert back to mm.
    depth = y * 24.0
    result = tails.how_rare(depth, "2005-01-01", ci="profile", unit="mm", name="Québec City")
    assert result.date is not None and result.date.year == 2005
    np.testing.assert_allclose(result.value, 105.4, atol=0.05)
    assert 50 < result.return_period < 400
    assert result.lower < result.return_period < result.upper
    assert "Québec City" in result.sentence
    assert "in 2005" in result.sentence
    assert "mm" in result.sentence


def test_how_rare_plot_and_float():
    pytest.importorskip("matplotlib").use("Agg")
    rng = np.random.default_rng(1)
    y = rng.gumbel(30, 8, size=80)
    result = tails.how_rare(y, float(np.quantile(y, 0.9)), ci="delta", plot=True)
    assert result.ax is not None
    assert float(result) == result.return_period
    import matplotlib.pyplot as plt

    plt.close(result.ax.figure)


def test_how_rare_pot_profile_interval():
    rng = np.random.default_rng(2)
    x = rng.normal(size=1600)
    x[:120] = 6.0 + rng.exponential(1.5, size=120)
    result = tails.how_rare(x, 12.0, model="pot", threshold=6.0, npy=365.25, ci="profile")
    assert result.model == "pot"
    assert result.lower < result.return_period < result.upper
    assert "95% CI" in result.sentence


def test_how_rare_rejects_date_without_index():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        tails.how_rare(np.arange(20.0), "2005-07-01")


def _constructed_gev(start, end, loc, scale, xi, spike_year, spike, seed):
    """Weibull quantiles of a GEV, shuffled, with one year replaced.

    Matches the constructed heat and wind snippets in ``docs/howto.md``.
    """
    years = np.arange(start, end)
    p = (np.arange(1, len(years) + 1) - 0.5) / len(years)
    values = np.round(loc + scale / xi * ((-np.log(p)) ** (-xi) - 1.0), 1)
    np.random.default_rng(seed).shuffle(values)
    values[years == spike_year] = spike
    return pd.Series(values, index=pd.to_datetime(years.astype(str), format="%Y"))


def test_how_rare_constructed_heat_and_wind():
    # Published numbers from docs/howto.md and docs/index.md.
    heat = _constructed_gev(1978, 2026, 32.0, 1.45, -0.18, 2021, 37.5, 11)
    heat_result = tails.how_rare(heat, "2021-01-01", ci="profile", unit="°C")
    assert heat_result.date is not None and heat_result.date.year == 2021
    np.testing.assert_allclose(heat_result.value, 37.5)
    np.testing.assert_allclose(heat_result.return_period, 124.4237339595553, rtol=1e-6)
    np.testing.assert_allclose(heat_result.lower, 20.43750126172065, rtol=1e-6)
    np.testing.assert_allclose(heat_result.upper, 4579.87261097234, rtol=1e-6)
    assert heat_result.sentence == (
        "A value of 37.5 °C in 2021 is a 1-in-124-year event (95% CI: 20–4580 years)."
    )

    wind = _constructed_gev(1978, 2026, 82.0, 7.5, -0.12, 2019, 118.0, 19)
    wind_result = tails.how_rare(wind, "2019-01-01", ci="profile", unit="km/h")
    assert wind_result.date is not None and wind_result.date.year == 2019
    np.testing.assert_allclose(wind_result.value, 118.0)
    np.testing.assert_allclose(wind_result.return_period, 154.1470027958524, rtol=1e-6)
    np.testing.assert_allclose(wind_result.lower, 24.011442238242225, rtol=1e-6)
    np.testing.assert_allclose(wind_result.upper, 5155.165603518771, rtol=1e-6)
    assert wind_result.sentence == (
        "A value of 118 km/h in 2019 is a 1-in-154-year event (95% CI: 24–5155 years)."
    )


def test_how_rare_docs_quote_constructed_examples():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    heat = "A value of 37.5 °C in 2021 is a 1-in-124-year event (95% CI: 20–4580 years)."
    wind = "A value of 118 km/h in 2019 is a 1-in-154-year event (95% CI: 24–5155 years)."
    howto = (root / "docs" / "howto.md").read_text(encoding="utf-8")
    index = (root / "docs" / "index.md").read_text(encoding="utf-8")
    assert heat in howto
    assert wind in howto
    assert "37.5 °C" in index and "1-in-124-year" in index
    assert "118 km/h" in index and "1-in-154-year" in index
    assert 'unit="°C"' in howto and 'unit="km/h"' in howto
