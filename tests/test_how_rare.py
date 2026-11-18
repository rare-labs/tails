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
