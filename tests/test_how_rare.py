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


def _annual(start, values):
    years = range(start, start + len(values))
    return pd.Series(values, index=pd.to_datetime([f"{y}-01-01" for y in years]))


# Annual maxima copied in docs/howto.md. Each list is a published series, not a draw.
_TREASURY_BP = [
    28, 11, 17, 15, 39, 18, 34, 18, 22, 21, 16, 25, 24, 24, 24, 14, 12, 16,
    24, 24, 20, 19, 15, 21, 10, 15, 19, 10, 10, 11, 29, 16, 28, 15, 19, 14,
]
_HENRY_HUB = [
    4.71, 2.61, 3.10, 10.48, 10.31, 5.31, 18.48, 8.12, 14.84, 9.90,
    8.16, 13.31, 6.10, 7.51, 4.92, 3.77, 4.52, 8.15, 3.32, 3.80,
    3.71, 6.24, 4.25, 3.14, 23.86, 9.85, 3.78, 13.20, 9.86,
]
_CLAIMS_MILLION = [
    0.310, 0.251, 0.232, 0.374, 0.359, 0.350, 0.326, 0.537, 0.575, 0.423,
    0.565, 0.429, 0.471, 0.642, 0.558, 0.695, 0.534, 0.439, 0.426, 0.416,
    0.370, 0.361, 0.407, 0.474, 0.509, 0.564, 0.415, 0.406, 0.390, 0.426,
    0.347, 0.376, 0.345, 0.364, 0.517, 0.479, 0.450, 0.380, 0.424, 0.349,
    0.360, 0.587, 0.665, 0.507, 0.468, 0.446, 0.375, 0.345, 0.317, 0.290,
    0.297, 0.252, 0.247, 6.137, 0.890, 0.250, 0.259, 0.258, 0.259,
]
_USGS_MW = [
    8.60, 7.81, 9.00, 7.90, 7.80, 7.52, 7.70, 8.60, 8.30, 7.90,
    9.50, 7.60, 7.47, 8.50, 9.20, 8.70, 8.10, 7.30, 8.20, 8.20,
    8.00, 8.10, 8.00, 7.70, 7.60, 7.90, 8.00, 8.30, 7.70, 7.90,
    7.90, 7.70, 7.29, 7.60, 7.60, 8.00, 8.00, 7.90, 7.80, 8.02,
    7.80, 7.60, 7.80, 7.80, 8.30, 8.00, 8.09, 7.80, 8.10, 7.70,
    8.00, 8.40, 7.90, 8.16, 9.10, 8.60, 8.30, 8.40, 7.90, 8.10,
    8.80, 9.10, 8.60, 8.30, 8.20, 8.30, 7.90, 8.20, 8.20, 8.00,
    7.80, 8.20, 7.60, 7.80, 7.50, 8.80,
]


@pytest.mark.parametrize(
    ("start", "values", "date", "unit", "sentence", "period", "lower", "upper"),
    [
        (
            1990,
            _TREASURY_BP,
            "1994-01-01",
            "basis points",
            "A value of 39 basis points in 1994 is a 1-in-74-year event (95% CI: 13–758 years).",
            73.672219,
            12.891339,
            757.699316,
        ),
        (
            1997,
            _HENRY_HUB,
            "2021-01-01",
            "USD per million Btu",
            "A value of 23.86 USD per million Btu in 2021 is a 1-in-24-year event (95% CI: 7.1–70 years).",
            23.639831,
            7.080600,
            70.353989,
        ),
        (
            1967,
            _CLAIMS_MILLION,
            "2020-01-01",
            "million claims",
            "A value of 6.137 million claims in 2020 is a 1-in-6654-year event (95% CI: 4251–50099 years).",
            6653.506723,
            4250.794578,
            50099.412687,
        ),
        (
            1950,
            _USGS_MW,
            "1960-01-01",
            "Mw",
            "A value of 9.5 Mw in 1960 is a 1-in-123-year event (95% CI: 29–1004 years).",
            123.446509,
            29.326637,
            1003.947013,
        ),
    ],
)
def test_how_rare_public_annual_maxima(start, values, date, unit, sentence, period, lower, upper):
    result = tails.how_rare(_annual(start, values), date, ci="profile", unit=unit)
    assert result.sentence == sentence
    assert result.date is not None and result.date.year == int(date[:4])
    np.testing.assert_allclose(result.return_period, period, rtol=1e-4)
    np.testing.assert_allclose(result.lower, lower, rtol=1e-4)
    np.testing.assert_allclose(result.upper, upper, rtol=1e-4)


def test_how_rare_docs_quote_public_examples():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    howto = (root / "docs" / "howto.md").read_text(encoding="utf-8")
    index = (root / "docs" / "index.md").read_text(encoding="utf-8")
    sentences = [
        "A value of 39 basis points in 1994 is a 1-in-74-year event (95% CI: 13–758 years).",
        "A value of 23.86 USD per million Btu in 2021 is a 1-in-24-year event (95% CI: 7.1–70 years).",
        "A value of 6.137 million claims in 2020 is a 1-in-6654-year event (95% CI: 4251–50099 years).",
        "A value of 9.5 Mw in 1960 is a 1-in-123-year event (95% CI: 29–1004 years).",
    ]
    for sentence in sentences:
        assert sentence in howto
    for fragment in ("39-basis-point", "23.86 USD per million Btu", "6.137 million", "Mw 9.5"):
        assert fragment in index
    assert "37.5 °C" not in howto
    assert "118 km/h" not in howto
