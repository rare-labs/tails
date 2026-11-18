import json
from pathlib import Path

import numpy as np
import pytest

from tails.datasets import load_eccc_quebec, load_idfcurves, parse_eccc_idf
from tails.idf import IDFData, IDFModel, fit_idf, loglikelihood, scaling_test
from tails.idf.special import log_besselk_scaled, t_ppf

REF = json.loads((Path(__file__).parent / "data" / "idfcurves_reference.json").read_text())
SPECS = {
    "simple": ("simple", None, "gaussian"),
    "general": ("general", None, "gaussian"),
    "general_matern_gaussian": ("general", "matern", "gaussian"),
    "general_exponential_gaussian": ("general", "exponential", "gaussian"),
    "general_exponential_t5": ("general", "exponential", "t"),
}


def _data(st):
    return IDFData.from_dataframe(load_idfcurves(st).dropna(), "Year")


@pytest.mark.parametrize("key", list(SPECS))
def test_loglik_equals_idfcurves_at_julia_mle(key):
    for st in REF:
        r = REF[st][key]
        ll = loglikelihood(IDFModel(*SPECS[key], df=5), r["params"], _data(st))
        assert ll == pytest.approx(r["loglik"], abs=1e-7)


@pytest.mark.parametrize("key", ["simple", "general", "general_matern_gaussian"])
def test_mle_and_ci_match_idfcurves(key):
    st = "702S006"
    r = REF[st][key]
    fit = fit_idf(_data(st), *SPECS[key])
    np.testing.assert_allclose(fit.params[0], r["params"], rtol=1e-5, atol=1e-6)
    assert fit.loglik[0] == pytest.approx(r["loglik"], abs=1e-6)
    e, lo, hi = fit.quantile_ci([1.0], [100])
    assert e.ravel()[0] == pytest.approx(r["q100_1h"], rel=1e-6)
    np.testing.assert_allclose([lo.ravel()[0], hi.ravel()[0]], r["ci100_1h"], rtol=1e-5)


def test_godambe_interval_matches():
    st = "1108446"
    fit = fit_idf(_data(st), "general")
    e, lo, hi = fit.quantile_ci([1.0], [100], method="sandwich")
    np.testing.assert_allclose([lo.ravel()[0], hi.ravel()[0]], REF[st]["general"]["ci100_1h_godambe"], rtol=1e-4)


def test_scaling_test_matches():
    st = "1108446"
    assert scaling_test(_data(st), "simple") == pytest.approx(REF[st]["simple"]["scalingtest"], rel=1e-4)


def test_batched_equals_individual():
    a, b = _data("702S006"), _data("1108446")
    both = fit_idf([a, b], "general")
    one = fit_idf(b, "general")
    np.testing.assert_allclose(both.params[1], one.params[0], rtol=1e-5, atol=1e-8)


def test_missing_values_handled():
    d = _data("702S006")
    d.values[:5, 0] = np.nan
    fit = fit_idf(d, "general", "matern")
    assert fit.converged.all() and np.isfinite(fit.loglik).all()


def test_bootstrap_ci():
    fit = fit_idf(_data("1108446"), "general", "exponential")
    e, lo, hi = fit.quantile_ci([1.0], [100], method="bootstrap", n_boot=50)
    assert lo.ravel()[0] < e.ravel()[0] < hi.ravel()[0]


def test_special_functions():
    from scipy.special import kv
    from scipy.stats import t

    z = np.array([1e-3, 0.1, 1.0, 10.0])
    for nu in (0.4, 1.5, 6.0):
        np.testing.assert_allclose(np.exp(np.asarray(log_besselk_scaled(nu, z))), z**nu * kv(nu, z), rtol=1e-10)
    u = np.array([1e-8, 0.2, 0.5, 0.97])
    np.testing.assert_allclose(np.asarray(t_ppf(u, 5.0)), t.ppf(u, 5), rtol=1e-9, atol=1e-12)


def test_eccc_dataset_and_parser():
    df = load_eccc_quebec()
    assert df.station_id.nunique() == 151
    q = df[df.station_id == "701S001"]
    assert q.year.min() == 1961 and q.year.max() == 2024
    assert q.loc[q.year == 1979, "30min"].item() == pytest.approx(40.1)
    txt = """
 QUEBEC JEAN LESAGE INTL                                QC        701S001
 Latitude:  46 48'N    Longitude: 71 23'W    Elevation/Altitude: 60         m
Table 1 : Annual Maximum (mm)/Maximum annuel (mm)
          Year  5 min 10 min 15 min 30 min    1 h    2 h    6 h   12 h   24 h
          1961    6.1   10.4   10.9   12.2   19.0   19.0   34.5   53.1   61.7
          1962  -99.9   11.7   13.2   19.8   25.4   29.7   62.5   63.5   71.4
      -----------------------------------------------------------------------
"""
    meta, t = parse_eccc_idf(txt)
    assert meta["station_id"] == "701S001" and meta["lon"] == pytest.approx(-71 - 23 / 60)
    assert np.isnan(t.loc[1, "5min"]) and t.loc[0, "24h"] == 61.7
    data = IDFData.from_eccc("701S001")
    assert data.values[0, 0] == pytest.approx(6.1 * 12)
