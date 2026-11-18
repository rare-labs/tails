"""Published numbers for the bundled Québec City 24-hour series (docs prose)."""

import numpy as np

import tails
from tails.core.distributions import gev_cdf
from tails.datasets import load_eccc_quebec


def test_quebec_24h_published_numbers():
    df = load_eccc_quebec()
    y = df.loc[df.station_id == "701S001", "24h"].to_numpy()
    fit = tails.fit_gev(y)
    assert fit.converged
    assert fit.nobs == 64
    rl = fit.return_level([20, 100], ci="profile")
    np.testing.assert_allclose(rl.estimate, [84.8832015, 102.05684613], rtol=1e-6)
    loc, scale, shape = (float(fit.params[k]) for k in ("loc", "scale", "shape"))
    p = float(np.asarray(gev_cdf(np.asarray(105.4), loc, scale, shape)))
    assert abs(1.0 / (1.0 - p) - 138.3) < 0.1
