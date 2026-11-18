"""Optional Bayesian GEV inference with NumPyro (``pip install tails[bayes]``).

Uses the same stable log-density as the likelihood engine, NUTS sampling and
weakly-informative priors; the shape prior defaults to the Martins–Stedinger
Beta(6, 9) on (-0.5, 0.5) used by ``method="gmle"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._jax import jax, jnp
from .core.distributions import gev_logpdf, gev_ppf

__all__ = ["fit_gev_bayes", "BayesGEV"]


@dataclass
class BayesGEV:
    """Posterior draws of a stationary GEV."""

    samples: dict

    def return_level(self, period, alpha: float = 0.05):
        """Posterior median and equal-tailed credible interval of return levels."""
        T = np.atleast_1d(np.asarray(period, float))
        s = self.samples
        draws = np.asarray(gev_ppf(1 - 1 / T[None, :], s["loc"][:, None], s["scale"][:, None], s["shape"][:, None]))
        return {
            "period": T,
            "median": np.median(draws, 0),
            "lower": np.quantile(draws, alpha / 2, 0),
            "upper": np.quantile(draws, 1 - alpha / 2, 0),
            "draws": draws,
        }


def fit_gev_bayes(y, *, num_warmup: int = 500, num_samples: int = 1000, num_chains: int = 1, seed: int = 0, shape_prior: str = "martins-stedinger") -> BayesGEV:
    """Sample the posterior of a stationary GEV for one series with NUTS."""
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as e:  # pragma: no cover
        raise ImportError("fit_gev_bayes requires numpyro: pip install 'tails[bayes]'") from e

    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    m, s = float(np.mean(y)), float(np.std(y))

    def model(obs):
        loc = numpyro.sample("loc", dist.Normal(m, 10 * s))
        log_scale = numpyro.sample("log_scale", dist.Normal(np.log(s), 2.0))
        if shape_prior == "martins-stedinger":
            u = numpyro.sample("_u", dist.Beta(9.0, 6.0))  # xi = u - 0.5
            shape = numpyro.deterministic("shape", u - 0.5)
        else:
            shape = numpyro.sample("shape", dist.Normal(0.0, 0.3))
        scale = numpyro.deterministic("scale", jnp.exp(log_scale))
        numpyro.factor("ll", jnp.sum(gev_logpdf(obs, loc, scale, shape)))

    lm = sorted(y)
    init = {"loc": m - 0.45 * s, "log_scale": np.log(0.78 * s)}
    if shape_prior == "martins-stedinger":
        init["_u"] = 0.5
    else:
        init["shape"] = 0.0
    del lm
    kernel = NUTS(model, init_strategy=numpyro.infer.init_to_value(values=init))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(y))
    smp = {k: np.asarray(v) for k, v in mcmc.get_samples().items() if not k.startswith("_")}
    return BayesGEV(smp)
