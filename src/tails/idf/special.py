"""Differentiable special functions needed by the IDF dependence models.

* ``log_matern``: Matérn correlation via a trapezoidal quadrature of
  ``K_nu(z) = int_0^inf exp(-z cosh t) cosh(nu t) dt`` (spectrally accurate for
  this analytic, doubly-exponentially decaying integrand), differentiable in
  ``nu``, ``rho`` and the distance. JAX has no ``besselk`` for real order.
* ``t_ppf``: Student-t quantile with exact implicit derivative
  ``dx/du = 1 / f(x)`` (``jax.custom_jvp``), primal by bisection on the
  regularised incomplete beta function.
"""

from __future__ import annotations

from jax.scipy.special import betainc, gammaln
from jax.scipy.stats import t as jt

from .._jax import jax, jnp

_NODES = 192


def log_besselk_scaled(nu, z):
    """``log(z**nu * K_nu(z))`` for scalar ``nu > 0`` and array ``z > 0``."""
    z = jnp.asarray(z, dtype=jnp.float64)
    T = jnp.maximum(jnp.log(4.0 * (nu + 50.0) / z), 0.5)
    k = jnp.arange(_NODES + 1, dtype=jnp.float64)
    h = T / _NODES
    t = h[..., None] * k
    nt = nu * t
    logcosh = nt + jnp.log1p(jnp.exp(-2.0 * nt)) - jnp.log(2.0)
    f = -z[..., None] * jnp.cosh(t) + logcosh
    w = jnp.where((k == 0) | (k == _NODES), jnp.log(0.5), 0.0)
    logK = jax.scipy.special.logsumexp(f + w, axis=-1) + jnp.log(h)
    return nu * jnp.log(z) + logK


def matern(h, nu, rho):
    """Matérn correlation ``2^(1-nu)/Gamma(nu) (sqrt(2 nu) h/rho)^nu K_nu(.)``; 1 at ``h = 0``."""
    pos = h > 0
    hs = jnp.where(pos, h, 1.0)
    z = jnp.sqrt(2.0 * nu) * hs / rho
    val = jnp.exp((1.0 - nu) * jnp.log(2.0) - gammaln(nu) + log_besselk_scaled(nu, z))
    return jnp.where(pos, val, 1.0)


def exponential(h, rho):
    """Exponential correlation ``exp(-h / rho)``."""
    return jnp.exp(-h / rho)


@jax.custom_jvp
def t_ppf(u, df):
    """Student-t quantile function (``df`` may be non-integer)."""
    q = jnp.minimum(u, 1.0 - u)
    target = 2.0 * q

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        val = betainc(df / 2.0, 0.5, jnp.exp(mid))
        go_up = val < target
        return jnp.where(go_up, mid, lo), jnp.where(go_up, hi, mid)

    lo = jnp.full_like(q, -745.0)
    hi = jnp.zeros_like(q)
    lo, hi = jax.lax.fori_loop(0, 110, body, (lo, hi))
    y = jnp.exp(0.5 * (lo + hi))
    x = jnp.sqrt(df * (1.0 - y) / y)
    return jnp.where(u < 0.5, -x, x)


@t_ppf.defjvp
def _t_ppf_jvp(primals, tangents):
    u, df = primals
    du, _ = tangents
    x = t_ppf(u, df)
    return x, du / jt.pdf(x, df)
