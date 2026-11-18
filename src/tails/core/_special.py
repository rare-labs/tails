"""Numerically stable special functions with well-defined gradients.

The GEV and GPD densities contain ``log(1 + xi*z) / xi`` and
``((1 + xi*z)^(-1/xi))`` which are 0/0 at ``xi = 0``. Naive code either branches
(and then breaks autodiff) or loses precision for ``|xi| < 1e-4``. The helpers
below evaluate the removable singularities by Taylor series near zero and use
the "double-where" trick so that gradients and Hessians stay finite.
"""

from __future__ import annotations

from .._jax import jnp

_SMALL = 1e-5


def log1p_over_x(x):
    """Return ``log1p(x) / x`` with the limit 1 at ``x = 0``."""
    small = jnp.abs(x) < _SMALL
    xs = jnp.where(small, 1.0, x)
    series = 1.0 - x / 2.0 + x * x / 3.0 - x * x * x / 4.0
    return jnp.where(small, series, jnp.log1p(xs) / xs)


def expm1_over_x(x):
    """Return ``expm1(x) / x`` with the limit 1 at ``x = 0``."""
    small = jnp.abs(x) < _SMALL
    xs = jnp.where(small, 1.0, x)
    series = 1.0 + x / 2.0 + x * x / 6.0 + x * x * x / 24.0
    return jnp.where(small, series, jnp.expm1(xs) / xs)


def safe_log(x, tiny=1e-300):
    """``log(max(x, tiny))`` (avoids -inf leaking into gradients)."""
    return jnp.log(jnp.maximum(x, tiny))
