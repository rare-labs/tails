"""JAX configuration shared by every tails module.

tails enables 64-bit floats in JAX on import. Extreme-value likelihoods are
evaluated deep in the tails of distributions, where float32 silently loses the
digits that matter (e.g. 1 - F(x) for a 1000-year return level).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

__all__ = ["jax", "jnp"]
