"""JAX configuration shared by every tails module.

tails enables 64-bit floats in JAX on import. Extreme-value likelihoods are
evaluated deep in the tails of distributions, where float32 silently loses the
digits that matter (e.g. 1 - F(x) for a 1000-year return level).

It also turns on JAX's persistent compilation cache, so the likelihoods and
Hessians compiled in one Python session are reused by the next one instead of
being compiled again. The cache lives in ``$TAILS_CACHE/jax`` (default
``$XDG_CACHE_HOME/tails/jax`` or ``~/.cache/tails/jax``). Set
``TAILS_JAX_CACHE=0`` to turn it off. If you already configured a JAX cache
directory (``JAX_COMPILATION_CACHE_DIR`` or ``jax.config``), tails leaves it
alone.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

__all__ = ["jax", "jnp", "compilation_cache_dir"]


def _default_cache_dir() -> Path:
    if os.environ.get("TAILS_CACHE"):
        base = Path(os.environ["TAILS_CACHE"])
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"]) / "tails"
    else:
        base = Path.home() / ".cache" / "tails"
    return base / "jax"


def _enable_compilation_cache() -> str | None:
    if os.environ.get("TAILS_JAX_CACHE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    current = jax.config.jax_compilation_cache_dir
    if current:  # the user (or JAX_COMPILATION_CACHE_DIR) already chose one
        return current
    try:
        path = _default_cache_dir()
        path.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(path))
        # tails kernels are small and compile in well under a second each;
        # cache all of them so a new session starts warm.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    except Exception:  # read-only home, sandbox, ...: just run uncached
        return None
    return str(path)


_CACHE_DIR = _enable_compilation_cache()


def compilation_cache_dir() -> str | None:
    """Directory of the persistent JAX compilation cache, or ``None`` if off."""
    return _CACHE_DIR
