"""Batched second-order optimiser.

Extreme-value models have few parameters (3 to ~10) but users often need to fit
*many* of them (stations, grid cells, bootstrap replicates, thresholds). The
fastest robust strategy in that regime is a damped Newton method with exact
Hessians from automatic differentiation, compiled once with ``jax.jit`` and
vectorised over problems with ``jax.vmap``:

* exact gradient + Hessian (forward-over-reverse AD) – no finite differences;
* Levenberg–Marquardt damping ``(H + lambda I) s = -g``: the damping grows
  until the system is positive definite and the step decreases the objective,
  and shrinks after successful steps (pure Newton near the optimum);
* step rejection when the objective increases or leaves the parameter space
  (``+inf``/``nan``);
* small systems are solved with an unrolled Cholesky made of elementwise ops,
  which vectorises perfectly under ``vmap`` (no per-matrix LAPACK calls);
* a fixed-size ``lax.while_loop`` so thousands of problems advance in lock
  step on CPU or GPU.

The objective must return ``+inf`` (or ``nan``) for infeasible parameters; the
starting point must be feasible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from .._jax import jax, jnp

__all__ = ["minimize", "OptimResult"]


class OptimResult(NamedTuple):
    """Result of :func:`minimize`; every field carries the batch dimension."""

    x: Any
    fun: Any
    grad: Any
    hess: Any
    n_iter: Any
    converged: Any


@dataclass(frozen=True)
class _Settings:
    max_iter: int = 200
    gtol: float = 1e-7
    ftol: float = 1e-13
    lam0: float = 1e-3


def _cholesky_solve(A, b):
    """Unrolled Cholesky solve of ``A x = b`` for small ``p`` (vectorises under vmap).

    Returns ``(x, ok)`` where ``ok`` is False if ``A`` is not positive definite.
    Pure elementwise jnp ops: no per-matrix LAPACK calls, which dominate the
    cost of batched 3x3 solves on CPU.
    """
    p = A.shape[0]
    L = [[None] * p for _ in range(p)]
    ok = jnp.asarray(True)
    for j in range(p):
        s = A[j, j]
        for k in range(j):
            s = s - L[j][k] * L[j][k]
        ok = ok & (s > 0)
        d = jnp.sqrt(jnp.where(s > 0, s, 1.0))
        L[j][j] = d
        for i in range(j + 1, p):
            t = A[i, j]
            for k in range(j):
                t = t - L[i][k] * L[j][k]
            L[i][j] = t / d
    z = [None] * p
    for i in range(p):
        t = b[i]
        for k in range(i):
            t = t - L[i][k] * z[k]
        z[i] = t / L[i][i]
    x = [None] * p
    for i in reversed(range(p)):
        t = z[i]
        for k in range(i + 1, p):
            t = t - L[k][i] * x[k]
        x[i] = t / L[i][i]
    return jnp.stack(x), ok


def _damped_solve(A, g):
    if A.shape[0] <= 12:
        sol, ok = _cholesky_solve(A, g)
    else:
        w, V = jnp.linalg.eigh(A)
        ok = jnp.all(w > 0)
        sol = V @ ((V.T @ g) / jnp.where(w > 0, w, 1.0))
    return -sol, ok


def _single(fun, x0, args, s: _Settings, vgh=None):
    if vgh is None:
        vg = jax.value_and_grad(fun)
        hess = jax.hessian(fun)

        def vgh(x, *a):
            f, g = vg(x, *a)
            return f, g, hess(x, *a)

    def evaluate(x):
        f, g, H = vgh(x, *args)
        bad = ~jnp.isfinite(f) | ~jnp.all(jnp.isfinite(g)) | ~jnp.all(jnp.isfinite(H))
        f = jnp.where(bad, jnp.inf, f)
        return f, g, H, bad

    f0, g0, H0, bad0 = evaluate(x0)
    p = x0.shape[0]
    eye = jnp.eye(p, dtype=x0.dtype)

    def gnorm(g, f):
        return jnp.max(jnp.abs(g)) / (1.0 + 0.0 * jnp.abs(f))

    def cond(state):
        x, f, g, H, lam, it, done, conv = state
        return (~done) & (it < s.max_iter)

    def body(state):
        x, f, g, H, lam, it, done, conv = state
        Hs = jnp.where(jnp.isfinite(H), 0.5 * (H + H.T), eye)
        scale = 1.0 + jnp.max(jnp.abs(jnp.diag(Hs)))
        step, ok = _damped_solve(Hs + lam * scale * eye, g)
        xt = x + step
        ft, gt, Ht, badt = evaluate(xt)
        accept = ok & (~badt) & (ft <= f)
        x_new = jnp.where(accept, xt, x)
        f_new = jnp.where(accept, ft, f)
        g_new = jnp.where(accept, gt, g)
        H_new = jnp.where(accept, Ht, H)
        lam_new = jnp.where(accept, jnp.maximum(lam * 0.2, 1e-12), jnp.minimum(lam * 10.0, 1e12))
        small_g = gnorm(g_new, f_new) < s.gtol
        small_f = accept & (jnp.abs(f - ft) <= s.ftol * (1.0 + jnp.abs(f)))
        small_x = accept & (jnp.max(jnp.abs(step)) <= 1e-10 * (1.0 + jnp.max(jnp.abs(x))))
        stuck = lam_new >= 1e12
        conv_new = small_g | small_f | small_x
        done_new = conv_new | stuck
        return (x_new, f_new, g_new, H_new, lam_new, it + 1, done_new, conv_new)

    init_conv = gnorm(g0, f0) < s.gtol
    state = (x0, f0, g0, H0, jnp.asarray(s.lam0, x0.dtype), jnp.asarray(0), bad0 | init_conv, init_conv)
    x, f, g, H, lam, it, done, conv = jax.lax.while_loop(cond, body, state)
    conv = conv & jnp.isfinite(f)
    return OptimResult(x, f, g, H, it, conv)


def minimize(
    fun: Callable[..., Any],
    x0,
    *args,
    batched: bool = True,
    vgh: Callable[..., Any] | None = None,
    arg_axes: tuple | None = None,
    max_iter: int = 200,
    gtol: float = 1e-7,
    ftol: float = 1e-13,
) -> OptimResult:
    """Minimise ``fun(x, *args)`` for a batch of problems at once.

    Parameters
    ----------
    fun
        Scalar objective of one problem, ``fun(x, *args) -> float``. Must be
        JAX-traceable. Return ``inf`` for infeasible ``x``.
    x0
        Starting values, shape ``(B, p)`` when ``batched`` (default) else ``(p,)``.
    *args
        Extra per-problem arguments; with ``batched=True`` each must have a
        leading batch dimension ``B`` unless its entry in ``arg_axes`` is None
        (shared across problems).
    vgh
        Optional ``vgh(x, *args) -> (f, grad, hess)`` replacing autodiff of
        ``fun`` (e.g. a structure-exploiting implementation).
    arg_axes
        ``jax.vmap`` axes for ``args`` (default: all 0).
    max_iter, gtol, ftol
        Iteration cap, gradient tolerance (max-abs), relative objective tolerance.

    Returns
    -------
    OptimResult
        ``x`` (minimiser), ``fun``, ``grad``, ``hess``, ``n_iter``, ``converged``.
    """
    if arg_axes is None:
        arg_axes = (0,) * len(args)
    arg_axes = tuple(arg_axes)
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    if not batched:
        s = _Settings(max_iter=max_iter, gtol=gtol, ftol=ftol)
        return _compiled(fun, s, False, arg_axes, vgh)(x0, *args)
    return _staged(fun, x0, args, arg_axes, max_iter, gtol, ftol, vgh)


def _bucket(k: int) -> int:
    """Pad active-set sizes to powers of two to bound recompilation."""
    b = 1
    while b < k:
        b *= 2
    return b


def _staged(fun, x0, args, arg_axes, max_iter, gtol, ftol, vgh=None):
    """Run all problems for a few iterations, then only the unconverged ones.

    Most well-posed extreme-value fits converge in 3-6 Newton steps, but a
    batched ``while_loop`` advances every problem until the slowest finishes.
    Compacting the active set between stages cuts the work several-fold.
    """
    import numpy as np

    B, p = x0.shape
    caps = [c for c in (8, 32) if c < max_iter] + [max_iter]
    x = np.array(x0)
    f = np.full(B, np.inf)
    g = np.zeros((B, p))
    H = np.zeros((B, p, p))
    it = np.zeros(B, dtype=np.int64)
    conv = np.zeros(B, dtype=bool)
    active = np.arange(B)
    used = 0
    for cap in caps:
        s = _Settings(max_iter=cap - used, gtol=gtol, ftol=ftol)
        solver = _compiled(fun, s, True, arg_axes, vgh)
        k = len(active)
        full = k == B
        if full:
            idx = active
            sub_args = args
        else:
            pad = _bucket(k)
            idx = np.concatenate([active, np.full(pad - k, active[0])])
            sub_args = tuple(a if ax is None else jnp.take(jnp.asarray(a), jnp.asarray(idx), axis=ax) for a, ax in zip(args, arg_axes))
        r = solver(jnp.asarray(x[idx]), *sub_args)
        rx, rf, rg, rH = (np.asarray(r.x)[:k], np.asarray(r.fun)[:k], np.asarray(r.grad)[:k], np.asarray(r.hess)[:k])
        rit, rc = np.asarray(r.n_iter)[:k], np.asarray(r.converged)[:k]
        x[active], f[active], g[active], H[active] = rx, rf, rg, rH
        it[active] += rit
        conv[active] = rc
        # keep going only for problems that hit the iteration cap and are still finite
        still = (~rc) & (rit >= cap - used) & np.isfinite(rf)
        used = cap
        active = active[still]
        if len(active) == 0:
            break
    return OptimResult(x, f, g, H, it, conv)


_CACHE: dict = {}


def _compiled(fun, s, batched, arg_axes, vgh=None):
    key = (fun, s, batched, arg_axes, vgh)
    if key not in _CACHE:
        single = lambda x0, *args: _single(fun, x0, args, s, vgh)  # noqa: E731
        _CACHE[key] = jax.jit(jax.vmap(single, in_axes=(0, *arg_axes)) if batched else single)
    return _CACHE[key]
