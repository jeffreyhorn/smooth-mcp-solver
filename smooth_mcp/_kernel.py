"""Shared solver internals: validation, normalization, Newton, and continuation."""

import inspect
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import gmres

from smooth_mcp.smoothing import smoothed_residual


def validate_bounds_and_x0(l, u, x0):
    """Validate shapes, NaN, and ordering of bounds and initial guess.

    Shape checks always run. Value checks (NaN, ordering) are skipped
    when inside any JAX tracing context (for example, jit, grad,
    vmap, or pmap), since they require concrete values.
    """
    if l.shape != u.shape:
        raise ValueError(
            f"l and u must have the same shape, got {l.shape} and {u.shape}"
        )
    if x0.shape != l.shape:
        raise ValueError(
            f"x0 must have the same shape as l, got {x0.shape} and {l.shape}"
        )
    try:
        if jnp.any(jnp.isnan(l)) or jnp.any(jnp.isnan(u)):
            raise ValueError("Bounds l and u must not contain NaN")
        if jnp.any(l > u):
            raise ValueError("Lower bounds must not exceed upper bounds (l <= u)")
    except jax.errors.TracerBoolConversionError:
        pass


def preflight_validate(l, u, x0):
    """Eagerly validate bounds and initial guess, raising on invalid input.

    Intended for users whose bounds are static across a training loop: call
    this once before entering a jitted/vmapped inner loop to verify inputs
    up front, with zero per-call overhead inside the traced region.

    Accepts array-likes (lists, scalars, numpy or JAX arrays). Raises
    ``ValueError`` on invalid shapes, NaN bounds, or ``l > u``.

    Under tracing, value checks are silently skipped (same convention as
    the solver factories). Supported validation options are:

    * Call ``preflight_validate`` eagerly before entering traced code.
    * Use ``strict_validation=True`` on ``make_mcp_solver`` or
      ``make_mcp_solver_diff``.
    * Use ``strict_validation="checkify"`` on ``make_mcp_solver`` or
      ``make_mcp_solver_diff``.
    """
    l_arr = jnp.asarray(l)
    u_arr = jnp.asarray(u)
    x0_arr = jnp.asarray(x0)
    validate_bounds_and_x0(l_arr, u_arr, x0_arr)


def traced_invalid_mask(l, u):
    """Return a scalar JAX bool that is True when bounds are invalid.

    Detects: NaN in ``l``, NaN in ``u``, or any element with ``l > u``.
    Used by NaN-poisoning strict validation; safe to trace under jit,
    grad, and vmap.
    """
    return jnp.any(jnp.isnan(l)) | jnp.any(jnp.isnan(u)) | jnp.any(l > u)


def sanitize_bounds(l, u):
    """Replace NaNs and fix ordering so the solver sees well-defined inputs.

    This is only meaningful paired with ``traced_invalid_mask``: the
    sanitized values keep the solver from producing internal NaNs in the
    invalid branch, and the caller then replaces the output with NaN when
    the mask is True.
    """
    safe_l = jnp.where(jnp.isnan(l), jnp.zeros_like(l), l)
    safe_u = jnp.where(jnp.isnan(u), jnp.ones_like(u), u)
    lo = jnp.minimum(safe_l, safe_u)
    hi = jnp.maximum(safe_l, safe_u)
    return lo, hi


def normalize_F(F_fn):
    """Wrap F_fn to always accept (x, theta) as positional arguments.

    Supported signatures:
        - F(x)           -> wrapped to ignore theta
        - F(x, theta)    -> used as-is
        - F(x, *args)    -> used as-is (theta passed positionally via *args)

    Unsupported signatures (raise ValueError):
        - F(x, *, theta) -> keyword-only theta cannot be called positionally
        - F(x, **kwargs) -> ambiguous calling convention
    """
    sig = inspect.signature(F_fn)
    params = sig.parameters.values()
    has_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
    has_keyword_only = any(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)
    if has_keyword_only or has_var_keyword:
        raise ValueError(
            f"F_fn has keyword-only or **kwargs parameters which are not supported. "
            f"F_fn must accept positional arguments: F(x), F(x, theta), or F(x, *args). "
            f"Got signature: {sig}"
        )
    has_var_positional = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)
    if has_var_positional:
        return F_fn
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    n_positional = sum(1 for p in params if p.kind in positional_kinds)
    if n_positional <= 1:
        return lambda x, theta: F_fn(x)
    if n_positional > 2:
        raise ValueError(
            f"F_fn must accept 1 or 2 positional arguments, got {n_positional}. "
            f"Signature: {sig}"
        )
    return F_fn


def make_newton_solver(
    F_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    l: jnp.ndarray,
    u: jnp.ndarray,
    theta: jnp.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
    armijo_c: float = 1e-4,
    backtrack_rho: float = 0.5,
    max_ls_steps: int = 20,
    linear_solver: str = "dense",
    krylov_tol: float = 1e-6,
    krylov_maxiter: int = 500,
    krylov_restart: int = 30,
    regularize: float = 1e-12,
):
    """Build a Newton solver that accepts mu as a JAX array.

    Returns a function solve(x0, mu) -> x. The returned function is pure
    JAX and can be traced by jax.jit or used inside lax.while_loop.

    For repeated solves with the same problem, see ``make_mcp_solver_diff``
    for a differentiable MCP solver interface.
    """

    def _residual(x, mu):
        return smoothed_residual(x, F_fn, l, u, mu, theta)

    def _solve_linear_dense(x, H, mu):
        J = jax.jacfwd(lambda xx: _residual(xx, mu))(x)
        n = J.shape[0]
        reg = jnp.array(regularize, dtype=x.dtype)
        J_reg = J + reg * jnp.eye(n, dtype=x.dtype)
        return jnp.linalg.solve(J_reg, -H)

    def _solve_linear_gmres(x, H, mu):
        reg = jnp.array(regularize, dtype=x.dtype)

        def Jv(v):
            _, jvp_val = jax.jvp(lambda xx: _residual(xx, mu), (x,), (v,))
            return jvp_val + reg * v

        d, info = gmres(
            Jv,
            -H,
            tol=krylov_tol,
            maxiter=krylov_maxiter,
            restart=krylov_restart,
        )
        d = jnp.where(info == 0, d, jnp.full_like(d, jnp.nan))
        return d

    if linear_solver == "dense":
        _solve_linear = _solve_linear_dense
    elif linear_solver == "gmres":
        _solve_linear = _solve_linear_gmres
    else:
        raise ValueError(
            f"linear_solver must be 'dense' or 'gmres', got {linear_solver!r}"
        )

    def solve(x0, mu):
        def body(state):
            x, H, it = state
            d = _solve_linear(x, H, mu)

            phi0 = 0.5 * jnp.sum(H**2)
            _, Jd = jax.jvp(lambda xx: _residual(xx, mu), (x,), (d,))
            dir_deriv_raw = jnp.dot(H, Jd)
            dir_deriv = jnp.where(dir_deriv_raw < 0, dir_deriv_raw, -phi0)

            def ls_cond(ls_state):
                alpha, ls_it = ls_state
                x_trial = x + alpha * d
                phi_trial = 0.5 * jnp.sum(_residual(x_trial, mu) ** 2)
                sufficient = phi_trial <= phi0 + armijo_c * alpha * dir_deriv
                return jnp.logical_and(~sufficient, ls_it < max_ls_steps)

            def ls_body(ls_state):
                alpha, ls_it = ls_state
                return alpha * backtrack_rho, ls_it + 1

            alpha_final, _ = lax.while_loop(ls_cond, ls_body, (1.0, 0))
            x_new = x + alpha_final * d
            H_new = _residual(x_new, mu)
            return x_new, H_new, it + 1

        def cond(state):
            x, H, it = state
            return jnp.logical_and(jnp.max(jnp.abs(H)) > tol, it < max_iter)

        H0 = _residual(x0, mu)
        init_state = (x0, H0, 0)
        final_x, _, _ = lax.while_loop(cond, body, init_state)
        return final_x

    return solve


def make_continuation_solver(
    newton_solve,
    F_fn: Callable,
    l: jnp.ndarray,
    u: jnp.ndarray,
    theta: jnp.ndarray,
    mu_init: float,
    mu_min: float,
    mu_decay: float,
    newton_tol: float,
    max_mu_steps: int,
):
    """Build a pure-JAX mu-continuation solver.

    Returns a function kernel(x0) -> (x, mu_used, num_steps) that
    uses lax.while_loop for the outer continuation, keeping all values as
    JAX arrays. No Python control flow, no float() coercions — fully
    traceable by jax.jit, jax.grad, etc.
    """

    def _residual_norm_at(x, mu):
        return jnp.max(jnp.abs(smoothed_residual(x, F_fn, l, u, mu, theta)))

    def kernel(x0):
        dt = x0.dtype
        mu_min_arr = jnp.array(mu_min, dtype=dt)
        mu_decay_arr = jnp.array(mu_decay, dtype=dt)
        newton_tol_arr = jnp.array(newton_tol, dtype=dt)

        def cond(state):
            x, mu_next, mu_used, step, converged = state
            return jnp.logical_and(step < max_mu_steps, ~converged)

        def body(state):
            x, mu_next, _mu_used, step, _converged = state
            x_new = newton_solve(x, mu_next)
            res = _residual_norm_at(x_new, mu_min_arr)
            converged = res < newton_tol_arr
            mu_used_new = jnp.where(converged, mu_min_arr, mu_next)
            mu_next_new = jnp.maximum(mu_next * mu_decay_arr, mu_min_arr)
            return x_new, mu_next_new, mu_used_new, step + 1, converged

        mu_init_arr = jnp.array(mu_init, dtype=dt)
        init = (x0, mu_init_arr, mu_init_arr, jnp.array(0), jnp.array(False))
        x_final, _mu_next, mu_used, num_steps, _converged = lax.while_loop(
            cond, body, init
        )
        return x_final, mu_used, num_steps

    return kernel
