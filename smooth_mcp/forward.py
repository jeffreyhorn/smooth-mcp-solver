"""Forward-only MCP solver factory (no custom_vjp, no gradient options).

For repeated forward solves, wrap the returned callable in ``jax.jit``. For
differentiable solves, use ``smooth_mcp.make_mcp_solver_diff`` instead.
"""

from typing import Callable, Union

import jax.numpy as jnp
from jax.experimental import checkify

from smooth_mcp._kernel import (
    make_continuation_solver,
    make_newton_solver,
    normalize_F,
    sanitize_bounds,
    traced_invalid_mask,
    validate_bounds_and_x0,
)
from smooth_mcp.diff import SolveInfo
from smooth_mcp.smoothing import smoothed_residual


def make_mcp_solver(
    F_fn: Callable,
    mu_init: float = 1.0,
    mu_min: float = 1e-12,
    mu_decay: float = 0.5,
    newton_tol: float = 1e-10,
    max_mu_steps: int = 50,
    armijo_c: float = 1e-4,
    backtrack_rho: float = 0.5,
    max_ls_steps: int = 20,
    linear_solver: str = "dense",
    krylov_tol: float = 1e-6,
    krylov_maxiter: int = 500,
    krylov_restart: int = 30,
    regularize: float = 1e-12,
    return_aux: bool = False,
    strict_validation: Union[bool, str] = False,
):
    """Factory returning a reusable forward-only MCP solver.

    Non-JAX arguments (F_fn, solver options) are closed over so the returned
    callable only takes JAX-traceable arguments (l, u, x0, theta). Wrap the
    returned callable in ``jax.jit`` for repeated fast forward solves; each
    call in eager mode rebuilds the Newton and continuation kernels, which
    is correct but slower.

    Unlike ``make_mcp_solver_diff``, this factory does not install a
    ``custom_vjp``. Use ``make_mcp_solver_diff`` if you need ``jax.grad``.

    Args:
        F_fn: Residual map. Either F(x) -> array or F(x, theta) -> array.
        mu_init: Initial smoothing parameter.
        mu_min: Terminal smoothing parameter.
        mu_decay: Multiplicative decay factor for mu.
        newton_tol: Newton convergence tolerance.
        max_mu_steps: Maximum mu-reduction steps.
        armijo_c: Armijo sufficient decrease parameter.
        backtrack_rho: Backtracking contraction factor.
        max_ls_steps: Maximum line search steps.
        linear_solver: Newton linear solver. "dense" (default) or "gmres"
            (matrix-free, better for large problems).
        krylov_tol: GMRES tolerance (only when linear_solver="gmres").
        krylov_maxiter: GMRES max iterations (only when linear_solver="gmres").
        krylov_restart: GMRES restart (only when linear_solver="gmres").
        regularize: Tikhonov regularization for the Newton Jacobian.
        return_aux: If True, the returned function returns (x_star, SolveInfo)
            instead of just x_star.
        strict_validation: Controls validation of bounds inside traced code
            (jit, vmap). Eager validation always runs and raises on invalid
            inputs; this knob only affects what happens when the inputs are
            tracers.
              - False (default): value checks are skipped under tracing.
              - True: NaN-poisoning. Sanitizes l/u so the inner solve cannot
                blow up, runs the solve, then replaces x_star with NaN (and
                SolveInfo.converged=False, residual_norm=NaN) when inputs
                were invalid. Composes with jit and vmap at near-zero
                overhead. Failure surfaces as NaN, not an exception.
              - "checkify": uses jax.experimental.checkify to attach a
                runtime error. Signature changes to
                (l, u, x0, theta) -> (Error, x_star) (or
                (Error, (x_star, SolveInfo)) when return_aux=True). Call
                err.throw() to raise. For vmap, use vmap(solver), not
                checkify(vmap(...)) — the kernel uses lax.while_loop, which
                JAX rejects under checkify-of-vmap-of-while.

    Returns:
        A function solve(l, u, x0, theta) -> x_star when return_aux=False,
        or solve(l, u, x0, theta) -> (x_star, SolveInfo) when return_aux=True.
        If strict_validation == "checkify", the signature is wrapped to
        return (Error, ...) per jax.experimental.checkify conventions.
        If F_fn takes only x, pass any array for theta (e.g., jnp.zeros(0));
        theta is still required for JAX tracing consistency.
    """
    F_fn_normalized = normalize_F(F_fn)
    if mu_init <= 0:
        raise ValueError(f"mu_init must be positive, got {mu_init}")
    if mu_min <= 0:
        raise ValueError(f"mu_min must be positive, got {mu_min}")
    if mu_min > mu_init:
        raise ValueError(
            f"mu_min must be <= mu_init, got mu_min={mu_min}, mu_init={mu_init}"
        )
    if mu_decay <= 0 or mu_decay >= 1:
        raise ValueError(f"mu_decay must be in (0, 1), got {mu_decay}")
    if max_mu_steps < 1:
        raise ValueError(f"max_mu_steps must be >= 1, got {max_mu_steps}")
    if newton_tol < 0:
        raise ValueError(f"newton_tol must be non-negative, got {newton_tol}")
    if regularize < 0:
        raise ValueError(f"regularize must be non-negative, got {regularize}")
    if strict_validation not in (False, True, "checkify"):
        raise ValueError(
            f"strict_validation must be False, True, or 'checkify', "
            f"got {strict_validation!r}"
        )

    def _run_forward(l, u, x0, theta):
        """Pure-JAX forward solve."""
        newton = make_newton_solver(
            F_fn_normalized,
            l,
            u,
            theta,
            tol=newton_tol,
            armijo_c=armijo_c,
            backtrack_rho=backtrack_rho,
            max_ls_steps=max_ls_steps,
            linear_solver=linear_solver,
            krylov_tol=krylov_tol,
            krylov_maxiter=krylov_maxiter,
            krylov_restart=krylov_restart,
            regularize=regularize,
        )
        continuation = make_continuation_solver(
            newton,
            F_fn_normalized,
            l,
            u,
            theta,
            mu_init,
            mu_min,
            mu_decay,
            newton_tol,
            max_mu_steps,
        )
        return continuation(x0)  # (x_star, mu_used, num_steps)

    def _make_aux(x_star, mu_used, num_steps, l, u, theta):
        """Build SolveInfo from forward results. No stop_gradient needed."""
        mu_min_arr = jnp.array(mu_min, dtype=x_star.dtype)
        newton_tol_arr = jnp.array(newton_tol, dtype=x_star.dtype)
        residual = smoothed_residual(x_star, F_fn_normalized, l, u, mu_min_arr, theta)
        residual_norm = jnp.max(jnp.abs(residual))
        converged = residual_norm < newton_tol_arr
        return SolveInfo(
            mu_used=mu_used,
            num_steps=num_steps,
            residual_norm=residual_norm,
            converged=converged,
        )

    def _solve(l, u, x0, theta):
        x_star, mu_used, num_steps = _run_forward(l, u, x0, theta)
        if return_aux:
            return x_star, _make_aux(x_star, mu_used, num_steps, l, u, theta)
        return x_star

    def _poison_aux(aux, invalid):
        return SolveInfo(
            mu_used=aux.mu_used,
            num_steps=aux.num_steps,
            residual_norm=jnp.where(invalid, jnp.nan, aux.residual_norm),
            converged=jnp.where(invalid, jnp.bool_(False), aux.converged),
        )

    def _poisoned_solve(l, u, x0, theta):
        invalid = traced_invalid_mask(l, u)
        safe_l, safe_u = sanitize_bounds(l, u)
        result = _solve(safe_l, safe_u, x0, theta)
        if return_aux:
            x_star, aux = result
            x_star = jnp.where(invalid, jnp.full_like(x_star, jnp.nan), x_star)
            return x_star, _poison_aux(aux, invalid)
        return jnp.where(invalid, jnp.full_like(result, jnp.nan), result)

    def _checks(l, u):
        checkify.check(~jnp.any(jnp.isnan(l)), "l contains NaN")
        checkify.check(~jnp.any(jnp.isnan(u)), "u contains NaN")
        checkify.check(jnp.all(l <= u), "l must be <= u element-wise")

    def solve_checked(l, u, x0, theta):
        l, u, x0, theta = (
            jnp.asarray(l),
            jnp.asarray(u),
            jnp.asarray(x0),
            jnp.asarray(theta),
        )
        validate_bounds_and_x0(l, u, x0)
        if strict_validation is True:
            return _poisoned_solve(l, u, x0, theta)
        return _solve(l, u, x0, theta)

    if strict_validation == "checkify":
        def _checkify_target(l, u, x0, theta):
            l = jnp.asarray(l)
            u = jnp.asarray(u)
            x0 = jnp.asarray(x0)
            theta = jnp.asarray(theta)
            validate_bounds_and_x0(l, u, x0)
            _checks(l, u)
            return _solve(l, u, x0, theta)

        return checkify.checkify(_checkify_target)

    return solve_checked
