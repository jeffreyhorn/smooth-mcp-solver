"""Differentiable MCP solver with implicit differentiation via custom_vjp."""

from typing import Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.experimental import checkify
from jax.scipy.sparse.linalg import cg, gmres

from smooth_mcp._kernel import (
    make_continuation_solver,
    make_newton_solver,
    normalize_F,
    sanitize_bounds,
    traced_invalid_mask,
    validate_bounds_and_x0,
)
from smooth_mcp.smoothing import smoothed_residual


class SolveInfo(NamedTuple):
    """Auxiliary diagnostics from the differentiable solver.

    Attributes:
        mu_used: Terminal smoothing parameter actually reached.
        num_steps: Total continuation steps taken.
        residual_norm: Max absolute smoothed residual at x_star, evaluated at mu_min.
        converged: True if residual_norm < newton_tol.
    """

    mu_used: jnp.ndarray
    num_steps: jnp.ndarray
    residual_norm: jnp.ndarray
    converged: jnp.ndarray


def make_mcp_solver_diff(
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
    adjoint_method: str = "gmres",
    gmres_tol: float = 1e-8,
    gmres_restart: int = 30,
    gmres_maxiter: int = 500,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 1000,
    precond: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    differentiate_through_x0: bool = False,
    return_aux: bool = False,
    strict_validation: Union[bool, str] = False,
):
    """Factory that returns a differentiable MCP solver with custom_vjp.

    Non-JAX arguments (F_fn, precond, solver options) are closed over so that
    the returned function only takes JAX-traceable arguments (l, u, x0, theta).

    The backward pass uses implicit differentiation with a Jacobian-free
    iterative linear solve for the adjoint system. Gradients are computed
    at the actual terminal smoothing parameter from the forward solve (not
    necessarily mu_min). This means truncated solves (small max_mu_steps)
    produce gradients consistent with the smoothed system that was actually
    solved, rather than the fully-converged system.

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
        linear_solver: Forward Newton linear solver. "dense" (default) or "gmres"
            (matrix-free, better for large problems).
        krylov_tol: Forward GMRES tolerance (only used when linear_solver="gmres").
        krylov_maxiter: Forward GMRES max iterations (only when linear_solver="gmres").
        krylov_restart: Forward GMRES restart (only when linear_solver="gmres").
        regularize: Tikhonov regularization for the forward Newton Jacobian.
        adjoint_method: Backward adjoint linear solver. "gmres" (default,
            correct for general non-symmetric systems) or "cg" (only valid when
            the Jacobian dH/dx is symmetric positive-definite).
        gmres_tol: Adjoint GMRES tolerance.
        gmres_restart: Adjoint GMRES restart parameter.
        gmres_maxiter: Adjoint GMRES maximum iterations.
        cg_tol: Adjoint CG tolerance (only used when adjoint_method="cg").
        cg_maxiter: Adjoint CG max iterations (only used when adjoint_method="cg").
        precond: Optional preconditioner callable for the adjoint linear solve.
        differentiate_through_x0: If True, use straight-through estimator for x0 gradients.
        return_aux: If True, the returned function returns (x_star, SolveInfo)
            instead of just x_star. The SolveInfo fields are not differentiated.
        strict_validation: Controls validation of bounds inside traced code
            (jit, grad, vmap). Eager validation always runs and raises on
            invalid inputs; this knob only affects what happens when the
            inputs are tracers.
              - False (default): value checks are skipped under tracing
                (current behavior).
              - True: NaN-poisoning. Sanitizes l/u so the inner solve cannot
                blow up, runs the solve, then replaces x_star with NaN (and
                marks SolveInfo.converged=False, residual_norm=NaN) when
                inputs were invalid. Composes with jit, grad, and vmap with
                near-zero overhead. Failure surfaces as NaN output, not an
                exception.
              - "checkify": uses jax.experimental.checkify to attach a
                runtime error. The returned function's signature changes to
                (l, u, x0, theta) -> (Error, x_star) (or (Error, (x_star,
                SolveInfo)) when return_aux=True). Call err.throw() to
                raise. Composes with jit and grad. For vmap, use
                vmap(solver(...)) ordering, not checkify(vmap(...)) — the
                kernel uses lax.while_loop, which JAX rejects under
                checkify-of-vmap-of-while. See docs/api.md.

    Returns:
        If return_aux is False (default):
            A function solve(l, u, x0, theta) -> x_star that supports jax.grad.
        If return_aux is True:
            A function solve(l, u, x0, theta) -> (x_star, SolveInfo).
            Gradients flow through x_star only; SolveInfo is stopped.
        If strict_validation == "checkify":
            The function signature is wrapped to return (Error, ...) per
            jax.experimental.checkify conventions.
        If F_fn takes only x, pass any array for theta (it will be ignored)
        but theta is still required for JAX tracing.
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
    if adjoint_method not in ("gmres", "cg"):
        raise ValueError(
            f"adjoint_method must be 'gmres' or 'cg', got {adjoint_method!r}"
        )
    if strict_validation not in (False, True, "checkify"):
        raise ValueError(
            f"strict_validation must be False, True, or 'checkify', "
            f"got {strict_validation!r}"
        )

    def _run_forward(l, u, x0, theta):
        """Pure-JAX forward solve shared by solve() and _fwd()."""
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

    def _compute_grads(x_star, mu_final, l, u, theta, g):
        """Shared adjoint computation for the backward pass."""
        mu = mu_final

        def H_x(xx):
            return smoothed_residual(xx, F_fn_normalized, l, u, mu, theta)

        def H_theta(th):
            return smoothed_residual(x_star, F_fn_normalized, l, u, mu, th)

        def H_l(ll):
            return smoothed_residual(x_star, F_fn_normalized, ll, u, mu, theta)

        def H_u(uu):
            return smoothed_residual(x_star, F_fn_normalized, l, uu, mu, theta)

        _, vjp_x_fn = jax.vjp(H_x, x_star)

        def JTv(v):
            return vjp_x_fn(v)[0]

        if adjoint_method == "cg":
            lambda_star, _ = cg(JTv, g, tol=cg_tol, maxiter=cg_maxiter, M=precond)
        else:
            lambda_star, info = gmres(
                JTv,
                g,
                tol=gmres_tol,
                restart=gmres_restart,
                maxiter=gmres_maxiter,
                M=precond,
            )
            lambda_star = jnp.where(
                info == 0, lambda_star, jnp.full_like(lambda_star, jnp.nan)
            )

        _, vjp_theta = jax.vjp(H_theta, theta)
        dtheta = -vjp_theta(lambda_star)[0]

        _, vjp_l = jax.vjp(H_l, l)
        dl = -vjp_l(lambda_star)[0]

        _, vjp_u = jax.vjp(H_u, u)
        du = -vjp_u(lambda_star)[0]

        d_x0 = g if differentiate_through_x0 else None

        return (dl, du, d_x0, dtheta)

    def _make_aux(x_star, mu_used, num_steps, l, u, theta):
        """Build stop-gradiented SolveInfo from forward results."""
        mu_min_arr = jnp.array(mu_min, dtype=x_star.dtype)
        newton_tol_arr = jnp.array(newton_tol, dtype=x_star.dtype)
        residual = smoothed_residual(x_star, F_fn_normalized, l, u, mu_min_arr, theta)
        residual_norm = jnp.max(jnp.abs(residual))
        converged = residual_norm < newton_tol_arr
        return SolveInfo(
            mu_used=jax.lax.stop_gradient(mu_used),
            num_steps=jax.lax.stop_gradient(num_steps),
            residual_norm=jax.lax.stop_gradient(residual_norm),
            converged=jax.lax.stop_gradient(converged),
        )

    # One custom_vjp definition — always returns (x_star, mu_used, num_steps).
    # Aux packaging (return_aux) happens outside the custom_vjp boundary.

    @custom_vjp
    def _core_solve(l, u, x0, theta):
        x_star, mu_used, num_steps = _run_forward(l, u, x0, theta)
        return (
            x_star,
            jax.lax.stop_gradient(mu_used),
            jax.lax.stop_gradient(num_steps),
        )

    def _core_fwd(l, u, x0, theta):
        x_star, mu_final, num_steps = _run_forward(l, u, x0, theta)
        primal_out = (
            x_star,
            jax.lax.stop_gradient(mu_final),
            jax.lax.stop_gradient(num_steps),
        )
        residuals = (x_star, mu_final, l, u, theta)
        return primal_out, residuals

    def _core_bwd(res, cotangent):
        x_star, mu_final, l, u, theta = res
        g_x = cotangent[0]
        return _compute_grads(x_star, mu_final, l, u, theta, g_x)

    _core_solve.defvjp(_core_fwd, _core_bwd)

    def solve(l, u, x0, theta):
        x_star, mu_used, num_steps = _core_solve(l, u, x0, theta)
        if return_aux:
            return x_star, _make_aux(x_star, mu_used, num_steps, l, u, theta)
        return x_star

    def _poison_aux(aux, invalid):
        """Mark SolveInfo as failed when traced inputs were invalid."""
        return SolveInfo(
            mu_used=aux.mu_used,
            num_steps=aux.num_steps,
            residual_norm=jnp.where(invalid, jnp.nan, aux.residual_norm),
            converged=jnp.where(invalid, jnp.bool_(False), aux.converged),
        )

    def _poisoned_solve(l, u, x0, theta):
        """NaN-poisoning wrapper around the inner custom_vjp solve."""
        invalid = traced_invalid_mask(l, u)
        safe_l, safe_u = sanitize_bounds(l, u)
        result = solve(safe_l, safe_u, x0, theta)
        if return_aux:
            x_star, aux = result
            x_star = jnp.where(invalid, jnp.full_like(x_star, jnp.nan), x_star)
            return x_star, _poison_aux(aux, invalid)
        return jnp.where(invalid, jnp.full_like(result, jnp.nan), result)

    def _checks(l, u):
        checkify.check(~jnp.any(jnp.isnan(l)), "l contains NaN")
        checkify.check(~jnp.any(jnp.isnan(u)), "u contains NaN")
        checkify.check(
            jnp.all(l <= u), "l must be <= u element-wise"
        )

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
        return solve(l, u, x0, theta)

    if strict_validation == "checkify":
        def _checkify_target(l, u, x0, theta):
            l = jnp.asarray(l)
            u = jnp.asarray(u)
            x0 = jnp.asarray(x0)
            theta = jnp.asarray(theta)
            validate_bounds_and_x0(l, u, x0)
            _checks(l, u)
            return solve(l, u, x0, theta)

        return checkify.checkify(_checkify_target)

    return solve_checked
