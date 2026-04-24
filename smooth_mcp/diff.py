"""Differentiable MCP solver with implicit differentiation via custom_vjp."""

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.scipy.sparse.linalg import cg, gmres

from smooth_mcp._factory_common import (
    build_forward_kernel,
    build_make_aux,
    build_public_solve,
    validate_strict_validation,
)
from smooth_mcp._kernel import (
    normalize_F,
    validate_adjoint_options,
    validate_solver_options,
)
from smooth_mcp.smoothing import smoothed_residual


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
    strict_validation: Union[bool, str] = True,
):
    """Factory that returns a differentiable MCP solver with custom_vjp.

    Non-JAX arguments (F_fn, precond, solver options) are closed over so that
    the returned function only takes JAX-traceable arguments (l, u, x0, theta).

    The backward pass uses implicit differentiation with a Jacobian-free
    iterative linear solve for the adjoint system. Gradients are computed
    at ``SolveInfo.mu_used`` — the last smoothing parameter at which the
    Newton solve actually ran. This is the ``mu`` for which ``x_star`` is
    (approximately) the fixed point of H(x, mu)=0, so the implicit-
    differentiation adjoint is consistent with the returned solution.
    ``mu_used`` may be larger than ``mu_min`` when the residual at
    ``mu_min`` passed the tolerance before continuation reached
    ``mu_min``; in that case gradients reflect the coarser system that
    was actually solved, not the fully-smoothed limit.

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
        strict_validation: Controls validation of bounds and x0 inside
            traced code (jit, grad, vmap). Outside tracing, the usual eager
            validation still runs; for False/True modes invalid value
            inputs raise immediately, while in "checkify" mode invalid
            value checks are reported via the returned Error object rather
            than raising ValueError automatically (shape mismatches may
            still raise during tracing). This knob mainly affects what
            happens when the inputs are tracers.
              - True (default): NaN-poisoning. Sanitizes l/u/x0 so the
                inner solve cannot blow up, runs the solve, then replaces
                x_star with NaN (and marks SolveInfo.converged=False,
                residual_norm=NaN) when inputs were invalid. Composes with
                jit, grad, and vmap with near-zero overhead. Failure
                surfaces as NaN output, not an exception. This is the
                safe-by-default mode.
              - False: value checks are skipped under tracing. Invalid
                bounds or x0 flow through silently and may produce
                finite-looking but meaningless output. Use only when you
                know every call sees valid inputs (e.g., inside a tight
                inner loop after calling ``preflight_validate``).
              - "checkify": uses jax.experimental.checkify to attach a
                runtime error. The returned function's signature changes to
                (l, u, x0, theta) -> (Error, x_star) (or (Error, (x_star,
                SolveInfo)) when return_aux=True). Call err.throw() to
                raise. Composes with jit and grad. For vmap, first form a
                batched solver with jax.vmap(solver); avoid
                checkify.checkify(jax.vmap(...)) — the kernel uses
                lax.while_loop, which JAX rejects under
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
    validate_solver_options(
        mu_init=mu_init,
        mu_min=mu_min,
        mu_decay=mu_decay,
        max_mu_steps=max_mu_steps,
        newton_tol=newton_tol,
        regularize=regularize,
        armijo_c=armijo_c,
        backtrack_rho=backtrack_rho,
        max_ls_steps=max_ls_steps,
        linear_solver=linear_solver,
        krylov_tol=krylov_tol,
        krylov_maxiter=krylov_maxiter,
        krylov_restart=krylov_restart,
    )
    validate_adjoint_options(
        adjoint_method=adjoint_method,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
        gmres_restart=gmres_restart,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
    )
    validate_strict_validation(strict_validation)

    _run_forward = build_forward_kernel(
        F_fn_normalized,
        mu_init=mu_init,
        mu_min=mu_min,
        mu_decay=mu_decay,
        newton_tol=newton_tol,
        max_mu_steps=max_mu_steps,
        armijo_c=armijo_c,
        backtrack_rho=backtrack_rho,
        max_ls_steps=max_ls_steps,
        linear_solver=linear_solver,
        krylov_tol=krylov_tol,
        krylov_maxiter=krylov_maxiter,
        krylov_restart=krylov_restart,
        regularize=regularize,
    )
    _make_aux = build_make_aux(
        F_fn_normalized, mu_min=mu_min, newton_tol=newton_tol, stop_grad=True
    )

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

    def _inner_solve(l, u, x0, theta):
        x_star, mu_used, num_steps = _core_solve(l, u, x0, theta)
        if return_aux:
            return x_star, _make_aux(x_star, mu_used, num_steps, l, u, theta)
        return x_star

    return build_public_solve(
        _inner_solve,
        strict_validation=strict_validation,
        return_aux=return_aux,
    )
