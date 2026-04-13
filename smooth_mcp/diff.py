"""Differentiable MCP solver with implicit differentiation via custom_vjp."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.scipy.sparse.linalg import cg, gmres

from smooth_mcp.smoothing import smoothed_residual
from smooth_mcp.solver import (
    _make_continuation_solver,
    _make_newton_solver,
    _normalize_F,
)


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

    Returns:
        A function solve(l, u, x0, theta) -> x_star that supports jax.grad.
        If F_fn takes only x, pass any array for theta (it will be ignored)
        but theta is still required for JAX tracing.
    """
    F_fn_normalized = _normalize_F(F_fn)
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

    def _run_forward(l, u, x0, theta):
        """Pure-JAX forward solve shared by solve() and _fwd()."""
        newton = _make_newton_solver(
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
        continuation = _make_continuation_solver(
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

    @custom_vjp
    def solve(l, u, x0, theta):
        x_star, _mu_final, _num_steps = _run_forward(l, u, x0, theta)
        return x_star

    def _fwd(l, u, x0, theta):
        x_star, mu_final, _num_steps = _run_forward(l, u, x0, theta)
        return x_star, (x_star, mu_final, l, u, theta)

    def _bwd(res, cotangent):
        x_star, mu_final, l, u, theta = res
        g = cotangent

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
            lambda_star, _ = gmres(
                JTv,
                g,
                tol=gmres_tol,
                restart=gmres_restart,
                maxiter=gmres_maxiter,
                M=precond,
            )

        _, vjp_theta = jax.vjp(H_theta, theta)
        dtheta = -vjp_theta(lambda_star)[0]

        _, vjp_l = jax.vjp(H_l, l)
        dl = -vjp_l(lambda_star)[0]

        _, vjp_u = jax.vjp(H_u, u)
        du = -vjp_u(lambda_star)[0]

        d_x0 = g if differentiate_through_x0 else None

        return (dl, du, d_x0, dtheta)

    solve.defvjp(_fwd, _bwd)
    return solve
