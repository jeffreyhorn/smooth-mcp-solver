"""MCP solver: Newton-Armijo continuation with mu-smoothing."""

from typing import Callable, NamedTuple, Optional

import jax.numpy as jnp

from smooth_mcp._kernel import (
    make_continuation_solver,
    make_newton_solver,
    normalize_F,
    validate_bounds_and_x0,
)
from smooth_mcp.smoothing import smoothed_residual


class MCPResult(NamedTuple):
    """Result from solve_mcp.

    Attributes:
        x: Solution array.
        residual_norm: Max absolute value of the smoothed residual at the solution.
        num_steps: Total number of outer solver steps (mu-reduction and terminal-mu iterations).
        converged: True if the final residual norm is below newton_tol.
    """

    x: jnp.ndarray
    residual_norm: float
    num_steps: int
    converged: bool


def solve_mcp(
    F_fn: Callable,
    l: jnp.ndarray,
    u: jnp.ndarray,
    x0: jnp.ndarray,
    theta: Optional[jnp.ndarray] = None,
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
    verbose: bool = False,
) -> MCPResult:
    """Solve a Mixed Complementarity Problem via smoothing + Newton + Armijo line search.

    Finds x such that for each component i:
        l_i <= x_i <= u_i  and  the complementarity conditions hold with F_fn.

    Args:
        F_fn: Residual map. Either F(x) -> array or F(x, theta) -> array.
        l: Lower bounds.
        u: Upper bounds (use jnp.inf for unbounded).
        x0: Initial guess.
        theta: Parameters passed to F_fn. Optional if F_fn takes only x.
        mu_init: Initial smoothing parameter.
        mu_min: Terminal smoothing parameter.
        mu_decay: Multiplicative decay factor for mu each step.
        newton_tol: Newton convergence tolerance.
        max_mu_steps: Maximum number of mu-reduction steps.
        armijo_c: Armijo sufficient decrease parameter.
        backtrack_rho: Backtracking line search contraction factor.
        max_ls_steps: Maximum line search steps per Newton iteration.
        linear_solver: Linear solver for Newton steps. "dense" (default) forms the
            full Jacobian via jacfwd and uses linalg.solve. "gmres" uses matrix-free
            GMRES with JVPs, avoiding Jacobian construction (better for large problems).
        krylov_tol: GMRES tolerance (only used when linear_solver="gmres").
        krylov_maxiter: GMRES maximum iterations (only used when linear_solver="gmres").
        krylov_restart: GMRES restart parameter (only used when linear_solver="gmres").
        regularize: Tikhonov regularization added to the Newton Jacobian (J + reg*I).
            Prevents NaN from singular Jacobians. Set to 0 to disable.
        verbose: Print progress.

    Returns:
        MCPResult with fields x, residual_norm, num_steps, converged.
    """
    F_fn = normalize_F(F_fn)

    l = jnp.asarray(l)
    u = jnp.asarray(u)
    x0 = jnp.asarray(x0)
    if theta is None:
        theta = jnp.zeros(0, dtype=x0.dtype)
    else:
        theta = jnp.asarray(theta)

    validate_bounds_and_x0(l, u, x0)
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

    newton_solve = make_newton_solver(
        F_fn,
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

    if verbose:
        x = x0
        mu = mu_init
        num_steps = 0
        for step in range(max_mu_steps):
            num_steps = step + 1
            print(f"Step {step:2d} | μ = {mu:.2e}")
            x = newton_solve(x, jnp.array(mu, dtype=x0.dtype))
            res = float(
                jnp.max(jnp.abs(smoothed_residual(x, F_fn, l, u, mu_min, theta)))
            )
            if res < newton_tol:
                break
            mu = max(mu * mu_decay, mu_min)
    else:
        continuation = make_continuation_solver(
            newton_solve,
            F_fn,
            l,
            u,
            theta,
            mu_init,
            mu_min,
            mu_decay,
            newton_tol,
            max_mu_steps,
        )
        x, _mu_final, num_steps = continuation(x0)
        num_steps = int(num_steps)

    residual_norm = float(
        jnp.max(
            jnp.abs(
                smoothed_residual(
                    x, F_fn, l, u, jnp.array(mu_min, dtype=x0.dtype), theta  # type: ignore[arg-type]
                )
            )
        )
    )
    converged = residual_norm < newton_tol

    if verbose:
        print(f"Finished. Final residual norm ≈ {residual_norm:.2e}")

    return MCPResult(
        x=x, residual_norm=residual_norm, num_steps=num_steps, converged=converged
    )
