"""MCP solver: Newton-Armijo continuation with mu-smoothing."""

import inspect
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import gmres

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


def _normalize_F(F_fn):
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
    return F_fn


def _make_newton_solver(
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

    For repeated solves with the same problem, use ``make_mcp_solver_diff``
    which constructs the solver once at factory time and reuses it.
    """

    def _residual(x, mu):
        return smoothed_residual(x, F_fn, l, u, mu, theta)

    def _solve_linear_dense(x, H, mu):
        J = jax.jacfwd(lambda xx: _residual(xx, mu))(x)
        n = J.shape[0]
        J_reg = J + regularize * jnp.eye(n)
        return jnp.linalg.solve(J_reg, -H)

    def _solve_linear_gmres(x, H, mu):
        def Jv(v):
            _, jvp_val = jax.jvp(lambda xx: _residual(xx, mu), (x,), (v,))
            return jvp_val + regularize * v

        d, _ = gmres(
            Jv,
            -H,
            tol=krylov_tol,
            maxiter=krylov_maxiter,
            restart=krylov_restart,
        )
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


def _make_continuation_solver(
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
        def cond(state):
            x, mu_next, mu_used, step, converged = state
            return jnp.logical_and(step < max_mu_steps, ~converged)

        def body(state):
            x, mu_next, _mu_used, step, _converged = state
            x_new = newton_solve(x, mu_next)
            res = _residual_norm_at(x_new, mu_min)
            converged = res < newton_tol
            mu_used_new = mu_next
            mu_next_new = jnp.maximum(mu_next * mu_decay, mu_min)
            return x_new, mu_next_new, mu_used_new, step + 1, converged

        mu_init_arr = jnp.array(mu_init)
        init = (x0, mu_init_arr, mu_init_arr, jnp.array(0), jnp.array(False))
        x_final, _mu_next, mu_used, num_steps, _converged = lax.while_loop(
            cond, body, init
        )
        return x_final, mu_used, num_steps

    return kernel


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
    F_fn = _normalize_F(F_fn)

    l = jnp.asarray(l)
    u = jnp.asarray(u)
    x0 = jnp.asarray(x0)
    if theta is None:
        theta = jnp.zeros(0, dtype=x0.dtype)
    else:
        theta = jnp.asarray(theta)

    if l.shape != u.shape:
        raise ValueError(
            f"l and u must have the same shape, got {l.shape} and {u.shape}"
        )
    if x0.shape != l.shape:
        raise ValueError(
            f"x0 must have the same shape as l, got {x0.shape} and {l.shape}"
        )
    if jnp.any(l > u):
        raise ValueError("Lower bounds must not exceed upper bounds (l <= u)")
    if mu_init <= 0:
        raise ValueError(f"mu_init must be positive, got {mu_init}")
    if mu_decay <= 0 or mu_decay >= 1:
        raise ValueError(f"mu_decay must be in (0, 1), got {mu_decay}")
    if max_mu_steps < 1:
        raise ValueError(f"max_mu_steps must be >= 1, got {max_mu_steps}")
    if newton_tol < 0:
        raise ValueError(f"newton_tol must be non-negative, got {newton_tol}")
    if regularize < 0:
        raise ValueError(f"regularize must be non-negative, got {regularize}")

    newton_solve = _make_newton_solver(
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
        # Verbose mode: use Python loop for printing (not JIT-compatible)
        x = x0
        mu = mu_init
        num_steps = 0
        for step in range(max_mu_steps):
            num_steps = step + 1
            print(f"Step {step:2d} | μ = {mu:.2e}")
            x = newton_solve(x, jnp.array(mu))
            next_mu = max(mu * mu_decay, mu_min)
            if mu <= mu_min or next_mu <= mu_min:
                res = float(
                    jnp.max(jnp.abs(smoothed_residual(x, F_fn, l, u, mu_min, theta)))
                )
                if res < newton_tol:
                    break
            mu = next_mu
    else:
        # Non-verbose: pure-JAX kernel, JIT-compiled for performance
        continuation = _make_continuation_solver(
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
        x, _mu_final, num_steps = jax.jit(continuation)(x0)
        num_steps = int(num_steps)

    residual_norm = float(
        jnp.max(jnp.abs(smoothed_residual(x, F_fn, l, u, mu_min, theta)))
    )
    converged = residual_norm < newton_tol

    if verbose:
        print(f"Finished. Final residual norm ≈ {residual_norm:.2e}")

    return MCPResult(
        x=x, residual_norm=residual_norm, num_steps=num_steps, converged=converged
    )
