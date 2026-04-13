import inspect
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import custom_vjp, lax
from jax.scipy.sparse.linalg import cg, gmres


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


_BIG = 1e15


def _normalize_F(F_fn):
    """Wrap F_fn to always accept (x, theta).

    If F_fn takes only one positional parameter (x), wrap it to ignore theta.
    Counts all positional parameters (POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD),
    including those with defaults, so `def F(x, theta=None)` is treated as
    two-argument and not wrapped. Functions with *args (VAR_POSITIONAL) are
    also treated as multi-argument and not wrapped.
    """
    sig = inspect.signature(F_fn)
    has_var_positional = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    if has_var_positional:
        return F_fn
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    params = [p for p in sig.parameters.values() if p.kind in positional_kinds]
    if len(params) <= 1:
        return lambda x, theta: F_fn(x)
    return F_fn


@jax.jit
def smooth_max(a: jnp.ndarray, b: jnp.ndarray, mu: float) -> jnp.ndarray:
    """Smooth approximation to elementwise max(a, b).

    Uses the identity smooth_max(a, b, mu) = (a + b + sqrt((a-b)^2 + mu)) / 2.
    Converges to max(a, b) as mu -> 0, and is always >= max(a, b).

    Args:
        a: First input array.
        b: Second input array.
        mu: Smoothing parameter (> 0). Smaller values give a tighter approximation.

    Returns:
        Smooth approximation to max(a, b), same shape as inputs.
    """
    return (a + b + jnp.sqrt((a - b) ** 2 + mu)) / 2.0


@jax.jit
def smooth_min(a: jnp.ndarray, b: jnp.ndarray, mu: float) -> jnp.ndarray:
    """Smooth approximation to elementwise min(a, b).

    Uses a numerically stable reformulation of (a + b - sqrt((a-b)^2 + mu)) / 2
    that avoids catastrophic cancellation when |a - b| >> sqrt(mu).
    Converges to min(a, b) as mu -> 0, and is always <= min(a, b).

    Args:
        a: First input array.
        b: Second input array.
        mu: Smoothing parameter (> 0). Smaller values give a tighter approximation.

    Returns:
        Smooth approximation to min(a, b), same shape as inputs.
    """
    s = jnp.sqrt((a - b) ** 2 + mu)
    denom = a + b + s
    abs_denom = jnp.maximum(jnp.abs(denom), jnp.sqrt(mu))
    safe_denom = jnp.copysign(abs_denom, denom + 1e-300)
    return (4.0 * a * b - mu) / (2.0 * safe_denom)


@jax.jit
def smooth_proj(
    z: jnp.ndarray, l: jnp.ndarray, u: jnp.ndarray, mu: float
) -> jnp.ndarray:
    """Smooth approximation to elementwise clip(z, l, u).

    Composes smooth_max (for the lower bound) and smooth_min (for the upper bound).
    Infinite bounds are replaced with a large finite surrogate to avoid NaN.
    Converges to clip(z, l, u) as mu -> 0.

    Args:
        z: Input array.
        l: Lower bounds (use -jnp.inf for unbounded below).
        u: Upper bounds (use jnp.inf for unbounded above).
        mu: Smoothing parameter (> 0).

    Returns:
        Smooth approximation to clip(z, l, u), same shape as z.
    """
    l_safe = jnp.where(jnp.isfinite(l), l, -_BIG)
    u_safe = jnp.where(jnp.isfinite(u), u, _BIG)
    inner = smooth_max(l_safe, z, mu)
    return smooth_min(u_safe, inner, mu)


def smoothed_residual(
    x: jnp.ndarray,
    F_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    l: jnp.ndarray,
    u: jnp.ndarray,
    mu: float,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    """Smoothed MCP residual: H(x) = x - smooth_proj(x - F(x, theta), l, u, mu).

    At a solution x* of the MCP, H(x*) = 0. The smoothing parameter mu
    controls how closely this approximates the exact (non-smooth) MCP residual.

    Note: F_fn must accept two arguments (x, theta). If your function takes
    only x, wrap it first: ``lambda x, theta: my_F(x)``.

    Args:
        x: Current iterate.
        F_fn: Residual map with signature F(x, theta) -> array.
        l: Lower bounds.
        u: Upper bounds.
        mu: Smoothing parameter (> 0).
        theta: Parameters passed to F_fn.

    Returns:
        Smoothed residual vector, same shape as x.
    """
    Fx = F_fn(x, theta)
    z = x - Fx
    proj = smooth_proj(z, l, u, mu)
    return x - proj


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
    """Build a JIT-compiled Newton solver that accepts mu as a JAX array.

    Returns a function solve(x0, mu) -> x that is traced once and reused
    across all mu steps, avoiding recompilation.
    """

    def _residual(x, mu):
        return smoothed_residual(x, F_fn, l, u, mu, theta)

    def _solve_linear_dense(x, H, mu):
        J = jax.jacfwd(lambda xx: _residual(xx, mu))(x)
        # Levenberg-Marquardt-style regularization: add a small diagonal term
        # to prevent singular Jacobians from producing NaN.
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

    _solve_linear = (
        _solve_linear_dense if linear_solver == "dense" else _solve_linear_gmres
    )

    @jax.jit
    def solve(x0, mu):
        def body(state):
            x, H, it = state
            d = _solve_linear(x, H, mu)

            phi0 = 0.5 * jnp.sum(H**2)
            # Directional derivative of merit phi(x) = 0.5*||H(x)||^2 along d:
            # nabla(phi) . d = (J^T H)^T d = H^T (J d)
            _, Jd = jax.jvp(lambda xx: _residual(xx, mu), (x,), (d,))
            dir_deriv_raw = jnp.dot(H, Jd)
            # Guard: if d is not a descent direction (dir_deriv >= 0), fall back
            # to steepest descent direction -grad(phi) = -J^T H, which has
            # dir_deriv = -||J^T H||^2 < 0. To avoid an extra VJP computation,
            # we use -phi0 = -0.5*||H||^2 as a conservative negative estimate.
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

    x = x0
    mu = mu_init
    num_steps = 0

    for step in range(max_mu_steps):
        num_steps = step + 1
        if verbose:
            print(f"Step {step:2d} | μ = {mu:.2e}")

        x = newton_solve(x, jnp.array(mu))

        # Early exit: check if already converged at mu_min
        res = float(jnp.max(jnp.abs(smoothed_residual(x, F_fn, l, u, mu_min, theta))))
        if res < newton_tol:
            break

        mu = max(mu * mu_decay, mu_min)

    residual_norm = float(
        jnp.max(jnp.abs(smoothed_residual(x, F_fn, l, u, mu_min, theta)))
    )
    converged = residual_norm < newton_tol

    if verbose:
        print(f"Finished. Final residual norm ≈ {residual_norm:.2e}")

    return MCPResult(
        x=x, residual_norm=residual_norm, num_steps=num_steps, converged=converged
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
    iterative linear solve for the adjoint system.

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

    @custom_vjp
    def solve(l, u, x0, theta):
        result = solve_mcp(
            F_fn_normalized,
            l,
            u,
            x0,
            theta,
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
            verbose=False,
        )
        return result.x

    def _fwd(l, u, x0, theta):
        x_star = solve(l, u, x0, theta)
        return x_star, (x_star, l, u, theta)

    def _bwd(res, cotangent):
        x_star, l, u, theta = res
        g = cotangent

        mu = mu_min

        def H_x(xx):
            return smoothed_residual(xx, F_fn_normalized, l, u, mu, theta)

        def H_theta(th):
            return smoothed_residual(x_star, F_fn_normalized, l, u, mu, th)

        def H_l(ll):
            return smoothed_residual(x_star, F_fn_normalized, ll, u, mu, theta)

        def H_u(uu):
            return smoothed_residual(x_star, F_fn_normalized, l, uu, mu, theta)

        def JTv(v):
            _, vjp_x = jax.vjp(H_x, x_star)
            return vjp_x(v)[0]

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


def solve_mcp_diff(
    F_fn: Callable,
    l: jnp.ndarray,
    u: jnp.ndarray,
    x0: jnp.ndarray,
    theta: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """Convenience wrapper: differentiable MCP solver.

    Builds the custom_vjp solver via make_mcp_solver_diff and calls it.
    See make_mcp_solver_diff for available keyword arguments.

    Note: This rebuilds the solver on every call, causing JAX to retrace.
    If calling repeatedly (e.g., in a training loop), use make_mcp_solver_diff
    once and reuse the returned function instead.
    """
    if theta is None:
        theta = jnp.zeros(0, dtype=jnp.asarray(x0).dtype)
    solver = make_mcp_solver_diff(F_fn, **kwargs)
    return solver(l, u, x0, theta)
