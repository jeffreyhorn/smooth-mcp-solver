"""Forward-only MCP solver factory (no custom_vjp, no gradient options).

For repeated forward solves, wrap the returned callable in ``jax.jit``. For
differentiable solves, use ``smooth_mcp.make_mcp_solver_diff`` instead.
"""

from typing import Callable, Union

from smooth_mcp._factory_common import (
    build_forward_kernel,
    build_make_aux,
    build_public_solve,
    validate_strict_validation,
)
from smooth_mcp._kernel import normalize_F, validate_solver_options


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
    strict_validation: Union[bool, str] = True,
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
        strict_validation: Controls validation of bounds and x0 inside
            traced code (jit, vmap). For ordinary eager calls, invalid-value
            checks run before the solve in the default and NaN-poisoning
            modes and raise on invalid inputs. In "checkify" mode, invalid
            value checks are reported via the returned Error rather than an
            eager ValueError, although tracing-time issues such as shape
            mismatches can still raise during tracing.
              - True (default): NaN-poisoning. Sanitizes l/u/x0 so the
                inner solve cannot blow up, runs the solve, then replaces
                x_star with NaN (and SolveInfo.converged=False,
                residual_norm=NaN) when inputs were invalid. Composes with
                jit and vmap at near-zero overhead. Failure surfaces as
                NaN, not an exception. This is the safe-by-default mode.
              - False: value checks are skipped under tracing. Invalid
                bounds or x0 flow through silently and may produce
                finite-looking but meaningless output. Use only when you
                know every call sees valid inputs (e.g., inside a tight
                inner loop after calling ``preflight_validate``).
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
        F_fn_normalized, mu_min=mu_min, newton_tol=newton_tol, stop_grad=False
    )

    def _inner_solve(l, u, x0, theta):
        x_star, mu_used, num_steps = _run_forward(l, u, x0, theta)
        if return_aux:
            return x_star, _make_aux(x_star, mu_used, num_steps, l, u, theta)
        return x_star

    return build_public_solve(
        _inner_solve,
        strict_validation=strict_validation,
        return_aux=return_aux,
    )
