# Improvement Plan (2026-04-12)

Based on review-2026-04-12.md. Items ordered so each can be addressed without depending on items later in the list.

## Phase 1: Correctness Fixes

### 1. Fix `dir_deriv` to be correct with regularization
**Review ref:** 3b
`core.py:198` uses `dir_deriv = -jnp.sum(H**2)` which assumes `d = -J^{-1} H`. With regularization, `d = -(J + reg*I)^{-1} H`, so the correct directional derivative is `jnp.dot(H, jax.jvp(residual, (x,), (d,))[1])` or simply `jnp.dot(jax.grad(merit)(x), d)`. Compute it from the actual gradient and Newton direction.

### 2. Fix `_EMPTY_THETA` dtype issue
**Review ref:** 2b
`core.py:27` creates `jnp.zeros(0)` at import time, which may be float32. Replace with a lazy approach: create it inside `solve_mcp`/`solve_mcp_diff` when needed, using the same dtype as `x0`.

### 3. Fix `_normalize_F` for functions with default arguments
**Review ref:** 5e
`def F(x, theta=None)` currently counts 1 required param and wraps to ignore theta. Count all positional params (including those with defaults) to decide arity.

## Phase 2: API Consistency

### 4. Add forward solver parameters to `make_mcp_solver_diff`
**Review ref:** 6a
Add `linear_solver`, `regularize`, `krylov_tol`, `krylov_maxiter`, `krylov_restart` parameters to `make_mcp_solver_diff` and pass them through to the internal `solve_mcp` call.

### 5. Pass `newton_tol` as keyword argument to `_make_newton_solver`
**Review ref:** 3a
Change the positional `newton_tol,` at `core.py:305` to `tol=newton_tol,` to prevent silent breakage if parameters are reordered.

### 6. Normalize `F_fn` in `smoothed_residual` or document the requirement
**Review ref:** 1b, 6d
Either add `_normalize_F` call inside `smoothed_residual`, or remove it from `__all__` and document it as internal-only.

## Phase 3: Documentation

### 7. Update README parameter tables
**Review ref:** 1a, 1c
Add the missing `solve_mcp` parameters (`linear_solver`, `krylov_*`, `regularize`, `verbose`) to the table. Clarify which table applies to which function. Add a note explaining the forward vs backward solver parameter split.

### 8. Clean up `differentiable_lcp.py` dead parameters
**Review ref:** 1d
Remove the `cg_tol` and `cg_maxiter` arguments from the `make_mcp_solver_diff` call in the demo, since `adjoint_method` defaults to `"gmres"` and they have no effect.

## Phase 4: Demos

### 9. Rewrite KKT demo as a proper MCP formulation
**Review ref:** 3c
Replace the 4D manual `[stationarity, complementarity]` formulation with the standard 2D `F(x) = Qx + c, l=0, u=inf` formulation. Add a comment explaining how MCP naturally enforces KKT complementarity.

### 10. Add a higher-dimensional demo (10+ variables)
**Review ref:** 4b
Create a demo with a larger problem (e.g., a discretized obstacle problem, a multi-commodity network equilibrium, or a larger LCP). Use `linear_solver="gmres"` to showcase the matrix-free solver.

### 11. Add a demo showing gradients through bounds
**Review ref:** 4e
Create a demo (or extend `differentiable_lcp.py`) that optimizes over bounds and shows `jax.grad` w.r.t. `l` or `u`.

### 12. Add a demo using `solve_mcp_diff`
**Review ref:** 4c
Add a simple demo showing the convenience wrapper, or remove `solve_mcp_diff` from the public API if it's not worth maintaining.

### 13. Ensure all demos print `MCPResult` fields
**Review ref:** 4d
Update demos that only print `result.x` to also show `result.converged`, `result.residual_norm`, and `result.num_steps`.

## Phase 5: Performance

### 14. Add early-exit on convergence in the mu-continuation loop
**Review ref:** 2a
After each `newton_solve` call, check if the residual is already below tolerance. If so, break early instead of continuing to reduce mu.

## Phase 6: Testing

### 15. Add tests for `solve_mcp_diff`
**Review ref:** 5a
Test that `solve_mcp_diff` produces correct solutions and gradients.

### 16. Add tests for `adjoint_method="cg"`
**Review ref:** 5b
Test gradient correctness with CG adjoint solver on a problem where the Jacobian is symmetric positive-definite.

### 17. Add tests for `_normalize_F` edge cases
**Review ref:** 5e
Test with lambda functions, functions with default args, and `*args` signatures.

### 18. Make singular Jacobian NaN test less fragile
**Review ref:** 5f
Instead of asserting NaN, assert that `regularize=0` produces a worse result (higher residual or `not result.converged`) compared to the default.

### 19. Run full quality gate
Run `make format && make typecheck && make lint && make test` and fix any issues.
