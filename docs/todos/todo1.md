# Prioritized Improvement Plan

Items are ordered so that each can be addressed without depending on items later in the list.

## Phase 1: Correctness

These must be resolved first — later work (tests, performance) builds on correct behavior.

### 1. Verify and fix the backward pass sign for `dtheta`
**Review ref:** 1a
The implicit function theorem gives `dL/dtheta = -(dH/dtheta)^T lambda`, but the code computes `+(dH/dtheta)^T lambda`. Verify with finite-difference gradient check and fix if confirmed.

### 2. Fix the adjoint linear solver: use GMRES by default instead of CG
**Review ref:** 1b
CG requires a symmetric positive-definite operator, which `(dH/dx)^T` generally is not. Either default to GMRES (correct for non-symmetric systems) or add a symmetry check before choosing CG.

### 3. Handle `smooth_min` numerical instability for negative inputs
**Review ref:** 1f
The denominator `a + b + s` can approach zero when both inputs are negative and close. Add a safe fallback (e.g., switch to the original formula `(a + b - s) / 2` when `|a - b|` is small).

## Phase 2: Testing Infrastructure

Tests are needed before refactoring to catch regressions.

### 4. Set up test infrastructure
**Review ref:** 7d
Add pytest to `[project.optional-dependencies]` in `pyproject.toml`. Create `tests/` directory and `conftest.py`.

### 5. Add gradient correctness tests
**Review ref:** 6a
Use finite-difference gradient checks (or `jax.test_util.check_grads`) to verify `make_mcp_solver_diff` produces correct gradients for several problems. This validates fixes from items 1 and 2.

### 6. Add unit tests for `smooth_min`, `smooth_max`, `smooth_proj`
**Review ref:** 6a
Test with known values, extreme inputs (large/small `mu`, large `|a - b|`, negative inputs, infinite bounds).

### 7. Add solver correctness tests with known solutions
**Review ref:** 6a
Test against analytical solutions for simple LCPs and NCPs. Include edge cases: single variable, all variables at bounds, unbounded variables.

### 8. Add convergence failure tests
**Review ref:** 6a
Verify behavior when the solver fails to converge (singular Jacobian, bad initial guess).

## Phase 3: Performance

### 9. Remove redundant `@jax.jit` decorators in `_newton_solve_fixed_mu`
**Review ref:** 2a
Remove `@jax.jit` from `merit`, `body`, and `cond` — they are already traced by `lax.while_loop`.

### 10. Eliminate redundant residual evaluations in the Newton loop
**Review ref:** 2c
Carry `H` (and `merit` value) in the `while_loop` state so the residual is computed once per iteration instead of three times.

### 11. Fix recompilation per mu step
**Review ref:** 2b
Pass `mu` as a JAX array argument instead of closing over it as a Python float, so the while_loop is traced once and reused across mu steps.

### 12. Add a warning or cache to `solve_mcp_diff`
**Review ref:** 2d
Document that `solve_mcp_diff` rebuilds the solver on every call, or add caching so repeated calls with the same `F_fn` reuse the compiled solver.

## Phase 4: API Improvements

### 13. Make `theta` optional
**Review ref:** 3a
Allow `F_fn` to be `(x,) -> F(x)` or `(x, theta) -> F(x)`. If `theta` is not needed, the user should not have to create a dummy parameter.

### 14. Return convergence information from `solve_mcp`
**Review ref:** 3b, 1c
Return a result object (e.g., NamedTuple) with `x`, `converged`, `residual_norm`, and `num_steps` instead of just `x`.

### 15. Add input validation
**Review ref:** 3c
Check `l <= u`, matching dimensions, `mu_init > 0`, etc. Raise clear errors for invalid inputs.

### 16. Remove duplicate demo
**Review ref:** 4a
`demos/nonlinear_1d_mcp.py` and `demos/simple_nonlinear_1d_mcp_with_bounds.py` are identical. Remove one or differentiate them (e.g., give one different bounds or a different function).

## Phase 5: Documentation & Packaging

### 17. Add docstrings to all public API functions
**Review ref:** 5a
`smooth_max`, `smooth_min`, `smooth_proj`, and `smoothed_residual` are exported but undocumented.

### 18. Expand README "How it works" section
**Review ref:** 5c
Explain the smoothing parameter `mu`, the continuation strategy, the Newton-Armijo solver, and the implicit differentiation approach.

### 19. Improve README examples to avoid dummy `theta`
**Review ref:** 5b
Either show a parametrized example first, or wait until item 13 is done and show the simpler no-`theta` API.

### 20. Add `.gitignore`
**Review ref:** 7c
Exclude `__pycache__/`, `*.egg-info/`, `.venv/`, etc.

### 21. Add LICENSE file
**Review ref:** 5d

### 22. Fill in `pyproject.toml` metadata
**Review ref:** 7a
Add `description`, `authors`, `license`, `readme`, `urls`.

### 23. Add `__version__` to package
**Review ref:** 7b
Expose version in `smooth_mcp/__init__.py`.

## Phase 6: Future Enhancements (Lower Priority)

### 24. Add matrix-free Newton-Krylov option for the forward solve
**Review ref:** 1d
Replace dense `jacfwd` + `linalg.solve` with a Krylov solver (GMRES) using JVPs, enabling large-scale problems.

### 25. Support gradients through bounds (`l`, `u`)
**Review ref:** 3d
Compute and return `dl` and `du` gradients in the backward pass instead of `None`.

### 26. Add condition number check or regularization for the Newton Jacobian
**Review ref:** 1e
Detect near-singular Jacobians and either regularize (e.g., Levenberg-Marquardt) or warn.
