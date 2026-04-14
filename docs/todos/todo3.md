# Improvement Plan (2026-04-13)

Based on `docs/reviews/review.codex.2026-04-13.md`.

## Phase 1: Lock In Regressions Before Refactoring

### 1. Add a failing `jax.jit` regression test for the differentiable solver
**Review refs:** 1, 8, 9  
Create a focused test that wraps a small `make_mcp_solver_diff(...)` solve inside `jax.jit`. The goal is to prove the current traced path fails, then flip the test to passing once the implementation is fixed.

### 2. Add a regression test for truncated continuation gradients
**Review refs:** 2, 9  
Construct a small problem with `max_mu_steps=1` or `2` and compare `jax.grad` against finite differences for the same truncated solver. This should fail today if the backward pass is using the wrong `mu`.

### 3. Add tests for invalid continuation controls
**Review refs:** 6, 9  
Add explicit validation tests for:

- `mu_decay <= 0`
- `mu_decay >= 1`
- `max_mu_steps < 1`
- negative `newton_tol`
- negative `regularize`

### 4. Add tests for `smoothed_residual` API behavior
**Review refs:** 5, 9  
Decide whether `smoothed_residual` should accept `F(x)` directly. Add a test for the chosen contract so the public API cannot drift again.

### 5. Add coverage for `precond`
**Review refs:** 9  
Add at least one backward-pass test that supplies a simple identity or diagonal preconditioner and verifies the solve still returns the correct gradient.

## Phase 2: Fix Correctness and JAX Composability

### 6. Split the solver into a pure-JAX kernel and an eager wrapper
**Review refs:** 1, 8  
Introduce an internal solve path that keeps all outputs as JAX values and avoids Python `if` checks on traced arrays and `float(...)` coercions. Keep `MCPResult` construction in the outer eager wrapper.

### 7. Make `make_mcp_solver_diff` use the pure-JAX forward path
**Review refs:** 1  
Update the custom-VJP forward function so it never enters the eager-only wrapper. That should make `jax.jit` composition possible and remove the traced boolean/float conversions.

### 8. Carry the final continuation state into the backward pass
**Review refs:** 2  
Return the actual terminal `mu` (and any other needed solve metadata) from `_fwd`, then use that exact value in `_bwd` instead of hard-coding `mu_min`.

### 9. Decide policy for non-converged differentiable solves
**Review refs:** 2  
Choose one of these behaviors and implement it consistently:

- differentiate through the actual final smoothed system using the recorded `mu`
- raise a clear error when users ask for gradients from a non-converged solve
- return a warning/status flag that callers can check before differentiating

Document the chosen policy in code and README.

### 10. Normalize `F_fn` inside `smoothed_residual` or remove it from the public surface
**Review refs:** 5  
Pick one contract and align implementation, docs, and exports with it.

### 11. Add full validation for solver control parameters
**Review refs:** 6  
Validate all continuation/Newton control ranges up front and raise precise `ValueError`s with the admissible range in the message.

## Phase 3: Remove Avoidable Recompilation and Adjoint Overhead

### 12. Stop rebuilding the jitted Newton solver on every `solve_mcp` call
**Review refs:** 3  
Either memoize `_make_newton_solver(...)` or refactor so the compiled Newton kernel is defined once and reused across identical call signatures.

### 13. Hoist `jax.vjp(H_x, x_star)` out of `JTv`
**Review refs:** 4  
Build the reverse-mode closure once per backward solve and reuse it for every GMRES/CG matrix-vector product.

### 14. Benchmark before and after the performance refactor
**Review refs:** 3, 4  
Measure:

- repeated `solve_mcp` calls on the same problem
- repeated backward solves on the same differentiable problem

Capture baseline and post-fix timings in a short dev note or PR description so the improvement is explicit.

## Phase 4: Clarify and Simplify the Public API

### 15. Make an explicit decision about `solve_mcp_diff`
**Review refs:** 7, 8  
Either:

- export it from `smooth_mcp/__init__.py` and add it to the README API table
- or remove it and update tests to use only `make_mcp_solver_diff`

### 16. Add a dedicated “JAX integration” section to the README
**Review refs:** 1, 8  
Document what is supported after the fixes:

- `jax.grad`
- `jax.jit`
- any remaining caveats around eager wrappers or solver-status outputs

### 17. Document the callable contract consistently
**Review refs:** 5, 8  
Make sure `solve_mcp`, `make_mcp_solver_diff`, `solve_mcp_diff` (if retained), and `smoothed_residual` all describe the same `F(x)` / `F(x, theta)` behavior, or clearly explain the differences.

## Phase 5: Improve Maintainability Once Behavior Is Stable

### 18. Split `smooth_mcp/core.py` into focused modules
**Review refs:** 10  
Suggested breakdown:

- `smooth_mcp/smoothing.py`
- `smooth_mcp/solver.py`
- `smooth_mcp/diff.py`
- keep `__init__.py` as the public export layer

Do this only after the new regression tests are passing so the refactor stays low-risk.

### 19. Trim duplicated wrapper logic and centralize result/status handling
**Review refs:** 1, 7, 10  
Once the module split is in place, make one path responsible for:

- JAX-native solve outputs
- eager `MCPResult` conversion
- any warning/error policy for non-converged differentiable solves

## Phase 6: Final Verification

### 20. Run the full quality gate in the project virtualenv
Run:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check smooth_mcp tests demos
.venv/bin/python -m mypy smooth_mcp demos
.venv/bin/python -m black --check smooth_mcp tests demos
```

### 21. Re-run the targeted runtime checks that motivated the review
**Review refs:** 1, 2, 3, 4  
Verify that after the changes:

- the differentiable solver works under `jax.jit`
- truncated continuation gradients match finite differences
- repeated identical solves do not trigger repeated `jit(solve)` compilation
- the adjoint path no longer rebuilds the VJP closure on every Krylov matvec
