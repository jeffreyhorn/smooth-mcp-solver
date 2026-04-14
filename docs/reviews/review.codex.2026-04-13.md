# Code Review: smooth-mcp (2026-04-13)

Scope: `smooth_mcp/`, `README.md`, demos, and tests. I used static review plus targeted runtime checks in the repo virtualenv to verify behavior where the code path looked risky.

## Findings

### 1. `make_mcp_solver_diff` is not composable with `jax.jit` even though it is presented as a JAX-native differentiable solver
**Area:** correctness, usability  
**Refs:** `smooth_mcp/core.py:320-339`, `smooth_mcp/core.py:373-383`, `smooth_mcp/core.py:466-487`, `README.md:3`, `README.md:45-53`, `README.md:112-131`

`make_mcp_solver_diff` calls `solve_mcp` inside the custom-VJP forward path. `solve_mcp` performs Python-side boolean checks on JAX arrays (`if jnp.any(l > u):`) and converts JAX values to Python floats (`float(jnp.max(...))`). Under `jax.jit`, those host-side conversions fail with `TracerBoolConversionError` / concretization errors.

This is a real integration bug, not just a documentation gap. A library described as a JAX solver with implicit differentiation should either work under `jax.jit` or explicitly separate the eager-only and traceable APIs.

**Improvement:** factor the actual solve into a pure-JAX kernel that returns JAX values, and keep Python validation / pretty result packing in a thin eager wrapper.

### 2. The backward pass always linearizes at `mu_min`, even when the forward solve stopped earlier at a larger continuation value
**Area:** correctness  
**Refs:** `smooth_mcp/core.py:361-379`, `smooth_mcp/core.py:466-487`, `smooth_mcp/core.py:493-509`

The forward solve may exit before reaching `mu_min` when `max_mu_steps` is too small or the user intentionally truncates the continuation schedule. In that case the returned `x` solves the smoothed problem for the last actual `mu`, not necessarily `mu_min`.

The backward pass ignores that and hard-codes `mu = mu_min`. That means the implicit gradient can correspond to a different residual map than the one that produced the forward solution.

Current tests only differentiate well-converged solves, so this mismatch is not covered.

**Improvement:** save the final continuation value in `_fwd` residuals and use that exact `mu` in `_bwd`. If the team wants gradients only for converged terminal solves, detect and reject the non-terminal case explicitly.

### 3. `solve_mcp` recompiles the Newton solve on every call
**Area:** efficiency  
**Refs:** `smooth_mcp/core.py:163-262`, `smooth_mcp/core.py:341-366`

`solve_mcp` rebuilds `_make_newton_solver(...)` every time it is called, and `_make_newton_solver` creates a fresh nested `@jax.jit`-decorated `solve` function. In practice that means repeated identical solves pay repeated trace/compile cost. With `JAX_LOG_COMPILES=1`, the same 1D problem compiled `jit(solve)` twice across two identical `solve_mcp(...)` calls.

For a library whose natural use case is solving many nearby problems, this is a substantial avoidable cost.

**Improvement:** memoize solver construction by callable/options/shape-dtype signature, or hoist the compiled kernels out of `solve_mcp` so they can reuse the same compilation cache.

### 4. The adjoint GMRES/CG matvec rebuilds the reverse-mode closure on every Krylov iteration
**Area:** efficiency  
**Refs:** `smooth_mcp/core.py:499-516`

`JTv` calls `jax.vjp(H_x, x_star)` every time the Krylov solver asks for a matrix-vector product. That recreates the VJP closure repeatedly even though `H_x` and `x_star` are fixed for the whole adjoint solve.

This is pure overhead in the most iteration-heavy part of the backward pass.

**Improvement:** compute `_, vjp_x = jax.vjp(H_x, x_star)` once, outside `JTv`, then let `JTv(v)` return `vjp_x(v)[0]`.

### 5. `smoothed_residual` is exported as public API but does not accept the same `F` signatures as the main solvers
**Area:** correctness, usability  
**Refs:** `smooth_mcp/__init__.py:3-20`, `smooth_mcp/core.py:130-160`, `smooth_mcp/core.py:318`, `smooth_mcp/core.py:458`, `README.md:187`

`solve_mcp` and `make_mcp_solver_diff` normalize both `F(x)` and `F(x, theta)` through `_normalize_F`. `smoothed_residual` does not. A user can import it from the package root and then immediately hit `TypeError` by passing the same single-argument `F(x)` that works everywhere else.

That is an API inconsistency on an exported symbol.

**Improvement:** normalize inside `smoothed_residual`, or stop exporting it and document it as an internal helper that expects a pre-normalized callable.

### 6. Continuation controls accept nonsensical values without validation
**Area:** correctness, usability  
**Refs:** `smooth_mcp/core.py:271-275`, `smooth_mcp/core.py:338-339`, `smooth_mcp/core.py:361-379`, `README.md:139-146`

The solver validates `mu_init > 0`, but not the rest of the continuation contract:

- `mu_decay >= 1` silently repeats or increases `mu`
- `mu_decay <= 0` jumps directly to `mu_min`
- `max_mu_steps <= 0` is not guarded
- negative `newton_tol` / `regularize` are not guarded

These are not harmless edge cases. They can quietly turn continuation off or make convergence reporting meaningless.

**Improvement:** validate admissible ranges up front and document them clearly: e.g. `0 < mu_decay < 1`, `max_mu_steps >= 1`, `newton_tol >= 0`, `regularize >= 0`.

### 7. `solve_mcp_diff` is treated like a user-facing helper in code, but not surfaced coherently as public API
**Area:** maintainability, usability, documentation  
**Refs:** `smooth_mcp/core.py:544-564`, `smooth_mcp/__init__.py:3-20`, `tests/test_gradients.py:312-371`, `README.md:179-187`

There is a convenience wrapper with a public-quality docstring, dedicated tests, and a warning about retracing cost. But it is not exported from `smooth_mcp.__init__` and it does not appear in the README API table.

That leaves the library with a shadow API: implemented, tested, and discoverable in code, but not committed to in the package interface.

**Improvement:** make a deliberate choice. Either export and document `solve_mcp_diff`, or remove it and keep only the factory API.

### 8. The README does not document the current tracing/composability limits
**Area:** documentation  
**Refs:** `README.md:3`, `README.md:45-53`, `README.md:112-131`, `README.md:177-187`

The README explains `jax.grad` support, but it does not say:

- that the differentiable path is currently eager-only
- that `jax.jit` composition fails today
- that `solve_mcp_diff` exists but is intentionally omitted from the public API
- that `smoothed_residual` expects a narrower callable contract than `solve_mcp`

Users will reasonably assume ordinary JAX composition semantics unless the docs say otherwise.

**Improvement:** add an explicit “JAX integration limits” note and make the public API boundary unambiguous.

### 9. Test coverage misses the highest-risk integration paths
**Area:** testing  
**Refs:** `tests/test_solver.py`, `tests/test_gradients.py`, `tests/test_convergence.py`, `smooth_mcp/core.py:415`, `smooth_mcp/core.py:493-538`

The current test suite is solid on basic solver behavior and finite-difference gradients, but it misses several risky paths:

- no `jax.jit` regression coverage
- no gradient test for a non-terminal continuation solve
- no coverage for `precond`
- no validation tests for invalid `mu_decay` / control ranges
- no direct test for the exported `smoothed_residual` behavior mismatch

Those are exactly the areas where the library currently has the sharpest edges.

**Improvement:** add focused regression tests before refactoring the implementation.

### 10. `smooth_mcp/core.py` is carrying too many responsibilities
**Area:** maintainability  
**Refs:** `smooth_mcp/core.py`

The same module owns:

- smoothing primitives
- residual construction
- dense and Krylov Newton solves
- continuation scheduling
- result packaging
- custom-VJP backward logic
- convenience wrapper behavior

That is manageable at the current size, but it is already encouraging cross-cutting bugs such as compile-boundary mistakes and API inconsistencies. Future work on JIT compatibility will be harder than necessary if everything stays coupled.

**Improvement:** once the missing regression tests are in place, split the module into focused pieces such as `smoothing.py`, `solver.py`, and `diff.py`.

## Overall Assessment

The library has a solid numerical core and much better test coverage than the earlier reviews, but the current weak point is JAX integration discipline. The most important work now is not adding more features; it is making the differentiable solve path correct for the actual forward system, JIT-safe, and cheaper to reuse repeatedly.
