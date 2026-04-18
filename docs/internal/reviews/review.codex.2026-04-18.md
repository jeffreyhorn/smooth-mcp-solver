# Code Review: smooth-mcp (2026-04-18)

Scope: current repository state on `review.codex.2026-04-18`, with focus on `smooth_mcp/`, tests, docs, demos, benchmarks, and project tooling.

Validation performed:

- Static inspection of package modules, tests, docs, demos, benchmarks, and project metadata.
- `.venv/bin/python -m pytest tests/ -q` -> `197 passed in 535.59s (0:08:55)`.
- `.venv/bin/python -m ruff check smooth_mcp tests demos` -> clean.
- `.venv/bin/python -m mypy smooth_mcp demos` -> clean.
- Targeted runtime probes for invalid `x0`, invalid solver hyperparameters, strict-validation behavior, and forward/diff factory contracts.

## Executive Summary

The numerical core is in good shape. The repository is test-clean, type-check clean, and materially better structured than the earlier internal reviews. The remaining issues are mostly around API safety, validation completeness, release engineering, and keeping the documented public contract fully backed by automated checks.

The two most important problems are:

1. invalid `x0` values are not actually validated, despite the shared validator implying that they are, and
2. a large part of the public solver configuration surface is accepted without validation, which lets obviously bad settings either silently run or fail later with raw JAX/internal exceptions.

Those are fixable without changing the mathematical core. After that, the main work is product hardening: safer traced defaults, better forward-factory test coverage, CI/version policy, and tighter module boundaries.

## Findings

### 1. NaN in `x0` is not rejected anywhere, despite the shared validator implying that it is

**Severity:** High  
**Areas:** correctness, usability, testability  
**Refs:** `smooth_mcp/_kernel.py:14-35`, `smooth_mcp/_kernel.py:38-60`, `smooth_mcp/solver.py:95-111`, `smooth_mcp/forward.py:197-220`, `smooth_mcp/diff.py:323-346`, `tests/test_strict_validation.py:35-60`, `docs/api.md:204-214`

`validate_bounds_and_x0` says it validates "NaN, and ordering of bounds and initial guess", but the implementation only checks `l` and `u`. `x0` is shape-checked, then passed through untouched.

Observed behavior from a direct probe:

- `preflight_validate(l, u, x0_nan)` accepted `x0 = [nan]`
- `solve_mcp(...)` returned `MCPResult(x=[nan], residual_norm=nan, num_steps=50, converged=False)`
- `make_mcp_solver(...)` returned `[nan]`
- `make_mcp_solver_diff(...)` returned `[nan]`

That is a real API-quality defect. A bad initial guess is not an exotic internal condition; it is ordinary user input. Letting it flow through as `NaN` output makes failures harder to diagnose and undermines the value of `preflight_validate`.

**Recommendation:** treat `x0` NaN as invalid in the shared validation layer, propagate that through eager and traced validation modes, and add regression tests for all three entry points.

### 2. Public solver hyperparameters are only partially validated

**Severity:** High  
**Areas:** code quality, maintainability, usability  
**Refs:** `smooth_mcp/solver.py:96-127`, `smooth_mcp/forward.py:100-139`, `smooth_mcp/diff.py:142-185`, `smooth_mcp/_kernel.py:124-211`, `docs/api.md:164-182`

The code validates a few top-level continuation knobs (`mu_*`, `max_mu_steps`, `newton_tol`, `regularize`, `adjoint_method`, `strict_validation`) but leaves many public parameters unchecked:

- line search: `armijo_c`, `backtrack_rho`, `max_ls_steps`
- forward GMRES: `krylov_tol`, `krylov_maxiter`, `krylov_restart`
- adjoint solvers: `gmres_tol`, `gmres_restart`, `gmres_maxiter`, `cg_tol`, `cg_maxiter`

Observed behavior from runtime probes:

- `solve_mcp(..., armijo_c=-1.0)` was accepted and reported convergence
- `solve_mcp(..., backtrack_rho=1.5)` was accepted and reported convergence
- `solve_mcp(..., max_ls_steps=-3)` was accepted and reported convergence
- `solve_mcp(..., linear_solver="gmres", krylov_tol=-1.0)` was accepted
- `solve_mcp(..., linear_solver="gmres", krylov_restart=0)` failed later as raw `IndexError`
- `make_mcp_solver_diff(..., gmres_restart=0)` constructed successfully; the bad setting only surfaced when gradients were taken

This is exactly the wrong failure mode for a numerical library. Invalid public API values should fail immediately with clear `ValueError`s at the boundary, not drift into JAX internals or silently behave in undefined ways.

**Recommendation:** centralize option validation in one shared helper and reject invalid line-search, forward-Krylov, and adjoint-Krylov settings before solver construction or execution.

### 3. The default traced-validation path is still silently unsafe, and that behavior is now part of the contract

**Severity:** Medium-High  
**Areas:** correctness, usability, documentation  
**Refs:** `smooth_mcp/_kernel.py:29-35`, `smooth_mcp/forward.py:71-90`, `smooth_mcp/diff.py:104-128`, `docs/api.md:211-251`, `tests/test_strict_validation.py:104-117`

The repository now has opt-in safety mechanisms (`preflight_validate`, `strict_validation=True`, `strict_validation="checkify"`), which is good. The problem is that the default behavior under `jax.jit` / `jax.grad` / `jax.vmap` is still "silently skip value checks", and the test suite explicitly locks that in.

`tests/test_strict_validation.py::TestDefaultMode.test_jit_invalid_silently_slips_through` documents and preserves the behavior that invalid bounds under tracing can return a finite result instead of failing.

That is understandable as a compatibility choice, but it remains a product risk. The safest APIs are still opt-in, while the default API path is the unsafe one.

**Recommendation:** make a deliberate product decision here instead of leaving the unsafe default in place indefinitely. Either:

- switch the factory APIs to a safe-by-default traced mode, or
- require an explicit opt-out for the unsafe fast path, or
- at minimum surface a much stronger warning and a migration path.

### 4. The forward-only factory's documented traced-validation contract is under-tested

**Severity:** Medium  
**Areas:** testability, maintainability  
**Refs:** `docs/api.md:234-277`, `tests/test_forward_factory.py:278-353`, `tests/test_strict_validation.py:68-369`

The docs present `make_mcp_solver` and `make_mcp_solver_diff` as parallel APIs for `strict_validation=True` and `strict_validation="checkify"`. In practice, only the differentiable path has substantial strict-validation coverage.

Current test shape:

- `tests/test_strict_validation.py` exercises the differentiable factory under eager, `jit`, `vmap`, `grad`, and `checkify`
- `tests/test_forward_factory.py` validates construction-time args and eager runtime shape/bounds checks, but does not exercise the forward factory's traced `strict_validation` modes

I verified by direct probe that forward-factory strict validation does currently work. The issue is the lack of regression coverage for a documented public contract.

**Recommendation:** add forward-factory tests for:

- `strict_validation=True` under eager, `jit`, and `vmap`
- `strict_validation="checkify"` under eager and `jit`
- poisoned/aux behavior with `return_aux=True`
- invalid `x0` once finding 1 is fixed

### 5. The repository has no automated compatibility gate for JAX version drift

**Severity:** Medium  
**Areas:** maintainability, release engineering, documentation  
**Refs:** `pyproject.toml:11-16`, `smooth_mcp/forward.py:10`, `smooth_mcp/diff.py:8`

This package depends directly on `jax`, `jaxlib`, `jax.experimental.checkify`, and JAX sparse linear solvers, but the project metadata and automation story are thin:

- `pyproject.toml` has unbounded `jax` / `jaxlib` dependencies
- dev dependencies are also unbounded
- there is no CI workflow in the repository

A local green run on April 18, 2026 is useful, but it is not a compatibility policy. For a numerical library sitting on top of evolving JAX APIs, especially experimental ones, this is a real maintenance risk.

**Recommendation:** define a supported Python/JAX version matrix, add CI for tests/lint/mypy, and make the packaging metadata reflect what is actually supported and exercised.

### 6. Module boundaries are weaker than the public API suggests

**Severity:** Low-Medium  
**Areas:** architecture, maintainability  
**Refs:** `smooth_mcp/forward.py:20`, `smooth_mcp/diff.py:22-35`

The package-level layout is better than it used to be, but there is still avoidable coupling across API layers. The forward-only module imports `SolveInfo` from `diff.py`, which makes the non-differentiable path depend on the differentiable implementation module for a shared public type.

That is not broken today. It is still the wrong direction of dependency. Shared public result types should live in a neutral module, not inside one API implementation and then get imported outward.

**Recommendation:** move `SolveInfo` (and possibly `MCPResult`) into a small shared `_types.py` / `types.py` module and keep `forward.py` and `diff.py` as peers.

### 7. User-facing examples are not protected by smoke automation, and `bench_solve.py` is not import-safe

**Severity:** Low-Medium  
**Areas:** documentation, tooling, testability  
**Refs:** `benchmarks/bench_solve.py:49-115`

The demos and benchmarks are part of the user-facing surface, but they are not protected the way the core library is:

- there is no demo/benchmark smoke layer in the automated tests
- `benchmarks/bench_solve.py` executes immediately on import instead of using `if __name__ == "__main__":`

That makes the examples easier to rot than the library code, and the benchmark script harder to reuse programmatically.

**Recommendation:** add a small smoke layer for demos/examples, make benchmark scripts import-safe, and decide whether benchmark files are meant to be scripts only or reusable helpers.

## Strengths

- The solver core, smoothing layer, and implicit-diff path are in strong shape.
- The test suite is already substantial for a library of this size.
- Public docs are much better than in earlier reviews.
- The project structure is coherent enough that the remaining issues are mostly hardening work, not a rewrite.

## Overall Assessment

The codebase is technically credible and materially improved over the earlier review cycle. The next round of work should not be about changing the mathematics. It should be about tightening the public contract: reject bad inputs early, validate the whole configuration surface, back the documented factory behavior with tests, and add enough automation that compatibility and example health do not rely on one machine and one date.
