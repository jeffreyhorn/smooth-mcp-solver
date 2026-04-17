# Code Review: smooth-mcp (2026-04-16)

Scope: current repository state on `review-codex.2026-04-16`, with focus on `smooth_mcp/`, tests, demos, benchmarks, and user-facing documentation.

Validation performed:

- Static review of package modules, docs, demos, benchmarks, and tests.
- `.venv/bin/python -m pytest tests/ -q` -> `123 passed in 254.88s`.
- `.venv/bin/python -m ruff check smooth_mcp tests demos` -> clean.
- `.venv/bin/python -m mypy smooth_mcp demos` -> clean.
- Runtime probes for repeated eager solves, `jax.jit`, `jax.vmap`, `return_aux`, invalid traced inputs, and benchmark behavior.
- `.venv/bin/python benchmarks/bench_solve.py`
- `.venv/bin/python benchmarks/bench_continuation.py`

## Findings

### 1. The differentiable API still silently accepts invalid value-based inputs under JAX tracing

**Severity:** High  
**Areas:** correctness, usability  
**Refs:** `smooth_mcp/_kernel.py:14-35`, `smooth_mcp/diff.py:262-270`, `README.md:178-185`, `docs/api.md:151-156`

The eager validation story is much better than it was in the previous review. However, the core limitation remains: value-based checks are skipped whenever the arguments are tracers. That means invalid bounds can flow through `jax.jit`, `jax.grad`, or `jax.vmap` without any hard failure.

That is not just theoretical. In a probe, this jitted call:

```python
jax.jit(lambda l, u, x0, theta: solver(l, u, x0, theta))(
    jnp.array([5.0]), jnp.array([3.0]), jnp.array([4.0]), jnp.array([2.0])
)
```

returned `[3.]` instead of rejecting `l > u`.

The current docs do explain that traced value checks are skipped, so this is not a documentation bug. It is still a product risk because the library’s headline differentiable path is exactly the one people will use inside traced code.

**Recommendation:** add a stricter optional validation mode for traced workflows, or provide a documented `checkify`/preflight wrapper that users can place around production training and batched inference code. At minimum, promote this limitation from a note in the docs to a prominent warning with examples of safe usage patterns.

### 2. `solve_mcp` remains expensive for repeated forward-only use and still lacks a reusable compiled/factory API

**Severity:** Medium-High  
**Areas:** efficiency, usability  
**Refs:** `smooth_mcp/solver.py:85-175`, `smooth_mcp/_kernel.py:78-225`, `benchmarks/bench_solve.py:44-89`

The differentiable path has a reusable factory. The non-differentiable path still does not. Every `solve_mcp(...)` call rebuilds the normalized function, Newton solver, and continuation solver around the current problem data before running the solve.

Observed behavior on this machine:

- repeated `solve_mcp` on the same 2D LCP: first call about `5.0 s`, subsequent calls about `1.11 s`
- benchmark average: about `1107.6 ms/call`

The first-call cost is partly JAX compilation, but the broader issue is API shape: users who want repeated forward solves with diagnostics have no reusable "compiled forward solver" entry point analogous to `make_mcp_solver_diff`.

**Recommendation:** add a forward-only solver factory such as `make_mcp_solver(...)` or a lower-level compiled kernel wrapper that can be reused across repeated solves while still exposing eager diagnostics when needed.

### 3. The tuning documentation overstates the usefulness of smaller `mu_decay` values by focusing on step counts more than wall-clock cost

**Severity:** Medium  
**Areas:** documentation, efficiency, usability  
**Refs:** `README.md:234-258`, `docs/tuning.md:7-38`, `benchmarks/bench_continuation.py:78-117`

The new tuning documentation is a good addition, but the current framing still implies that smaller `mu_decay` is generally a speed optimization when continuation overhead matters.

The benchmark matrix shows that this is not reliably true:

- `ncp_2d`: `mu_decay=0.5` took `34` steps and about `1297 ms/call`, while `0.1` took only `11` steps but about `1948 ms/call`
- `spatial_eq`: `0.5` took `32` steps and about `1604 ms/call`, while `0.1` took `10` steps but about `2515 ms/call`
- only some problems, such as `obstacle_50d`, benefited clearly from more aggressive decay

So the real tuning story is: smaller `mu_decay` often reduces step count, but may increase per-step difficulty enough to make total runtime worse.

**Recommendation:** revise the tuning guide to separate "fewer continuation steps" from "faster runtime", and anchor the guidance to measured benchmark behavior rather than step-count intuition alone.

### 4. External documentation is now broader, but it is duplicated heavily between `README.md` and `docs/`, which creates maintenance drift risk

**Severity:** Medium  
**Areas:** external documentation, maintainability  
**Refs:** `README.md:57-267`, `docs/installation.md`, `docs/api.md`, `docs/tuning.md`, `docs/troubleshooting.md`, `Makefile:23-29`

This is a meaningful improvement over the previous state: there are now actual user docs for installation, API, tuning, and troubleshooting. The problem is that large parts of those docs are also repeated directly in the README.

That duplication already creates tension:

- `README.md` now contains installation, API, input validation, and tuning guidance in long form
- the same material exists again in `docs/installation.md`, `docs/api.md`, `docs/tuning.md`, and `docs/troubleshooting.md`
- `Makefile` still uses `pip install -e .` for `install`, while the user docs consistently present `pip install .` as the standard install path

The current content is mostly consistent, but the maintenance burden is now obvious: small behavior changes will have to be updated in several places.

**Recommendation:** decide what the README should do. The cleanest split is:

- README: overview, quickstart, feature summary, links
- `docs/`: installation, API, tuning, troubleshooting, performance notes

Then remove the duplicated long-form material from the README.

### 5. `make_mcp_solver_diff` has duplicated `custom_vjp` branches for `return_aux=False` and `return_aux=True`

**Severity:** Low-Medium  
**Areas:** maintainability, structure  
**Refs:** `smooth_mcp/diff.py:226-272`

The new `return_aux` feature is valuable, but it currently doubles the most delicate part of the implementation. The function defines two versions of `solve`, two forward rules, and two backward rules that are almost identical except for aux packaging.

That increases maintenance risk in exactly the code where subtle mistakes are expensive:

- adjoint behavior
- auxiliary diagnostics
- future API changes
- any future support for more metadata

This is not broken today, but it is brittle.

**Recommendation:** refactor the `custom_vjp` setup so there is one canonical forward/backward implementation and the aux packaging is handled in a thinner layer.

### 6. `jax.vmap` support is advertised but not covered by the automated test suite

**Severity:** Low  
**Areas:** testing, usability  
**Refs:** `README.md:170-173`, `docs/api.md:27-34`, `tests/test_gradients.py`, `tests/test_convergence.py`

The library docs explicitly advertise `jax.vmap` compatibility. In manual probes, it worked both for the plain differentiable solver and for `return_aux=True`.

However, there is no explicit `vmap` regression coverage in the test suite. For a JAX library, that is a real gap because batching is one of the first transforms users try after `jit` and `grad`.

**Recommendation:** add focused tests for:

- `jax.vmap` on `make_mcp_solver_diff`
- `jax.vmap` with `return_aux=True`
- batched gradients or batched aux output shapes

## Area Assessment

### Efficiency

Strong under JIT. Cached `jax.jit(jax.grad(...))` runs were about `1.4 ms/call` in the local benchmark, versus about `2240.7 ms/call` for eager `jax.grad`.

Fair in eager mode. Repeated forward solves are still expensive, and the continuation schedule tradeoffs are more nuanced than the current docs suggest.

### Maintainability

Good overall. The `_kernel.py` extraction is a real structural improvement, and the codebase is easier to reason about than the earlier monolithic layout.

Remaining weak points are mostly local:

- duplicated `custom_vjp` branches in `diff.py`
- duplicated long-form user documentation

### Usability

Much improved. There are clearer docs, runtime validation for the differentiable wrapper in eager mode, and optional auxiliary diagnostics via `SolveInfo`.

The main remaining usability issue is the traced-validation gap: the most production-oriented execution mode can still accept invalid values silently.

### Internal Documentation

Good. Public APIs and core internals are documented clearly enough for a small numerical library. The code is now teachable from the source alone.

### External Documentation

Good breadth, but only fair maintainability. The new docs are helpful; the issue is not missing content, it is duplicated content and some tuning guidance that is more certain than the benchmark data supports.

### Structure

Good. The separation into `smoothing.py`, `_kernel.py`, `solver.py`, and `diff.py` is appropriate for the current size of the package and is materially better than earlier versions.

The remaining structural problem is local complexity in `diff.py`, not the overall package layout.

## Strengths

- Core solver and gradient behavior are in solid shape.
- The test suite is stronger than most libraries of this size.
- The package structure is now coherent.
- User-facing docs exist and are useful.
- `return_aux` is a real improvement for diagnostics.
- JIT-compiled gradient performance is excellent.

## Overall Assessment

The project is now in a credible, technically mature state for research and advanced application use. The biggest remaining issues are not about the mathematical core; they are about the safety and ergonomics of the traced API, the lack of a reusable forward-only solver interface, and documentation that still needs one more pass from "complete" to "tight and durable."

---

## Resolution Status (2026-04-17)

All six findings have been addressed. Implementation tracked in
`docs/internal/todos/todo5.md`.

| # | Finding | Resolution |
|---|---------|------------|
| 1 | Traced validation gap | **Resolved (phase 1).** Three opt-in mechanisms: `preflight_validate`, `strict_validation=True` (NaN-poisoning), `strict_validation="checkify"`. All compose with `jit`, `grad`, `vmap`. 32 tests in `test_strict_validation.py`. Documented in `docs/api.md` §Input validation. |
| 2 | No reusable forward-only solver | **Resolved (phase 2).** `make_mcp_solver` factory added in `smooth_mcp/forward.py`. JIT-compiled forward solves ~970x faster than eager `solve_mcp`. 28 parity tests in `test_forward_factory.py`. |
| 3 | Tuning docs overstate `mu_decay` speed benefit | **Resolved (phase 3).** Benchmark matrix expanded to 6 problems including GMRES. Tuning docs rewritten with evidence-basis annotations, measured numbers, and profiling recipe. "Fewer steps does not mean faster" is now the lead message. |
| 4 | Documentation duplication between README and docs/ | **Resolved (phase 4).** README trimmed from 359 to 189 lines (overview + quickstarts). All reference material has one canonical home in `docs/`. Makefile install target aligned with documented behavior. Final consistency pass confirmed single source of truth per topic. |
| 5 | Duplicated `custom_vjp` branches for `return_aux` | **Resolved (phase 5).** Refactored to one `_core_solve` with `@custom_vjp`, one `_core_fwd`, one `_core_bwd`. Aux packaging happens outside the custom_vjp boundary. |
| 6 | No `vmap` test coverage | **Resolved (phase 6).** 14 tests in `test_vmap.py`: forward vmap (5), vmap with return_aux (5), batched gradients (4). Only constraint: checkify/vmap ordering (already documented). |

**Final state:** 197 tests pass, ruff clean, mypy clean, both benchmarks run successfully.
