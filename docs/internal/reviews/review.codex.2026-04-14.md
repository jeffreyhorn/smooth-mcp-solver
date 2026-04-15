# Code Review: smooth-mcp (2026-04-14)

Scope: `smooth_mcp/`, `README.md`, `demos/`, `tests/`, build tooling, and repository docs relevant to the `smooth_mcp` Python library.

Validation performed:

- Static inspection of the package modules, tests, demos, README, benchmark harness, and project metadata.
- `.venv/bin/python -m pytest tests/ -q` -> `110 passed in 279.89s`.
- `.venv/bin/python -m ruff check smooth_mcp tests demos` -> clean.
- `.venv/bin/python -m mypy smooth_mcp demos` -> clean.
- Runtime probes for `jax.jit`, `jax.vmap`, continuation step counts, benchmark behavior, and invalid-input handling.

## Executive Summary

Overall, the numerical core is in good shape. The project is small, focused, well-tested, and materially better structured than the earlier single-file versions. Internal docstrings are solid, the demos cover a useful range of MCP use cases, and the differentiable path now composes with `jax.jit` and `jax.vmap` in the scenarios I checked.

The current weak points are no longer fundamental solver correctness. They are productization issues around API symmetry, eager-path efficiency, user-facing documentation, and how much diagnostic information the differentiable API exposes. In other words: the library is technically credible, but still sharper at the edges than it should be for outside users.

## Area Ratings

| Area | Assessment |
|---|---|
| Correctness | Good |
| Efficiency | Good under JIT, Fair in eager mode |
| Maintainability | Good |
| Usability | Fair |
| Internal documentation | Good |
| External documentation | Fair |
| Structure | Good, but internal boundaries remain weak |

## Findings

### 1. `make_mcp_solver_diff` does not provide the same runtime input validation guarantees as `solve_mcp`

**Severity:** High  
**Area:** correctness, usability  
**Refs:** `smooth_mcp/solver.py:279-316`, `smooth_mcp/diff.py:86-146`, `tests/test_convergence.py:233-385`

`solve_mcp` validates shape compatibility, rejects NaN bounds, and rejects invalid bounds with `l > u`. The differentiable factory validates only construction-time hyperparameters. Once the returned solver is called, it runs directly through the pure-JAX path without matching input validation.

That difference is observable today:

- In a runtime probe, `make_mcp_solver_diff(F)(jnp.array([2.0]), jnp.array([1.0]), jnp.array([0.5]), jnp.array([2.0]))` returned `array([1.])`.
- The equivalent `solve_mcp(...)` call raised `ValueError: Lower bounds must not exceed upper bounds (l <= u)`.
- A mismatched-shape diff-solver probe failed with the raw JAX message `TypeError: sub got incompatible shapes for broadcasting: (3,), (2,)`, while `solve_mcp` raises a clear `ValueError`.

This is the sharpest current user-facing issue because it creates two different safety contracts for what appear to be parallel public APIs.

**Recommendation:** introduce a shared validation layer and either:

1. expose an eager checked wrapper around the differentiable solver, or
2. add `validate_inputs=True` behavior for non-jitted use while keeping the core traceable path intact.

Also add explicit regression tests for invalid runtime inputs on the differentiable path.

### 2. The default continuation schedule is robust but too conservative for eager use, and the docs do not help users tune it

**Severity:** Medium-High  
**Area:** efficiency, usability  
**Refs:** `smooth_mcp/solver.py:171-223`, `README.md:168-212`, `benchmarks/bench_solve.py:48-90`

The continuation kernel always uses a fixed geometric decay schedule. There is no adaptive schedule, no fast preset, and no documentation that helps users trade robustness for speed.

Evidence from local probes:

- The included benchmark measured `solve_mcp(F_lcp)` at about `1159.7 ms/call` for a 2D LCP on this machine.
- A simple 1D problem (`F(x) = 2x - 1`) took `37` outer continuation steps with the default `mu_decay=0.5`.
- The same problem reached essentially the same residual order with `19` steps at `mu_decay=0.25` and `11` steps at `mu_decay=0.1`.

This does not mean `0.1` should become the new universal default. It does mean the current defaults are conservative enough to have a real cost, and the library does not yet give users much guidance about that tradeoff.

**Recommendation:** benchmark a few schedule strategies, document tuning guidance, and consider adding an adaptive or "fast-but-less-conservative" preset for common small/medium problems.

### 3. External documentation is still centered on a single README, and parts of it have already drifted

**Severity:** Medium  
**Area:** documentation, usability  
**Refs:** `README.md:22`, `README.md:57-63`, `README.md:144-166`

The README is reasonably strong on solver mechanics, but it is still doing almost all user-facing documentation work by itself. That would be acceptable if it were very tight, but it already shows drift:

- `README.md:22` links to `smooth_mcp/core.py`, which no longer exists.
- The primary install command is `pip install -e .`, which is a contributor workflow rather than the clearest consumer-facing install path.
- The repository `docs/` tree currently contains benchmark notes, reviews, and todo files, but no actual user guide, API reference, or troubleshooting docs for library users.

The practical effect is that users must infer behavior from README prose plus demos, while the `docs/` directory mostly serves internal project management.

**Recommendation:** fix the broken README references immediately, then split user documentation from internal project notes. At minimum, add:

- an installation guide with JAX environment guidance,
- an API reference page,
- a "solver tuning and diagnostics" page,
- a troubleshooting page for common tracing / convergence / bounds issues.

### 4. The differentiable API is difficult to inspect in production use because it returns only `x`

**Severity:** Medium  
**Area:** usability, maintainability  
**Refs:** `smooth_mcp/diff.py:108-146`, `README.md:214-222`

Internally, `_run_forward` computes `(x_star, mu_used, num_steps)`, but the public differentiable API exposes only `x_star`. That makes several practical tasks harder than they need to be:

- logging convergence during training loops,
- telling whether a solve stopped early under truncated continuation,
- debugging bad gradients without rerunning `solve_mcp`,
- comparing eager and differentiable solver behavior at the same settings.

This is not a math bug, but it is an observability problem. Libraries that solve hard numerical problems usually need better diagnostics than "here is the primal variable."

**Recommendation:** add a companion API that exposes auxiliary info, for example:

- `make_mcp_solver_diff_with_aux(...)`,
- `return_info=True`,
- or a separate eager diagnostic wrapper that shares the same forward kernel.

The important part is to let users recover `mu_used`, `num_steps`, and ideally a final residual estimate without sacrificing autodiff support.

### 5. The package split is much better than the old monolith, but the internal boundaries are still weak

**Severity:** Medium-Low  
**Area:** structure, maintainability  
**Refs:** `smooth_mcp/diff.py:10-15`

The current three-module split is a real improvement:

- `smoothing.py` for smooth primitives,
- `solver.py` for the eager solve,
- `diff.py` for the custom-VJP path.

However, `diff.py` still imports `_make_continuation_solver`, `_make_newton_solver`, and `_normalize_F` from `solver.py`. That means the current structure is partly presentational; the internal contract is still a set of private cross-module dependencies.

This is workable today because the codebase is small. It will become a maintenance tax if the solver kernel, validation layer, or API surface keeps evolving.

**Recommendation:** extract shared internals into a dedicated private module such as `smooth_mcp/_kernel.py` or `smooth_mcp/_internal.py`, then keep `solver.py` and `diff.py` as thinner API-specific wrappers.

### 6. Performance messaging is partly hard-coded and more confident than it should be

**Severity:** Low  
**Area:** documentation, maintainability  
**Refs:** `README.md:155-157`, `benchmarks/bench_solve.py:87-90`, `docs/internal/benchmarks/benchmark-2026-04-13.md:7-18`

The performance story is directionally correct. On this machine, cached `jax.jit(jax.grad(...))` calls were about `1.1 ms`, versus about `2469.5 ms` for eager `jax.grad`. That is excellent.

The problem is how the result is communicated:

- the README says `~1000x speedup`,
- the benchmark script prints `~1000x speedup`,
- the stored benchmark note says `~2000x faster`.

Those are environment-specific observations, not library invariants. Hard-coding them in executable messaging guarantees drift over time.

**Recommendation:** keep the benchmark harness factual and dated. Phrase performance claims as "example results on machine X with config Y" rather than fixed universal claims.

### 7. Minor tooling polish: `Makefile` advertises `typecheck` but does not mark it phony

**Severity:** Low  
**Area:** maintainability, tooling  
**Refs:** `Makefile:1`, `Makefile:40-43`

This is small, but it is the kind of rough edge that accumulates. The help text advertises `typecheck`, but `.PHONY` omits it.

**Recommendation:** add `typecheck` to `.PHONY` and keep the help text and target declarations in sync.

## Strengths

- The numerical core is materially more mature than the earlier reviews suggest. I did not find a current correctness break in the forward or backward math.
- Test coverage is strong for a small numerical library. It covers solver behavior, smoothing primitives, convergence edge cases, gradients, truncated continuation, GMRES/CG choices, preconditioners, and JIT composability.
- Internal documentation is good. Public functions and major internal helpers have useful docstrings and reasonable parameter descriptions.
- The demo set is broad enough to teach the library: LCPs, nonlinear MCPs, KKT conditions, bounds optimization, and a higher-dimensional obstacle problem.
- JIT-compiled gradient performance is legitimately strong.

## Overall Assessment

The project is in a credible state for experimental and research use. The highest-priority work now is not reworking the math; it is tightening the user contract around the differentiable API and improving the documentation/product surface so users get the same safety, diagnostics, and guidance regardless of which public entry point they choose.
