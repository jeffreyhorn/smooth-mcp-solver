# Codex Review

Date: 2026-04-23

## Scope

Reviewed the full repository with special attention to documentation, testing, maintainability, efficiency, and correctness.

Files and areas inspected:
- `smooth_mcp/`
- `tests/`
- `docs/`
- `benchmarks/`
- `demos/`
- `README.md`, `pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`

Checks run during review:
- `ruff check smooth_mcp tests demos` -> passed
- `mypy smooth_mcp demos` -> passed
- `black --check smooth_mcp tests demos` -> passed with a Python-3.14 grammar warning under Python 3.12
- `pytest` -> unit/integration portions passed while review was in progress; the long-running demo smoke tail was still executing when this document was finalized

## Findings

### 1. High: The public API accepts arbitrary same-shaped arrays, but the solver only works for 1D vectors

Refs:
- `docs/api.md:33-39`
- `smooth_mcp/_kernel.py:14-38`
- `smooth_mcp/_kernel.py:256-277`
- `smooth_mcp/_kernel.py:295-317`

The public contract only requires `l`, `u`, and `x0` to have the same shape. The implementation never enforces `ndim == 1`, but the Newton and GMRES kernels assume vector state.

Concrete reproduction from this review:

```text
solve_mcp ValueError jvp called with different primal and tangent shapes;Got primal shape (2, 1) and tangent shape as (2, 1, 2, 1)
make_mcp_solver ValueError jvp called with different primal and tangent shapes;Got primal shape (2, 1) and tangent shape as (2, 1, 2, 1)
```

Impact:
- Users get an opaque JAX shape error instead of an API-level validation error.
- Documentation implies broader shape support than the solver actually has.
- The current test suite has no regression coverage for this contract edge.

Recommendation:
- Either validate that solver state is 1D at the public boundary, or implement explicit flatten/unflatten support throughout the Newton and adjoint paths.

### 2. High: `mu_used` is misdocumented, and the backward pass can differentiate at `mu_min` instead of the last solved `mu`

Refs:
- `smooth_mcp/_kernel.py:360-367`
- `smooth_mcp/_types.py:37-42`
- `smooth_mcp/diff.py:56-61`
- `smooth_mcp/diff.py:194-203`
- `smooth_mcp/diff.py:269-282`
- `docs/api.md:148-153`
- `README.md:55`
- `tests/test_forward_factory.py:259-270`

`SolveInfo.mu_used` is documented as the "terminal smoothing parameter actually reached". That is not what the continuation kernel returns on convergence: it clamps `mu_used` to `mu_min` whenever the residual-at-`mu_min` test passes.

Concrete reproduction from this review:

```text
x [1.99611637]
num_steps 6
mu_used 1e-10
residual 0.0038836262708028535
```

This run used `mu_init=1.0`, `mu_decay=0.5`, `mu_min=1e-10`, `newton_tol=1e-1`. After 6 continuation steps, the last solved `mu` is `0.03125`, not `1e-10`, but `mu_used` still reports `1e-10`.

Because `make_mcp_solver_diff` uses that returned `mu_final` in `_compute_grads`, the README/docstring claim that gradients are taken at the "actual terminal smoothing parameter from the forward solve" is not always true.

Impact:
- `SolveInfo.mu_used` is semantically misleading.
- The differentiable solver's documented gradient semantics are stronger than the implementation.
- Tests currently lock in the clamped behavior instead of clarifying the distinction.

Recommendation:
- Separate "last solved mu" from "evaluation mu".
- Use the true last-solved `mu` in the backward pass if that is the intended contract.
- Otherwise rename and document the current field/behavior precisely.

### 3. Medium: The Armijo implementation accepts a step after budget exhaustion without rechecking acceptance

Refs:
- `smooth_mcp/_kernel.py:299-313`
- `README.md:41`

The line search loop stops when either the Armijo condition is satisfied or `ls_it == max_ls_steps`. The code then always applies `x + alpha_final * d`; it never rejects the step if the budget was exhausted while the sufficient-decrease condition was still false.

Concrete reproduction from this review with `max_ls_steps=0`:

```text
x0 [0.] phi0 0.46945887601128483
x1 [0.95080286] phi1 0.6695238233013745
phi increased True
```

That means the docs currently overstate the guarantee when they say the line search "ensures global convergence".

Impact:
- Users can silently disable sufficient-decrease enforcement.
- Budget exhaustion can accept a non-Armijo step and increase the merit function.
- There is no explicit failure flag for this condition.

Recommendation:
- Recheck Armijo after the loop and reject/fail if the condition is still unmet.
- Document `max_ls_steps=0` as "line search disabled" if the current behavior is kept.
- Add tests around monotonic merit decrease and line-search failure semantics.

### 4. Medium: The API docs have drifted from the code for `strict_validation`

Refs:
- `docs/api.md:76-79`
- `docs/api.md:113-127`
- `smooth_mcp/forward.py:25-42`
- `smooth_mcp/diff.py:25-50`

Two documentation issues are visible in the main API reference:
- `make_mcp_solver` still lists `strict_validation` defaulting to `False`, but the code defaults it to `True`.
- The `make_mcp_solver_diff` argument table omits `strict_validation` entirely even though it is a public parameter.

Impact:
- The public safety default is misstated in the main parameter table.
- Readers comparing the two factories do not get a complete signature from the docs.
- This is documentation drift on a recently changed user-facing contract.

Recommendation:
- Update the tables immediately.
- Longer term, generate or centralize shared option tables so forward/diff docs cannot drift independently.

### 5. Medium: The float64 contract is unclear, unenforced, and effectively untested

Refs:
- `README.md:69-76`
- `docs/installation.md:47-56`
- `tests/conftest.py:1-3`

The docs say the solver "requires float64 precision". The implementation does not enforce this, emit a warning, or expose a supported float32 policy. Meanwhile the tests globally enable x64 in `tests/conftest.py`, so the suite never exercises the default user configuration.

Impact:
- Users are told float64 is mandatory, but the package does nothing to protect them if they forget it.
- The test suite masks the real user-default environment.
- The project currently cannot tell users whether float32 is unsupported, partially supported, or merely lower accuracy.

Recommendation:
- Decide the real contract.
- If float64 is required, validate or warn at runtime.
- If float32 is only discouraged, soften the docs and describe the tradeoff.
- Add explicit float32 behavior tests either way.

### 6. Medium: Test coverage is broad on happy-path parity, but shallow on contract edges and failure modes

Refs:
- `tests/test_demos.py:23-82`
- `tests/test_forward_factory.py:317-360`
- `Makefile:56-59`

The core parity and gradient tests are good, but several user-facing edges are not covered:
- no regression test for multidimensional input rejection/support
- no test for line-search budget exhaustion semantics
- no test of `precond`
- no float32 coverage
- demo tests are smoke-only and do not assert the numerical claims shown in docs/examples

There is also a tooling mismatch: `Makefile` says `test-fast` skips "slow gradient tests", but the target actually ignores all of `tests/test_gradients.py`.

Impact:
- Public-contract bugs can slip through while parity tests remain green.
- Documentation claims in demos and examples are not actually verified.
- The developer workflow wording in `Makefile` does not match behavior.

Recommendation:
- Add targeted contract tests for the missing edges.
- Tighten demo assertions or add a smaller set of numeric example tests.
- Align the `Makefile` comment and behavior.

### 7. Low: `forward.py` and `diff.py` duplicate a large amount of plumbing

Refs:
- `smooth_mcp/forward.py:128-228`
- `smooth_mcp/diff.py:163-340`

The forward and differentiable factories duplicate substantial logic:
- forward solve construction
- aux construction
- NaN-poisoning wrappers
- checkify wrappers
- validation entry points

Impact:
- Fixes can drift between the two paths.
- Shared semantics become harder to keep consistent.
- This duplication likely contributed to documentation drift and semantic confusion around shared options.

Recommendation:
- Extract the shared wrapper/plumbing into a single internal helper and leave only VJP-specific logic in `diff.py`.

## Summary

The project’s core solver is compact, and the existing parity/gradient test coverage is better than average for a small numerical package. The main remaining risks are not basic syntax or style problems; they are contract problems at the boundary:
- the solver shape contract is under-specified and currently fails badly off the happy path
- the documented `mu_used` / gradient semantics do not match the continuation kernel
- the line-search behavior is weaker than the docs claim
- the docs and tests overstate certainty in areas that are not actually enforced or covered

The remediation plan in `docs/internal/todos/todo.2026-04-23.md` is ordered to fix correctness and contract clarity first, then strengthen tests and reduce duplication.
