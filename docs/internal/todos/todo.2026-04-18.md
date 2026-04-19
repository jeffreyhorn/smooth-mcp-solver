# Remediation Plan (2026-04-18)

This task list addresses every shortcoming identified in `docs/internal/reviews/review.codex.2026-04-18.md`.

## P0

### 1. Reject `x0` NaN in every public entry point and validation mode

**Why:** current validation treats `l`/`u` as checked inputs but lets `x0` NaN pass through into solver output as `NaN`.

**Work:**

- Update `smooth_mcp/_kernel.py::validate_bounds_and_x0` to reject NaN in `x0`.
- Update `traced_invalid_mask` and any strict-validation helpers if `x0` should also poison/checkify under traced execution.
- Ensure `preflight_validate`, `solve_mcp`, `make_mcp_solver`, and `make_mcp_solver_diff` all share the same rule.
- Update `docs/api.md`, `docs/troubleshooting.md`, and any relevant README wording so the validation contract matches the implementation.
- Add tests for:
  - `preflight_validate(..., x0_nan)` raising
  - `solve_mcp(..., x0_nan)` raising
  - `make_mcp_solver(..., x0_nan)` raising eagerly
  - `make_mcp_solver_diff(..., x0_nan)` raising eagerly
  - traced strict modes handling `x0_nan` correctly

**Acceptance criteria:**

- No public solver path silently accepts `x0 = NaN` in eager mode.
- Traced behavior is explicit and tested for both strict modes.
- The docs no longer imply checks that the code does not perform.

**Done (2026-04-18).**

**Code changes:**
- `smooth_mcp/_kernel.py::validate_bounds_and_x0`: added NaN check on
  `x0` with message `"x0 must not contain NaN"` (eager only, same
  tracing convention as the bounds checks).
- `smooth_mcp/_kernel.py::traced_invalid_mask`: signature changed from
  `(l, u)` to `(l, u, x0)`; now also flags NaN in `x0`.
- `smooth_mcp/_kernel.py`: renamed `sanitize_bounds(l, u) → (safe_l,
  safe_u)` to `sanitize_inputs(l, u, x0) → (safe_l, safe_u, safe_x0)`.
  The internal strict-mode solve now receives sanitized `x0` too, so
  the inner kernel never sees NaN state in the invalid branch.
- `smooth_mcp/diff.py` and `smooth_mcp/forward.py`:
  - `_poisoned_solve` updated to use `traced_invalid_mask(l, u, x0)`
    and `sanitize_inputs(l, u, x0)`.
  - `_checks(l, u)` → `_checks(l, u, x0)`: adds
    `checkify.check(~jnp.any(jnp.isnan(x0)), "x0 contains NaN")`.
  - `_checkify_target` passes `x0` into `_checks`.
- `preflight_validate` docstring updated to mention NaN in `x0`.

**Doc changes:**
- `docs/api.md` §Input validation: "NaN checks" list entry now says
  "`l`, `u`, and `x0` must not contain NaN". The `strict_validation=True`
  section updated to say all three inputs are sanitized and poisoned.
- `docs/troubleshooting.md`: "NaN in bounds" entry expanded to cover
  `x0`, and the jit/vmap symptom section updated accordingly.
- README: no changes (no validation contract details duplicated there).

**Tests added (9 new, 206 total):**
- `tests/test_strict_validation.py`:
  - `TestPreflightValidate::test_rejects_nan_x0`
  - `TestDefaultMode::test_eager_nan_x0_raises`
  - `TestStrictNanPoisoning::test_eager_nan_x0_still_raises`
  - `TestStrictNanPoisoning::test_jit_nan_x0_returns_nan`
  - `TestStrictCheckify::test_eager_nan_x0_error_set`
  - `TestStrictCheckify::test_jit_nan_x0_error_set`
- `tests/test_convergence.py`:
  - `TestInputValidation::test_nan_x0` (solve_mcp)
  - `TestDiffSolverRuntimeValidation::test_nan_x0` (make_mcp_solver_diff)
- `tests/test_forward_factory.py`:
  - `TestRuntimeValidation::test_nan_x0` (make_mcp_solver)

**Regression checks:**
- `pytest tests/` → `206 passed in 475.62s` (197 prior + 9 new).
- `ruff check smooth_mcp tests demos` → clean.
- `mypy smooth_mcp demos` → clean.

### 2. Centralize and complete public solver-option validation

**Why:** several invalid public parameters are currently accepted, and some fail later as opaque internal exceptions.

**Work:**

- Add a shared option-validation helper used by `solve_mcp`, `make_mcp_solver`, and `make_mcp_solver_diff`.
- Validate at minimum:
  - `armijo_c` in `(0, 1)`
  - `backtrack_rho` in `(0, 1)`
  - `max_ls_steps >= 0`
  - `krylov_tol > 0`
  - `krylov_maxiter >= 1`
  - `krylov_restart >= 1`
  - `gmres_tol > 0`
  - `gmres_restart >= 1`
  - `gmres_maxiter >= 1`
  - `cg_tol > 0`
  - `cg_maxiter >= 1`
- Decide whether validation should be construction-time, call-time, or split by API shape.
- Replace raw downstream failures like `IndexError` with clear `ValueError`s naming the bad argument.
- Add regression tests for every new validation rule.

**Acceptance criteria:**

- Invalid public settings fail immediately at the API boundary.
- The same validation rules apply across eager, forward-factory, and differentiable APIs.
- No bad solver knob reaches JAX internals before being rejected.

**Done (2026-04-18).**

**Design:**
- Two shared helpers added to `smooth_mcp/_kernel.py`:
  - `validate_solver_options(...)` — continuation, Newton,
    line-search, and forward-Krylov knobs shared by all three APIs.
  - `validate_adjoint_options(...)` — adjoint-only knobs for
    `make_mcp_solver_diff`.
- `solve_mcp` validates call-time (matching its one-shot API shape);
  both factories validate at construction time (fail-fast factory
  pattern). Both sites use the same helper.
- Adjoint options are validated unconditionally regardless of
  `adjoint_method`, so a bad `cg_*` knob is rejected even when
  `adjoint_method="gmres"` is selected.

**New validation rules added (all with clear ValueErrors naming
the offending argument):**
- `armijo_c` must be in (0, 1)
- `backtrack_rho` must be in (0, 1)
- `max_ls_steps >= 0`
- `krylov_tol > 0`
- `krylov_maxiter >= 1`
- `krylov_restart >= 1`
- `gmres_tol > 0`
- `gmres_maxiter >= 1`
- `gmres_restart >= 1`
- `cg_tol > 0`
- `cg_maxiter >= 1`
- `linear_solver` in ("dense", "gmres") — promoted from internal
  `make_newton_solver` to the public-API boundary.

**Code changes:**
- `smooth_mcp/_kernel.py`: new `validate_solver_options` and
  `validate_adjoint_options` helpers.
- `smooth_mcp/solver.py`: inline mu_*/newton_tol/regularize checks
  replaced with a single `validate_solver_options(...)` call (which
  also covers armijo, line-search, and krylov knobs).
- `smooth_mcp/forward.py`: same refactor as solver.py.
- `smooth_mcp/diff.py`: refactored to call both helpers; inline
  `adjoint_method` check moved into `validate_adjoint_options`.

**Tests added (32 new, 238 total):**
- `tests/test_convergence.py::TestExtendedOptionValidation` covers:
  - `solve_mcp`: armijo_c (3 variants), backtrack_rho, max_ls_steps,
    krylov_tol (2), krylov_maxiter, krylov_restart, linear_solver.
  - `make_mcp_solver`: armijo_c, backtrack_rho, max_ls_steps,
    krylov_tol, krylov_maxiter, krylov_restart, linear_solver.
  - `make_mcp_solver_diff`: same forward knobs as above, plus
    gmres_tol (2), gmres_maxiter, gmres_restart, cg_tol (2),
    cg_maxiter, and the adjoint-validated-even-when-using-gmres
    regression case.

**Replaced raw downstream failures:**
- `krylov_restart=0` previously raised `IndexError` inside JAX
  internals; now raises `ValueError("krylov_restart must be >= 1")`
  at the public API boundary.
- `gmres_restart=0` on the diff factory previously only surfaced
  when gradients were taken; now rejected at factory-construction
  time with a clear ValueError.

**Regression checks:**
- `pytest tests/` → `238 passed in 622.67s` (206 prior + 32 new).
- `ruff check smooth_mcp tests demos` → clean.
- `mypy smooth_mcp demos` → clean.
- `black --check smooth_mcp tests demos` → clean.

## P1

### 3. Make a deliberate policy decision on traced invalid-input safety

**Why:** the default traced path still silently skips value checks, and the test suite currently codifies that unsafe behavior.

**Work:**

- Choose one of these directions:
  - make traced safety the default for factory APIs
  - keep the fast unsafe path, but require an explicit opt-out or explicit mode selection
  - keep the current behavior but emit stronger warnings and make the unsafe choice unmistakable
- Update factory signatures, docs, examples, and tests to match the chosen policy.
- If behavior changes, add migration notes in README/docs/api/troubleshooting.

**Acceptance criteria:**

- The repository has one explicit, documented policy for traced invalid inputs.
- Users do not get silent finite outputs from invalid traced bounds by accident.
- Tests describe the intended long-term behavior, not a temporary compromise.

**Done (2026-04-18). Direction chosen: option 1 — flip the factory
default from `strict_validation=False` to `strict_validation=True`.**

**Policy:**
- Factory default for `make_mcp_solver` and `make_mcp_solver_diff` is
  now `strict_validation=True` (NaN-poisoning). Invalid traced inputs
  produce `NaN` output and `SolveInfo.converged=False`.
- The old fast path remains available as an explicit opt-out:
  `strict_validation=False`. This is the only way to get the legacy
  silent-finite behavior, which the docs now describe as a tight-inner-
  loop optimization intended for use after `preflight_validate`.
- `solve_mcp` is unchanged (always eager, always checks).

**Rationale:** NaN-poisoning has near-zero measured overhead, so the
cost of making it the default is negligible. Users who want the fast
path can ask for it; users who forget get safety for free. This
matches the acceptance criterion that invalid traced bounds should
not produce silent finite outputs by accident.

**Code changes:**
- `smooth_mcp/forward.py`: `strict_validation: Union[bool, str] = True`.
  Docstring rewritten so `True` is described as the default/safe mode
  and `False` is described as the explicit opt-out with its safety
  caveat.
- `smooth_mcp/diff.py`: same default flip and docstring update.

**Doc changes:**
- `docs/api.md` §Input validation:
  - Added a migration-note admonition at the top of the section
    calling out the default flip.
  - Rewrote the eager/traced paragraphs so the factory default is
    described as safe.
  - Rewrote §2 ("NaN-poisoning") to note it is now the default.
  - Added §4 ("`strict_validation=False` — explicit opt-out") with
    guidance and example.
  - Updated the "Which to use" table so the first row is the safe
    default and the opt-out is called out explicitly.
- `docs/troubleshooting.md`: "Invalid bounds not rejected under jit"
  section rewritten — factory default now safe, opt-out path
  described as the expected source of silent-finite results.
- `README.md`: no changes (no specific claims about the default).

**Test changes:**
- `tests/test_strict_validation.py::TestDefaultMode`:
  - `test_jit_invalid_silently_slips_through` renamed to
    `test_jit_invalid_poisons_to_nan` and rewritten to lock in the
    new safe contract (invalid traced inputs → NaN, not finite).
- `tests/test_strict_validation.py::TestUncheckedMode` (new class,
  3 tests): locks in the explicit opt-out path —
  - `test_eager_invalid_still_raises`: eager validation unaffected.
  - `test_jit_invalid_silently_slips_through`: traced invalid inputs
    flow through as finite (documented fast-path behavior).
  - `test_jit_valid_matches_strict`: on valid inputs, unchecked and
    strict modes agree.

**Verification probe:** direct smoke run confirms both factories
poison invalid `l > u` to NaN under `jit` by default, and both
preserve the old silent-finite behavior with
`strict_validation=False` explicit.

**Regression checks:**
- `pytest tests/` → `241 passed in 655.49s` (238 prior − 1 renamed
  + 1 renamed + 3 new = 241).
- `ruff check smooth_mcp tests demos` → clean.
- `mypy smooth_mcp demos` → clean.
- `black --check smooth_mcp tests demos` → clean.

### 4. Bring `make_mcp_solver` test coverage up to its documented public contract

**Why:** the forward-only factory is documented as supporting the same traced validation modes as the differentiable factory, but that contract is not fully covered by tests.

**Work:**

- Extend `tests/test_forward_factory.py` or add a dedicated forward-validation test module.
- Cover:
  - `strict_validation=True` with eager, `jit`, `vmap`, and `return_aux=True`
  - `strict_validation="checkify"` with eager and `jit`
  - forward-factory invalid `x0` cases after task 1 lands
  - forward-factory `vmap` smoke tests if the docs continue to imply batching support
- Keep parity with the differentiable test matrix where the APIs claim parallel behavior.

**Acceptance criteria:**

- Every public `make_mcp_solver` validation mode described in `docs/api.md` is exercised by automated tests.
- Regressions in the forward-only path are caught without relying on manual probes.

**Done (2026-04-18).**

**New test module: `tests/test_forward_strict_validation.py`**

Mirrors `tests/test_strict_validation.py` (which covers the diff
factory) for `make_mcp_solver`. Gradient tests are omitted since the
forward factory has no backward path.

Coverage (27 tests across 4 classes):

- **`TestForwardDefaultMode`** (5 tests): eager `l > u`, NaN `l`,
  NaN `u`, NaN `x0` all raise; `jit` with invalid inputs poisons to
  NaN (locks in the 2026-04-18 safe-by-default contract).
- **`TestForwardUncheckedMode`** (3 tests): `strict_validation=False`
  eager still raises, `jit` silently slips through, and on valid
  inputs unchecked matches strict.
- **`TestForwardStrictNanPoisoning`** (10 tests): eager valid matches
  default; `jit` with `l > u` / NaN `l` / NaN `u` / NaN `x0` returns
  NaN; `jit` valid returns finite; `vmap` mixed batch gives per-row
  NaN; `jit(vmap)` mixed batch gives per-row NaN; `return_aux=True`
  with invalid input sets `converged=False` and `residual_norm=NaN`;
  `return_aux=True` with valid input reports `converged=True`.
- **`TestForwardStrictCheckify`** (9 tests): eager valid → `err.get()
  is None`; eager `l > u` / NaN `l` / NaN `x0` and `jit` invalid all
  set a matching error message; `jit` valid → no error; `err.throw()`
  raises on invalid, no-op on valid; `checkify.checkify(jax.vmap(
  raw))` raises `ValueError` at trace time.

**New vmap coverage: `tests/test_forward_factory.py::TestForwardVmap`**

3 tests covering:
- `vmap(solver, in_axes=(None, None, None, 0))` over a batch of
  thetas: shape, finiteness, and per-row parity with unbatched calls.
- `jax.jit(jax.vmap(solver))` composition: shape, finiteness,
  compiled-graph reuse on a second call.
- `return_aux=True` under `vmap`: `SolveInfo` fields all have the
  expected batch dimension and every row reports `converged=True`.

**Results:**
- `pytest tests/test_forward_strict_validation.py -v` → `27 passed`.
- `pytest tests/test_forward_factory.py::TestForwardVmap -v` →
  `3 passed`.
- Full suite: `pytest tests/` → `271 passed in 630.39s`
  (241 prior + 27 + 3 = 271).
- `ruff check smooth_mcp tests demos` → clean.
- `mypy smooth_mcp demos` → clean.
- `black --check smooth_mcp tests demos` → clean.

### 5. Add CI and publish a tested version-support policy

**Why:** the package depends on JAX and experimental JAX APIs but has no automated compatibility gate and no encoded support matrix.

**Work:**

- Add a CI workflow under `.github/workflows/` that runs at least:
  - `pytest`
  - `ruff check`
  - `mypy`
- Decide the supported Python and JAX/JAXLIB versions.
- Reflect that support policy in:
  - `pyproject.toml`
  - README / installation docs
  - CI matrix
- If full version pinning is too strict, at least add sensible lower bounds and an explicitly tested matrix.

**Acceptance criteria:**

- New changes are automatically validated on the supported version matrix.
- The supported environment is documented and matches what CI actually runs.

**Done (2026-04-18).**

**Support policy:**
- Python: `3.11`, `3.12`, `3.13` (CI matrix). Python 3.10 may work but
  is not exercised; `requires-python = ">=3.11"` in `pyproject.toml`.
- JAX / jaxlib: `>=0.4.38` (pinned lower bound, matching the version
  this codebase was developed and tested against).

Rationale for the lower bound: stable composition of
`jax.experimental.checkify` with `jit`/`grad`/`vmap` and the current
sparse linear solver API (`jax.scipy.sparse.linalg.gmres`, `cg`)
are both load-bearing for this library's public contract.

**New file: `.github/workflows/ci.yml`**

Two jobs:
- **`lint`** (single runner on Python 3.12): `ruff check`,
  `black --check`, `mypy`. Fails fast on formatting or static checks.
- **`test`** (matrix over Python 3.11 / 3.12 / 3.13): installs the
  package with dev deps, prints the resolved JAX version for
  traceability, runs the full pytest suite.

Triggers: push to `main` and pull requests targeting `main`.
Concurrency group cancels superseded runs on the same ref so
re-pushed branches do not stack up.

**`pyproject.toml` changes:**
- `requires-python = ">=3.10"` → `">=3.11"`.
- Dependencies pinned: `jax>=0.4.38`, `jaxlib>=0.4.38`.
- Added PyPI classifiers (Python versions, MIT, OS-independent,
  scientific/mathematics topic).

**`docs/installation.md` changes:**
- Added a "Supported versions" table at the top documenting the
  Python and JAX ranges CI actually exercises, plus a short
  explanation of why the JAX lower bound exists.

**Verification:**
- `.venv/bin/pip install -e .` succeeds with the updated metadata.
- `import smooth_mcp; smooth_mcp.__version__` and public-API imports
  all resolve cleanly.
- `ruff check` / `mypy` / `black --check` on the repository all pass
  (no code changes — metadata only).

**Deferred / out of scope:**
- No JAX-version matrix yet. The `test` job uses whatever JAX pip
  resolves (currently 0.4.38 or newer). Adding a multi-JAX matrix
  is possible later if JAX drift becomes a real issue.

### 6. Add smoke automation for examples and make benchmark scripts import-safe

**Why:** demos and benchmarks are part of the user-facing surface but are easier to rot than the core package.

**Work:**

- Add `if __name__ == "__main__":` around the executable body of `benchmarks/bench_solve.py`.
- Decide whether benchmark modules should be importable helpers or script-only entry points; structure them consistently.
- Add a small smoke layer for demos/examples. Options:
  - execute a selected subset with reduced sizes
  - import and run cheap code paths only
  - add a dedicated `make test-examples` or `pytest -m examples` path
- Wire the smoke layer into CI if runtime is reasonable.

**Acceptance criteria:**

- Importing a benchmark module has no side effects.
- At least one automated path exercises the example surface so demo breakage is caught early.

**Done (2026-04-18).**

**`benchmarks/bench_solve.py` — import-safe:**
- Module-level benchmark body moved into a `run_benchmark()` function.
- Execution gated behind `if __name__ == "__main__":`.
- Helpers (`F_lcp`, `F_param`, `bench`) and constants (`l`, `u`, `x0`,
  `theta`, `N_REPEATS`) still live at module scope so consumers can
  reuse them, but nothing runs on import.
- `bench_continuation.py` was already import-safe; both benchmark
  modules are now consistent.

**New test module: `tests/test_demos.py`**

Each demo is run end-to-end as a subprocess via `subprocess.run`
with a 15-minute timeout. Exit code 0 is required; otherwise stdout
and stderr are surfaced in the pytest failure message so a broken
demo's first stack trace is visible in CI logs.

- **`TestDemoSmoke`** (8 tests) — the fast demos, run on every CI
  push:
  - `lcp_as_mcp`, `differentiable_lcp`, `nonlinear_1d_mcp`,
    `2d_nonlinear_complementarity_problem`, `kkt_conditions`,
    `obstacle_problem`, `spatial_price_equilibrium`,
    `traffic_route_choice`.
  - Wall-time range: ~19s to ~75s per demo (measured before
    committing).
- **`TestSlowDemoSmoke`** (1 test) — `bound_optimization`, tagged
  `@pytest.mark.slow` because it does 50 eager gradient-descent
  steps (~6.5 min). Run on demand with `make test-slow` or
  `pytest -m slow`.

**Slow-marker infrastructure:**
- `pyproject.toml` now configures `[tool.pytest.ini_options]` with
  `addopts = "-m 'not slow'"` so slow-tagged tests are deselected by
  default, and registers the `slow` marker.
- CI naturally inherits this — the `pytest tests/ -q` step in the
  workflow picks up the default marker exclusion from pyproject.
- Local escape hatches:
  - `make test` — default; slow tests deselected.
  - `make test-slow` — only slow tests.
  - `make test-examples` — only demo smoke tests.

**Verification:**
- Import-safety: `python -c "import sys; sys.path.insert(0,
  'benchmarks'); import bench_solve"` prints nothing and exposes
  `run_benchmark`.
- Marker deselection: `pytest tests/test_demos.py --collect-only`
  shows 8/9 tests collected (1 deselected).
- Full suite: `pytest tests/ -q` → `279 passed, 1 deselected in
  1185.51s` (271 prior + 8 new). Runtime grew by ~9 min for the
  demo smoke layer.
- `ruff check` / `mypy` / `black --check` → clean.

**CI time note:** Adding the demo smoke layer roughly doubled the
`test` job runtime per Python version (from ~10 min to ~19 min).
With three Python versions in the CI matrix, this is ~60 min of CI
time per run. Acceptable for a library of this size; if it becomes
a pain point, the demo subset can be pruned or moved to a
nightly-only workflow later.

## P2

### 7. Extract shared result types into a neutral module

**Why:** `make_mcp_solver` currently depends on `diff.py` for `SolveInfo`, which is the wrong architectural direction.

**Work:**

- Create `smooth_mcp/_types.py` or `smooth_mcp/types.py`.
- Move `SolveInfo` there.
- Optionally move `MCPResult` there as well if the team wants all public result containers together.
- Update imports in `forward.py`, `diff.py`, and `__init__.py`.
- Keep behavior and public exports unchanged.

**Acceptance criteria:**

- `forward.py` no longer imports from `diff.py`.
- Shared public types live in a module that does not depend on solver implementation details.

**Done (2026-04-18).**

**New module: `smooth_mcp/_types.py`**

Holds the two public result containers (`MCPResult`, `SolveInfo`) as
`NamedTuple`s. Has no dependencies on any solver module — only
`jax.numpy` for the annotation types. `_types.py` is internal (the
underscore prefix signals "don't import the submodule directly"); the
types themselves are public and re-exported from `smooth_mcp`.

**Code changes:**
- `smooth_mcp/solver.py`: `MCPResult` definition removed; now
  imported from `smooth_mcp._types`. Unused `NamedTuple` import
  removed.
- `smooth_mcp/diff.py`: `SolveInfo` definition removed; now imported
  from `smooth_mcp._types`. Unused `NamedTuple` import removed.
- `smooth_mcp/forward.py`: `from smooth_mcp.diff import SolveInfo`
  replaced with `from smooth_mcp._types import SolveInfo`. No more
  cross-factory dependency.
- `smooth_mcp/__init__.py`: `MCPResult` and `SolveInfo` re-exported
  from `smooth_mcp._types`. Public API (`from smooth_mcp import
  MCPResult, SolveInfo`) is unchanged.
- `tests/test_vmap.py`: import switched from
  `from smooth_mcp.diff import SolveInfo` to `from smooth_mcp import
  SolveInfo` (uses the public API directly).

**Backward compatibility preserved:**
- `from smooth_mcp.diff import SolveInfo` still works (re-exported
  through the submodule's namespace).
- `from smooth_mcp.solver import MCPResult` still works.
- `from smooth_mcp.forward import SolveInfo` still works.
- Verified by a direct probe: all three submodule paths resolve to
  the canonical `smooth_mcp._types.SolveInfo` / `.MCPResult`.

**Verification:**
- `grep "from smooth_mcp.diff"` in `smooth_mcp/forward.py`: no hits.
  Acceptance criterion met.
- `grep "from smooth_mcp.diff" tests/`: no hits.
- Public-API import smoke test confirms every entry point still
  imports cleanly and type identity is preserved across paths.
- `pytest tests/` → `279 passed, 1 deselected in 1011.59s`
  (unchanged from before the refactor).
- `ruff check smooth_mcp tests demos` → clean.
- `mypy smooth_mcp demos` → clean (16 source files vs 15 before —
  the new `_types.py` adds one).
- `black --check smooth_mcp tests demos` → clean.

## Suggested Execution Order

1. Task 1
2. Task 2
3. Task 4
4. Task 3
5. Task 5
6. Task 6
7. Task 7
