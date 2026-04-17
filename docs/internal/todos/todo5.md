# Todo 5: Address Review Findings from 2026-04-16

This plan addresses the issues documented in `docs/internal/reviews/review.codex.2026-04-16.md`.

## Phase 1: Reduce the traced-validation risk

1. Decide on the intended safety contract for traced execution.
   Clarify whether the library should:
   - reject invalid values under tracing when possible,
   - offer an optional strict mode,
   - or keep the current behavior but expose safer wrappers.

   **Decision (2026-04-16):** adopt **optional strict mode as the primary
   mechanism, with a documented eager preflight helper as a supporting path.**
   Default behavior stays as it is today: shape checks always run, value
   checks (NaN, `l <= u`) run eagerly, value checks are skipped under tracing.

   Concretely, this commits subsequent phase-1 tasks to:
   - Add an opt-in `strict_validation: bool = False` knob on
     `make_mcp_solver_diff` (and any forward-only factory introduced in
     phase 2). When `True`, the returned solver enforces NaN-free bounds
     and `l <= u` even when `l`, `u`, `x0` are tracers. Implementation should
     prefer `jax.experimental.checkify` if it composes cleanly with `jit`,
     `grad`, and `vmap`; otherwise fall back to a `lax.cond` / `jnp.where`
     based NaN-poisoning scheme that surfaces as `NaN` outputs and a
     `converged=False` flag in `SolveInfo`.
   - Expose a public `preflight_validate(l, u, x0)` helper for users whose
     bounds are static across a training loop. This is a pure-eager check
     and adds zero per-call overhead inside the JIT region.
   - Promote the traced-validation gap to a prominent warning in
     `README.md` and `docs/api.md` until strict mode ships, then update the
     same sections to point at the new knob.

   **Rationale:**
   - Always-on traced rejection (option 1) imposes `checkify` overhead on
     every call. In the common training-loop case the bounds are static,
     so users would pay a recurring cost for a check that only needs to
     run once.
   - A preflight-only path (option 3) is insufficient on its own: when
     `l` or `u` are themselves traced (e.g. `vmap` over per-element bounds,
     learned bounds, or batched inference), there is no eager moment to
     check them.
   - An optional strict mode is the right point in the design space: keep
     the fast default, give safety-critical workflows an explicit knob,
     and let the eager preflight serve the static-bound case cheaply.

   **Out of scope for this decision:** validating `F_fn` outputs (NaN in
   the residual) and validating `theta`. Those are separate concerns and
   should be handled by user code or a later iteration.

2. Evaluate JAX-native options for strict validation in traced code.
   Check whether `jax.experimental.checkify` or a comparable mechanism can be used without breaking `jit`, `grad`, or `vmap`.

   **Findings (2026-04-16, JAX 0.4.38):** probed both `jax.experimental.checkify`
   and a `jnp.where`-based NaN-poisoning fallback against the real solver.

   **`checkify` results:**
   - Eager: catches `l > u`, NaN bounds. Returns `(Error, out)`.
   - `jit(checkify(strict_solve))`: works. Error survives compilation.
   - `jit(grad(checkify(strict_solve)))`: works. Error reported alongside gradient.
   - `checkify(vmap(strict_solve))`: **FAILS** with
     `ValueError: Checkify does not support batched while-loops
     (checkify-of-vmap-of-while)`. Our continuation kernel is built on
     `lax.while_loop`, so this combination is structurally rejected by JAX.
   - `vmap(checkify(strict_solve))` and `jit(vmap(checkify(...)))`: both
     work. Error message includes the offending mapped index
     (`'at mapped index 1: l must be <= u'`). This is the JAX-recommended
     ordering for any kernel that uses `while_loop`.
   - Per-call overhead on a 1D toy problem under warm `jit`: about
     `+85 us/call` (~`1.16x` baseline `527 us`). Negligible for real work,
     non-trivial for tiny problems in tight loops.
   - API impact: the strict-mode return value is `(Error, x_star)` (or
     `(Error, (x_star, SolveInfo))`), not `x_star`. Users must call
     `err.throw()` or `err.get()` themselves.

   **NaN-poisoning fallback results:**
   Pattern: compute `invalid = any(NaN(l)) | any(NaN(u)) | any(l > u)` as a
   JAX boolean, sanitize the inputs with `jnp.where` so the inner solve is
   well-defined, then `jnp.where(invalid, NaN, x_star)` on the output.
   - Composes natively with `jit`, `grad`, `vmap`, and any combination —
     no `while_loop` interaction issue.
   - Overhead under warm `jit`: about `+2 us/call` (~`1.00x` baseline).
     Effectively free.
   - API impact: return type unchanged. Failure surfaces as `NaN` in the
     output and (already wired) `converged=False` in `SolveInfo`. Users
     must inspect the result rather than catch an exception.
   - Tradeoff: silent failure mode. A user who ignores the output and
     `SolveInfo` will not notice bad inputs.

   **Implications for phase-1 design:**
   - Strict mode via `checkify` is viable for the `jit` and `grad` paths
     and is the right primitive when the user wants an *exception-style*
     contract. For `vmap`, users must structure as `vmap(checkify(...))`,
     not `checkify(vmap(...))`. We can document this; we cannot fix it
     without removing `lax.while_loop` from the kernel, which is not on
     the table.
   - NaN-poisoning is the natural complement: it gives a strict guarantee
     ("invalid input cannot produce a numerically meaningful answer")
     while remaining a drop-in transformation that composes with anything.
   - Recommendation for task 3: prototype both. Default `strict_validation=True`
     to NaN-poisoning (composes everywhere, near-zero overhead, no API
     change). Add `strict_validation="checkify"` for users who want
     `(Error, out)` semantics and accept the `vmap` ordering constraint.
     Document the `vmap` ordering as the only checkify caveat.
   - The eager `preflight_validate(l, u, x0)` helper from task 1 remains
     the cheapest option for static-bound training loops and is unaffected
     by this evaluation.

   **Probe scripts retained at:** `/tmp/eval_checkify.py`,
   `/tmp/eval_checkify_vmap.py`, `/tmp/eval_overhead_and_poison.py`.
   Re-run with `.venv/bin/python <path>` to reproduce.

3. Prototype a strict traced validation path for the differentiable solver.
   Minimum target:
   - detect `l > u`,
   - detect NaN bounds,
   - preserve JAX compatibility.

   **Implemented (2026-04-16).** Changes:
   - `smooth_mcp/_kernel.py`: added `preflight_validate`, `traced_invalid_mask`,
     `sanitize_bounds` helpers.
   - `smooth_mcp/diff.py`: `make_mcp_solver_diff` gained a
     `strict_validation: bool | str = False` parameter with three modes:
       - `False` (default): unchanged behavior.
       - `True`: NaN-poisoning. `_poisoned_solve` computes a per-batch
         `invalid` mask, sanitizes `l`/`u` with `jnp.where`, runs the inner
         custom_vjp solve, then replaces `x_star` with NaN where invalid.
         When `return_aux=True`, `_poison_aux` also flips
         `converged → False` and `residual_norm → NaN` so aux stays
         consistent with the poisoned output.
       - `"checkify"`: factory returns `checkify.checkify(_checkify_target)`
         with `checkify.check` calls for NaN and ordering. Signature becomes
         `(l, u, x0, theta) -> (Error, x_star)` (or `(Error, (x_star, SolveInfo))`).
   - `smooth_mcp/__init__.py`: exports `preflight_validate` so users can
     run eager validation once before entering a jitted/vmapped loop.

   **Verified (smoke test at `/tmp/smoke_strict.py`):**
   - default behavior: invalid under `jit` slips through (x=[3.] for l=5, u=3) —
     unchanged, as intended.
   - `strict_validation=True`:
     - eager invalid still raises `ValueError` via the existing
       `validate_bounds_and_x0` path.
     - `jit(strict)` with `l > u` → `NaN`.
     - `jit(strict)` with NaN in `l` → `NaN`.
     - `vmap(strict)` on mixed batch → per-element NaN only for bad rows.
     - `jit(vmap(strict))` on mixed batch → same.
     - `grad(strict)` and `jit(grad(strict))` on valid input → real gradients.
     - with `return_aux=True`: `SolveInfo.converged=False`,
       `residual_norm=NaN` for invalid rows.
   - `strict_validation="checkify"`:
     - eager / jit invalid → `(Error, x)` with the offending check message.
     - `vmap(checked)` on mixed batch → aggregated error with "at mapped
       index N" prefixes (per-row reporting).
     - `err.throw()` raises `JaxRuntimeError`.
     - invalid value checks under checkify replace the eager `ValueError`
       raise path with an `Error` return, which is consistent with the
       checkify contract.

   **Regression checks:**
   - `pytest tests/ -q` → `123 passed in 339.62s`.
   - `ruff check smooth_mcp tests demos` → clean.
   - `mypy smooth_mcp demos` → clean.

   Tasks 4 and 5 build on this:
   - Task 4 ("documented preflight wrapper") is subsumed — `preflight_validate`
     is already exposed.
   - Task 5 will formalize the smoke test into the pytest suite.

4. If strict traced validation is too invasive, add a documented preflight wrapper.
   The wrapper should perform eager value checks immediately before traced execution in training/inference pipelines.

   **Status (2026-04-16): conditional not triggered, but preflight still
   shipped and documented.** Strict traced validation landed cleanly in
   task 3 — both `strict_validation=True` (NaN-poisoning) and
   `strict_validation="checkify"` compose with `jit`, `grad`, and `vmap`
   at manageable cost. So the fallback this task describes is not needed
   as a rescue, but the `preflight_validate` helper remains the cheapest
   option for the common "bounds are static across the training loop"
   case and is now part of the public API.

   Changes for this task:
   - `smooth_mcp.preflight_validate(l, u, x0)` is exported from the
     package (added in task 3). Accepts array-likes and raises
     `ValueError` on shape mismatch, NaN bounds, or `l > u`. Under
     tracing, value checks are silently skipped (same convention as the
     internal validator), so accidental use inside a jitted region is a
     no-op rather than an error.
   - `docs/api.md` "Input validation" section rewritten to describe the
     three tiers explicitly, with a usage example for each:
       1. `preflight_validate` before the loop — static bounds.
       2. `strict_validation=True` — NaN-poisoning, composes everywhere.
       3. `strict_validation="checkify"` — `(Error, out)` semantics, with
          the `vmap(solver)` vs `checkify(vmap(...))` ordering caveat
          called out.
   - The `solve_mcp` vs `make_mcp_solver_diff` comparison row for input
     validation now mentions the opt-in strict modes.

   README updates are deferred to task 6 (high-visibility warning pass).

5. Add tests for the chosen behavior.
   Cover:
   - eager invalid bounds,
   - jitted invalid bounds,
   - NaN bounds under traced and non-traced execution,
   - expected failure or warning semantics.

   **Implemented (2026-04-16).** New file: `tests/test_strict_validation.py`
   with 32 tests across five classes.

   Coverage map:
   - **`TestPreflightValidate`** (7 tests): valid arrays, Python lists,
     shape mismatch on `l/u`, shape mismatch on `x0`, `l > u`, NaN in `l`,
     NaN in `u`.
   - **`TestStrictValidationArg`** (2 tests): bad string and `None`
     rejected at factory-construction time. (Integer `1`/`0` are
     deliberately accepted as Python bool aliases — the `in (False, True,
     "checkify")` check relies on `1 == True`.)
   - **`TestDefaultMode`** (3 tests): eager invalid raises, eager NaN
     raises, jit invalid silently returns finite output. The last one
     pins the *current* limitation as a test so a silent behavior change
     in default mode shows up as a regression.
   - **`TestStrictNanPoisoning`** (12 tests): eager invalid still raises,
     eager valid matches default, jit `l > u` / NaN-`l` / NaN-`u` all
     return NaN, jit valid matches default, vmap mixed batch gives
     per-row NaN, jit(vmap) mixed batch gives per-row NaN, grad and
     jit(grad) on valid input return finite gradients, return_aux on
     invalid sets `converged=False` and `residual_norm=NaN`, return_aux
     on valid reports `converged=True`.
   - **`TestStrictCheckify`** (8 tests): eager valid yields `err.get() is
     None`, eager `l > u` / NaN in `l` / jit invalid all set a matching
     error message, vmap aggregates per-index errors
     (`"mapped index"` appears in the message), `err.throw()` raises on
     invalid, `err.throw()` is a no-op on valid, and the documented
     `checkify(vmap(...))` caveat is locked in by asserting that this
     ordering raises `ValueError: ... batched while-loops` at trace.

   **Results:**
   - `pytest tests/test_strict_validation.py -q` → `32 passed in 50.7s`.
   - `pytest tests/ -q` → `155 passed in 322.91s` (123 prior + 32 new).
   - `ruff check smooth_mcp tests demos` → clean.
   - `mypy smooth_mcp demos` → clean.

6. Promote the traced-validation limitation to a high-visibility warning in user docs until the stricter path exists.

   **Done (2026-04-16).** The stricter path does exist now (tasks 3–4),
   so this task evolved from "promote a warning" to "surface the gap and
   the three new opt-in mechanisms prominently in user-facing docs."

   Changes:
   - `README.md` — "Input validation" section rewritten. The old text
     described the limitation passively ("value-dependent checks are
     skipped"); the new text calls out the gap in bold ("Value checks
     are skipped by default under `jax.jit`, `jax.grad`, and `jax.vmap`")
     and lists the three opt-in fixes (`preflight_validate`,
     `strict_validation=True`, `strict_validation="checkify"`) with
     one-line descriptions and the checkify/vmap ordering caveat. Links
     to `docs/api.md#input-validation` for the full treatment.
   - `docs/troubleshooting.md`:
     - Added a new top-level entry "Invalid bounds not rejected under
       `jax.jit` / `jax.grad` / `jax.vmap`" that describes the symptom,
       the cause, and the three fixes.
     - Extended the existing "NaN in the solution" section with a bullet
       explaining that NaN output can come from `strict_validation=True`
       poisoning (point users at `SolveInfo.converged`).
   - `docs/api.md` — already rewritten in task 4 to cover the three
     tiers with examples and a "which to use" table. Confirmed the
     anchor (`#input-validation`) matches the README link.

   **Regression check:** `pytest tests/ -q` → `155 passed`.

## Phase 2: Add a reusable forward-only solver interface

7. Design a forward-only factory API.
   Candidate shapes:
   - `make_mcp_solver(F_fn, ...)`,
   - or `compile_mcp_solver(F_fn, ...)`.

   **Decision (2026-04-16): `make_mcp_solver`.**

   **Name.** `make_mcp_solver` is parallel to the existing
   `make_mcp_solver_diff`. `compile_mcp_solver` was considered but
   rejected because "compile" implies auto-JIT, and we're deliberately
   leaving JIT to the user (see below). The three-way naming after this
   change is:
   - `solve_mcp(F_fn, l, u, x0, ...)` — eager one-shot with Python-scalar
     result and `verbose` support. Kept for debugging / one-off use.
   - `make_mcp_solver(F_fn, ...)` — reusable forward-only factory. **new.**
   - `make_mcp_solver_diff(F_fn, ...)` — reusable differentiable factory.

   **Factory signature.** Mirror `make_mcp_solver_diff` exactly for the
   shared options, drop the gradient / adjoint options:

   ```python
   solver = make_mcp_solver(
       F_fn,
       mu_init=1.0,
       mu_min=1e-12,
       mu_decay=0.5,
       newton_tol=1e-10,
       max_mu_steps=50,
       armijo_c=1e-4,
       backtrack_rho=0.5,
       max_ls_steps=20,
       linear_solver="dense",
       krylov_tol=1e-6,
       krylov_maxiter=500,
       krylov_restart=30,
       regularize=1e-12,
       return_aux=False,
       strict_validation=False,
   )
   ```

   Deliberately **excluded** (not applicable to forward-only):
   `adjoint_method`, `gmres_tol`, `gmres_restart`, `gmres_maxiter`,
   `cg_tol`, `cg_maxiter`, `precond`, `differentiate_through_x0`.

   Deliberately **excluded** (not JIT-compatible):
   `verbose`. Users who want per-step prints use `solve_mcp` instead.

   **Returned callable.**

   ```python
   solver(l, u, x0, theta) -> x_star                   # return_aux=False
   solver(l, u, x0, theta) -> (x_star, SolveInfo)      # return_aux=True
   ```

   Signature matches `make_mcp_solver_diff`:
   - `theta` is required (pass `jnp.zeros(0)` for single-arg `F(x)` — same
     convention as the diff factory).
   - `SolveInfo` has the same four fields (`mu_used`, `num_steps`,
     `residual_norm`, `converged`) but the `stop_gradient` calls aren't
     needed (no gradient flow). Reuse the existing `SolveInfo` NamedTuple.
   - `strict_validation` supports the same three modes as the diff
     factory (False / True / "checkify"). Checkify mode changes the
     signature to `(Error, x_star)` / `(Error, (x_star, SolveInfo))`,
     matching the diff factory's behavior exactly.

   **JIT behavior.** The factory does **not** auto-wrap the returned
   callable in `jax.jit`. Rationale:
   - Consistent with `make_mcp_solver_diff`.
   - Preserves user control over `static_argnums`, caching, and
     re-tracing semantics.
   - Documented pattern: `solver = jax.jit(make_mcp_solver(F))` for
     repeated fast forward solves.

   **Implementation.** Reuse `smooth_mcp/_kernel.py` internals
   (`normalize_F`, `make_newton_solver`, `make_continuation_solver`,
   `validate_bounds_and_x0`, `traced_invalid_mask`, `sanitize_bounds`,
   and the strict-validation helpers from `diff.py`). No duplication of
   solver logic. The implementation is effectively
   `make_mcp_solver_diff`'s `_run_forward` + `_make_aux` + strict
   wrapping, minus the `custom_vjp` setup and the `_compute_grads`
   adjoint code.

   **Why not just use `make_mcp_solver_diff` for forward-only work?**
   1. `custom_vjp` adds tracing overhead the user pays for even when
      they never call `jax.grad`.
   2. The gradient-specific options (`adjoint_method`, `gmres_tol`, etc.)
      clutter the API for users who only care about forward solves.
   3. A clear name for "repeated forward solves" is itself valuable
      documentation: users currently reach for `solve_mcp` because
      `make_mcp_solver_diff` sounds gradient-specific.

   **Why not duplicate solver logic?** `_kernel.py` already exposes the
   right seams. The diff factory demonstrates the pattern.

   This design sets up tasks 8–12: task 8 pins the aux contract, task 9
   implements against `_kernel.py`, task 10 adds parity tests with
   `solve_mcp`, task 11 benchmarks, task 12 updates the user docs.

8. Define what the forward-only factory should return.
   It should preserve the useful ergonomics of `solve_mcp`, including diagnostics, while avoiding repeated setup cost.

   **Decision (2026-04-16).**

   **Return type.** Reuse the existing `SolveInfo` NamedTuple from
   `diff.py`. No new diagnostic type.

   **Contract.**

   | `return_aux` | `strict_validation` | Returned callable signature |
   |---|---|---|
   | `False` (default) | `False` (default) | `solver(l, u, x0, theta) -> x_star` |
   | `True` | `False` | `solver(l, u, x0, theta) -> (x_star, SolveInfo)` |
   | `False` | `True` (NaN-poisoning) | `solver(l, u, x0, theta) -> x_star` (NaN on invalid) |
   | `True` | `True` | `solver(l, u, x0, theta) -> (x_star, SolveInfo)` (NaN + converged=False on invalid) |
   | `False` | `"checkify"` | `solver(l, u, x0, theta) -> (Error, x_star)` |
   | `True` | `"checkify"` | `solver(l, u, x0, theta) -> (Error, (x_star, SolveInfo))` |

   This mirrors `make_mcp_solver_diff` exactly.

   **`SolveInfo` fields.** Unchanged from the current definition:
   - `mu_used: jnp.ndarray` — terminal smoothing parameter actually reached.
   - `num_steps: jnp.ndarray` — total continuation steps taken.
   - `residual_norm: jnp.ndarray` — max absolute smoothed residual at
     `x_star` evaluated at `mu_min`.
   - `converged: jnp.ndarray` — True if `residual_norm < newton_tol`.

   All four are JAX array scalars (not Python scalars). This is the
   central ergonomic difference from `solve_mcp`'s `MCPResult`:
   - `solve_mcp` coerces to `float`/`int`/`bool` so you can inline the
     values in Python expressions. Not JIT-compatible.
   - The forward-only factory keeps them as JAX scalars so the function
     is JIT-compatible. Users who want Python scalars convert explicitly
     (`float(info.residual_norm)`).

   **No `stop_gradient` in the forward-only aux construction.** The
   diff factory wraps each `SolveInfo` field in `jax.lax.stop_gradient`
   because the aux is returned alongside `x_star` through `custom_vjp`.
   The forward-only factory has no gradient path, so the wrappers are
   redundant. Drop them in the forward-only implementation for clarity.

   **What the "useful ergonomics of `solve_mcp`" means here:**
   - **Diagnostics available when you want them** — `return_aux=True`
     hands back the same four signals (`converged`, `residual_norm`,
     `num_steps`, plus `mu_used` which `solve_mcp` didn't expose).
   - **Truncation awareness** — `mu_used` surfaces cases where
     `max_mu_steps` was truncated, something `solve_mcp` hid. Minor
     ergonomic upgrade.
   - **Convergence check is obvious** — `info.converged` is the single
     boolean signal; same as `MCPResult.converged`.

   **What is deliberately not preserved from `solve_mcp`:**
   - `verbose=True` per-step printing (not JIT-compatible).
   - Python-scalar aux fields (breaks JIT).
   - Optional `theta` (the factory requires `theta` for tracing
     consistency with `make_mcp_solver_diff`).

   **Avoiding repeated setup cost.** The factory closes over `F_fn`,
   the normalized version `F_fn_normalized`, and all solver options at
   construction time. The returned callable builds `newton_solve` and
   `continuation` per call, but those are pure-JAX factories — under
   `jax.jit` the builds happen once at trace time and are reused across
   calls with matching input shapes/dtypes. Without JIT, each call
   rebuilds (correct but slower). This is exactly how
   `make_mcp_solver_diff._run_forward` already works.

   The explicit guidance to users: **wrap the returned callable in
   `jax.jit` for repeated fast forward solves.** The docstring and
   `docs/api.md` will spell this out (task 12).

   **Test implication for task 10.** Parity with `solve_mcp` is
   checked field-by-field:
   - `x` parity: `jnp.allclose(solver_out, solve_mcp_out.x)`.
   - `converged` parity: `bool(info.converged) == solve_mcp_out.converged`.
   - `residual_norm` parity: `jnp.allclose(info.residual_norm,
     solve_mcp_out.residual_norm)`.
   - `num_steps` parity: `int(info.num_steps) == solve_mcp_out.num_steps`.
   - `mu_used` has no `solve_mcp` counterpart — test that it's finite
     and `<= mu_init`.

9. Implement the factory on top of the existing `_kernel.py` internals instead of duplicating solver logic.

   **Implemented (2026-04-16).** New file: `smooth_mcp/forward.py` with
   `make_mcp_solver`, exported from `smooth_mcp/__init__.py`.

   Structure:
   - Option validation (mu_init, mu_min, mu_decay, max_mu_steps,
     newton_tol, regularize, strict_validation) — mirrors the diff
     factory's checks.
   - `_run_forward(l, u, x0, theta)` — builds `make_newton_solver` and
     `make_continuation_solver` via `_kernel.py`, runs continuation.
     No duplication of solver logic.
   - `_make_aux(...)` — constructs `SolveInfo` from forward results.
     Reuses the existing `SolveInfo` from `diff.py`. No `stop_gradient`
     wrappers (no gradient path exists).
   - `_solve(...)` — the inner solve that picks return_aux shape.
   - `_poisoned_solve(...)` + `_poison_aux(...)` — NaN-poisoning path
     for `strict_validation=True`. Mirrors `diff.py`.
   - `_checks(l, u)` + `_checkify_target` — checkify path.
   - `solve_checked(...)` — outer wrapper with eager validation.

   Deliberately **omitted** vs `make_mcp_solver_diff`:
   - `custom_vjp` + `_fwd` + `_bwd` + `_compute_grads`.
   - Adjoint options: `adjoint_method`, `gmres_tol`, `gmres_restart`,
     `gmres_maxiter`, `cg_tol`, `cg_maxiter`, `precond`,
     `differentiate_through_x0`.

   **Smoke verification (`/tmp/smoke_forward.py`):**
   - eager valid call returns correct `x_star`.
   - `return_aux=True` exposes `SolveInfo` with finite `mu_used` and
     `num_steps=22`, matching `solve_mcp`.
   - `jax.jit(solver)` works; second call reuses compiled graph.
   - `jax.vmap(solver)` works.
   - Field-level parity with `solve_mcp` on a 1D toy problem: same `x`,
     same `converged`, same `residual_norm`, same `num_steps`.
   - `strict_validation=True` under `jit` returns `NaN` on invalid.
   - `strict_validation="checkify"` returns `(Error, x)` with matching
     message.
   - eager invalid inputs raise `ValueError` via `validate_bounds_and_x0`.

   **Regression checks:**
   - `pytest tests/ -q` → `155 passed in 409.86s`.
   - `ruff check smooth_mcp tests demos` → clean.
   - `mypy smooth_mcp demos` → clean (15 source files).

   Tasks 10–12 build on this: task 10 formalizes parity tests, task 11
   benchmarks, task 12 updates user docs.

10. Add tests that compare the factory-produced solver against `solve_mcp` for:
    - solution values,
    - convergence flags,
    - residual norms,
    - step counts.

    **Implemented (2026-04-16).** New file:
    `tests/test_forward_factory.py` with 28 tests across 8 classes.

    Helper `_assert_parity(result, x, info)` compares all four parity
    fields in one place: `jnp.allclose(result.x, x)`,
    `int(info.num_steps) == result.num_steps`,
    `bool(info.converged) == result.converged`, and
    `jnp.isclose(info.residual_norm, result.residual_norm)`.

    Coverage:
    - **`TestConvergedProblems`** (3): 2D LCP (with `M @ x + q`), nonlinear
      1D cubic, finite-bounds clipping (solution pinned to upper bound).
    - **`TestParametricProblems`** (2): parametric LCP with `theta` as
      the matrix entries, and parametric scaling.
    - **`TestTruncatedContinuation`** (2): `max_mu_steps=1` and `=2` —
      `solve_mcp` and the factory must both report the same
      `num_steps` and `converged` values, including the unconverged case.
    - **`TestLinearSolvers`** (2): `linear_solver="dense"` and
      `linear_solver="gmres"` (looser tolerance for the iterative path).
    - **`TestJitComposability`** (2): `jax.jit(solver)` matches eager
      output field-by-field; second call with matching shapes reuses
      the compiled graph and still returns the right answer.
    - **`TestReturnShape`** (2): `return_aux=False` returns a bare
      `jax.Array`; `return_aux=True` returns a 2-tuple with a
      `SolveInfo` whose four fields are present.
    - **`TestMuUsed`** (2): `mu_used` is finite, bounded by `mu_init`;
      on full convergence it equals `mu_min`.
    - **`TestFactoryValidation`** (8): construction-time rejection of
      bad `mu_init`, `mu_min`, `mu_min > mu_init`, `mu_decay`,
      `max_mu_steps`, `newton_tol`, `regularize`, and
      `strict_validation`. Mirrors the diff factory's validation suite.
    - **`TestRuntimeValidation`** (5): eager rejection of `l > u`,
      shape mismatches on `(l, u)` and `x0`, NaN in `l`, NaN in `u`.

    **Results:**
    - `pytest tests/test_forward_factory.py -v` → `28 passed in 32.1s`.
    - `pytest tests/ -q` → `183 passed in 368.58s`
      (155 prior + 28 new).
    - `ruff check smooth_mcp tests demos` → clean.
    - `mypy smooth_mcp demos` → clean.

11. Add a benchmark section comparing:
    - repeated `solve_mcp(...)`,
    - repeated calls through the new forward-only factory,
    - differentiable forward solves through `make_mcp_solver_diff`.

    **Implemented (2026-04-16).** Extended `benchmarks/bench_solve.py`:
    - Added a "make_mcp_solver (forward-only factory)" section that
      benchmarks factory creation, eager solves, and `jax.jit`-wrapped
      solves.
    - Renamed the existing "make_mcp_solver_diff" section to make the
      forward-only-path framing explicit, and added a jitted forward
      benchmark there too (previously only eager).
    - Extended the summary with a side-by-side table of five forward
      paths and a derived "make_mcp_solver(jit) vs solve_mcp(eager)"
      speedup. The final "Guidance" block points users at the right
      entry point by use case.

    **Captured numbers (`python benchmarks/bench_solve.py`, macOS
    Intel, CPU, JAX 0.4.38, float64, 2D LCP):**

    | Path                              | ms/call |
    |-----------------------------------|--------:|
    | `solve_mcp` (eager)               |  1361.9 |
    | `make_mcp_solver` (eager)         |  1421.1 |
    | `make_mcp_solver` (jit, warm)     |     1.8 |
    | `make_mcp_solver_diff` (eager)    |  1575.7 |
    | `make_mcp_solver_diff` (jit, warm)|     1.6 |
    | `jax.grad` (eager)                |  2661.6 |
    | `jit(grad)` (warm)                |    14.5 |

    Derived:
    - `make_mcp_solver(jit)` vs `solve_mcp(eager)`: **737x** speedup.
    - `jit(grad)` vs eager `grad`: 184x speedup.
    - Eager `make_mcp_solver` and eager `solve_mcp` are the same order
      of magnitude (~1.4s): the factory's value is entirely realized
      under `jax.jit`. The factory is not a magic eager speedup.
    - Forward-only `make_mcp_solver(jit)` and differentiable
      `make_mcp_solver_diff(jit)` forward-only path have near-identical
      wall time (~1.7 ms). The `custom_vjp` wrapping adds negligible
      cost when gradients are not computed.

    These numbers replace the review finding's observation that
    "repeated `solve_mcp` costs about 1107 ms/call" with a specific
    fix: wrapping `make_mcp_solver` in `jax.jit` reduces per-call cost
    by almost three orders of magnitude on the same problem.

    **Regression check:** `ruff check benchmarks/bench_solve.py` → clean.

    Note: the full benchmark-notes file under
    `docs/internal/benchmarks/` is left for task 32 ("update any
    internal benchmark notes") at the end of phase 7.

12. Update user docs to recommend the right entry point for:
    - one-off debugging,
    - repeated forward solves,
    - differentiable/JIT workflows.

    **Done (2026-04-16).**

    **`docs/api.md`:**
    - New "### Choosing an entry point" section at the top of "## Solvers"
      with a use-case-to-entry-point table. Puts the guidance above all
      three per-function sections so it's the first thing readers see.
    - New "### `make_mcp_solver(F_fn, ...)`" section between `solve_mcp`
      and `make_mcp_solver_diff`. Quickstart shows `jax.jit(make_mcp_solver(F))`
      and the `return_aux=True` variant. Links to the shared option
      tables and the validation section.
    - Rewrote "## Comparing `solve_mcp` and `make_mcp_solver_diff`" to
      "## Comparing the three entry points" with a three-column table
      covering returns, gradients, JIT-compatibility, validation,
      diagnostics, verbose, `theta` handling, reusability, and
      `custom_vjp` overhead. Adds a "When to reach for which" bullet
      list.
    - Updated shared-option section headers from "Both `solve_mcp` and
      `make_mcp_solver_diff`" to "all three entry points" where the
      options apply to `make_mcp_solver` as well.
    - Strict-validation section now shows the knob used on both
      `make_mcp_solver` and `make_mcp_solver_diff` so readers don't
      infer it is diff-specific.

    **`README.md`:**
    - New "### Choosing an entry point" block directly above the
      first quickstart. Three-row table with a link to
      `docs/api.md#choosing-an-entry-point`.
    - New "### Repeated forward solves" quickstart between "Solving an
      MCP" and "Differentiable solving". Example uses
      `jax.jit(make_mcp_solver(F))` in a sweep loop, then shows the
      `return_aux=True` variant.
    - Updated existing section-level references from "Both `solve_mcp`
      and `make_mcp_solver_diff`" to "all three entry points" in the
      function-signatures block, input-validation section, and the
      common-parameters header.
    - Caveat mentioning `solve_mcp` not being JIT-compatible now points
      at both factories ("Use `make_mcp_solver` (forward-only) or
      `make_mcp_solver_diff` (differentiable) for JIT-compatible code").
    - Top-level API table now lists four entries: `solve_mcp`,
      `make_mcp_solver`, `make_mcp_solver_diff`, `preflight_validate`.

    **`docs/troubleshooting.md`:**
    - "Invalid bounds not rejected under `jax.jit` ..." section updated
      so `strict_validation` advice mentions both factories.
    - "`TracerBoolConversionError` inside `jax.jit`" section updated to
      offer both the forward-only and differentiable replacements for
      `solve_mcp` inside JIT.

    **Regression checks:**
    - `pytest tests/ -q` → `183 passed in 401.20s`.
    - `ruff check smooth_mcp tests demos` → clean.
    - `mypy smooth_mcp demos` → clean.

## Phase 3: Tighten the tuning guidance

13. Rework the tuning docs so they distinguish:
    - continuation step count,
    - Newton difficulty per step,
    - total runtime.

    **Done (2026-04-16).** Reworked `docs/tuning.md`.

    Changes:
    - **New "Three dimensions to think about" section** at the top of
      the file (before any solver-specific option). Defines each of the
      three metrics, calls out that they "often move in opposite
      directions", and explicitly warns that the "smaller `mu_decay` is
      faster" folklore doesn't hold on this repo's benchmark matrix.
    - **Rewrote the `mu_decay` section** into two subsections:
      - "Step count vs. wall time (benchmarked)" — a 12-row table from
        `benchmarks/bench_continuation.py` showing steps and ms/call
        side-by-side for `lcp_2d`, `ncp_2d`, `spatial_eq`, and
        `obstacle_50d` at `mu_decay=0.5 / 0.25 / 0.10`. Reading notes
        below the table show how to interpret each row.
      - "When to reach for a different value" — the old decision-table
        rewritten to flag that `0.25` is "flat or slightly slower" and
        `0.1` is "experimental — don't assume it's faster, measure"
        based on the benchmark matrix.
    - **Added a "Rule of thumb: profile before changing `mu_decay`"**
      line that separates deterministic step-count claims from
      problem-dependent wall-time claims.
    - Updated `return_aux` / `verbose` references to mention both
      factories now that `make_mcp_solver` exists.

    The concrete numbers came from the existing
    `docs/internal/benchmarks/benchmark-2026-04-14.md` record; task 14
    will expand the benchmark matrix, task 15 will formally tag which
    statements are measured versus heuristic, and task 16 will
    reconcile the README's tuning section with the updated framing.

    **Regression checks (doc-only changes; sanity pass):**
    - `pytest tests/ -q` → `183 passed in 404.23s`.
    - `ruff check smooth_mcp tests demos` → clean.
    - `mypy smooth_mcp demos` → clean.

14. Expand the continuation benchmark matrix with at least one medium-sized dense problem and one larger GMRES-oriented problem.

    **Done (2026-04-17).** Two new problems added to
    `benchmarks/bench_continuation.py` and fully benchmarked.

    **New problems:**
    - `random_lcp_30d` (medium dense): `M = A A^T + 0.1I` with random
      `A` and `q` (n=30, seed=0). Exercises the dense linear solver on
      a genuinely dense, full-rank coupling problem with no spatial
      structure. Uses default solver settings.
    - `obstacle_100d` (GMRES): 100D discretized obstacle problem with
      `linear_solver="gmres"`, `krylov_tol=1e-10`, `krylov_maxiter=1000`,
      `krylov_restart=50`. Tighter `krylov_tol` is required because the
      default `1e-6` is too loose to converge `newton_tol=1e-10` — the
      residual stalls at ~1e-6 regardless of continuation step count.

    **Key findings:**
    - `random_lcp_30d` is the one problem where `mu_decay=0.1` is
      marginally faster (~10% wall-time improvement). Well-conditioned
      SPD system where per-step Newton cost barely changes with
      aggressive decay.
    - `obstacle_100d` (GMRES) dramatically confirms "fewer steps ≠
      faster": `mu_decay=0.1` takes 3x fewer steps but is **3.3x
      slower** (183.7s vs 54.9s per call). `mu_decay=0.25` is 2.2x
      slower. Each Newton step requires far more GMRES iterations when
      the mu jump is larger.
    - GMRES `krylov_tol` must be ≤ `newton_tol` for the outer solve to
      converge. This is a useful finding for the tuning docs.

    **Updated files:**
    - `benchmarks/bench_continuation.py` — replaced `obstacle_200d`
      (which never converged in 50 steps at default krylov_tol) with
      `obstacle_100d` using tuned GMRES settings. Fixed pre-existing
      ruff lint (f-string without placeholders).
    - `docs/internal/benchmarks/benchmark-2026-04-14.md` — expanded
      continuation table from 4 to 6 problems (12 to 18 rows),
      updated observations section with GMRES-specific findings and
      the `krylov_tol` guidance.
    - `docs/tuning.md` — expanded the "Step count vs. wall time
      (benchmarked)" table to include both new problems, updated
      reading notes and "When to reach for a different value" table
      with GMRES-specific guidance.

    **Regression checks:**
    - `ruff check benchmarks/bench_continuation.py` → clean.

15. Record which guidance is benchmark-backed and which guidance is heuristic.
    Avoid wording that implies `mu_decay=0.25` or `0.1` is generally faster.

    **Done (2026-04-17).** Audited all tuning guidance across three files
    and annotated evidence basis throughout.

    **`docs/tuning.md`:**
    - Added "Evidence basis" section at the top defining two categories:
      **Benchmarked** (scoped to the in-repo problem matrix, sections
      cite the table explicitly) and **Heuristic** (standard numerical-
      methods advice, marked where it appears).
    - The `mu_decay` continuation table and reading notes were already
      well-scoped to the benchmark matrix — no changes needed.
    - `mu_decay=0.7` row: annotated "(heuristic — not benchmarked)".
    - Linear solver section: added italic heuristic notice ("n ≈ 100
      crossover is a rough rule of thumb, not a benchmarked threshold").
      Added benchmarked `krylov_tol` warning: must be ≤ `newton_tol`
      for the outer solve to converge (measured on `obstacle_100d`).
    - Regularization section: added italic heuristic notice.
    - Adjoint solver section: added italic heuristic notice with carve-
      out for the GMRES-vs-CG SPD requirement (mathematical property).

    **`README.md`:**
    - Rewrote `mu_decay` guidance to lead with the key insight:
      "**Fewer steps does not mean faster**". Each bullet now states the
      benchmark evidence rather than implying speed improvement:
      - `0.5`: "never meaningfully beaten on wall time".
      - `0.25`: "Wall time is typically flat vs 0.5 on dense problems;
        on GMRES problems it can be significantly slower (benchmarked:
        2.2x slower on obstacle_100d)."
      - `0.1`: "Don't assume it's faster — measure on your problem."
      Links to `docs/tuning.md` for measured numbers.

    **`docs/troubleshooting.md`:**
    - Replaced "**Faster continuation**: Try `mu_decay=0.25` or `0.1`
      to reduce step count" with "**Continuation tuning**" heading and
      text that warns "Profile before assuming fewer steps is faster"
      with a link to the tuning doc.

    **Problematic wording removed:**
    - "good for easy problems" (README, implied speed for `0.1`)
    - "Faster continuation" (troubleshooting, directly implied speed)
    - "when profiling shows continuation overhead dominates" (README,
      implied `0.1` addresses overhead — it often worsens it)

16. Update `README.md` and `docs/tuning.md` to reflect the actual measured tradeoff:
    fewer continuation steps do not necessarily imply lower wall time.

    **Done (2026-04-17).** Substantively completed as part of tasks 14
    and 15; this task verified no remaining gaps.

    The measured tradeoff is now reflected throughout:
    - **`README.md`** `mu_decay` section leads with "**Fewer steps does
      not mean faster**", references the benchmark matrix for every
      decay value, and links to `docs/tuning.md` for the full table.
    - **`docs/tuning.md`** has the 18-row benchmark table with Steps
      and Wall time side-by-side, reading notes that state "fewer
      steps ≠ faster runtime" explicitly, and the "When to reach for a
      different value" decision table referencing measured data.
    - **`docs/troubleshooting.md`** "Slow performance" section now
      warns to profile before assuming fewer steps is faster.
    - **`docs/api.md`** — checked; contains only neutral parameter
      descriptions, no speed claims about `mu_decay`.

    No additional edits were needed beyond tasks 14 and 15.

17. Add a short "how to tune" example that walks users through profiling two or three `mu_decay` values on their own problem.

    **Done (2026-04-17).** Added a "How to profile `mu_decay` on your
    problem" code block to `README.md`, placed immediately after the
    `mu_decay` bullet list in the Tuning guide section.

    The example:
    - Loops over `mu_decay` ∈ {0.5, 0.25, 0.1}.
    - Builds a JIT-compiled `make_mcp_solver` with `return_aux=True`
      for each decay value.
    - Warms up with one call (trace + compile), then times 10 calls
      with `block_until_ready()`.
    - Prints steps, ms/call, and converged for each decay.
    - Closes with: "Pick the `mu_decay` with the lowest wall time
      *that still converges*. Don't assume fewer steps is faster —
      measure."

    `docs/tuning.md` already references this example at line 37–38
    ("the 'how to tune' example in the README"). No changes needed
    there.

## Phase 4: Remove external documentation duplication

18. Decide what belongs in `README.md` versus `docs/`.

    **Decision (2026-04-17).**

    **Principle.** The README is the front door: project pitch, install,
    quickstart, orientation, links. It should be scannable in under 5
    minutes. Deep reference material lives in `docs/` and the README
    points there. No topic should have its authoritative treatment in
    both places — one is the source of truth, the other summarizes and
    links.

    **Current state.** README is 359 lines. The following sections
    duplicate content that already has a canonical home in `docs/`:

    | README section | Lines | Canonical home | Overlap |
    |---|---|---|---|
    | Function signatures | 86–93 | `docs/api.md` §Function signatures | Fully duplicated |
    | JAX integration | 190–212 | `docs/api.md` §Comparing the three entry points | Duplicated |
    | Input validation | 214–227 | `docs/api.md` §Input validation | Summary of same content |
    | Solver options (param tables) | 229–274 | `docs/api.md` §Common/Forward/Adjoint options | Identical tables |
    | Tuning guide | 276–324 | `docs/tuning.md` | Abridged version of same content |

    ~130 lines of README are duplicated material. The three quickstart
    examples (solve_mcp, make_mcp_solver, make_mcp_solver_diff) overlap
    with `docs/api.md` per-function sections, but quickstarts are the
    right kind of overlap — the README shows "how to get started" and
    `docs/api.md` shows "full reference."

    **What stays in README:**
    - What is an MCP? (project pitch, ~13 lines)
    - How it works (smoothing, continuation, implicit diff — ~40 lines;
      this is a differentiator and helps users decide if the library
      fits their problem)
    - Installation (quick: `pip install .`, float64 note, link to
      `docs/installation.md` for platform details — ~15 lines)
    - Choosing an entry point (3-row table + link to `docs/api.md`)
    - Three quickstart examples (~80 lines total: solve_mcp,
      make_mcp_solver, make_mcp_solver_diff)
    - API summary table (~12 lines)
    - Demos listing (~20 lines)
    - A "Documentation" links section pointing to all four docs/ files

    **What moves out of README (replaced by brief summary + link):**
    - Function signatures → 1-line note + link to `docs/api.md`
    - JAX integration (supported transforms, caveats) → fold the
      essential `jax.jit` note into the quickstarts, drop the rest,
      link to `docs/api.md`
    - Input validation → 2-line summary + link to `docs/api.md`
    - Solver options parameter tables → drop from README, link to
      `docs/api.md`
    - Tuning guide (mu_decay bullets, profiling snippet, linear_solver,
      regularize, adjoint) → drop from README, link to
      `docs/tuning.md`. The profiling snippet from task 17 moves to
      `docs/tuning.md` since that is the canonical tuning reference.

    **Estimated post-trim README:** ~180 lines (down from 359).

    **Makefile inconsistency (noted for task 22):** `make install` runs
    `pip install -e .` (editable, no dev deps) — a middle ground not
    documented in README or `docs/installation.md`. Either change the
    Makefile target to `pip install .` (matching "standard install") or
    document the editable-without-dev pattern.

    Tasks 19–21 execute this plan.

19. Trim the README to an overview and quickstart.
    Keep:
    - project summary,
    - quick install,
    - minimal usage example,
    - links to deeper docs.

    **Done (2026-04-17).** Rewrote `README.md` from 359 lines to 189
    lines following the task 18 decision.

    **Kept:**
    - What is an MCP? (project pitch)
    - How it works (smoothing, continuation, implicit diff)
    - Installation (condensed: `pip install .`, dev install, float64
      note, link to `docs/installation.md`)
    - Quickstart section with three examples:
      - Solving an MCP (`solve_mcp`)
      - Repeated forward solves (`make_mcp_solver` + `jax.jit`)
      - Differentiable solving (`make_mcp_solver_diff` + `jax.grad`)
    - Documentation links table (all four docs/ files)
    - API summary table (four public entry points + link to api.md for
      low-level building blocks)
    - Demos listing

    **Removed (canonical home is `docs/`):**
    - Function signatures section → `docs/api.md` §Function signatures
    - JAX integration section (supported transforms, caveats) →
      `docs/api.md` §Comparing the three entry points. The essential
      `jax.jit` example is now folded into the Differentiable solving
      quickstart.
    - Input validation section → `docs/api.md` §Input validation
    - Solver options parameter tables (common, forward, adjoint) →
      `docs/api.md` §Common/Forward solver options
    - Tuning guide (mu_decay, linear_solver, regularize, adjoint
      settings, profiling snippet) → `docs/tuning.md`. The profiling
      snippet from task 17 was moved to `docs/tuning.md`
      §How to profile mu_decay on your problem.
    - `MCPResult` / `SolveInfo` type descriptions →
      `docs/api.md` §Data types

    **Also updated:** `docs/tuning.md` — internal reference that
    pointed to "the how to tune example in the README" now links to
    the local `#how-to-profile-mu_decay-on-your-problem` anchor.

20. Move the long-form API, tuning, and troubleshooting details fully into `docs/`.

    **Done (2026-04-17).** Verified that all content removed from the
    README in task 19 already has a canonical home in `docs/`. Two
    small gaps were found and filled:

    - `docs/api.md` `make_mcp_solver_diff` section: added
      `jax.value_and_grad` to the supported-transforms list and a
      sentence noting that gradients flow w.r.t. `theta`, `l`, `u` by
      default plus `x0` with `differentiate_through_x0=True`. This
      information was in the old README's "JAX integration" section
      but not in `docs/api.md`.
    - `docs/api.md` comparison table: added `jax.value_and_grad` to
      the Gradients row.
    - `docs/api.md` JIT-incompatibility note: now mentions both
      `make_mcp_solver` and `make_mcp_solver_diff` as alternatives
      (previously only mentioned the diff factory).

    **Audit result — all content accounted for:**

    | Old README section | Canonical home | Status |
    |---|---|---|
    | Function signatures | `docs/api.md` §Function signatures | Already present |
    | JAX integration | `docs/api.md` §make_mcp_solver_diff + §Comparing | Filled gap (value_and_grad, grad targets) |
    | Input validation | `docs/api.md` §Input validation | Already present |
    | Solver options tables | `docs/api.md` §Common/Forward/Adjoint | Already present |
    | Tuning guide | `docs/tuning.md` | Already present (profiling snippet moved in task 19) |

    No content was lost in the README trim.

21. Add explicit links from the README to:
    - `docs/installation.md`,
    - `docs/api.md`,
    - `docs/tuning.md`,
    - `docs/troubleshooting.md`.

    **Done (2026-04-17).** Already completed as part of task 19. The
    trimmed README has all four links in a Documentation table (lines
    153–156) plus contextual links: `docs/installation.md` from the
    Installation section, `docs/api.md` from the entry-point table,
    differentiable-solving quickstart, and API summary, and
    `docs/tuning.md` / `docs/troubleshooting.md` in the table. No
    additional changes needed.

22. Resolve current guidance inconsistencies.
    In particular, make `Makefile` install behavior line up with the documented "standard install" story, or document clearly why they differ.

    **Done (2026-04-17).** Audited all docs for inconsistencies and
    fixed three issues.

    **Makefile install target:**
    - `make install` changed from `pip install -e .` (editable) to
      `pip install .` (non-editable), matching the "standard install"
      documented in README and `docs/installation.md`.
    - `make install-dev` remains `pip install -e ".[dev]"` — matches
      the documented "development install."
    - Added a comment to the Makefile target clarifying the match.

    **`docs/api.md` adjoint parameter order:**
    - Swapped `gmres_restart` and `gmres_maxiter` rows in the adjoint
      parameters table so the order is tol → maxiter → restart,
      matching the forward solver's `krylov_tol` → `krylov_maxiter` →
      `krylov_restart` order.
    - The different naming (`krylov_*` vs `gmres_*`) is by design —
      forward and adjoint are separate solver instances with separate
      tuning. Not changed.

    **`docs/troubleshooting.md` cross-reference link styles:**
    - Line 24: changed `[api.md](api.md#input-validation)` to
      `[Input validation](api.md#input-validation) in \`docs/api.md\``
    - Line 61: changed `` [`docs/tuning.md`](tuning.md#...) `` to
      `[Continuation schedule](tuning.md#...) in \`docs/tuning.md\``
    - Both now use descriptive link text + explicit file path.

    **No other inconsistencies found:** parameter defaults, function
    names, and API descriptions are consistent across all five files.

23. Do a final consistency pass so one source of truth exists for each topic.

    **Done (2026-04-17).** Audited all ten documentation topics across
    five files for single-source-of-truth compliance.

    **One duplication fixed:**
    - Float64 requirement was substantively duplicated in three places
      (README, `docs/installation.md`, `docs/troubleshooting.md` —
      same code block and explanation in each).
    - `docs/installation.md` is the authoritative source (includes
      the placement guidance and consequence of forgetting).
    - README keeps its 2-line snippet (quickstart-essential — users
      need it to get started).
    - `docs/troubleshooting.md` §Float32 section trimmed from a full
      code block to a one-line fix + link to `installation.md#float64`.

    **Audit result — all ten topics have clear ownership:**

    | Topic | Authoritative source | Other files |
    |---|---|---|
    | Installation | `docs/installation.md` | README: brief snippet + link |
    | Float64 | `docs/installation.md` | README: quickstart snippet; troubleshooting: link only (fixed) |
    | Function signatures | `docs/api.md` §Function signatures | README: link only |
    | Entry-point comparison | `docs/api.md` §Choosing + §Comparing | README: brief table + link |
    | Parameter tables | `docs/api.md` §Common/Forward/Adjoint | No duplication |
    | Input validation | `docs/api.md` §Input validation | troubleshooting: symptom→fix + link |
    | Tuning guidance | `docs/tuning.md` | troubleshooting: brief cross-ref |
    | Troubleshooting | `docs/troubleshooting.md` | No duplication |
    | Data types | `docs/api.md` §Data types | No duplication |
    | Low-level building blocks | `docs/api.md` §Low-level | README: math (complementary, not duplicative) |

## Phase 5: Simplify `return_aux` implementation

24. Refactor `make_mcp_solver_diff` so the `custom_vjp` logic is defined once.

    **Done (2026-04-17).** Replaced the two-branch `if return_aux` /
    `else` custom_vjp definition with a single `_core_solve` +
    `_core_fwd` + `_core_bwd` triple.

    **Before:** two `@custom_vjp` definitions (lines 258–292), one for
    `return_aux=True` returning `(x_star, SolveInfo)` and one for
    `return_aux=False` returning `x_star`. Each had its own `_fwd` and
    `_bwd`. The `_bwd` functions differed only in how they unpacked the
    cotangent: `(g_x, _g_aux)` vs bare `cotangent`.

    **After:** one `@custom_vjp` on `_core_solve` that always returns
    `(x_star, mu_used_sg, num_steps_sg)` — the raw forward results
    with `mu_used` and `num_steps` stop-gradiented. One `_core_fwd`,
    one `_core_bwd` (extracts `g_x = cotangent[0]`). A plain `solve`
    function wraps `_core_solve` and optionally packages aux via
    `_make_aux` outside the custom_vjp boundary.

    **Why this works:** `_make_aux` is called outside the custom_vjp,
    so JAX's auto-diff traces through it, but all SolveInfo fields are
    stop-gradiented — zero gradient flows through the aux path.
    Gradients for `x_star` flow exclusively through `_core_bwd`'s
    implicit differentiation, exactly as before.

    **Regression checks:**
    - `pytest tests/ -q` → `183 passed in 327.96s`.
    - `ruff check smooth_mcp/diff.py` → clean.
    - `mypy smooth_mcp/` → clean.

25. Isolate aux packaging from the forward/backward solver rules.
    The implementation goal is:
    - one forward path,
    - one backward path,
    - one aux construction path.

    **Done (2026-04-17).** Already achieved by task 24's refactor. The
    current structure satisfies all three requirements:

    - **One forward path:** `_run_forward` (line 164). Called from
      `_core_solve` (primal) and `_core_fwd` (forward rule). No other
      forward-solve code exists.
    - **One backward path:** `_core_bwd` → `_compute_grads` (lines
      280, 195). Single definition, extracts `g_x = cotangent[0]`
      regardless of whether aux was returned.
    - **One aux construction path:** `_make_aux` (line 244). Called
      only from `solve` (outside the custom_vjp boundary), only when
      `return_aux=True`.

    The remaining `if return_aux` branches in `solve` (line 289) and
    `_poisoned_solve` (line 307) are thin output-shape wrappers — they
    decide whether to package the aux tuple, not how the forward,
    backward, or aux construction works. No logic is duplicated.

    No code changes needed beyond task 24.

26. Re-run the full gradient and JIT tests after the refactor.
    Pay special attention to:
    - `return_aux=False`,
    - `return_aux=True`,
    - jitted gradients,
    - truncated continuation.

    **Done (2026-04-17).** All four concern areas verified after the
    task 24 refactor.

    **Targeted test suites:**
    - `pytest tests/test_gradients.py tests/test_strict_validation.py
      tests/test_convergence.py -v` → `107 passed in 293.48s`.
    - Full suite (`pytest tests/ -q`) → `183 passed in 327.96s`
      (run during task 24).

    **Manual smoke test covering all four concern areas:**
    - `return_aux=False`: forward solve returns correct `x`, finite.
    - `return_aux=True`: forward solve returns correct `x` +
      `SolveInfo` with `converged=True`, `num_steps=35`.
    - `jit(grad)` with `return_aux=False`: finite gradients.
    - `jit(grad)` with `return_aux=True`: finite gradients, match
      the `return_aux=False` gradients exactly (`jnp.allclose`).
    - Truncated continuation (`max_mu_steps=2`): `converged=False`,
      `num_steps=2`, `jit(grad)` produces finite gradients.

## Phase 6: Add missing `vmap` coverage

27. Add explicit tests for `jax.vmap` over the differentiable solver.

    **Done (2026-04-17).** New file: `tests/test_vmap.py` with 5 tests
    in `TestVmapDiffSolver` (return_aux=False).

    Coverage:
    - `test_vmap_over_theta`: batch over 3 different theta values with
      fixed bounds. Checks shape, finiteness, and parity with
      unbatched calls.
    - `test_vmap_over_bounds`: batch over 3 different lower bounds.
    - `test_vmap_over_all_inputs`: batch over l, u, x0, theta
      simultaneously (4 batch elements).
    - `test_jit_vmap`: `jax.jit(jax.vmap(solver))` composes; second
      call reuses compiled graph and matches first call.
    - `test_vmap_1d_problem`: 1D scalar problem (F(x)=x-theta, l=0)
      with 3-element batch. Verifies solution is `max(theta, 0)`.

    **Results:**
    - `pytest tests/test_vmap.py -v` → `5 passed in 30.32s`.
    - `ruff check tests/test_vmap.py` → clean.

28. Add explicit tests for `jax.vmap` with `return_aux=True`.

    **Done (2026-04-17).** Added `TestVmapDiffSolverReturnAux` class
    to `tests/test_vmap.py` with 5 tests.

    Coverage:
    - `test_vmap_return_aux_shape`: verifies `(x_star, SolveInfo)`
      structure with batch dim on all four SolveInfo fields.
    - `test_vmap_return_aux_converged`: all batch elements converge,
      residual_norm < 1e-8.
    - `test_vmap_return_aux_parity_with_unbatched`: per-element parity
      with unbatched calls on x, residual_norm, num_steps, converged.
    - `test_jit_vmap_return_aux`: `jit(vmap(solver))` with
      return_aux=True composes.
    - `test_vmap_return_aux_truncated`: `max_mu_steps=2` reports
      per-element `converged=False` and `num_steps=2`.

    **Results:**
    - `pytest tests/test_vmap.py -v` → `10 passed in 66.77s`
      (5 from task 27 + 5 new).
    - `ruff check tests/test_vmap.py` → clean.

29. Add at least one batched gradient test that confirms shapes and finiteness under `vmap`.

    **Done (2026-04-17).** Added `TestVmapBatchedGradients` class to
    `tests/test_vmap.py` with 4 tests.

    Coverage:
    - `test_vmap_grad_over_theta`: `vmap(grad(loss))` over 3 theta
      values. Checks shape, finiteness, and parity with unbatched
      `grad` per-element.
    - `test_jit_vmap_grad`: `jit(vmap(grad(loss)))` composes. Shape
      and finiteness.
    - `test_grad_of_vmapped_loss`: `grad` of a loss that internally
      uses `vmap(solver)` — the "outer grad, inner vmap" pattern.
      Shape and finiteness on a 1D scalar problem.
    - `test_vmap_grad_with_return_aux`: `vmap(grad(loss))` with
      `return_aux=True`. Verifies finiteness and exact match with
      `return_aux=False` gradients.

    **Results:**
    - `pytest tests/test_vmap.py::TestVmapBatchedGradients -v` →
      `4 passed in 110.31s`.
    - Full vmap file: `14 passed` (5 + 5 + 4).
    - `ruff check tests/test_vmap.py` → clean.

30. If any batching constraints exist, document them clearly in `docs/api.md` and `README.md`.

    **Done (2026-04-17).** One batching constraint exists and is
    already documented.

    **Known constraint:** `strict_validation="checkify"` requires
    `vmap(solver)` ordering, not `checkify(vmap(...))`. This is a JAX
    limitation (`lax.while_loop` rejects checkify-of-vmap-of-while).
    Already documented in `docs/api.md` §strict_validation="checkify"
    (lines 263–274) with correct/incorrect code examples and the
    `vmap(solver)` workaround. Also tested in
    `tests/test_strict_validation.py::test_checkify_of_vmap_raises_at_trace`.

    **No other constraints found.** Tasks 27–29 confirmed that all
    standard vmap patterns work without restrictions:
    - `vmap(solver)` — both `return_aux` modes
    - `jit(vmap(solver))` — both `return_aux` modes
    - `vmap(grad(loss))` — both `return_aux` modes
    - `jit(vmap(grad(loss)))` — composes
    - `grad(vmapped_loss)` — outer grad, inner vmap

    **README** (line 147) mentions `jax.vmap` support and links to
    `docs/api.md` for caveats. No additional documentation needed.

## Phase 7: Final verification and housekeeping

31. Re-run:
    - `pytest`,
    - `ruff`,
    - `mypy`,
    - `benchmarks/bench_solve.py`,
    - `benchmarks/bench_continuation.py`.

    **Done (2026-04-17).** All five checks pass.

    - `pytest tests/ -q` → **197 passed** in 724.03s (183 prior +
      14 new vmap tests from tasks 27–29).
    - `ruff check smooth_mcp tests demos benchmarks` → **clean**.
    - `mypy smooth_mcp demos` → **clean** (15 source files).
    - `benchmarks/bench_solve.py` → runs cleanly. Key numbers:
      `solve_mcp` eager 1652 ms, `make_mcp_solver(jit)` 1.7 ms
      (970x), `make_mcp_solver_diff(jit)` 1.4 ms, `jit(grad)` warm
      1.7 ms (2074x vs eager grad).
    - `benchmarks/bench_continuation.py` → runs cleanly. All 6
      problems × 3 mu_decay values converge. Step counts and
      convergence behavior identical to the task 14 baseline. Wall
      times vary by run (machine load) but relative ordering is
      consistent — GMRES `obstacle_100d` still shows the dramatic
      "fewer steps = slower" effect (mu_decay=0.1 is ~2x slower
      than 0.5).

32. Update any internal benchmark notes and review artifacts after the changes so they reflect the new state rather than the pre-fix behavior.

    **Done (2026-04-17).** Two artifacts updated.

    **`docs/internal/benchmarks/benchmark-2026-04-14.md`:**
    - Solver performance table updated with task 31 bench_solve
      numbers. Now includes all 8 benchmark paths: `solve_mcp` eager,
      `make_mcp_solver` eager/jit, `make_mcp_solver_diff` eager/jit,
      `jax.grad` eager, `jit(grad)` first/cached. Previously only had
      5 rows and was missing `make_mcp_solver` entirely.
    - Added "Last updated: 2026-04-17" and a note that absolute wall
      times vary between runs.
    - Continuation section was already updated in task 14.

    **`docs/internal/reviews/review.codex.2026-04-16.md`:**
    - Added "Resolution Status (2026-04-17)" appendix with a 6-row
      table mapping each finding to its resolution (phase, mechanism,
      test coverage, doc location).
    - Final state: 197 tests, ruff clean, mypy clean, both benchmarks
      pass.

    **`docs/internal/benchmarks/benchmark-2026-04-13.md`:**
    - Not updated — historical record from a different machine (Apple
      Silicon). Kept as-is for provenance.
