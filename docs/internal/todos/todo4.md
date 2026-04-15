# Todo 4: Address Review Findings from 2026-04-14

This plan addresses the shortcomings documented in `docs/internal/reviews/review.codex.2026-04-14.md`.

## Phase 1: Fix the highest-risk API inconsistency

1. Add failing tests for differentiable-solver runtime input validation.
   Cover at least:
   - `l > u`,
   - mismatched `l` / `u` shapes,
   - mismatched `x0` shape,
   - NaN bounds.

2. Extract the current eager validation rules from `solve_mcp` into a shared helper.
   The helper should own shape checks, NaN checks, and bound ordering checks so the rules cannot drift.

3. Decide how the differentiable API should expose validation.
   Preferred options:
   - an eager checked wrapper around the jittable core, or
   - a `validate_inputs=True` code path that is documented as eager-only.

4. Implement the chosen validation strategy for `make_mcp_solver_diff` users.
   Preserve the traceable core for `jax.jit`, but make the default non-jitted experience as friendly as `solve_mcp`.

5. Re-run the full test suite and add a short note to the README clarifying how validation behaves on the eager and jitted paths.

## Phase 2: Improve solver observability

6. Design an auxiliary-info API for the differentiable solver.
   Minimum useful fields:
   - `mu_used`,
   - `num_steps`,
   - final residual estimate,
   - optionally a `converged` flag.

7. Implement that API without breaking the existing `make_mcp_solver_diff` signature.
   Good options:
   - `make_mcp_solver_diff_with_aux`,
   - `return_info=True`,
   - or a separate eager diagnostic wrapper that shares the same kernel.

8. Add tests that verify the auxiliary diagnostics match `solve_mcp` on representative problems, including truncated continuation cases.

## Phase 3: Improve eager-path efficiency and tuning guidance

9. Add a benchmark matrix for continuation settings.
   Measure a few representative problems across:
   - `mu_decay=0.5`,
   - one or two faster schedules such as `0.25` and `0.1`,
   - any adaptive schedule candidate.

10. Decide whether to change defaults, add presets, or add an adaptive continuation heuristic.
    The goal is not "smallest step count at all costs"; the goal is a documented tradeoff between robustness and speed.

11. Document solver tuning guidance in user-facing docs.
    Explain when to adjust:
    - `mu_decay`,
    - `max_mu_steps`,
    - `linear_solver`,
    - `regularize`,
    - adjoint solver settings.

12. Re-run the benchmark harness and record dated, reproducible results with environment details.

## Phase 4: Clean up package structure

13. Create a dedicated private module for shared internals.
    Suggested names:
    - `smooth_mcp/_kernel.py`,
    - `smooth_mcp/_internal.py`.

14. Move shared helpers out of `solver.py`.
    At minimum:
    - `_normalize_F`,
    - `_make_newton_solver`,
    - `_make_continuation_solver`.

15. Update `solver.py` and `diff.py` to depend on the new internal module instead of each other.
    Keep public API wrappers small and clearly separated from shared mechanics.

16. Re-run mypy, ruff, and tests after the refactor to ensure the module split did not change behavior.

## Phase 5: Repair and expand user-facing documentation

17. Fix all stale README references immediately.
    Start with the broken `smooth_mcp/core.py` link.

18. Replace contributor-centric install guidance with clearer user guidance.
    The docs should distinguish:
    - editable local development installs,
    - standard local installs,
    - any JAX platform-specific setup notes that users need.

19. Create actual user-facing docs under `docs/`.
    Minimum set:
    - installation,
    - API reference,
    - solver tuning and diagnostics,
    - troubleshooting.

20. Separate user docs from internal project notes.
    Reviews, benchmark notes, and todo files should not be the primary contents of `docs/`.

21. Document the validation contract and diagnostic differences between `solve_mcp` and the differentiable solver.

## Phase 6: Make benchmark messaging more precise

22. Remove hard-coded universal speedup claims from executable output.
    Report measured numbers, environment, and date instead of canned ratios.

23. Update the README performance section to describe benchmark results as examples, not guarantees.

24. Keep benchmark notes reproducible.
    Include:
    - hardware,
    - JAX version,
    - dtype mode,
    - command used to reproduce.

## Phase 7: Finish the small tooling cleanup

25. Add `typecheck` to `.PHONY` in `Makefile`.

26. Do a final consistency pass across README, Makefile help text, benchmark docs, and public API docs so all published guidance matches the codebase.
