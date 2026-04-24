# API change checklist

Before merging a change that touches the public API (solver entry
points, public types, option tables, or user-visible behavior), walk
this list. It is short on purpose — it only names the checks that
past reviews have caught drift on.

## 1. Code reaches all three entry points

If the change touches solver behavior, make sure it applies to all
three entry points:

- `smooth_mcp/solver.py::solve_mcp`
- `smooth_mcp/forward.py::make_mcp_solver`
- `smooth_mcp/diff.py::make_mcp_solver_diff`

The two factories share plumbing via `smooth_mcp/_factory_common.py`;
most contract changes live there or in `smooth_mcp/_kernel.py`.

## 2. Option tables in `docs/api.md` match the code

For any new or changed factory argument:

- The argument name, default, and description in the factory's table
  in `docs/api.md` match the current signature.
- If the argument is shared across factories (e.g. `strict_validation`),
  both factory tables list it with the same wording.
- If the argument validation lives in `validate_solver_options` or
  `validate_adjoint_options`, the docstring in `docs/api.md` matches
  the accepted range.

The 2026-04-18 `strict_validation` migration drifted between code and
docs because only one factory's table was updated.

## 3. README claims still hold

The README is the first thing users read. Re-read these sections if
the change touches:

- **Algorithm overview** (Newton / Armijo / continuation wording) —
  behavioral changes to the line search or continuation should be
  reflected here, not only in docstrings.
- **Implicit differentiation** — claims about "gradients at X" should
  match what `_compute_grads` actually evaluates at.
- **Installation / float64** — if you change dtype behavior (enforce,
  warn, or silently accept), update the Installation section and the
  Troubleshooting entry together.

## 4. Types and diagnostics

If you change a field on `MCPResult` or `SolveInfo`:

- Update the docstring in `smooth_mcp/_types.py`.
- Update the corresponding table in `docs/api.md`.
- Check the README for any inline references to the field (e.g.
  `SolveInfo.mu_used`, `result.converged`).

## 5. Tests cover eager, jit, and vmap where relevant

Validation and shape-contract changes behave differently at each
transform level:

- **Eager** — most ValueError tests live here.
- **`jax.jit`** — traced validation path; shape-only checks still
  fire at trace time.
- **`jax.vmap`** — per-row body sees unbatched shapes; rank checks
  fire on those.

For any input-boundary change, add regression tests in all three
contexts. `tests/test_shape_contract.py` is the template for the
rank-check rollout.

## 6. Forward / diff parity

If the change affects a shared option (`return_aux`, `strict_validation`,
diagnostics packaging), extend `tests/test_factory_parity.py` with a
case that asserts both factories produce the same observable result.
Shared plumbing in `_factory_common.py` makes accidental drift rarer
but not impossible.

## 7. Demo comments match demo output

If the change alters which solution a representative demo converges
to, update both:

- The demo header comment stating the expected solution.
- The corresponding assertion in `tests/test_demos_numeric.py`.

The 2D NCP demo had a wrong "expected" comment for months because
nothing checked it until `test_demos_numeric.py` was added.

## 8. Full validation matrix before merge

At the end of any non-trivial branch, run:

```
ruff check smooth_mcp tests demos
mypy smooth_mcp demos
black --check smooth_mcp tests demos
pytest tests/ -m 'slow or not slow'
```

All four must be green. The slow-demo suite is gated behind `-m slow`
and defaults off — include it explicitly on the final validation run.
