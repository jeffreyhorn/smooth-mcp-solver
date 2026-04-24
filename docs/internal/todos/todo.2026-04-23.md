# Remediation Plan

Date: 2026-04-23

This plan addresses the issues documented in `docs/internal/reviews/review.codex.2026-04-23.md`.

## Phase 1: Fix the public contract and correctness gaps

1. Decide the supported solver-state shape contract.
   - Option A: explicitly support only 1D vectors.
   - Option B: support arbitrary shapes by flattening/unflattening consistently through forward and backward solves.

2. Implement that shape contract at the public API boundary.
   - If choosing 1D-only, raise a clear `ValueError` before entering JAX internals.
   - If choosing arbitrary shapes, refactor Jacobian, JVP, GMRES, and adjoint code to operate on flattened vectors and restore original shape on output.

3. Add regression tests for the chosen shape behavior.
   - Cover `solve_mcp`, `make_mcp_solver`, and `make_mcp_solver_diff`.
   - Cover eager, `jit`, and `vmap` where applicable.

4. Define the intended semantics of `mu_used`.
   - Separate "last solved mu" from "evaluation mu at which convergence/residual is measured".
   - Decide which one belongs in `SolveInfo`, and whether both should be exposed.

5. Correct the differentiable solver to use the intended `mu` in the backward pass.
   - If gradients should reflect the actual last solved smoothing level, feed that value through `_core_fwd` and `_compute_grads`.
   - If the intended behavior is evaluation at `mu_min`, rename the field/docs so the contract is honest.

6. Add targeted tests for `mu_used` semantics.
   - Include an early-stop case where convergence occurs before `mu_min` is reached.
   - Assert both diagnostics and gradient semantics for that case.

7. Harden the Armijo line-search implementation.
   - Recheck the sufficient-decrease condition after the backtracking loop.
   - If the condition still fails, do not silently accept the step.
   - Decide on failure behavior: keep current iterate, mark non-convergence, or surface an explicit failure signal.

8. Add line-search regression tests.
   - Test that accepted steps reduce the merit function.
   - Test the `max_ls_steps=0` contract explicitly.
   - Test budget exhaustion semantics.

## Phase 2: Repair documentation and user-facing guidance

9. Update `docs/api.md` to match the implementation.
   - Fix the `make_mcp_solver` `strict_validation` default.
   - Add the missing `strict_validation` row to the `make_mcp_solver_diff` argument table.
   - Clarify the supported shape contract.
   - Clarify `mu_used`, convergence, and gradient semantics.

10. Update `README.md` to remove overstatements.
   - Replace "ensures global convergence" with wording that matches the implemented Armijo behavior.
   - Fix the backward-pass description so it matches the actual `mu` semantics.
   - Clarify whether float64 is required or recommended.

11. Update `docs/installation.md` and `docs/troubleshooting.md` for the chosen dtype policy.
   - If float64 is required, document the runtime validation/warning behavior.
   - If float32 is allowed-but-risky, describe when and why it may fail.

## Phase 3: Strengthen testing and developer workflow

12. Add explicit dtype-policy tests.
   - Cover at least one float32 execution path.
   - Assert the intended behavior: warning, error, or supported lower-accuracy execution.

13. Improve demo verification.
   - Keep subprocess smoke tests if desired.
   - Add a smaller deterministic test layer that asserts key numeric outputs from representative demos instead of only exit code 0.

14. Add missing edge-case tests.
   - `precond` plumbing
   - multidimensional input rejection/support
   - line-search failure behavior
   - early-stop `mu_used` semantics

15. Fix the `Makefile` developer workflow mismatch.
   - Either change `test-fast` so it really skips only slow gradient tests, or change the comment/help text to say it skips the full gradient module.

16. Re-run the full validation matrix after the above changes.
   - `ruff`
   - `mypy`
   - `black --check`
   - targeted new regression tests
   - full `pytest`

## Phase 4: Reduce maintenance drift

17. Refactor shared factory plumbing out of `forward.py` and `diff.py`.
   - Extract shared forward-kernel construction.
   - Extract shared validation entry points.
   - Extract shared `return_aux`, NaN-poisoning, and checkify wrappers where possible.

18. Keep VJP-specific logic isolated in `diff.py`.
   - The backward linear solve and cotangent rules should remain the only major diff-specific pieces.

19. After the refactor, add parity tests that compare forward and diff wrapper behavior on shared options.
   - `return_aux`
   - `strict_validation`
   - diagnostics packaging

20. Add a lightweight review checklist for future API changes.
   - Update code
   - update docs tables
   - update README claims
   - update tests for eager/jit/vmap where relevant
