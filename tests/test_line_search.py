"""Regression tests for the Armijo line-search guarantee.

Option A + X (review codex.2026-04-23 finding #3, remediation todo #7-#8):
- After backtracking, the Armijo sufficient-decrease condition is
  rechecked; if unmet, the step is rejected (alpha_effective=0,
  iterate unchanged).
- Applied uniformly, including max_ls_steps=0 (which means "try
  alpha=1 only, Armijo-checked" — not "disable line search").
- The merit function phi = 0.5 * ||H(x, mu)||^2 is never increased
  by an accepted Newton step.
"""

import jax.numpy as jnp

from smooth_mcp import solve_mcp
from smooth_mcp._kernel import make_newton_solver
from smooth_mcp.smoothing import smoothed_residual


# Stress problem: F(x) = exp(x) - 2 at x0 = -1.5, mu = 1.0. The
# unsmoothed Newton direction overshoots — the full step lands at
# x ~ 4.08 where phi jumps from ~1.51 to ~18.86. An unchecked full
# step (or a too-small backtracking budget) would increase phi.
def _F_exp(x, theta):
    return jnp.exp(x) - jnp.array([2.0])


_L = jnp.array([-2.0])
_U = jnp.array([2.0])
_X0_OVERSHOOT = jnp.array([-1.5])
_THETA = jnp.zeros(0)
_MU = jnp.array(1.0)


class TestMaxLsSteps0Contract:
    def test_armijo_failure_at_alpha_1_rejects_step(self):
        """max_ls_steps=0 tries alpha=1 Armijo-checked. On this
        problem alpha=1 drives phi from 1.5 to 18.9 — fails Armijo —
        so the step is rejected and Newton stalls at x0."""
        newton = make_newton_solver(_F_exp, _L, _U, _THETA, max_ls_steps=0)
        x_final = newton(_X0_OVERSHOOT, _MU)
        assert jnp.allclose(x_final, _X0_OVERSHOOT)

    def test_armijo_pass_at_alpha_1_accepts_step(self):
        """max_ls_steps=0 does NOT silently disable the line search.
        On an easy problem where alpha=1 satisfies Armijo, the step
        is accepted and Newton moves the iterate."""

        def F(x, theta):
            return x - jnp.array([2.0])

        newton = make_newton_solver(
            F, jnp.array([0.0]), jnp.array([3.0]), _THETA, max_ls_steps=0
        )
        x0 = jnp.array([1.0])
        x_final = newton(x0, _MU)
        # Step accepted: x moved substantially away from x0 toward the
        # smoothed solution at mu=1.
        assert not jnp.allclose(x_final, x0, atol=1e-3)


class TestBudgetExhaustion:
    def test_exhausted_budget_rejects_step(self):
        """All alphas in the backtracking budget can fail Armijo.
        With backtrack_rho=0.99 and max_ls_steps=2, the alphas tried
        are 1.0, 0.99, 0.9801 — all close to alpha=1 where phi is
        still ~18. Budget exhausted without sufficient decrease ->
        step rejected, Newton stalls at x0."""
        newton = make_newton_solver(
            _F_exp, _L, _U, _THETA, max_ls_steps=2, backtrack_rho=0.99
        )
        x_final = newton(_X0_OVERSHOOT, _MU)
        assert jnp.allclose(x_final, _X0_OVERSHOOT)

    def test_sufficient_budget_accepts_step(self):
        """max_ls_steps>=1 with default backtrack_rho=0.5 gives
        alpha=0.5 as the first backtrack, which satisfies Armijo on
        this problem (phi drops to ~1.30). Rejection fires only when
        the budget is genuinely too small — not spuriously."""
        newton = make_newton_solver(_F_exp, _L, _U, _THETA, max_ls_steps=1)
        x_final = newton(_X0_OVERSHOOT, _MU)
        # Newton proceeds past x0 toward the solution x* ~ 0.69.
        assert float(x_final[0]) > 0.0


class TestMonotoneMerit:
    def test_merit_is_non_increasing_after_newton_solve(self):
        """After a full Newton solve at a single mu, phi(x_final) does
        not exceed phi(x0). Every accepted step satisfies Armijo, and
        rejected steps leave phi unchanged — so the cumulative
        sequence can only decrease."""
        newton = make_newton_solver(_F_exp, _L, _U, _THETA, max_ls_steps=0)
        x_final = newton(_X0_OVERSHOOT, _MU)

        phi0 = 0.5 * jnp.sum(
            smoothed_residual(_X0_OVERSHOOT, _F_exp, _L, _U, _MU, _THETA) ** 2
        )
        phi_final = 0.5 * jnp.sum(
            smoothed_residual(x_final, _F_exp, _L, _U, _MU, _THETA) ** 2
        )
        assert float(phi_final) <= float(phi0) + 1e-12

    def test_public_api_stalls_gracefully_under_max_ls_0(self):
        """End-to-end: with max_ls_steps=0 on an overshoot-prone
        problem, solve_mcp reports converged=False with the iterate
        unchanged and residual finite — not a blown-up NaN or a merit
        function that spiraled upward."""
        result = solve_mcp(
            lambda x: jnp.exp(x) - jnp.array([2.0]),
            _L,
            _U,
            _X0_OVERSHOOT,
            max_ls_steps=0,
        )
        assert bool(result.converged) is False
        assert jnp.isfinite(result.residual_norm)
        assert jnp.allclose(result.x, _X0_OVERSHOOT)
