"""Dtype-policy tests.

Remediation plan todo.2026-04-23.md #11 adopted Option C:
float32 execution is not runtime-rejected, but it is also not
validated — the test suite (via ``tests/conftest.py``) globally
enables x64 so float64 is the only tested configuration. These
tests cover the execution path at float32 to confirm the solver
still builds, runs, and produces finite output without silently
upcasting. Numerical accuracy is intentionally not asserted.
"""

import jax
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver, make_mcp_solver_diff, solve_mcp


def _F(x, theta):
    return x - theta


def _f32(*values):
    return jnp.array(list(values), dtype=jnp.float32)


# Easy problem: F(x) = x - 2 on [0, 3] with x0 = 1. Interior root x* = 2.
# Chosen so the solver converges comfortably within float32 precision.
_L = jnp.array([0.0], dtype=jnp.float32)
_U = jnp.array([3.0], dtype=jnp.float32)
_X0 = jnp.array([1.0], dtype=jnp.float32)
_THETA = jnp.array([2.0], dtype=jnp.float32)


class TestFloat32ExecutionPath:
    def test_solve_mcp_runs_at_float32(self):
        """solve_mcp preserves float32 dtype end-to-end and returns a
        finite MCPResult. No dtype upcast, no NaN output."""

        def F(x):
            return x - jnp.array([2.0], dtype=jnp.float32)

        result = solve_mcp(F, _L, _U, _X0)
        assert result.x.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(result.x)))
        assert jnp.isfinite(result.residual_norm)

    def test_make_mcp_solver_runs_at_float32(self):
        """The forward factory preserves float32 end-to-end including
        SolveInfo fields."""
        solver = make_mcp_solver(_F, return_aux=True)
        x, info = solver(_L, _U, _X0, _THETA)
        assert x.dtype == jnp.float32
        assert info.mu_used.dtype == jnp.float32
        assert info.residual_norm.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(x)))

    def test_make_mcp_solver_diff_runs_at_float32(self):
        """The diff factory runs at float32 and returns a finite x."""
        solver = make_mcp_solver_diff(_F)
        x = solver(_L, _U, _X0, _THETA)
        assert x.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(x)))

    def test_gradient_path_at_float32(self):
        """jax.grad through make_mcp_solver_diff produces a finite
        float32 gradient. This exercises the adjoint GMRES solve at
        float32 — the path most likely to degrade per docs/installation.md,
        but one the solver does still execute."""
        solver = make_mcp_solver_diff(_F)

        def loss(t):
            return jnp.sum(solver(_L, _U, _X0, t) ** 2)

        grad = jax.grad(loss)(_THETA)
        assert grad.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(grad)))
