"""Tests for the 1D-only solver-state shape contract.

The public API accepts only 1D vector state for ``l``, ``u``, and ``x0``.
Higher-rank inputs are rejected at the public boundary with a clear
``ValueError`` rather than falling through into JAX internals as an
opaque JVP shape error.
"""

import jax
import jax.numpy as jnp
import pytest

from smooth_mcp import (
    make_mcp_solver,
    make_mcp_solver_diff,
    preflight_validate,
    solve_mcp,
)


def _F(x, theta):
    return x - theta


# ---------------------------------------------------------------------------
# 1D input is accepted (happy path)
# ---------------------------------------------------------------------------


class TestOneDimensionalAccepted:
    def test_solve_mcp_1d(self):
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)
        theta = jnp.array([1.0, 2.0])
        result = solve_mcp(_F, l, u, x0, theta)
        assert result.x.shape == (2,)
        assert result.converged

    def test_make_mcp_solver_1d(self):
        solver = make_mcp_solver(_F)
        x = solver(
            jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(2), jnp.array([1.0, 2.0])
        )
        assert x.shape == (2,)

    def test_make_mcp_solver_diff_1d(self):
        solver = make_mcp_solver_diff(_F)
        x = solver(
            jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(2), jnp.array([1.0, 2.0])
        )
        assert x.shape == (2,)

    def test_preflight_validate_1d(self):
        preflight_validate(jnp.zeros(2), jnp.ones(2), jnp.zeros(2))


# ---------------------------------------------------------------------------
# Higher-rank input is rejected at the public API boundary
# ---------------------------------------------------------------------------


class TestHigherRankRejected:
    def _bad_2d(self):
        l = jnp.zeros((2, 1))
        u = jnp.full((2, 1), jnp.inf)
        x0 = jnp.zeros((2, 1))
        theta = jnp.array([1.0, 2.0]).reshape(2, 1)
        return l, u, x0, theta

    def _bad_3d(self):
        l = jnp.zeros((2, 1, 1))
        u = jnp.full((2, 1, 1), jnp.inf)
        x0 = jnp.zeros((2, 1, 1))
        theta = jnp.zeros((2, 1, 1))
        return l, u, x0, theta

    def test_solve_mcp_rejects_2d(self):
        l, u, x0, theta = self._bad_2d()
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            solve_mcp(_F, l, u, x0, theta)

    def test_solve_mcp_rejects_3d(self):
        l, u, x0, theta = self._bad_3d()
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            solve_mcp(_F, l, u, x0, theta)

    def test_solve_mcp_rejects_scalar(self):
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            solve_mcp(
                _F,
                jnp.array(0.0),
                jnp.array(1.0),
                jnp.array(0.5),
                jnp.array(1.0),
            )

    def test_make_mcp_solver_rejects_2d_eager(self):
        solver = make_mcp_solver(_F)
        l, u, x0, theta = self._bad_2d()
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            solver(l, u, x0, theta)

    def test_make_mcp_solver_rejects_2d_jit(self):
        solver = jax.jit(make_mcp_solver(_F))
        l, u, x0, theta = self._bad_2d()
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            solver(l, u, x0, theta)

    def test_make_mcp_solver_diff_rejects_2d_eager(self):
        solver = make_mcp_solver_diff(_F)
        l, u, x0, theta = self._bad_2d()
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            solver(l, u, x0, theta)

    def test_make_mcp_solver_diff_rejects_2d_jit(self):
        solver = jax.jit(make_mcp_solver_diff(_F))
        l, u, x0, theta = self._bad_2d()
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            solver(l, u, x0, theta)

    def test_preflight_validate_rejects_2d(self):
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            preflight_validate(jnp.zeros((2, 1)), jnp.ones((2, 1)), jnp.zeros((2, 1)))

    def test_error_names_the_offending_arg(self):
        # x0 is the 2D one; the message should identify it.
        with pytest.raises(ValueError, match=r"x0 must be a 1D array"):
            solve_mcp(
                _F,
                jnp.zeros(2),
                jnp.full(2, jnp.inf),
                jnp.zeros((2, 1)),
                jnp.array([1.0, 2.0]),
            )


# ---------------------------------------------------------------------------
# vmap is unaffected: batching adds a leading axis that vmap consumes,
# so the body still sees 1D state.
# ---------------------------------------------------------------------------


class TestVmapStillWorks:
    def test_vmap_over_batch_of_1d_problems(self):
        solver = make_mcp_solver_diff(_F)
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)
        thetas = jnp.array([[1.0, 2.0], [0.5, 1.5], [3.0, 4.0]])
        batched = jax.vmap(solver, in_axes=(None, None, None, 0))
        xs = batched(l, u, x0, thetas)
        assert xs.shape == (3, 2)
        # Solutions are max(theta, 0) for the identity-minus-theta map.
        assert jnp.allclose(xs, jnp.maximum(thetas, 0.0))


# ---------------------------------------------------------------------------
# vmap rejection: when the batched input makes the per-body state rank >= 2,
# the boundary check must still fire inside the vmapped body.
# ---------------------------------------------------------------------------


class TestVmapRejects2DBody:
    def _rank3_batch(self):
        # Shape (B=3, N=2, M=1): vmap over axis 0 leaves (2, 1) inside the body.
        l = jnp.zeros((3, 2, 1))
        u = jnp.full((3, 2, 1), jnp.inf)
        x0 = jnp.zeros((3, 2, 1))
        theta = jnp.zeros((3, 2, 1))
        return l, u, x0, theta

    def test_vmap_forward_rejects_2d_body(self):
        solver = make_mcp_solver(_F)
        batched = jax.vmap(solver, in_axes=(0, 0, 0, 0))
        l, u, x0, theta = self._rank3_batch()
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            batched(l, u, x0, theta)

    def test_vmap_diff_rejects_2d_body(self):
        solver = make_mcp_solver_diff(_F)
        batched = jax.vmap(solver, in_axes=(0, 0, 0, 0))
        l, u, x0, theta = self._rank3_batch()
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            batched(l, u, x0, theta)

    def test_jit_of_vmap_rejects_2d_body(self):
        solver = make_mcp_solver_diff(_F)
        batched = jax.jit(jax.vmap(solver, in_axes=(0, 0, 0, 0)))
        l, u, x0, theta = self._rank3_batch()
        with pytest.raises(ValueError, match=r"must be a 1D array"):
            batched(l, u, x0, theta)
