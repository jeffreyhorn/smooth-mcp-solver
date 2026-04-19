"""Parity tests for `make_mcp_solver` vs `solve_mcp`.

The forward-only factory must produce field-level matches with the eager
`solve_mcp` path across a range of problems and configurations.
"""

import jax
import jax.numpy as jnp
import pytest

from smooth_mcp import make_mcp_solver, solve_mcp


def _assert_parity(result, x, info, atol=1e-10, residual_rtol=1e-3):
    """Compare MCPResult (eager) against (x_star, SolveInfo) (factory)."""
    assert jnp.allclose(result.x, x, atol=atol)
    assert int(info.num_steps) == result.num_steps
    assert bool(info.converged) == result.converged
    assert jnp.isclose(info.residual_norm, result.residual_norm, rtol=residual_rtol)


# ---------------------------------------------------------------------------
# Converged problems
# ---------------------------------------------------------------------------


class TestConvergedProblems:
    def test_lcp_2d(self):
        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)

        result = solve_mcp(F, l, u, x0)
        solver = make_mcp_solver(F, return_aux=True)
        x, info = solver(l, u, x0, jnp.zeros(0))

        _assert_parity(result, x, info)

    def test_nonlinear_1d(self):
        def F(x):
            return x**3 - x - 1.0

        l = jnp.array([0.0])
        u = jnp.array([2.0])
        x0 = jnp.array([1.0])

        result = solve_mcp(F, l, u, x0)
        solver = make_mcp_solver(F, return_aux=True)
        x, info = solver(l, u, x0, jnp.zeros(0))

        _assert_parity(result, x, info)

    def test_finite_bounds_clipping(self):
        """Problem where solution sits at the upper bound."""

        def F(x):
            return x - jnp.array([5.0])  # unconstrained root at 5

        l = jnp.array([0.0])
        u = jnp.array([1.0])  # force clipping to upper bound
        x0 = jnp.array([0.5])

        result = solve_mcp(F, l, u, x0)
        solver = make_mcp_solver(F, return_aux=True)
        x, info = solver(l, u, x0, jnp.zeros(0))

        _assert_parity(result, x, info)
        assert jnp.isclose(x[0], 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Parametric problems (F takes theta)
# ---------------------------------------------------------------------------


class TestParametricProblems:
    def test_parametric_lcp(self):
        def F(x, theta):
            M = theta.reshape(2, 2)
            return M @ x + jnp.array([1.0, -2.0])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)
        theta = jnp.array([2.0, -1.0, -1.0, 3.0])

        result = solve_mcp(F, l, u, x0, theta=theta)
        solver = make_mcp_solver(F, return_aux=True)
        x, info = solver(l, u, x0, theta)

        _assert_parity(result, x, info)

    def test_parametric_scaling(self):
        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        result = solve_mcp(F, l, u, x0, theta=theta)
        solver = make_mcp_solver(F, return_aux=True)
        x, info = solver(l, u, x0, theta)

        _assert_parity(result, x, info)


# ---------------------------------------------------------------------------
# Truncated continuation
# ---------------------------------------------------------------------------


class TestTruncatedContinuation:
    def _problem(self):
        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        return F, jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(2)

    def test_max_mu_steps_1(self):
        F, l, u, x0 = self._problem()
        result = solve_mcp(F, l, u, x0, max_mu_steps=1)
        solver = make_mcp_solver(F, max_mu_steps=1, return_aux=True)
        x, info = solver(l, u, x0, jnp.zeros(0))

        _assert_parity(result, x, info)
        assert int(info.num_steps) == 1

    def test_max_mu_steps_2(self):
        F, l, u, x0 = self._problem()
        result = solve_mcp(F, l, u, x0, max_mu_steps=2)
        solver = make_mcp_solver(F, max_mu_steps=2, return_aux=True)
        x, info = solver(l, u, x0, jnp.zeros(0))

        _assert_parity(result, x, info)


# ---------------------------------------------------------------------------
# Linear solver variants
# ---------------------------------------------------------------------------


class TestLinearSolvers:
    def _problem(self):
        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        return F, jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(2)

    def test_dense_parity(self):
        F, l, u, x0 = self._problem()
        result = solve_mcp(F, l, u, x0, linear_solver="dense")
        solver = make_mcp_solver(F, linear_solver="dense", return_aux=True)
        x, info = solver(l, u, x0, jnp.zeros(0))
        _assert_parity(result, x, info)

    def test_gmres_parity(self):
        F, l, u, x0 = self._problem()
        result = solve_mcp(F, l, u, x0, linear_solver="gmres")
        solver = make_mcp_solver(F, linear_solver="gmres", return_aux=True)
        x, info = solver(l, u, x0, jnp.zeros(0))
        # GMRES is iterative, looser residual tolerance.
        _assert_parity(result, x, info, atol=1e-6, residual_rtol=1e-2)


# ---------------------------------------------------------------------------
# JIT composability
# ---------------------------------------------------------------------------


class TestJitComposability:
    def test_jit_matches_eager(self):
        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        solver = make_mcp_solver(F, return_aux=True)
        x_eager, info_eager = solver(l, u, x0, theta)
        x_jit, info_jit = jax.jit(solver)(l, u, x0, theta)

        assert jnp.allclose(x_eager, x_jit, atol=1e-10)
        assert bool(info_eager.converged) == bool(info_jit.converged)
        assert int(info_eager.num_steps) == int(info_jit.num_steps)
        assert jnp.allclose(info_eager.residual_norm, info_jit.residual_norm, rtol=1e-6)

    def test_jit_reuses_compiled_graph(self):
        """Second call with matching shapes should reuse the compiled kernel."""

        def F(x):
            return x - jnp.array([2.0])

        solver = jax.jit(make_mcp_solver(F))
        x1 = solver(jnp.array([0.0]), jnp.array([3.0]), jnp.array([1.0]), jnp.zeros(0))
        x2 = solver(jnp.array([-1.0]), jnp.array([5.0]), jnp.array([0.0]), jnp.zeros(0))
        assert jnp.isclose(x1[0], 2.0, atol=1e-10)
        assert jnp.isclose(x2[0], 2.0, atol=1e-10)


# ---------------------------------------------------------------------------
# return_aux=False shape
# ---------------------------------------------------------------------------


class TestReturnShape:
    def test_returns_array_when_return_aux_false(self):
        def F(x):
            return x - jnp.array([2.0])

        solver = make_mcp_solver(F)
        out = solver(jnp.array([0.0]), jnp.array([3.0]), jnp.array([1.0]), jnp.zeros(0))
        assert hasattr(out, "shape")
        assert jnp.asarray(out).shape == (1,)

    def test_returns_tuple_when_return_aux_true(self):
        def F(x):
            return x - jnp.array([2.0])

        solver = make_mcp_solver(F, return_aux=True)
        out = solver(jnp.array([0.0]), jnp.array([3.0]), jnp.array([1.0]), jnp.zeros(0))
        assert isinstance(out, tuple)
        assert len(out) == 2
        x, info = out
        assert x.shape == (1,)
        assert hasattr(info, "mu_used")
        assert hasattr(info, "num_steps")
        assert hasattr(info, "residual_norm")
        assert hasattr(info, "converged")


# ---------------------------------------------------------------------------
# mu_used semantics (no solve_mcp counterpart)
# ---------------------------------------------------------------------------


class TestMuUsed:
    def test_mu_used_finite_and_bounded(self):
        def F(x):
            return x - jnp.array([2.0])

        solver = make_mcp_solver(F, return_aux=True)
        _, info = solver(
            jnp.array([0.0]), jnp.array([3.0]), jnp.array([1.0]), jnp.zeros(0)
        )
        assert jnp.isfinite(info.mu_used)
        assert float(info.mu_used) <= 1.0  # mu_init default
        assert float(info.mu_used) > 0.0

    def test_mu_used_equals_mu_min_on_full_convergence(self):
        """When converged, terminal mu is clamped to mu_min."""

        def F(x):
            return x - jnp.array([2.0])

        solver = make_mcp_solver(F, mu_min=1e-10, return_aux=True)
        _, info = solver(
            jnp.array([0.0]), jnp.array([3.0]), jnp.array([1.0]), jnp.zeros(0)
        )
        assert bool(info.converged) is True
        assert jnp.isclose(info.mu_used, 1e-10, rtol=1e-6)


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


class TestFactoryValidation:
    def test_invalid_mu_init(self):
        with pytest.raises(ValueError, match="mu_init must be positive"):
            make_mcp_solver(lambda x: x, mu_init=0.0)

    def test_invalid_mu_min(self):
        with pytest.raises(ValueError, match="mu_min must be positive"):
            make_mcp_solver(lambda x: x, mu_min=0.0)

    def test_mu_min_greater_than_init(self):
        with pytest.raises(ValueError, match="mu_min must be <= mu_init"):
            make_mcp_solver(lambda x: x, mu_init=0.1, mu_min=1.0)

    def test_invalid_mu_decay(self):
        with pytest.raises(ValueError, match="mu_decay must be in"):
            make_mcp_solver(lambda x: x, mu_decay=1.5)

    def test_invalid_max_mu_steps(self):
        with pytest.raises(ValueError, match="max_mu_steps must be >= 1"):
            make_mcp_solver(lambda x: x, max_mu_steps=0)

    def test_invalid_newton_tol(self):
        with pytest.raises(ValueError, match="newton_tol must be non-negative"):
            make_mcp_solver(lambda x: x, newton_tol=-1.0)

    def test_invalid_regularize(self):
        with pytest.raises(ValueError, match="regularize must be non-negative"):
            make_mcp_solver(lambda x: x, regularize=-0.1)

    def test_invalid_strict_validation(self):
        with pytest.raises(ValueError, match="strict_validation must be"):
            make_mcp_solver(lambda x: x, strict_validation="oops")


# ---------------------------------------------------------------------------
# Eager runtime validation (shape, NaN, ordering)
# ---------------------------------------------------------------------------


class TestRuntimeValidation:
    def _solver(self):
        return make_mcp_solver(lambda x: x)

    def test_l_greater_than_u(self):
        with pytest.raises(ValueError, match="Lower bounds must not exceed"):
            self._solver()(
                jnp.array([5.0]), jnp.array([3.0]), jnp.array([4.0]), jnp.zeros(0)
            )

    def test_mismatched_lu_shape(self):
        with pytest.raises(ValueError, match="same shape"):
            self._solver()(jnp.zeros(2), jnp.zeros(3), jnp.zeros(2), jnp.zeros(0))

    def test_mismatched_x0_shape(self):
        with pytest.raises(ValueError, match="same shape as l"):
            self._solver()(
                jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(3), jnp.zeros(0)
            )

    def test_nan_l(self):
        with pytest.raises(ValueError, match="must not contain NaN"):
            self._solver()(
                jnp.array([jnp.nan]),
                jnp.array([1.0]),
                jnp.array([0.5]),
                jnp.zeros(0),
            )

    def test_nan_u(self):
        with pytest.raises(ValueError, match="must not contain NaN"):
            self._solver()(
                jnp.array([0.0]),
                jnp.array([jnp.nan]),
                jnp.array([0.5]),
                jnp.zeros(0),
            )

    def test_nan_x0(self):
        with pytest.raises(ValueError, match="x0 must not contain NaN"):
            self._solver()(
                jnp.array([0.0]),
                jnp.array([1.0]),
                jnp.array([jnp.nan]),
                jnp.zeros(0),
            )


# ---------------------------------------------------------------------------
# jax.vmap support
# ---------------------------------------------------------------------------


class TestForwardVmap:
    """Batching over the forward-only factory via ``jax.vmap``.

    The forward factory documents ``jax.vmap`` support; these tests pin
    the contract on shapes, finiteness, per-batch parity, and composition
    with ``jax.jit``.
    """

    _L = jnp.zeros(2)
    _U = jnp.full(2, jnp.inf)
    _X0 = jnp.zeros(2)

    def _lcp_F(self, x, theta):
        M = theta.reshape(2, 2)
        return M @ x + jnp.array([1.0, -2.0])

    def test_vmap_over_theta(self):
        solver = make_mcp_solver(self._lcp_F)
        theta0 = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()
        thetas = jnp.stack([theta0, theta0 + 0.1, theta0 - 0.05])

        batched = jax.vmap(solver, in_axes=(None, None, None, 0))
        xs = batched(self._L, self._U, self._X0, thetas)

        assert xs.shape == (3, 2)
        assert jnp.all(jnp.isfinite(xs))

        # Per-row parity with unbatched calls
        for i in range(3):
            x_ref = solver(self._L, self._U, self._X0, thetas[i])
            assert jnp.allclose(xs[i], x_ref)

    def test_jit_vmap(self):
        solver = make_mcp_solver(self._lcp_F)
        theta0 = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()
        thetas = jnp.stack([theta0, theta0 + 0.1])
        ls = jnp.tile(self._L, (2, 1))
        us = jnp.tile(self._U, (2, 1))
        x0s = jnp.tile(self._X0, (2, 1))

        jit_vmap = jax.jit(jax.vmap(solver))
        xs = jit_vmap(ls, us, x0s, thetas)

        assert xs.shape == (2, 2)
        assert jnp.all(jnp.isfinite(xs))

        # Second call reuses the compiled graph and matches
        xs2 = jit_vmap(ls, us, x0s, thetas)
        assert jnp.allclose(xs, xs2)

    def test_vmap_return_aux_shape(self):
        solver = make_mcp_solver(self._lcp_F, return_aux=True)
        theta0 = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()
        thetas = jnp.stack([theta0, theta0 + 0.1, theta0 - 0.05])

        batched = jax.vmap(solver, in_axes=(None, None, None, 0))
        xs, infos = batched(self._L, self._U, self._X0, thetas)

        assert xs.shape == (3, 2)
        assert infos.mu_used.shape == (3,)
        assert infos.num_steps.shape == (3,)
        assert infos.residual_norm.shape == (3,)
        assert infos.converged.shape == (3,)
        assert jnp.all(infos.converged)
