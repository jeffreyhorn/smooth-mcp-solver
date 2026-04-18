"""Tests for convergence failure behavior."""

import jax
import jax.numpy as jnp
import pytest

from smooth_mcp import make_mcp_solver, make_mcp_solver_diff, solve_mcp


class TestSingularJacobian:
    """Problems where the Jacobian is singular at the initial guess."""

    def test_symmetric_initial_guess_regularized(self):
        """2D NCP with symmetric F and x0 on the symmetry line.
        Jacobian is singular at x0=[1,1]. Default regularization (1e-12)
        prevents NaN and allows the solver to find a finite solution."""

        def F(x):
            x1, x2 = x[0], x[1]
            return jnp.array(
                [
                    x1**2 + x1 * x2 - 3 * x1 + 2,
                    x2**2 + x1 * x2 - 3 * x2 + 2,
                ]
            )

        x0 = jnp.array([1.0, 1.0])
        result = solve_mcp(F, jnp.zeros(2), jnp.full(2, jnp.inf), x0)
        assert jnp.all(jnp.isfinite(result.x)), f"Got non-finite: {result.x}"

    def test_symmetric_initial_guess_no_regularization(self):
        """Without regularization, singular Jacobian produces a worse result."""

        def F(x):
            x1, x2 = x[0], x[1]
            return jnp.array(
                [
                    x1**2 + x1 * x2 - 3 * x1 + 2,
                    x2**2 + x1 * x2 - 3 * x2 + 2,
                ]
            )

        x0 = jnp.array([1.0, 1.0])
        result_reg = solve_mcp(
            F, jnp.zeros(2), jnp.full(2, jnp.inf), x0, regularize=1e-12
        )
        result_noreg = solve_mcp(
            F, jnp.zeros(2), jnp.full(2, jnp.inf), x0, regularize=0.0
        )
        # Without regularization, the solver should do worse:
        # either non-finite, not converged, or converged with a worse residual
        reg_ok = result_reg.converged and jnp.all(jnp.isfinite(result_reg.x))
        noreg_ok = result_noreg.converged and jnp.all(jnp.isfinite(result_noreg.x))
        assert reg_ok, "Regularized solver should converge"
        assert (not noreg_ok) or (
            result_noreg.residual_norm > result_reg.residual_norm
        ), "Unregularized solver should be worse than regularized on singular Jacobian"

    def test_symmetric_problem_with_asymmetric_x0(self):
        """Same problem with asymmetric x0 to avoid singularity entirely."""

        def F(x):
            x1, x2 = x[0], x[1]
            return jnp.array(
                [
                    x1**2 + x1 * x2 - 3 * x1 + 2,
                    x2**2 + x1 * x2 - 3 * x2 + 2,
                ]
            )

        x0 = jnp.array([0.5, 1.5])
        result = solve_mcp(F, jnp.zeros(2), jnp.full(2, jnp.inf), x0)
        assert jnp.all(jnp.isfinite(result.x)), f"Got non-finite solution: {result.x}"


class TestInsufficientIterations:
    """Solver given too few iterations to converge."""

    def test_single_mu_step(self):
        """With max_mu_steps=1, solver does one Newton pass at mu=1.0.
        Result should be finite but not fully converged."""

        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        result = solve_mcp(
            F, jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(2), max_mu_steps=1
        )
        assert jnp.all(jnp.isfinite(result.x)), f"Non-finite with 1 mu step: {result.x}"
        assert result.num_steps == 1

    def test_zero_newton_iters(self):
        """With large newton_tol, the Newton loop exits immediately."""

        def F(x):
            return 2.0 * x - 1.0

        result = solve_mcp(
            F, jnp.array([0.0]), jnp.array([10.0]), jnp.array([5.0]), newton_tol=1e6
        )
        assert jnp.all(jnp.isfinite(result.x))


class TestDifficultProblems:
    """Problems that stress the solver."""

    def test_nearly_degenerate_bounds(self):
        """Very tight bounds: l=1.0, u=1.0+1e-8. The smoothing approximation
        (mu up to ~1e-12) is wider than the bound interval, so the solution
        may slightly violate bounds. We check it's finite and close."""

        def F(x):
            return x - 0.5

        result = solve_mcp(
            F, jnp.array([1.0]), jnp.array([1.0 + 1e-8]), jnp.array([1.0])
        )
        assert jnp.all(jnp.isfinite(result.x))
        assert jnp.isclose(result.x[0], 1.0, atol=1e-4)

    def test_large_initial_residual(self):
        """x0 very far from solution."""

        def F(x):
            return 2.0 * x - 1.0

        result = solve_mcp(F, jnp.array([0.0]), jnp.array([100.0]), jnp.array([100.0]))
        assert jnp.isclose(result.x[0], 0.5, atol=1e-6)

    def test_stiff_problem(self):
        """F with very different scales across components."""

        def F(x):
            return jnp.array([1000.0 * x[0] - 500.0, 0.001 * x[1] - 0.0005])

        result = solve_mcp(F, jnp.zeros(2), jnp.full(2, jnp.inf), jnp.ones(2))
        assert jnp.isclose(result.x[0], 0.5, atol=1e-6)
        assert jnp.isclose(result.x[1], 0.5, atol=1e-6)


class TestMCPResult:
    """Test the MCPResult return type."""

    def test_result_fields(self):
        def F(x):
            return 2.0 * x - 1.0

        result = solve_mcp(F, jnp.array([0.0]), jnp.full(1, jnp.inf), jnp.array([1.0]))
        assert hasattr(result, "x")
        assert hasattr(result, "residual_norm")
        assert hasattr(result, "num_steps")
        assert hasattr(result, "converged")

    def test_converged_true(self):
        def F(x):
            return 2.0 * x - 1.0

        result = solve_mcp(F, jnp.array([0.0]), jnp.full(1, jnp.inf), jnp.array([1.0]))
        assert result.converged
        assert result.residual_norm < 1e-10
        assert result.num_steps > 0

    def test_not_converged(self):
        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        result = solve_mcp(
            F, jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(2), max_mu_steps=1
        )
        assert not result.converged
        assert result.num_steps == 1


class TestDiffSolverFailureModes:
    """Ensure the differentiable solver doesn't crash on edge cases."""

    def test_zero_cotangent(self):
        """When loss doesn't depend on the solution, gradient should be zero."""

        def F(x, theta):
            return 2.0 * x - theta

        solver = make_mcp_solver_diff(F)
        l = jnp.array([0.0])
        u = jnp.full(1, jnp.inf)
        x0 = jnp.array([1.0])

        def loss(th):
            _ = solver(l, u, x0, th)
            return 0.0

        grad = jax.grad(loss)(jnp.array([1.0]))
        assert jnp.allclose(grad, 0.0)

    def test_solution_at_bound_gradient(self):
        """Gradient should be finite even when solution sits exactly at a bound."""

        def F(x, theta):
            return x + theta

        solver = make_mcp_solver_diff(F)
        l = jnp.array([0.0])
        u = jnp.full(1, jnp.inf)
        x0 = jnp.array([0.0])

        def loss(th):
            sol = solver(l, u, x0, th)
            return jnp.sum(sol**2)

        grad = jax.grad(loss)(jnp.array([1.0]))
        assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient at bound: {grad}"


class TestSolveInfoMatchesSolveMcp:
    """Verify that return_aux diagnostics match solve_mcp on the same problems."""

    def test_converged_lcp(self):
        """Fully converged LCP: all SolveInfo fields should match MCPResult."""

        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)

        result = solve_mcp(F, l, u, x0)
        solver = make_mcp_solver_diff(F, return_aux=True)
        x_diff, info = solver(l, u, x0, jnp.zeros(0))

        assert jnp.allclose(result.x, x_diff, atol=1e-10)
        assert int(info.num_steps) == result.num_steps
        assert bool(info.converged) == result.converged
        assert jnp.isclose(info.residual_norm, result.residual_norm, rtol=1e-3)

    def test_converged_nonlinear(self):
        """Converged nonlinear MCP with finite bounds."""

        def F(x):
            return x**3 - x - 1.0

        l = jnp.array([0.0])
        u = jnp.array([2.0])
        x0 = jnp.array([1.0])

        result = solve_mcp(F, l, u, x0)
        solver = make_mcp_solver_diff(F, return_aux=True)
        x_diff, info = solver(l, u, x0, jnp.zeros(0))

        assert jnp.allclose(result.x, x_diff, atol=1e-10)
        assert int(info.num_steps) == result.num_steps
        assert bool(info.converged) == result.converged

    def test_truncated_single_step(self):
        """With max_mu_steps=1, solver is truncated. Info should reflect this."""

        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)

        result = solve_mcp(F, l, u, x0, max_mu_steps=1)
        solver = make_mcp_solver_diff(F, max_mu_steps=1, return_aux=True)
        x_diff, info = solver(l, u, x0, jnp.zeros(0))

        assert jnp.allclose(result.x, x_diff, atol=1e-10)
        assert int(info.num_steps) == result.num_steps
        assert int(info.num_steps) == 1
        assert bool(info.converged) == result.converged

    def test_truncated_two_steps(self):
        """With max_mu_steps=2, partially converged."""

        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)

        result = solve_mcp(F, l, u, x0, max_mu_steps=2)
        solver = make_mcp_solver_diff(F, max_mu_steps=2, return_aux=True)
        x_diff, info = solver(l, u, x0, jnp.zeros(0))

        assert jnp.allclose(result.x, x_diff, atol=1e-10)
        assert int(info.num_steps) == result.num_steps
        assert bool(info.converged) == result.converged

    def test_parametric_problem(self):
        """Parametric F(x, theta) with return_aux."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        result = solve_mcp(F, l, u, x0, theta=theta)
        solver = make_mcp_solver_diff(F, return_aux=True)
        x_diff, info = solver(l, u, x0, theta)

        assert jnp.allclose(result.x, x_diff, atol=1e-10)
        assert int(info.num_steps) == result.num_steps
        assert bool(info.converged) == result.converged

    def test_grad_works_with_aux(self):
        """jax.grad should work when return_aux=True (aux is not differentiated)."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0])

        solver = make_mcp_solver_diff(F, return_aux=True)
        l = jnp.array([0.0])
        u = jnp.full(1, jnp.inf)
        x0 = jnp.ones(1)

        def loss(th):
            x, _info = solver(l, u, x0, th)
            return jnp.sum(x**2)

        grad = jax.grad(loss)(jnp.array([2.0]))
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_works_with_aux(self):
        """jax.jit should work when return_aux=True."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        result = solve_mcp(F, l, u, x0, theta=theta)
        solver = make_mcp_solver_diff(F, return_aux=True)
        jit_solver = jax.jit(solver)
        x_diff, info = jit_solver(l, u, x0, theta)

        assert jnp.allclose(result.x, x_diff, atol=1e-10)
        assert int(info.num_steps) == result.num_steps
        assert bool(info.converged) == result.converged

    def test_jit_grad_works_with_aux(self):
        """jax.jit(jax.grad(...)) should work when loss ignores aux output."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0])

        solver = make_mcp_solver_diff(F, return_aux=True)
        l = jnp.array([0.0])
        u = jnp.full(1, jnp.inf)
        x0 = jnp.ones(1)

        def loss(th):
            x, _info = solver(l, u, x0, th)
            return jnp.sum(x**2)

        grad_fn = jax.jit(jax.grad(loss))
        grad = grad_fn(jnp.array([2.0]))
        assert jnp.all(jnp.isfinite(grad))


class TestInputValidation:
    """Test that invalid inputs produce clear errors."""

    def test_mismatched_l_u_shape(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="same shape"):
            solve_mcp(F, jnp.zeros(2), jnp.zeros(3), jnp.zeros(2))

    def test_mismatched_x0_shape(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="same shape as l"):
            solve_mcp(F, jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(3))

    def test_l_greater_than_u(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="Lower bounds must not exceed"):
            solve_mcp(F, jnp.array([5.0]), jnp.array([3.0]), jnp.array([4.0]))

    def test_nan_x0(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="x0 must not contain NaN"):
            solve_mcp(F, jnp.array([0.0]), jnp.array([1.0]), jnp.array([jnp.nan]))

    def test_negative_mu_init(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="mu_init must be positive"):
            solve_mcp(
                F, jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.5]), mu_init=-1.0
            )

    def test_negative_mu_min(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="mu_min must be positive"):
            solve_mcp(
                F, jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.5]), mu_min=-1.0
            )

    def test_mu_min_zero(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="mu_min must be positive"):
            solve_mcp(
                F, jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.5]), mu_min=0.0
            )

    def test_mu_min_greater_than_mu_init(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="mu_min must be <= mu_init"):
            solve_mcp(
                F,
                jnp.array([0.0]),
                jnp.array([1.0]),
                jnp.array([0.5]),
                mu_init=0.1,
                mu_min=1.0,
            )

    def test_mu_decay_zero(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="mu_decay must be in"):
            solve_mcp(
                F, jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.5]), mu_decay=0.0
            )

    def test_mu_decay_one(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="mu_decay must be in"):
            solve_mcp(
                F, jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.5]), mu_decay=1.0
            )

    def test_mu_decay_negative(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="mu_decay must be in"):
            solve_mcp(
                F,
                jnp.array([0.0]),
                jnp.array([1.0]),
                jnp.array([0.5]),
                mu_decay=-0.5,
            )

    def test_max_mu_steps_zero(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="max_mu_steps must be >= 1"):
            solve_mcp(
                F,
                jnp.array([0.0]),
                jnp.array([1.0]),
                jnp.array([0.5]),
                max_mu_steps=0,
            )

    def test_negative_newton_tol(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="newton_tol must be non-negative"):
            solve_mcp(
                F,
                jnp.array([0.0]),
                jnp.array([1.0]),
                jnp.array([0.5]),
                newton_tol=-1e-10,
            )

    def test_negative_regularize(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="regularize must be non-negative"):
            solve_mcp(
                F,
                jnp.array([0.0]),
                jnp.array([1.0]),
                jnp.array([0.5]),
                regularize=-1.0,
            )


class TestDiffSolverRuntimeValidation:
    """Validate that the differentiable solver rejects invalid runtime inputs.

    solve_mcp validates l/u shapes, x0 shape, NaN bounds, and bound ordering
    at call time. The solver returned by make_mcp_solver_diff should apply the
    same checks so users get clear errors instead of silent wrong results.
    """

    def _make_solver(self):
        return make_mcp_solver_diff(lambda x: x)

    def test_l_greater_than_u(self):
        solver = self._make_solver()
        with pytest.raises(ValueError, match="Lower bounds must not exceed"):
            solver(jnp.array([5.0]), jnp.array([3.0]), jnp.array([4.0]), jnp.zeros(0))

    def test_mismatched_l_u_shapes(self):
        solver = self._make_solver()
        with pytest.raises(ValueError, match="same shape"):
            solver(jnp.zeros(2), jnp.zeros(3), jnp.zeros(2), jnp.zeros(0))

    def test_mismatched_x0_shape(self):
        solver = self._make_solver()
        with pytest.raises(ValueError, match="same shape as l"):
            solver(jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(3), jnp.zeros(0))

    def test_nan_lower_bound(self):
        solver = self._make_solver()
        with pytest.raises(ValueError, match="must not contain NaN"):
            solver(
                jnp.array([jnp.nan]), jnp.array([1.0]), jnp.array([0.5]), jnp.zeros(0)
            )

    def test_nan_upper_bound(self):
        solver = self._make_solver()
        with pytest.raises(ValueError, match="must not contain NaN"):
            solver(
                jnp.array([0.0]), jnp.array([jnp.nan]), jnp.array([0.5]), jnp.zeros(0)
            )

    def test_nan_x0(self):
        solver = self._make_solver()
        with pytest.raises(ValueError, match="x0 must not contain NaN"):
            solver(
                jnp.array([0.0]), jnp.array([1.0]), jnp.array([jnp.nan]), jnp.zeros(0)
            )


class TestDiffSolverValidation:
    """Validate that make_mcp_solver_diff rejects invalid parameters at construction."""

    def test_invalid_mu_init(self):
        with pytest.raises(ValueError, match="mu_init must be positive"):
            make_mcp_solver_diff(lambda x: x, mu_init=0.0)

    def test_invalid_mu_min(self):
        with pytest.raises(ValueError, match="mu_min must be positive"):
            make_mcp_solver_diff(lambda x: x, mu_min=0.0)

    def test_invalid_mu_min_greater_than_init(self):
        with pytest.raises(ValueError, match="mu_min must be <= mu_init"):
            make_mcp_solver_diff(lambda x: x, mu_init=0.1, mu_min=1.0)

    def test_invalid_mu_decay(self):
        with pytest.raises(ValueError, match="mu_decay must be in"):
            make_mcp_solver_diff(lambda x: x, mu_decay=1.5)

    def test_invalid_max_mu_steps(self):
        with pytest.raises(ValueError, match="max_mu_steps must be >= 1"):
            make_mcp_solver_diff(lambda x: x, max_mu_steps=0)

    def test_invalid_newton_tol(self):
        with pytest.raises(ValueError, match="newton_tol must be non-negative"):
            make_mcp_solver_diff(lambda x: x, newton_tol=-1.0)

    def test_invalid_regularize(self):
        with pytest.raises(ValueError, match="regularize must be non-negative"):
            make_mcp_solver_diff(lambda x: x, regularize=-0.1)

    def test_invalid_adjoint_method(self):
        with pytest.raises(ValueError, match="adjoint_method must be"):
            make_mcp_solver_diff(lambda x: x, adjoint_method="bicgstab")


class TestExtendedOptionValidation:
    """Tests for the public option validators introduced in todo.2026-04-18 #2.

    Verifies that every public solver knob is rejected at the API
    boundary with a clear ``ValueError`` instead of drifting into JAX
    internals as an opaque failure.
    """

    _L = jnp.array([0.0])
    _U = jnp.array([1.0])
    _X0 = jnp.array([0.5])

    # -- solve_mcp (call-time) --------------------------------------------

    def test_solve_mcp_armijo_c_out_of_range_zero(self):
        with pytest.raises(ValueError, match="armijo_c must be in"):
            solve_mcp(lambda x: x, self._L, self._U, self._X0, armijo_c=0.0)

    def test_solve_mcp_armijo_c_out_of_range_one(self):
        with pytest.raises(ValueError, match="armijo_c must be in"):
            solve_mcp(lambda x: x, self._L, self._U, self._X0, armijo_c=1.0)

    def test_solve_mcp_armijo_c_negative(self):
        with pytest.raises(ValueError, match="armijo_c must be in"):
            solve_mcp(lambda x: x, self._L, self._U, self._X0, armijo_c=-0.1)

    def test_solve_mcp_backtrack_rho_out_of_range(self):
        with pytest.raises(ValueError, match="backtrack_rho must be in"):
            solve_mcp(lambda x: x, self._L, self._U, self._X0, backtrack_rho=1.5)

    def test_solve_mcp_max_ls_steps_negative(self):
        with pytest.raises(ValueError, match="max_ls_steps must be >= 0"):
            solve_mcp(lambda x: x, self._L, self._U, self._X0, max_ls_steps=-3)

    def test_solve_mcp_krylov_tol_negative(self):
        with pytest.raises(ValueError, match="krylov_tol must be positive"):
            solve_mcp(
                lambda x: x,
                self._L,
                self._U,
                self._X0,
                linear_solver="gmres",
                krylov_tol=-1.0,
            )

    def test_solve_mcp_krylov_tol_zero(self):
        with pytest.raises(ValueError, match="krylov_tol must be positive"):
            solve_mcp(
                lambda x: x,
                self._L,
                self._U,
                self._X0,
                linear_solver="gmres",
                krylov_tol=0.0,
            )

    def test_solve_mcp_krylov_maxiter_zero(self):
        with pytest.raises(ValueError, match="krylov_maxiter must be >= 1"):
            solve_mcp(
                lambda x: x,
                self._L,
                self._U,
                self._X0,
                linear_solver="gmres",
                krylov_maxiter=0,
            )

    def test_solve_mcp_krylov_restart_zero(self):
        with pytest.raises(ValueError, match="krylov_restart must be >= 1"):
            solve_mcp(
                lambda x: x,
                self._L,
                self._U,
                self._X0,
                linear_solver="gmres",
                krylov_restart=0,
            )

    def test_solve_mcp_invalid_linear_solver(self):
        with pytest.raises(ValueError, match="linear_solver must be"):
            solve_mcp(lambda x: x, self._L, self._U, self._X0, linear_solver="bicgstab")

    # -- make_mcp_solver (construction-time) ------------------------------

    def test_forward_factory_armijo_c_out_of_range(self):
        with pytest.raises(ValueError, match="armijo_c must be in"):
            make_mcp_solver(lambda x: x, armijo_c=0.0)

    def test_forward_factory_backtrack_rho_out_of_range(self):
        with pytest.raises(ValueError, match="backtrack_rho must be in"):
            make_mcp_solver(lambda x: x, backtrack_rho=1.1)

    def test_forward_factory_max_ls_steps_negative(self):
        with pytest.raises(ValueError, match="max_ls_steps must be >= 0"):
            make_mcp_solver(lambda x: x, max_ls_steps=-1)

    def test_forward_factory_krylov_tol_zero(self):
        with pytest.raises(ValueError, match="krylov_tol must be positive"):
            make_mcp_solver(lambda x: x, krylov_tol=0.0)

    def test_forward_factory_krylov_maxiter_zero(self):
        with pytest.raises(ValueError, match="krylov_maxiter must be >= 1"):
            make_mcp_solver(lambda x: x, krylov_maxiter=0)

    def test_forward_factory_krylov_restart_zero(self):
        with pytest.raises(ValueError, match="krylov_restart must be >= 1"):
            make_mcp_solver(lambda x: x, krylov_restart=0)

    def test_forward_factory_invalid_linear_solver(self):
        with pytest.raises(ValueError, match="linear_solver must be"):
            make_mcp_solver(lambda x: x, linear_solver="bicgstab")

    # -- make_mcp_solver_diff (construction-time) -------------------------

    def test_diff_factory_armijo_c_out_of_range(self):
        with pytest.raises(ValueError, match="armijo_c must be in"):
            make_mcp_solver_diff(lambda x: x, armijo_c=1.0)

    def test_diff_factory_backtrack_rho_out_of_range(self):
        with pytest.raises(ValueError, match="backtrack_rho must be in"):
            make_mcp_solver_diff(lambda x: x, backtrack_rho=0.0)

    def test_diff_factory_max_ls_steps_negative(self):
        with pytest.raises(ValueError, match="max_ls_steps must be >= 0"):
            make_mcp_solver_diff(lambda x: x, max_ls_steps=-5)

    def test_diff_factory_krylov_tol_zero(self):
        with pytest.raises(ValueError, match="krylov_tol must be positive"):
            make_mcp_solver_diff(lambda x: x, krylov_tol=0.0)

    def test_diff_factory_krylov_maxiter_zero(self):
        with pytest.raises(ValueError, match="krylov_maxiter must be >= 1"):
            make_mcp_solver_diff(lambda x: x, krylov_maxiter=0)

    def test_diff_factory_krylov_restart_zero(self):
        with pytest.raises(ValueError, match="krylov_restart must be >= 1"):
            make_mcp_solver_diff(lambda x: x, krylov_restart=0)

    def test_diff_factory_invalid_linear_solver(self):
        with pytest.raises(ValueError, match="linear_solver must be"):
            make_mcp_solver_diff(lambda x: x, linear_solver="bicgstab")

    # -- Adjoint options (diff factory only) ------------------------------

    def test_diff_factory_gmres_tol_zero(self):
        with pytest.raises(ValueError, match="gmres_tol must be positive"):
            make_mcp_solver_diff(lambda x: x, gmres_tol=0.0)

    def test_diff_factory_gmres_tol_negative(self):
        with pytest.raises(ValueError, match="gmres_tol must be positive"):
            make_mcp_solver_diff(lambda x: x, gmres_tol=-1e-6)

    def test_diff_factory_gmres_maxiter_zero(self):
        with pytest.raises(ValueError, match="gmres_maxiter must be >= 1"):
            make_mcp_solver_diff(lambda x: x, gmres_maxiter=0)

    def test_diff_factory_gmres_restart_zero(self):
        with pytest.raises(ValueError, match="gmres_restart must be >= 1"):
            make_mcp_solver_diff(lambda x: x, gmres_restart=0)

    def test_diff_factory_cg_tol_zero(self):
        with pytest.raises(ValueError, match="cg_tol must be positive"):
            make_mcp_solver_diff(lambda x: x, cg_tol=0.0)

    def test_diff_factory_cg_tol_negative(self):
        with pytest.raises(ValueError, match="cg_tol must be positive"):
            make_mcp_solver_diff(lambda x: x, cg_tol=-1e-6)

    def test_diff_factory_cg_maxiter_zero(self):
        with pytest.raises(ValueError, match="cg_maxiter must be >= 1"):
            make_mcp_solver_diff(lambda x: x, cg_maxiter=0)

    def test_adjoint_validated_even_when_using_gmres(self):
        """Bad cg_* must be rejected even when adjoint_method='gmres'."""
        with pytest.raises(ValueError, match="cg_maxiter must be >= 1"):
            make_mcp_solver_diff(lambda x: x, adjoint_method="gmres", cg_maxiter=0)


class TestNormalizeF:
    """Test _normalize_F edge cases."""

    def test_single_arg_function(self):
        """F(x) should be wrapped to ignore theta."""

        def F(x):
            return 2.0 * x - 1.0

        sol = solve_mcp(F, jnp.array([0.0]), jnp.full(1, jnp.inf), jnp.array([1.0]))
        assert jnp.isclose(sol.x[0], 0.5, atol=1e-8)

    def test_two_arg_function(self):
        """F(x, theta) should be passed through unchanged."""

        def F(x, theta):
            return theta[0] * x - 1.0

        sol = solve_mcp(
            F,
            jnp.array([0.0]),
            jnp.full(1, jnp.inf),
            jnp.array([1.0]),
            theta=jnp.array([2.0]),
        )
        assert jnp.isclose(sol.x[0], 0.5, atol=1e-8)

    def test_function_with_default_arg(self):
        """F(x, theta=None) has 2 positional params and should NOT be wrapped."""

        def F(x, theta=None):
            if theta is None:
                return 2.0 * x - 1.0
            return theta[0] * x - 1.0

        # With explicit theta
        sol = solve_mcp(
            F,
            jnp.array([0.0]),
            jnp.full(1, jnp.inf),
            jnp.array([1.0]),
            theta=jnp.array([4.0]),
        )
        assert jnp.isclose(sol.x[0], 0.25, atol=1e-8)

    def test_lambda_single_arg(self):
        """Lambda with one arg should be wrapped."""
        sol = solve_mcp(
            lambda x: 2.0 * x - 1.0,
            jnp.array([0.0]),
            jnp.full(1, jnp.inf),
            jnp.array([1.0]),
        )
        assert jnp.isclose(sol.x[0], 0.5, atol=1e-8)

    def test_lambda_two_args(self):
        """Lambda with two args should not be wrapped."""
        sol = solve_mcp(
            lambda x, theta: theta[0] * x - 1.0,
            jnp.array([0.0]),
            jnp.full(1, jnp.inf),
            jnp.array([1.0]),
            theta=jnp.array([2.0]),
        )
        assert jnp.isclose(sol.x[0], 0.5, atol=1e-8)

    def test_var_positional_args(self):
        """F(x, *args) should NOT be wrapped — *args can accept theta."""

        def F(x, *args):
            theta = args[0]
            return theta[0] * x - 1.0

        sol = solve_mcp(
            F,
            jnp.array([0.0]),
            jnp.full(1, jnp.inf),
            jnp.array([1.0]),
            theta=jnp.array([2.0]),
        )
        assert jnp.isclose(sol.x[0], 0.5, atol=1e-8)

    def test_keyword_only_raises(self):
        """F(x, *, theta) is unsupported and should raise ValueError."""

        def F(x, *, theta):
            return theta[0] * x - 1.0

        with pytest.raises(ValueError, match="keyword-only"):
            solve_mcp(F, jnp.array([0.0]), jnp.full(1, jnp.inf), jnp.array([1.0]))

    def test_var_keyword_raises(self):
        """F(x, **kwargs) is unsupported and should raise ValueError."""

        def F(x, **kwargs):
            return x - 1.0

        with pytest.raises(ValueError, match="keyword-only"):
            solve_mcp(F, jnp.array([0.0]), jnp.full(1, jnp.inf), jnp.array([1.0]))
