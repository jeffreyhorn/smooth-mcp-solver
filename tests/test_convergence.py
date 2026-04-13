"""Tests for convergence failure behavior."""

import jax
import jax.numpy as jnp
import pytest

from smooth_mcp import make_mcp_solver_diff, solve_mcp


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

    def test_negative_mu_init(self):
        def F(x):
            return x

        with pytest.raises(ValueError, match="mu_init must be positive"):
            solve_mcp(
                F, jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.5]), mu_init=-1.0
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


class TestDiffSolverValidation:
    """Validate that make_mcp_solver_diff rejects invalid parameters at construction."""

    def test_invalid_mu_init(self):
        with pytest.raises(ValueError, match="mu_init must be positive"):
            make_mcp_solver_diff(lambda x: x, mu_init=0.0)

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
