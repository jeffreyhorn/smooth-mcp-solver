"""Solver correctness tests against known analytical solutions."""

import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp

# -- LCP tests (F(x) = Mx + q, l=0, u=inf) --------------------------------


class TestLCP:
    """Linear complementarity problems with known solutions."""

    def test_2x2_one_active_bound(self):
        """M = [[2,-1],[-1,3]], q = [1,-2]. Solution: x=[0, 2/3], F=[1/3, 0]."""

        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)

        result = solve_mcp(F, l, u, x0)
        sol = result.x
        assert result.converged
        assert jnp.isclose(sol[0], 0.0, atol=1e-8)
        assert jnp.isclose(sol[1], 2.0 / 3.0, atol=1e-8)

        Fx = F(sol)
        assert Fx[0] > -1e-8, "F[0] should be >= 0 at lower bound"
        assert jnp.isclose(Fx[1], 0.0, atol=1e-8), "F[1] should be 0 at interior"

    def test_diagonal_all_interior(self):
        """Diagonal M with q chosen so solution is strictly interior."""

        def F(x):
            return jnp.array([2.0, 3.0]) * x + jnp.array([-1.0, -1.5])

        sol = solve_mcp(F, jnp.zeros(2), jnp.full(2, jnp.inf), jnp.ones(2)).x
        assert jnp.allclose(sol, jnp.array([0.5, 0.5]), atol=1e-8)

    def test_diagonal_all_at_bound(self):
        """q > 0 means F(0) > 0, so x* = 0 is the solution."""

        def F(x):
            return jnp.array([2.0, 3.0]) * x + jnp.array([1.0, 2.0])

        sol = solve_mcp(F, jnp.zeros(2), jnp.full(2, jnp.inf), jnp.ones(2)).x
        assert jnp.allclose(sol, jnp.zeros(2), atol=1e-8)

    def test_1d_lcp_interior(self):
        """1D: F(x) = 2x - 1, l=0, u=inf. Solution: x=0.5."""

        def F(x):
            return 2.0 * x - 1.0

        sol = solve_mcp(F, jnp.array([0.0]), jnp.full(1, jnp.inf), jnp.array([1.0])).x
        assert jnp.isclose(sol[0], 0.5, atol=1e-8)

    def test_1d_lcp_at_lower_bound(self):
        """1D: F(x) = 2x + 1, l=0, u=inf. F(0) = 1 > 0, so x*=0."""

        def F(x):
            return 2.0 * x + 1.0

        sol = solve_mcp(F, jnp.array([0.0]), jnp.full(1, jnp.inf), jnp.array([0.0])).x
        assert jnp.isclose(sol[0], 0.0, atol=1e-8)


# -- Nonlinear complementarity problems ------------------------------------


class TestNonlinearMCP:

    def test_1d_cubic_interior(self):
        """F(x) = x^3 - x - 1, l=0, u=2. Root at x ≈ 1.3247 is interior."""

        def F(x):
            return x**3 - x - 1.0

        result = solve_mcp(F, jnp.array([0.0]), jnp.array([2.0]), jnp.array([1.0]))
        assert jnp.isclose(result.x[0], 1.3247179572, atol=1e-6)
        assert jnp.isclose(F(result.x)[0], 0.0, atol=1e-6)

    def test_1d_root_outside_bounds_lower(self):
        """F(x) = x - 5, l=0, u=2. Root at x=5 is outside [0,2].
        At u=2: F(2) = -3 < 0, so x*=2 (at upper bound)."""

        def F(x):
            return x - 5.0

        sol = solve_mcp(F, jnp.array([0.0]), jnp.array([2.0]), jnp.array([1.0])).x
        assert jnp.isclose(sol[0], 2.0, atol=1e-8)

    def test_1d_root_outside_bounds_upper(self):
        """F(x) = x + 5, l=0, u=10. Root at x=-5 is outside [0,10].
        At l=0: F(0) = 5 > 0, so x*=0 (at lower bound)."""

        def F(x):
            return x + 5.0

        sol = solve_mcp(F, jnp.array([0.0]), jnp.array([10.0]), jnp.array([1.0])).x
        assert jnp.isclose(sol[0], 0.0, atol=1e-8)

    def test_2d_ncp_symmetric(self):
        """Standard 2D NCP: x^2 + xy - 3x + 2, with asymmetric x0."""

        def F(x):
            x1, x2 = x[0], x[1]
            return jnp.array(
                [
                    x1**2 + x1 * x2 - 3 * x1 + 2,
                    x2**2 + x1 * x2 - 3 * x2 + 2,
                ]
            )

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.array([0.5, 1.5])

        sol = solve_mcp(F, l, u, x0).x
        Fx = F(sol)

        for i in range(2):
            assert sol[i] >= -1e-8, f"x[{i}] = {sol[i]} < 0"
            assert Fx[i] >= -1e-8, f"F[{i}] = {Fx[i]} < 0"
            assert jnp.isclose(
                sol[i] * Fx[i], 0.0, atol=1e-6
            ), f"Complementarity violated: x[{i}]={sol[i]}, F[{i}]={Fx[i]}"


# -- Finite bounds on both sides -------------------------------------------


class TestFiniteBounds:

    def test_solution_respects_bounds(self):
        """Solution must satisfy l <= x <= u."""

        def F(x):
            return jnp.array([x[0] - 3.0, x[1] + 1.0])

        l = jnp.array([0.0, 0.0])
        u = jnp.array([2.0, 5.0])
        x0 = jnp.array([1.0, 1.0])

        sol = solve_mcp(F, l, u, x0).x
        assert jnp.all(sol >= l - 1e-8), f"Below lower bound: {sol}"
        assert jnp.all(sol <= u + 1e-8), f"Above upper bound: {sol}"

    def test_mixed_active_bounds(self):
        """One variable at lower bound, one at upper bound."""

        def F(x):
            return jnp.array([x[0] + 1.0, x[1] - 10.0])

        sol = solve_mcp(
            F, jnp.array([0.0, 0.0]), jnp.array([5.0, 5.0]), jnp.array([2.0, 2.0])
        ).x
        assert jnp.isclose(sol[0], 0.0, atol=1e-8), f"Expected x[0]=0, got {sol[0]}"
        assert jnp.isclose(sol[1], 5.0, atol=1e-8), f"Expected x[1]=5, got {sol[1]}"

    def test_tight_bounds(self):
        """When l == u, solution must be l (regardless of F)."""

        def F(x):
            return x * 100.0 - 50.0

        sol = solve_mcp(F, jnp.array([3.0]), jnp.array([3.0]), jnp.array([3.0])).x
        assert jnp.isclose(sol[0], 3.0, atol=1e-6)


# -- Single variable -------------------------------------------------------


class TestSingleVariable:

    def test_scalar_interior(self):
        def F(x):
            return 2.0 * x - 1.0

        sol = solve_mcp(F, jnp.array([0.0]), jnp.array([10.0]), jnp.array([5.0])).x
        assert jnp.isclose(sol[0], 0.5, atol=1e-8)

    def test_scalar_at_lower(self):
        """F(x) = x + 5, l=-2, u=10. Root at x=-5 is below l.
        At l=-2: F(-2) = 3 > 0, so x*=-2 (at lower bound)."""

        def F(x):
            return x + 5.0

        sol = solve_mcp(F, jnp.array([-2.0]), jnp.array([10.0]), jnp.array([0.0])).x
        assert jnp.isclose(sol[0], -2.0, atol=1e-6)

    def test_scalar_at_upper(self):
        def F(x):
            return x - 100.0

        sol = solve_mcp(F, jnp.array([0.0]), jnp.array([5.0]), jnp.array([2.0])).x
        assert jnp.isclose(sol[0], 5.0, atol=1e-8)


# -- Differentiable solver produces same solution --------------------------


class TestDiffSolverMatchesForward:

    def test_lcp_matches(self):
        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)

        sol_fwd = solve_mcp(F, l, u, x0).x
        solver = make_mcp_solver_diff(F)
        sol_diff = solver(l, u, x0, jnp.zeros(0))

        assert jnp.allclose(sol_fwd, sol_diff, atol=1e-10)

    def test_nonlinear_matches(self):
        def F(x):
            return x**3 - x - 1.0

        l = jnp.array([0.0])
        u = jnp.array([2.0])
        x0 = jnp.array([1.0])

        sol_fwd = solve_mcp(F, l, u, x0).x
        solver = make_mcp_solver_diff(F)
        sol_diff = solver(l, u, x0, jnp.zeros(0))

        assert jnp.allclose(sol_fwd, sol_diff, atol=1e-10)


# -- Matrix-free Newton-Krylov solver --------------------------------------


class TestKrylovSolver:

    def test_lcp_gmres_matches_dense(self):
        """GMRES linear solver should produce same solution as dense."""

        def F(x):
            M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
            return M @ x + jnp.array([1.0, -2.0])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)

        sol_dense = solve_mcp(F, l, u, x0, linear_solver="dense").x
        sol_gmres = solve_mcp(F, l, u, x0, linear_solver="gmres").x

        assert jnp.allclose(sol_dense, sol_gmres, atol=1e-6)

    def test_nonlinear_gmres(self):
        def F(x):
            return x**3 - x - 1.0

        sol = solve_mcp(
            F,
            jnp.array([0.0]),
            jnp.array([2.0]),
            jnp.array([1.0]),
            linear_solver="gmres",
        )
        assert jnp.isclose(sol.x[0], 1.3247179572, atol=1e-5)
        assert sol.converged

    def test_2d_ncp_gmres(self):
        def F(x):
            x1, x2 = x[0], x[1]
            return jnp.array(
                [
                    x1**2 + x1 * x2 - 3 * x1 + 2,
                    x2**2 + x1 * x2 - 3 * x2 + 2,
                ]
            )

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.array([0.5, 1.5])

        sol = solve_mcp(F, l, u, x0, linear_solver="gmres").x
        Fx = F(sol)

        for i in range(2):
            assert sol[i] >= -1e-6, f"x[{i}] = {sol[i]} < 0"
            assert Fx[i] >= -1e-6, f"F[{i}] = {Fx[i]} < 0"

    def test_diff_solver_gmres(self):
        """make_mcp_solver_diff with linear_solver='gmres' should match dense."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        solver_dense = make_mcp_solver_diff(F, linear_solver="dense")
        solver_gmres = make_mcp_solver_diff(F, linear_solver="gmres")

        sol_dense = solver_dense(l, u, x0, theta)
        sol_gmres = solver_gmres(l, u, x0, theta)
        assert jnp.allclose(sol_dense, sol_gmres, atol=1e-6)

        # Check gradients match too
        def loss_dense(th):
            return jnp.sum(solver_dense(l, u, x0, th) ** 2)

        def loss_gmres(th):
            return jnp.sum(solver_gmres(l, u, x0, th) ** 2)

        import jax

        grad_dense = jax.grad(loss_dense)(theta)
        grad_gmres = jax.grad(loss_gmres)(theta)
        assert jnp.allclose(
            grad_dense, grad_gmres, atol=1e-5
        ), f"dense={grad_dense}, gmres={grad_gmres}"
