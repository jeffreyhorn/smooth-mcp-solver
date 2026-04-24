"""Deterministic numeric tests for representative demo setups.

``tests/test_demos.py`` runs each demo as a subprocess and asserts
exit code 0, which catches crashes but not wrong answers. Per
remediation plan todo.2026-04-23.md #13, this module adds a smaller
in-process layer that replicates the numeric setup of representative
demos and asserts the key outputs documented in each demo's header
comment. This way the documented "expected solution" claims are
verified on every test run.

The subprocess smoke tests are intentionally kept — they catch
top-level import and scripting regressions that a direct setup
replication does not.
"""

import jax.numpy as jnp

from smooth_mcp import solve_mcp


class TestLcpAsMcp:
    """Mirror of demos/lcp_as_mcp.py.

    F(x) = Mx + q with x >= 0. Expected per the demo header:
        x* = [0, 2/3], F(x*) = [1/3, 0].
    """

    def test_matches_demo_expected_solution(self):
        M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
        q = jnp.array([1.0, -2.0])

        def F(x):
            return M @ x + q

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)

        result = solve_mcp(F, l, u, x0)
        assert result.converged
        assert jnp.allclose(result.x, jnp.array([0.0, 2.0 / 3.0]), atol=1e-8)
        assert jnp.allclose(F(result.x), jnp.array([1.0 / 3.0, 0.0]), atol=1e-8)
        # Complementarity: x_i * F(x_i) = 0 for each i.
        assert jnp.allclose(result.x * F(result.x), jnp.zeros(2), atol=1e-8)


class TestNonlinear1dMcp:
    """Mirror of demos/nonlinear_1d_mcp.py.

    F(x) = x^3 - x - 1 with 0 <= x <= 2. Interior root:
        x* ≈ 1.3247 (the real root of x^3 - x - 1).
    """

    def test_matches_demo_expected_solution(self):
        def F(x):
            return x**3 - x - 1.0

        l = jnp.array([0.0])
        u = jnp.array([2.0])
        x0 = jnp.array([1.0])

        result = solve_mcp(F, l, u, x0)
        assert result.converged
        # x* is the real root of x^3 - x - 1 = 0; solution is interior
        # so F(x*) = 0 exactly (modulo solver tolerance).
        expected = 1.324717957244746  # Wolfram: real root of x^3 - x - 1
        assert jnp.isclose(result.x[0], expected, atol=1e-8)
        assert jnp.isclose(F(result.x)[0], 0.0, atol=1e-8)


class TestNonlinearNcp2d:
    """Mirror of demos/2d_nonlinear_complementarity_problem.py.

    F_1(x) = x1^2 + x1 x2 - 3 x1 + 2
    F_2(x) = x2^2 + x1 x2 - 3 x2 + 2
    with x >= 0. The problem has two MCP solutions: [0, 2] and
    [2, 0]. From x0 = [0.5, 1.5] the solver converges to [0, 2].
    """

    def test_matches_demo_expected_solution(self):
        def F(x):
            x1, x2 = x[0], x[1]
            f1 = x1**2 + x1 * x2 - 3 * x1 + 2
            f2 = x2**2 + x1 * x2 - 3 * x2 + 2
            return jnp.array([f1, f2])

        l = jnp.array([0.0, 0.0])
        u = jnp.array([jnp.inf, jnp.inf])
        x0 = jnp.array([0.5, 1.5])

        result = solve_mcp(F, l, u, x0)
        assert result.converged
        assert jnp.allclose(result.x, jnp.array([0.0, 2.0]), atol=1e-8)
        # Complementarity: x_i * F_i = 0 component-wise.
        assert jnp.allclose(result.x * F(result.x), jnp.zeros(2), atol=1e-8)


class TestObstacleProblemDenseGmresParity:
    """Mirror of demos/obstacle_problem.py.

    50-variable discretized obstacle problem. The demo demonstrates
    linear_solver='dense' and linear_solver='gmres' should agree on
    the same problem; verify that parity is tight (the demo prints
    the max difference but does not assert anything).
    """

    def test_dense_and_gmres_agree(self):
        n = 50
        h = 1.0 / (n + 1)
        x_grid = jnp.linspace(h, 1.0 - h, n)
        diag = jnp.full(n, -2.0 / h**2)
        off = jnp.full(n - 1, 1.0 / h**2)
        A = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
        f = jnp.full(n, -1.0)
        phi = 0.2 * jnp.exp(-((x_grid - 0.5) ** 2) / 0.005)

        def F(u):
            return -A @ u - f

        l = phi
        u_bound = jnp.full(n, jnp.inf)
        u0 = jnp.zeros(n)
        tol = 1e-8

        result_dense = solve_mcp(
            F, l, u_bound, u0, linear_solver="dense", newton_tol=tol
        )
        result_gmres = solve_mcp(
            F, l, u_bound, u0, linear_solver="gmres", newton_tol=tol
        )

        assert result_dense.converged
        assert result_gmres.converged
        # The demo prints max_diff but doesn't assert; we do.
        max_diff = float(jnp.max(jnp.abs(result_dense.x - result_gmres.x)))
        assert max_diff < 1e-6, f"dense vs gmres disagreement: {max_diff:.2e}"

        # Contact region is a demo talking point; confirm the membrane
        # touches the obstacle somewhere (problem-specific sanity).
        active = jnp.abs(result_dense.x - phi) < 1e-6
        assert int(jnp.sum(active)) > 0
