"""Gradient correctness tests via finite-difference comparison."""

import jax
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff


def _fd_grad(loss_fn, x, eps=1e-5):
    """Compute gradient via central finite differences."""
    g = jnp.zeros_like(x)
    for i in range(len(x)):
        e = jnp.zeros_like(x).at[i].set(eps)
        g = g.at[i].set((loss_fn(x + e) - loss_fn(x - e)) / (2 * eps))
    return g


# -- Problem definitions ------------------------------------------------


def _lcp_F(x, theta):
    M = theta.reshape(2, 2)
    q = jnp.array([1.0, -2.0])
    return M @ x + q


def _nonlinear_1d_F(x, theta):
    return x**3 - theta * x - 1.0


def _ncp_2d_F(x, theta):
    """2D NCP with theta scaling the coupling term."""
    x1, x2 = x[0], x[1]
    f1 = x1**2 + theta[0] * x1 * x2 - 3 * x1 + 2
    f2 = x2**2 + theta[0] * x1 * x2 - 3 * x2 + 2
    return jnp.array([f1, f2])


def _network_F(x, theta):
    """Spatial price equilibrium with parametrized coefficients."""
    x1, x2 = x
    f1 = x1 - theta[0] + theta[1] * x1**2 - 0.1 * x2
    f2 = x2 - theta[2] + theta[3] * x2**2 - 0.05 * x1
    return jnp.array([f1, f2])


# -- Tests ---------------------------------------------------------------


class TestGradientTheta:
    """Test gradients w.r.t. theta (the main use case for implicit diff)."""

    def test_lcp_grad_theta(self):
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)
        theta = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()

        solver = make_mcp_solver_diff(_lcp_F)

        def loss(th):
            sol = solver(l, u, x0, th)
            return jnp.sum(sol**2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"LCP grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_nonlinear_1d_grad_theta(self):
        l = jnp.array([0.0])
        u = jnp.array([2.0])
        x0 = jnp.array([1.0])
        theta = jnp.array([1.0])

        solver = make_mcp_solver_diff(_nonlinear_1d_F)

        def loss(th):
            sol = solver(l, u, x0, th)
            return jnp.sum(sol**2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"1D NCP grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_ncp_2d_grad_theta(self):
        l = jnp.array([0.0, 0.0])
        u = jnp.full(2, jnp.inf)
        x0 = jnp.array([0.5, 1.5])
        theta = jnp.array([1.0])

        solver = make_mcp_solver_diff(_ncp_2d_F)

        def loss(th):
            sol = solver(l, u, x0, th)
            return jnp.sum(sol**2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"2D NCP grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_network_grad_theta(self):
        l = jnp.array([0.0, 0.0])
        u = jnp.array([10.0, 10.0])
        x0 = jnp.array([4.0, 3.0])
        theta = jnp.array([5.0, 0.2, 3.0, 0.3])

        solver = make_mcp_solver_diff(_network_F)

        def loss(th):
            sol = solver(l, u, x0, th)
            return jnp.sum(sol**2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"Network grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_finite_upper_bounds_grad_theta(self):
        """Problem with finite bounds on both sides."""

        def F(x, theta):
            return theta[0] * x - theta[1]

        l = jnp.array([0.0])
        u = jnp.array([5.0])
        x0 = jnp.array([2.0])
        theta = jnp.array([2.0, 3.0])

        solver = make_mcp_solver_diff(F)

        def loss(th):
            sol = solver(l, u, x0, th)
            return jnp.sum(sol**2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"Finite bounds grad mismatch: auto={grad_auto}, fd={grad_fd}"


class TestGradientX0:
    """Test straight-through gradients w.r.t. x0."""

    def test_lcp_grad_x0(self):
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        theta = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()

        solver = make_mcp_solver_diff(_lcp_F, differentiate_through_x0=True)

        def loss(x0):
            sol = solver(l, u, x0, theta)
            return jnp.sum(sol**2)

        x0 = jnp.zeros(2)
        # Straight-through estimator: d_x0 = g (the cotangent).
        # This is NOT the true mathematical gradient, so we just check
        # it's finite and has the right shape.
        grad = jax.grad(loss)(x0)
        assert grad.shape == x0.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_no_x0_grad_by_default(self):
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        theta = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()

        solver = make_mcp_solver_diff(_lcp_F, differentiate_through_x0=False)

        def loss(x0):
            sol = solver(l, u, x0, theta)
            return jnp.sum(sol**2)

        x0 = jnp.zeros(2)
        grad = jax.grad(loss)(x0)
        assert jnp.allclose(grad, 0.0), "x0 gradient should be zero when disabled"


class TestGradientMultipleLosses:
    """Test gradients with different loss functions to catch sign/scale errors."""

    def test_weighted_loss(self):
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)
        theta = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()

        solver = make_mcp_solver_diff(_lcp_F)

        def loss(th):
            sol = solver(l, u, x0, th)
            return 3.0 * sol[0] ** 2 + 7.0 * sol[1] ** 2

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"Weighted loss grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_linear_loss(self):
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.zeros(2)
        theta = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()

        solver = make_mcp_solver_diff(_lcp_F)

        def loss(th):
            sol = solver(l, u, x0, th)
            return jnp.sum(sol)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"Linear loss grad mismatch: auto={grad_auto}, fd={grad_fd}"


class TestGradientBounds:
    """Test gradients w.r.t. lower and upper bounds."""

    def test_grad_lower_bound(self):
        """Gradient w.r.t. l when solution is at the lower bound."""

        def F(x, theta):
            return theta[0] * x + theta[1]

        # With theta = [2, 1], F(x) = 2x + 1. F(l) > 0 for l >= 0, so x* = l.
        # loss = sum(x*^2) = sum(l^2), so d(loss)/dl = 2*l.
        theta = jnp.array([2.0, 1.0])
        x0 = jnp.array([0.0])
        u = jnp.full(1, jnp.inf)
        solver = make_mcp_solver_diff(F)

        def loss(l):
            sol = solver(l, u, x0, theta)
            return jnp.sum(sol**2)

        l = jnp.array([0.5])
        grad_auto = jax.grad(loss)(l)
        grad_fd = _fd_grad(loss, l)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"Lower bound grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_grad_upper_bound(self):
        """Gradient w.r.t. u when solution is at the upper bound."""

        def F(x, theta):
            return theta[0] * x + theta[1]

        # With theta = [2, -10], F(x) = 2x - 10. F(u) < 0 for u < 5, so x* = u.
        # loss = sum(x*^2) = sum(u^2), so d(loss)/du = 2*u.
        theta = jnp.array([2.0, -10.0])
        x0 = jnp.array([1.0])
        l = jnp.array([0.0])
        solver = make_mcp_solver_diff(F)

        def loss(u):
            sol = solver(l, u, x0, theta)
            return jnp.sum(sol**2)

        u = jnp.array([3.0])
        grad_auto = jax.grad(loss)(u)
        grad_fd = _fd_grad(loss, u)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"Upper bound grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_grad_bounds_interior_solution(self):
        """When solution is interior, bound gradients should be ~zero."""

        def F(x, theta):
            return 2.0 * x - 1.0

        theta = jnp.array([0.0])
        x0 = jnp.array([0.0])
        solver = make_mcp_solver_diff(F)

        def loss_l(l):
            sol = solver(l, jnp.array([10.0]), x0, theta)
            return jnp.sum(sol**2)

        def loss_u(u):
            sol = solver(jnp.array([-10.0]), u, x0, theta)
            return jnp.sum(sol**2)

        # Solution x*=0.5 is far from bounds, so gradients should be ~0
        grad_l = jax.grad(loss_l)(jnp.array([-10.0]))
        grad_u = jax.grad(loss_u)(jnp.array([10.0]))
        assert jnp.allclose(grad_l, 0.0, atol=1e-5), f"dl should be ~0, got {grad_l}"
        assert jnp.allclose(grad_u, 0.0, atol=1e-5), f"du should be ~0, got {grad_u}"


class TestAdjointCG:
    """Test gradients using adjoint_method='cg' on SPD problems."""

    def test_diagonal_lcp_cg_gradient(self):
        """Diagonal LCP has a symmetric Jacobian, so CG is valid."""

        def F(x, theta):
            # F(x) = diag(theta) @ x + q — Jacobian is diag(theta), which is SPD
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        solver = make_mcp_solver_diff(F, adjoint_method="cg")

        def loss(th):
            sol = solver(l, u, x0, th)
            return jnp.sum(sol**2)

        grad_cg = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_cg, grad_fd, atol=1e-5
        ), f"CG grad mismatch: cg={grad_cg}, fd={grad_fd}"

    def test_cg_matches_gmres_on_spd(self):
        """On an SPD problem, CG and GMRES should give the same gradient."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        solver_cg = make_mcp_solver_diff(F, adjoint_method="cg")
        solver_gmres = make_mcp_solver_diff(F, adjoint_method="gmres")

        def loss_cg(th):
            return jnp.sum(solver_cg(l, u, x0, th) ** 2)

        def loss_gmres(th):
            return jnp.sum(solver_gmres(l, u, x0, th) ** 2)

        grad_cg = jax.grad(loss_cg)(theta)
        grad_gmres = jax.grad(loss_gmres)(theta)

        assert jnp.allclose(
            grad_cg, grad_gmres, atol=1e-6
        ), f"CG={grad_cg}, GMRES={grad_gmres}"


class TestPreconditioner:
    """Test that supplying a preconditioner to the adjoint solve works."""

    def test_identity_precond_matches_no_precond(self):
        """Identity preconditioner should give the same gradient as no preconditioner."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        solver_none = make_mcp_solver_diff(F, precond=None)
        solver_id = make_mcp_solver_diff(F, precond=lambda v: v)

        def loss_none(th):
            return jnp.sum(solver_none(l, u, x0, th) ** 2)

        def loss_id(th):
            return jnp.sum(solver_id(l, u, x0, th) ** 2)

        grad_none = jax.grad(loss_none)(theta)
        grad_id = jax.grad(loss_id)(theta)

        assert jnp.allclose(
            grad_none, grad_id, atol=1e-8
        ), f"none={grad_none}, identity={grad_id}"

    def test_diagonal_precond_gradient_correct(self):
        """Diagonal preconditioner should still produce correct gradients."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        # Diagonal preconditioner (approximate inverse of J^T)
        solver = make_mcp_solver_diff(F, precond=lambda v: v / theta)

        def loss(th):
            return jnp.sum(solver(l, u, x0, th) ** 2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"precond grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_cg_adjoint_with_precond(self):
        """precond must plumb through adjoint_method='cg', not only the
        GMRES default. Verify the CG adjoint produces FD-agreeing
        gradients under a diagonal preconditioner."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        solver = make_mcp_solver_diff(
            F, adjoint_method="cg", precond=lambda v: v / theta
        )

        def loss(th):
            return jnp.sum(solver(l, u, x0, th) ** 2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"cg+precond grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_precond_composes_with_jit(self):
        """precond is a closure-time argument captured by the factory;
        jitting the returned solver must keep that closure intact and
        still produce FD-agreeing gradients."""

        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        solver = make_mcp_solver_diff(F, precond=lambda v: v / theta)
        jitted = jax.jit(solver)

        def loss(th):
            return jnp.sum(jitted(l, u, x0, th) ** 2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)

        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-5
        ), f"jit+precond grad mismatch: auto={grad_auto}, fd={grad_fd}"


class TestTruncatedContinuationGradients:
    """Test gradients when the continuation is truncated (few mu steps).

    The backward pass uses the actual terminal mu from the forward solve,
    so truncated solves should still produce correct gradients.
    """

    def test_truncated_1_step(self):
        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        solver = make_mcp_solver_diff(F, max_mu_steps=1)
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        def loss(th):
            return jnp.sum(solver(l, u, x0, th) ** 2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)
        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-4
        ), f"Truncated 1-step grad mismatch: auto={grad_auto}, fd={grad_fd}"

    def test_truncated_2_steps(self):
        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        solver = make_mcp_solver_diff(F, max_mu_steps=2)
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])

        def loss(th):
            return jnp.sum(solver(l, u, x0, th) ** 2)

        grad_auto = jax.grad(loss)(theta)
        grad_fd = _fd_grad(loss, theta)
        assert jnp.allclose(
            grad_auto, grad_fd, atol=1e-4
        ), f"Truncated 2-step grad mismatch: auto={grad_auto}, fd={grad_fd}"


class TestJitComposability:
    """Test that the differentiable solver composes with jax.jit."""

    def _make_problem(self):
        def F(x, theta):
            return theta * x + jnp.array([-1.0, -1.5])

        solver = make_mcp_solver_diff(F)
        l = jnp.zeros(2)
        u = jnp.full(2, jnp.inf)
        x0 = jnp.ones(2)
        theta = jnp.array([2.0, 3.0])
        return solver, l, u, x0, theta

    def test_jit_forward(self):
        """jax.jit(solver)(...) should produce a finite solution."""
        solver, l, u, x0, theta = self._make_problem()
        jitted = jax.jit(solver)
        result = jitted(l, u, x0, theta)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_grad(self):
        """jax.jit(jax.grad(loss))(...) should produce a finite gradient."""
        solver, l, u, x0, theta = self._make_problem()

        def loss(th):
            return jnp.sum(solver(l, u, x0, th) ** 2)

        grad = jax.jit(jax.grad(loss))(theta)
        assert jnp.all(jnp.isfinite(grad))
