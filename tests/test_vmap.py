"""Tests for jax.vmap over the differentiable solver."""

import jax
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff
from smooth_mcp.diff import SolveInfo


# -- Problem definitions ---------------------------------------------------


def _lcp_F(x, theta):
    M = theta.reshape(2, 2)
    q = jnp.array([1.0, -2.0])
    return M @ x + q


def _scalar_F(x, theta):
    """1D: F(x) = x - theta."""
    return x - theta


# -- Fixtures --------------------------------------------------------------

_L = jnp.zeros(2)
_U = jnp.full(2, jnp.inf)
_X0 = jnp.zeros(2)
_THETA = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()


class TestVmapDiffSolver:
    """jax.vmap over make_mcp_solver_diff (return_aux=False)."""

    def test_vmap_over_theta(self):
        """Batch over different theta values, same bounds."""
        solver = make_mcp_solver_diff(_lcp_F)
        thetas = jnp.stack([_THETA, _THETA + 0.1, _THETA - 0.05])  # (3, 4)

        batched = jax.vmap(solver, in_axes=(None, None, None, 0))
        xs = batched(_L, _U, _X0, thetas)

        assert xs.shape == (3, 2)
        assert jnp.all(jnp.isfinite(xs))

        # Each row should match an unbatched call
        for i in range(3):
            x_ref = solver(_L, _U, _X0, thetas[i])
            assert jnp.allclose(xs[i], x_ref)

    def test_vmap_over_bounds(self):
        """Batch over different lower bounds."""
        solver = make_mcp_solver_diff(_lcp_F)
        ls = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.2]])  # (3, 2)

        batched = jax.vmap(solver, in_axes=(0, None, None, None))
        xs = batched(ls, _U, _X0, _THETA)

        assert xs.shape == (3, 2)
        assert jnp.all(jnp.isfinite(xs))

    def test_vmap_over_all_inputs(self):
        """Batch over l, u, x0, theta simultaneously."""
        solver = make_mcp_solver_diff(_lcp_F)
        n_batch = 4
        ls = jnp.tile(_L, (n_batch, 1))
        us = jnp.tile(_U, (n_batch, 1))
        x0s = jnp.tile(_X0, (n_batch, 1))
        thetas = jnp.stack([_THETA + 0.01 * i for i in range(n_batch)])

        xs = jax.vmap(solver)(ls, us, x0s, thetas)

        assert xs.shape == (n_batch, 2)
        assert jnp.all(jnp.isfinite(xs))

    def test_jit_vmap(self):
        """jax.jit(jax.vmap(...)) composes."""
        solver = make_mcp_solver_diff(_lcp_F)
        thetas = jnp.stack([_THETA, _THETA + 0.1])
        ls = jnp.tile(_L, (2, 1))
        us = jnp.tile(_U, (2, 1))
        x0s = jnp.tile(_X0, (2, 1))

        jit_vmap = jax.jit(jax.vmap(solver))
        xs = jit_vmap(ls, us, x0s, thetas)

        assert xs.shape == (2, 2)
        assert jnp.all(jnp.isfinite(xs))

        # Second call reuses compiled graph
        xs2 = jit_vmap(ls, us, x0s, thetas)
        assert jnp.allclose(xs, xs2)

    def test_vmap_1d_problem(self):
        """vmap works on a 1D scalar problem."""
        solver = make_mcp_solver_diff(_scalar_F)
        l = jnp.array([0.0])
        u = jnp.full(1, jnp.inf)
        x0 = jnp.array([0.5])
        thetas = jnp.array([[1.0], [2.0], [-1.0]])

        batched = jax.vmap(solver, in_axes=(None, None, None, 0))
        xs = batched(l, u, x0, thetas)

        assert xs.shape == (3, 1)
        assert jnp.all(jnp.isfinite(xs))
        # F(x)=x-theta, l=0, u=inf: solution is max(theta, 0)
        assert jnp.allclose(xs[:, 0], jnp.array([1.0, 2.0, 0.0]), atol=1e-6)


class TestVmapDiffSolverReturnAux:
    """jax.vmap over make_mcp_solver_diff with return_aux=True."""

    def test_vmap_return_aux_shape(self):
        """vmap with return_aux=True returns (x_star, SolveInfo) with batch dim."""
        solver = make_mcp_solver_diff(_lcp_F, return_aux=True)
        thetas = jnp.stack([_THETA, _THETA + 0.1, _THETA - 0.05])
        batched = jax.vmap(solver, in_axes=(None, None, None, 0))
        xs, infos = batched(_L, _U, _X0, thetas)

        assert xs.shape == (3, 2)
        assert isinstance(infos, SolveInfo)
        assert infos.mu_used.shape == (3,)
        assert infos.num_steps.shape == (3,)
        assert infos.residual_norm.shape == (3,)
        assert infos.converged.shape == (3,)

    def test_vmap_return_aux_converged(self):
        """All batch elements converge on a well-posed problem."""
        solver = make_mcp_solver_diff(_lcp_F, return_aux=True)
        thetas = jnp.stack([_THETA, _THETA + 0.1])
        ls = jnp.tile(_L, (2, 1))
        us = jnp.tile(_U, (2, 1))
        x0s = jnp.tile(_X0, (2, 1))

        xs, infos = jax.vmap(solver)(ls, us, x0s, thetas)

        assert jnp.all(jnp.isfinite(xs))
        assert jnp.all(infos.converged)
        assert jnp.all(jnp.isfinite(infos.residual_norm))
        assert jnp.all(infos.residual_norm < 1e-8)

    def test_vmap_return_aux_parity_with_unbatched(self):
        """Batched aux matches unbatched per-element calls."""
        solver = make_mcp_solver_diff(_lcp_F, return_aux=True)
        thetas = jnp.stack([_THETA, _THETA + 0.1, _THETA - 0.05])
        batched = jax.vmap(solver, in_axes=(None, None, None, 0))
        xs, infos = batched(_L, _U, _X0, thetas)

        for i in range(3):
            x_ref, info_ref = solver(_L, _U, _X0, thetas[i])
            assert jnp.allclose(xs[i], x_ref)
            assert jnp.isclose(infos.residual_norm[i], info_ref.residual_norm)
            assert int(infos.num_steps[i]) == int(info_ref.num_steps)
            assert bool(infos.converged[i]) == bool(info_ref.converged)

    def test_jit_vmap_return_aux(self):
        """jax.jit(jax.vmap(...)) with return_aux=True composes."""
        solver = make_mcp_solver_diff(_lcp_F, return_aux=True)
        thetas = jnp.stack([_THETA, _THETA + 0.1])
        ls = jnp.tile(_L, (2, 1))
        us = jnp.tile(_U, (2, 1))
        x0s = jnp.tile(_X0, (2, 1))

        jit_vmap = jax.jit(jax.vmap(solver))
        xs, infos = jit_vmap(ls, us, x0s, thetas)

        assert xs.shape == (2, 2)
        assert jnp.all(jnp.isfinite(xs))
        assert jnp.all(infos.converged)

    def test_vmap_return_aux_truncated(self):
        """Truncated continuation reports per-element converged=False."""
        solver = make_mcp_solver_diff(_lcp_F, max_mu_steps=2, return_aux=True)
        thetas = jnp.stack([_THETA, _THETA + 0.1])
        batched = jax.vmap(solver, in_axes=(None, None, None, 0))
        xs, infos = batched(_L, _U, _X0, thetas)

        assert xs.shape == (2, 2)
        assert jnp.all(jnp.isfinite(xs))
        # With only 2 mu steps, should not converge
        assert not jnp.any(infos.converged)
        assert jnp.all(infos.num_steps == 2)


class TestVmapBatchedGradients:
    """Batched gradients: vmap(grad(...)) and grad(vmap(...))."""

    def test_vmap_grad_over_theta(self):
        """vmap(grad(loss)) over a batch of theta values."""
        solver = make_mcp_solver_diff(_lcp_F)

        def loss(theta):
            x = solver(_L, _U, _X0, theta)
            return jnp.sum(x**2)

        grad_fn = jax.grad(loss)
        thetas = jnp.stack([_THETA, _THETA + 0.1, _THETA - 0.05])

        batched_grad = jax.vmap(grad_fn)
        grads = batched_grad(thetas)

        assert grads.shape == thetas.shape  # (3, 4)
        assert jnp.all(jnp.isfinite(grads))

        # Parity: each row matches unbatched grad
        for i in range(3):
            g_ref = grad_fn(thetas[i])
            assert jnp.allclose(grads[i], g_ref, atol=1e-10)

    def test_jit_vmap_grad(self):
        """jax.jit(jax.vmap(jax.grad(loss))) composes."""
        solver = make_mcp_solver_diff(_lcp_F)

        def loss(theta):
            return jnp.sum(solver(_L, _U, _X0, theta) ** 2)

        thetas = jnp.stack([_THETA, _THETA + 0.1])
        jit_vmap_grad = jax.jit(jax.vmap(jax.grad(loss)))
        grads = jit_vmap_grad(thetas)

        assert grads.shape == (2, 4)
        assert jnp.all(jnp.isfinite(grads))

    def test_grad_of_vmapped_loss(self):
        """grad of a loss that internally uses vmap."""
        solver = make_mcp_solver_diff(_scalar_F)
        l = jnp.array([0.0])
        u = jnp.full(1, jnp.inf)
        x0 = jnp.array([0.5])

        def batched_loss(theta_batch):
            # theta_batch is (3, 1); vmap over batch dim
            xs = jax.vmap(solver, in_axes=(None, None, None, 0))(
                l, u, x0, theta_batch
            )
            return jnp.sum(xs**2)

        thetas = jnp.array([[1.0], [2.0], [3.0]])
        grad = jax.grad(batched_loss)(thetas)

        assert grad.shape == thetas.shape  # (3, 1)
        assert jnp.all(jnp.isfinite(grad))

    def test_vmap_grad_with_return_aux(self):
        """vmap(grad) works when solver has return_aux=True."""
        solver = make_mcp_solver_diff(_lcp_F, return_aux=True)

        def loss(theta):
            x, _info = solver(_L, _U, _X0, theta)
            return jnp.sum(x**2)

        thetas = jnp.stack([_THETA, _THETA + 0.1])
        grads = jax.vmap(jax.grad(loss))(thetas)

        assert grads.shape == (2, 4)
        assert jnp.all(jnp.isfinite(grads))

        # Should match the return_aux=False gradients
        solver_no_aux = make_mcp_solver_diff(_lcp_F)

        def loss_no_aux(theta):
            return jnp.sum(solver_no_aux(_L, _U, _X0, theta) ** 2)

        grads_ref = jax.vmap(jax.grad(loss_no_aux))(thetas)
        assert jnp.allclose(grads, grads_ref)
