"""Targeted tests for the SolveInfo.mu_used contract and the
backward-pass gradient semantics.

Option A (review codex.2026-04-23 finding #2, remediation todo #4-#6):
- mu_used reports the last mu at which the Newton solve ran — not a
  clamped value of mu_min.
- The backward pass of make_mcp_solver_diff differentiates at that
  mu_used, so the adjoint is consistent with the returned x_star.
"""

import jax
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver, make_mcp_solver_diff, smoothed_residual


def _F(x, theta):
    return x - theta


# Newton converges to ~theta at mu_used ~ 1.2e-4 on this problem. The
# looser tol is what makes early stop actually fire; with the default
# newton_tol=1e-10 the continuation walks mu all the way down to mu_min
# in float64.
_EARLY_STOP_KWARGS = dict(mu_init=1.0, mu_min=1e-10, newton_tol=1e-4)


# ---------------------------------------------------------------------------
# Diagnostic semantics
# ---------------------------------------------------------------------------


class TestMuUsedDiagnostic:
    def test_diff_factory_early_stop(self):
        """The diff factory sees the same early-stop mu_used as the
        forward factory — both come from the shared continuation
        kernel. Complements the forward-factory test in
        test_forward_factory.py::TestMuUsed.
        """
        solver = make_mcp_solver_diff(_F, return_aux=True, **_EARLY_STOP_KWARGS)
        l = jnp.array([0.0])
        u = jnp.array([3.0])
        x0 = jnp.array([1.0])
        _, info = solver(l, u, x0, jnp.array([2.0]))
        assert bool(info.converged) is True
        # mu_used sits well above mu_min=1e-10; the old clamped
        # semantics would have reported ~1e-10.
        assert float(info.mu_used) > 1e-5
        assert float(info.mu_used) <= 1.0

    def test_max_mu_steps_exhaustion_reports_last_tried_mu(self):
        """Non-convergence: mu_used is the mu the Newton solve last ran
        at, not clamped to mu_min. With max_mu_steps=3 and default tight
        tol, the solver takes exactly three body iterations at
        mu_next = 1.0, 0.5, 0.25 and exits without converging.
        """
        solver = make_mcp_solver(
            _F,
            mu_init=1.0,
            mu_decay=0.5,
            mu_min=1e-10,
            max_mu_steps=3,
            return_aux=True,
        )
        l = jnp.array([0.0])
        u = jnp.array([3.0])
        x0 = jnp.array([1.0])
        _, info = solver(l, u, x0, jnp.array([2.0]))
        assert bool(info.converged) is False
        assert int(info.num_steps) == 3
        # The third body iteration solved Newton at mu_init * decay^2 = 0.25.
        assert jnp.isclose(info.mu_used, 0.25, rtol=1e-6)


# ---------------------------------------------------------------------------
# Gradient semantics
# ---------------------------------------------------------------------------


class TestMuUsedGradient:
    def test_backward_pass_differentiates_at_mu_used(self):
        """The gradient produced by make_mcp_solver_diff matches an
        independent implicit-function-theorem adjoint computed at
        mu_used. Proves the backward pass is taken at the last-solved
        mu, consistent with the returned x_star — not at mu_min.
        """
        solver = make_mcp_solver_diff(_F, return_aux=True, **_EARLY_STOP_KWARGS)
        l = jnp.array([0.0])
        u = jnp.array([3.0])
        x0 = jnp.array([1.0])
        theta = jnp.array([2.0])

        x_star, info = solver(l, u, x0, theta)
        mu_used = info.mu_used
        # Sanity: we are in the early-stop regime.
        assert float(mu_used) > 1e-5

        def loss(t):
            x, _ = solver(l, u, x0, t)
            return jnp.sum(x**2)

        impl_grad = jax.grad(loss)(theta)

        # Independent adjoint at mu_used:
        #   H(x, mu, theta) = 0  ⇒  dx/dtheta = -(dH/dx)^{-1} dH/dtheta
        #   d(||x||^2)/dtheta   = 2 * x · dx/dtheta
        def H(x_in, t_in, mu):
            return smoothed_residual(x_in, _F, l, u, mu, t_in)

        dH_dx = jax.jacfwd(lambda xx: H(xx, theta, mu_used))(x_star)
        dH_dt = jax.jacfwd(lambda tt: H(x_star, tt, mu_used))(theta)
        dx_dt = -jnp.linalg.solve(dH_dx, dH_dt)
        expected_grad = 2.0 * (x_star @ dx_dt)

        assert jnp.allclose(impl_grad, expected_grad, rtol=1e-6, atol=1e-8)
