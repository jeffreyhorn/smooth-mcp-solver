import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp


def F_parametrized(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    M = theta.reshape(2, 2)
    q = jnp.array([1.0, -2.0])
    return M @ x + q


l = jnp.zeros(2)
u = jnp.full(2, jnp.inf)
x0 = jnp.zeros(2)
theta_init = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()

# Non-differentiable solve
result = solve_mcp(F_parametrized, l, u, x0, theta_init, verbose=True)
print("Solution (non-diff):", result.x)
print(
    f"Converged: {result.converged} (residual={result.residual_norm:.2e}, steps={result.num_steps})"
)

# Differentiable solve
diff_solver = make_mcp_solver_diff(
    F_parametrized,
    cg_tol=1e-10,
    cg_maxiter=2000,
    gmres_tol=1e-9,
    gmres_restart=20,
    gmres_maxiter=300,
    differentiate_through_x0=True,
)


sol_diff = diff_solver(l, u, x0, theta_init)
print("Solution (diff):", sol_diff)


def loss(theta, x0):
    sol = diff_solver(l, u, x0, theta)
    return jnp.sum(sol**2) + 0.1 * jnp.sum(theta**2)


grad_loss = jax.grad(loss, argnums=(0, 1))(theta_init, x0)
print("Gradient w.r.t. theta:", grad_loss[0])
print("Gradient w.r.t. x0 (straight-through):", grad_loss[1])
