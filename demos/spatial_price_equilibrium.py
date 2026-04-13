"""Spatial price equilibrium network model."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp


def F_network(x):
    x1, x2 = x
    f1 = x1 - 5.0 + 0.2 * x1**2 - 0.1 * x2
    f2 = x2 - 3.0 + 0.3 * x2**2 - 0.05 * x1
    return jnp.array([f1, f2])


l = jnp.array([0.0, 0.0])
u = jnp.array([10.0, 10.0])
x0 = jnp.array([4.0, 3.0])

# Non-differentiable solve
result = solve_mcp(F_network, l, u, x0, verbose=True)
print("Solution:", result.x)
print("F(x*):", F_network(result.x))
print(
    f"Converged: {result.converged} (residual={result.residual_norm:.2e}, steps={result.num_steps})"
)

# Differentiable solve
diff_solver = make_mcp_solver_diff(F_network)
sol_diff = diff_solver(l, u, x0, jnp.zeros(0))
print("Solution (diff):", sol_diff)
