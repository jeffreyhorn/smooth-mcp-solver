"""Traffic route choice equilibrium with flow conservation."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp


def F_traffic(x):
    # x = flows on two routes
    # Cost functions (increasing)
    cost1 = 2.0 + 0.5 * x[0] ** 2 + 0.1 * x[0] * x[1]
    cost2 = 3.0 + 0.8 * x[1] ** 2 + 0.05 * x[0] * x[1]
    demand = 10.0  # fixed demand
    return jnp.array(
        [cost1 - cost2, x[0] + x[1] - demand]  # route choice equilibrium
    )  # flow conservation


l = jnp.array([0.0, 0.0])
u = jnp.array([10.0, 10.0])
x0 = jnp.array([5.0, 5.0])

# Non-differentiable solve
result = solve_mcp(F_traffic, l, u, x0, verbose=True)
print("Solution:", result.x)
print("F(x*):", F_traffic(result.x))
print(f"  Route flows: x1={result.x[0]:.4f}, x2={result.x[1]:.4f}")
print(f"  Flow sum: {result.x[0] + result.x[1]:.4f} (demand=10)")
print(
    f"Converged: {result.converged} (residual={result.residual_norm:.2e}, steps={result.num_steps})"
)

# Differentiable solve
diff_solver = make_mcp_solver_diff(F_traffic)
sol_diff = diff_solver(l, u, x0, jnp.zeros(0))
print("Solution (diff):", sol_diff)
