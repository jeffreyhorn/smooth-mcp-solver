"""LCP as MCP: F(x) = Mx + q with x >= 0."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp

M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
q = jnp.array([1.0, -2.0])


def F_lcp(x):
    return M @ x + q


l = jnp.zeros(2)
u = jnp.full(2, jnp.inf)
x0 = jnp.zeros(2)

# Non-differentiable solve
result = solve_mcp(F_lcp, l, u, x0, verbose=True)
print("Solution:", result.x)
print("F(x*):", F_lcp(result.x))
print(
    f"Converged: {result.converged} (residual={result.residual_norm:.2e}, steps={result.num_steps})"
)

# Expected: one variable at 0 and the other positive.
# x* ≈ [0, 2/3], F(x*) ≈ [1/3, 0]

# Differentiable solve
diff_solver = make_mcp_solver_diff(F_lcp)
sol_diff = diff_solver(l, u, x0, jnp.zeros(0))
print("Solution (diff):", sol_diff)
