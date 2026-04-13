"""Nonlinear 1D MCP: F(x) = x^3 - x - 1 with 0 <= x <= 2."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp


def F_nonlinear_1d(x):
    return x**3 - x - 1.0


l = jnp.array([0.0])
u = jnp.array([2.0])
x0 = jnp.array([1.0])

# Non-differentiable solve
result = solve_mcp(F_nonlinear_1d, l, u, x0, verbose=True)
print("Solution:", result.x)
print("F(x*):", F_nonlinear_1d(result.x))
print(
    f"Converged: {result.converged} (residual={result.residual_norm:.2e}, steps={result.num_steps})"
)

# Expected solution ≈ 1.3247 (interior point, so F(x*) ≈ 0)

# Differentiable solve
diff_solver = make_mcp_solver_diff(F_nonlinear_1d)
sol_diff = diff_solver(l, u, x0, jnp.zeros(0))
print("Solution (diff):", sol_diff)
