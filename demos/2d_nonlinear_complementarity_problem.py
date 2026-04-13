"""2D nonlinear complementarity problem from the literature."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp


def F_ncp_2d(x):
    x1, x2 = x[0], x[1]
    f1 = x1**2 + x1 * x2 - 3 * x1 + 2
    f2 = x2**2 + x1 * x2 - 3 * x2 + 2
    return jnp.array([f1, f2])


l = jnp.array([0.0, 0.0])
u = jnp.array([jnp.inf, jnp.inf])
x0 = jnp.array([0.5, 1.5])

# Non-differentiable solve
result = solve_mcp(F_ncp_2d, l, u, x0, verbose=True)
print("Solution:", result.x)
print("F(x*):", F_ncp_2d(result.x))
print(
    f"Converged: {result.converged} (residual={result.residual_norm:.2e}, steps={result.num_steps})"
)

# Expected solution: x* ≈ [1.0, 1.0]  (F(x*) = 0 and x* > 0)

# Differentiable solve
diff_solver = make_mcp_solver_diff(F_ncp_2d)
sol_diff = diff_solver(l, u, x0, jnp.zeros(0))
print("Solution (diff):", sol_diff)
