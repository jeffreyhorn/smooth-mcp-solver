"""KKT conditions from bound-constrained QP as a natural MCP.

Solves: min 0.5 x'Qx + c'x  s.t.  x >= 0

The KKT conditions for this problem are:
    x >= 0,  F(x) = Qx + c >= 0,  x . F(x) = 0

This is exactly an MCP with F(x) = Qx + c, l = 0, u = inf.
The MCP structure automatically enforces the complementarity conditions --
no need to manually encode primal/dual variables or x*lambda terms.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp

# Problem data
Q = jnp.diag(jnp.array([2.0, 4.0]))
c = jnp.array([-1.0, 2.0])


def F_kkt(x):
    """Gradient of the QP objective: F(x) = Qx + c."""
    return Q @ x + c


l = jnp.zeros(2)
u = jnp.full(2, jnp.inf)
x0 = jnp.ones(2) * 0.5

# Solve
result = solve_mcp(F_kkt, l, u, x0, verbose=True)
print("Solution x*:", result.x)
print("F(x*) = Qx* + c:", F_kkt(result.x))
print(
    f"Converged: {result.converged} (residual={result.residual_norm:.2e}, steps={result.num_steps})"
)

# Verify KKT conditions:
#   x* >= 0 (feasibility)
#   F(x*) >= 0 (stationarity + dual feasibility)
#   x* . F(x*) = 0 (complementarity)
print()
print("KKT verification:")
print("  x* >= 0:", bool(jnp.all(result.x >= -1e-8)))
print("  F(x*) >= 0:", bool(jnp.all(F_kkt(result.x) >= -1e-8)))
print("  x* . F(x*) ≈ 0:", float(jnp.dot(result.x, F_kkt(result.x))))

# Expected: x* ≈ [0.5, 0.0]
#   x*[0] = 0.5 > 0, so F(x*)[0] = 2*0.5 - 1 = 0 (interior)
#   x*[1] = 0.0 at bound, so F(x*)[1] = 0 + 2 = 2 >= 0 (complementary)

# Differentiable solve
diff_solver = make_mcp_solver_diff(F_kkt)
sol_diff = diff_solver(l, u, x0, jnp.zeros(0))
print("\nSolution (diff):", sol_diff)
