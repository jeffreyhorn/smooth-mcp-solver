"""KKT conditions from bound-constrained QP: min 0.5 x'Qx + c'x s.t. x >= 0."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp


def F_kkt_qp(x):
    """x[:n] = primal, x[n:] = dual for x >= 0"""
    n = len(x) // 2
    primal = x[:n]
    dual = x[n:]
    Q = jnp.diag(jnp.array([2.0, 4.0]))
    c = jnp.array([-1.0, 2.0])
    stationarity = Q @ primal + c - dual
    complementarity = primal * dual
    return jnp.concatenate([stationarity, complementarity])


l = jnp.zeros(4)
u = jnp.full(4, jnp.inf)
x0 = jnp.ones(4) * 0.5

# Non-differentiable solve
result = solve_mcp(F_kkt_qp, l, u, x0, verbose=True)
print("Solution:", result.x)
print("  primal:", result.x[:2])
print("  dual:  ", result.x[2:])
print("F(x*):", F_kkt_qp(result.x))
print(
    f"Converged: {result.converged} (residual={result.residual_norm:.2e}, steps={result.num_steps})"
)

# Expected: one or more primal components at bound with corresponding dual > 0

# Differentiable solve
diff_solver = make_mcp_solver_diff(F_kkt_qp)
sol_diff = diff_solver(l, u, x0, jnp.zeros(0))
print("Solution (diff):", sol_diff)
print("  primal:", sol_diff[:2])
print("  dual:  ", sol_diff[2:])
