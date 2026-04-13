"""Discretized obstacle problem (higher-dimensional MCP).

A classic free-boundary problem: find the displacement u(x) of an elastic
membrane constrained to lie above an obstacle phi(x), subject to tension.

Discretized on n interior points, the problem becomes an MCP:
    F(u) = -A @ u - f  (negative of the discrete Laplacian residual)
    l = phi             (obstacle height at each grid point)
    u = +inf            (no upper constraint)

where A is the tridiagonal discrete Laplacian and f is the applied load.

This demo uses n=50 variables and showcases linear_solver="gmres" for
matrix-free Newton-Krylov solving.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import solve_mcp

# Grid
n = 50
h = 1.0 / (n + 1)
x_grid = jnp.linspace(h, 1.0 - h, n)

# Discrete Laplacian (tridiagonal): A[i,i] = -2/h^2, A[i,i±1] = 1/h^2
diag = jnp.full(n, -2.0 / h**2)
off = jnp.full(n - 1, 1.0 / h**2)
A = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)

# Applied load (uniform downward force)
f = jnp.full(n, -1.0)

# Obstacle: a bump in the middle, tall enough to be active
phi = 0.2 * jnp.exp(-((x_grid - 0.5) ** 2) / 0.005)


def F_obstacle(u):
    """Residual: F(u) = -A @ u - f (so F(u) = 0 means A @ u = -f)."""
    return -A @ u - f


# Lower bound = obstacle, upper bound = unbounded
l = phi
u_bound = jnp.full(n, jnp.inf)
u0 = jnp.zeros(n)

# Solve with dense linear solver
tol = 1e-8
print(f"Obstacle problem: {n} variables")
print("Solving with linear_solver='dense'...")
result_dense = solve_mcp(
    F_obstacle, l, u_bound, u0, linear_solver="dense", newton_tol=tol
)
print(
    f"  Converged: {result_dense.converged} "
    f"(residual={result_dense.residual_norm:.2e}, steps={result_dense.num_steps})"
)

# Solve with GMRES (matrix-free) linear solver
print("Solving with linear_solver='gmres'...")
result_gmres = solve_mcp(
    F_obstacle, l, u_bound, u0, linear_solver="gmres", newton_tol=tol
)
print(
    f"  Converged: {result_gmres.converged} "
    f"(residual={result_gmres.residual_norm:.2e}, steps={result_gmres.num_steps})"
)

# Verify solutions match
max_diff = float(jnp.max(jnp.abs(result_dense.x - result_gmres.x)))
print(f"  Max difference dense vs gmres: {max_diff:.2e}")

# Report solution properties
sol = result_dense.x
tol_contact = 1e-6
active = jnp.abs(sol - phi) < tol_contact  # points where membrane touches obstacle
n_contact = int(jnp.sum(active))
print("\nSolution:")
print(f"  Range: [{float(jnp.min(sol)):.6f}, {float(jnp.max(sol)):.6f}]")
print(f"  Contact points: {n_contact} of {n}")
if n_contact > 0:
    contact_x = x_grid[active]
    print(
        f"  Contact region: x in [{float(contact_x[0]):.3f}, {float(contact_x[-1]):.3f}]"
    )
