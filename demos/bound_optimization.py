"""Optimizing bounds via gradients through the MCP solution.

Demonstrates jax.grad w.r.t. lower and upper bounds.

Problem: F(x) = x + q with l <= x <= u.
- When q > 0: F(l) > 0, so x* = l (at lower bound). Gradient dl flows through.
- When q < 0 and |q| > u: F(u) < 0, so x* = u (at upper bound). Gradient du flows through.
- When the root -q is interior: x* = -q, independent of bounds. Gradients are ~zero.

We optimize bounds to push the solution toward a target.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp

# F(x) = x + q. With q = [2, -10, -0.5]:
#   Component 0: root at -2. With l=0: F(0)=2>0, so x*=l=0 (at lower bound)
#   Component 1: root at 10. With u=5: F(5)=-5<0, so x*=u=5 (at upper bound)
#   Component 2: root at 0.5. Interior, so x*=0.5 (bound-independent)
q = jnp.array([2.0, -10.0, -0.5])


def F(x, theta):
    return x + theta


theta = q
x0 = jnp.ones(3)
target = jnp.array([1.0, 3.0, 0.5])

solver = make_mcp_solver_diff(F)

# --- Optimize lower bounds ---
print("=== Optimizing lower bounds ===")
u_fixed = jnp.array([10.0, 5.0, 10.0])
l_init = jnp.zeros(3)

result = solve_mcp(F, l_init, u_fixed, x0, theta)
print(f"Initial l:        {l_init}")
print(f"Initial solution: {result.x}")
print("  (x[0]=l[0]=0 at lower bound, x[1]=u[1]=5 at upper, x[2]=0.5 interior)")


def loss_l(l):
    sol = solver(l, u_fixed, x0, theta)
    return jnp.sum((sol - target) ** 2)


grad_l = jax.grad(loss_l)(l_init)
print(f"Gradient d(loss)/dl: {grad_l}")
print("  (dl[0]=-2: pushing l[0] up moves x[0] toward target=1)")
print("  (dl[1]≈0: x[1] is at upper bound, not lower)")
print("  (dl[2]≈0: x[2] is interior, independent of bounds)")

# Gradient descent on l
l_opt = l_init
for _ in range(50):
    l_opt = l_opt - 0.1 * jax.grad(loss_l)(l_opt)

result_opt = solve_mcp(F, l_opt, u_fixed, x0, theta)
print("\nAfter optimization:")
print(f"  Optimized l: {l_opt}")
print(f"  Solution:    {result_opt.x}")
print(f"  Loss: {float(loss_l(l_init)):.4f} -> {float(loss_l(l_opt)):.4f}")

# --- Optimize upper bounds ---
print("\n=== Optimizing upper bounds ===")
l_fixed = jnp.zeros(3)
u_init = jnp.array([10.0, 5.0, 10.0])


def loss_u(u):
    sol = solver(l_fixed, u, x0, theta)
    return jnp.sum((sol - target) ** 2)


grad_u = jax.grad(loss_u)(u_init)
print(f"Initial u:        {u_init}")
print(f"Gradient d(loss)/du: {grad_u}")
print("  (du[0]≈0: x[0] is at lower bound, not upper)")
print("  (du[1]=4: pushing u[1] down moves x[1] toward target=3)")
print("  (du[2]��0: x[2] is interior, independent of bounds)")

u_opt = u_init
for _ in range(50):
    u_opt = u_opt - 0.1 * jax.grad(loss_u)(u_opt)

result_opt_u = solve_mcp(F, l_fixed, u_opt, x0, theta)
print("\nAfter optimization:")
print(f"  Optimized u: {u_opt}")
print(f"  Solution:    {result_opt_u.x}")
print(f"  Loss: {float(loss_u(u_init)):.4f} -> {float(loss_u(u_opt)):.4f}")
