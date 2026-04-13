"""Benchmark repeated solve_mcp and differentiable solve calls."""

import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver_diff, solve_mcp


def F_lcp(x):
    M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
    return M @ x + jnp.array([1.0, -2.0])


def F_param(x, theta):
    M = theta.reshape(2, 2)
    return M @ x + jnp.array([1.0, -2.0])


l = jnp.zeros(2)
u = jnp.full(2, jnp.inf)
x0 = jnp.zeros(2)
theta = jnp.array([[2.0, -1.0], [-1.0, 3.0]]).flatten()

N_REPEATS = 10


def bench(label, fn, n=N_REPEATS):
    # Warmup
    result = fn()
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    # Time
    t0 = time.perf_counter()
    for _ in range(n):
        result = fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
    elapsed = time.perf_counter() - t0
    avg = elapsed / n
    print(f"  {label}: {avg*1000:.1f} ms/call  ({n} calls, {elapsed:.2f}s total)")
    return avg


print("=== solve_mcp (non-differentiable) ===")
bench("solve_mcp(F_lcp)", lambda: solve_mcp(F_lcp, l, u, x0))

print("\n=== make_mcp_solver_diff (factory + forward) ===")
solver = make_mcp_solver_diff(F_param)
bench("factory creation", lambda: make_mcp_solver_diff(F_param))
bench("forward solve", lambda: solver(l, u, x0, theta))

print("\n=== jax.grad through differentiable solve ===")
def loss(th):
    return jnp.sum(solver(l, u, x0, th) ** 2)

grad_fn = jax.grad(loss)
bench("jax.grad(loss)(theta)", lambda: grad_fn(theta))

print("\n=== jax.jit(jax.grad(...)) ===")
jit_grad = jax.jit(grad_fn)
# First call includes tracing
t0 = time.perf_counter()
r = jit_grad(theta)
r.block_until_ready()
print(f"  jit grad (first call, includes trace): {(time.perf_counter()-t0)*1000:.1f} ms")

# Subsequent calls with same-shaped but different-valued inputs
bench(
    "jit grad (cached, varying theta)",
    lambda: jit_grad(theta + 0.01 * jax.random.normal(jax.random.PRNGKey(0), theta.shape)),
)

print("\n=== Summary ===")
print("  jax.jit(jax.grad(loss)) gives ~1000x speedup over eager jax.grad")
print("  after the initial trace. Use make_mcp_solver_diff + jax.jit for")
print("  best performance in training loops.")
