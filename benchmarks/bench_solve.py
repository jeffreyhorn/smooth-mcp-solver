"""Benchmark repeated solve_mcp, forward-only factory, and differentiable solves."""

import platform
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import make_mcp_solver, make_mcp_solver_diff, solve_mcp


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


print("=== solve_mcp (non-differentiable, eager) ===")
solve_mcp_avg = bench("solve_mcp(F_lcp)", lambda: solve_mcp(F_lcp, l, u, x0))

print("\n=== make_mcp_solver (forward-only factory) ===")
fwd_solver = make_mcp_solver(F_param)
bench("factory creation", lambda: make_mcp_solver(F_param))
fwd_eager_avg = bench("eager solve", lambda: fwd_solver(l, u, x0, theta))
fwd_jit = jax.jit(fwd_solver)
fwd_jit_avg = bench("jit solve (warm)", lambda: fwd_jit(l, u, x0, theta))

print("\n=== make_mcp_solver_diff (differentiable factory, forward path) ===")
solver = make_mcp_solver_diff(F_param)
bench("factory creation", lambda: make_mcp_solver_diff(F_param))
diff_eager_avg = bench("eager forward solve", lambda: solver(l, u, x0, theta))
diff_jit = jax.jit(solver)
diff_jit_avg = bench("jit forward solve (warm)", lambda: diff_jit(l, u, x0, theta))

print("\n=== jax.grad through differentiable solve ===")
def loss(th):
    return jnp.sum(solver(l, u, x0, th) ** 2)

grad_fn = jax.grad(loss)
eager_avg = bench("jax.grad(loss)(theta)", lambda: grad_fn(theta))

print("\n=== jax.jit(jax.grad(...)) ===")
jit_grad = jax.jit(grad_fn)
# First call includes tracing
t0 = time.perf_counter()
r = jit_grad(theta)
r.block_until_ready()
print(f"  jit grad (first call, includes trace): {(time.perf_counter()-t0)*1000:.1f} ms")

# Subsequent calls with same-shaped but different-valued inputs
theta_variants = [
    theta + 0.01 * jax.random.normal(jax.random.PRNGKey(i), theta.shape)
    for i in range(N_REPEATS)
]
_variant_idx = [0]


def _bench_varying():
    i = _variant_idx[0] % len(theta_variants)
    _variant_idx[0] += 1
    return jit_grad(theta_variants[i])


jit_avg = bench("jit grad (cached, varying theta)", _bench_varying)

print(f"\n=== Summary ({time.strftime('%Y-%m-%d')}) ===")
print(f"  Platform: {platform.platform()}")
print(f"  JAX: {jax.__version__}, Devices: {jax.devices()}")
print("\n  Forward solves (ms/call):")
print(f"    solve_mcp (eager):               {solve_mcp_avg*1000:.1f}")
print(f"    make_mcp_solver (eager):         {fwd_eager_avg*1000:.1f}")
print(f"    make_mcp_solver (jit):           {fwd_jit_avg*1000:.1f}")
print(f"    make_mcp_solver_diff (eager):    {diff_eager_avg*1000:.1f}")
print(f"    make_mcp_solver_diff (jit):      {diff_jit_avg*1000:.1f}")
if fwd_jit_avg > 0:
    fwd_speedup = solve_mcp_avg / fwd_jit_avg
    print(f"\n  make_mcp_solver(jit) vs solve_mcp(eager): {fwd_speedup:.1f}x")
if jit_avg > 0:
    speedup = eager_avg / jit_avg
    print(f"  jit grad vs eager grad:                    {speedup:.0f}x")
print("\n  Guidance:")
print("    - One-off or debugging:       solve_mcp")
print("    - Repeated forward solves:    make_mcp_solver wrapped in jax.jit")
print("    - Gradient-based training:    make_mcp_solver_diff wrapped in jax.jit")
