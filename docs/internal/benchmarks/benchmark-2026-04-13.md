# Performance Benchmark — 2026-04-13

## Environment

- Platform: macOS (Apple Silicon)
- JAX: 0.4.38
- Device: CpuDevice
- Dtype: float64 (jax_enable_x64=True)

## Results

2D LCP: M = [[2,-1],[-1,3]], q = [1,-2], bounds [0, inf).

| Benchmark                          | ms/call | notes                        |
|------------------------------------|---------|------------------------------|
| solve_mcp (eager)                  | ~1400   | non-differentiable           |
| make_mcp_solver_diff factory       | < 0.1   | essentially free             |
| make_mcp_solver_diff forward       | ~1500   | differentiable, no grad      |
| jax.grad (eager)                   | ~6800   | forward + backward, no JIT   |
| jax.jit(jax.grad) — first call     | ~5500   | includes JAX tracing         |
| jax.jit(jax.grad) — cached         | ~3.1    | ~2000x measured speedup      |

## Observations

- JIT-compiled gradients measured ~2000x faster than eager `jax.grad` on this problem (3 ms vs 6.8 s).
- Factory creation (`make_mcp_solver_diff`) is essentially free.
- The initial JIT trace takes ~5.5 s, then subsequent calls with different `theta` values run in ~3 ms.

## Changes that enabled this

1. **Pure-JAX continuation kernel** (item 6): `lax.while_loop` replaces Python `for` loop, making the forward path fully traceable
2. **JIT-compatible diff solver** (item 7): `make_mcp_solver_diff` uses the pure kernel directly, bypassing the eager `solve_mcp` wrapper
3. **Hoisted VJP closure** (item 13): `jax.vjp(H_x, x_star)` built once per backward pass instead of per GMRES iteration

## Reproducing

```bash
python benchmarks/bench_solve.py
```
