# Performance Benchmark (2026-04-13)

2D LCP problem on CPU (Apple Silicon, JAX 0.4.38, float64).

## Results

| Operation | Time per call |
|---|---|
| `solve_mcp` (non-diff, eager) | ~1.4 s |
| `make_mcp_solver_diff` factory | < 0.1 ms |
| Diff forward solve (eager) | ~1.5 s |
| `jax.grad(loss)` (eager) | ~6.8 s |
| `jax.jit(jax.grad(loss))` first call (trace) | ~5.5 s |
| `jax.jit(jax.grad(loss))` cached | **3.1 ms** |

## Key takeaways

- **JIT-compiled gradients are ~2000x faster** than eager `jax.grad` after the initial trace (3 ms vs 6.8 s)
- The initial JIT trace takes ~5.5 s (compiling the full forward + backward), then subsequent calls with different `theta` values execute in ~3 ms
- Factory creation (`make_mcp_solver_diff`) is essentially free
- For training loops: create the solver once, JIT the grad, and amortize the trace cost

## Changes that enabled this

1. **Pure-JAX continuation kernel** (item 6): `lax.while_loop` replaces Python `for` loop, making the forward path fully traceable
2. **JIT-compatible diff solver** (item 7): `make_mcp_solver_diff` uses the pure kernel directly, bypassing the eager `solve_mcp` wrapper
3. **Hoisted VJP closure** (item 13): `jax.vjp(H_x, x_star)` built once per backward pass instead of per GMRES iteration

## Reproduce

```bash
python benchmarks/bench_solve.py
```
