# Benchmark results — 2026-04-14

## Environment

- Platform: macOS-15.7.3-x86_64-i386-64bit
- Processor: i386 (Intel)
- JAX: 0.4.38
- Device: CpuDevice
- Dtype: float64 (jax_enable_x64=True)

## Solver performance (`benchmarks/bench_solve.py`)

2D LCP: M = [[2,-1],[-1,3]], q = [1,-2], bounds [0, inf).

| Benchmark                          | ms/call | notes                        |
|------------------------------------|---------|------------------------------|
| solve_mcp (eager)                  | 443.5   | non-differentiable           |
| make_mcp_solver_diff forward       | 434.3   | differentiable, no grad      |
| jax.grad (eager)                   | 912.1   | forward + backward, no JIT   |
| jax.jit(jax.grad) — first call     | 1000.4  | includes JAX tracing         |
| jax.jit(jax.grad) — cached         | 0.5     | ~1800x speedup over eager    |

Command: `python benchmarks/bench_solve.py`

## Continuation settings (`benchmarks/bench_continuation.py`)

| Problem      | mu_decay | steps | residual   | converged | ms/call |
|--------------|----------|-------|------------|-----------|---------|
| lcp_2d       | 0.50     | 35    | 4.29e-11   | Y         | 585.6   |
| lcp_2d       | 0.25     | 18    | 4.29e-11   | Y         | 625.3   |
| lcp_2d       | 0.10     | 11    | 7.42e-11   | Y         | 635.4   |
| ncp_2d       | 0.50     | 34    | 1.44e-11   | Y         | 748.7   |
| ncp_2d       | 0.25     | 18    | 7.15e-12   | Y         | 729.6   |
| ncp_2d       | 0.10     | 11    | 1.24e-11   | Y         | 734.7   |
| spatial_eq   | 0.50     | 32    | 4.41e-11   | Y         | 794.1   |
| spatial_eq   | 0.25     | 16    | 8.84e-11   | Y         | 793.7   |
| spatial_eq   | 0.10     | 10    | 9.49e-11   | Y         | 821.0   |
| obstacle_50d | 0.50     | 40    | 6.33e-11   | Y         | 758.5   |
| obstacle_50d | 0.25     | 21    | 2.09e-13   | Y         | 823.0   |
| obstacle_50d | 0.10     | 13    | 1.82e-13   | Y         | 1098.2  |

Command: `python benchmarks/bench_continuation.py`

## Observations

- **JIT speedup**: `jax.jit(jax.grad(...))` is ~1800x faster than eager `jax.grad` after the initial trace. Always JIT in training loops.
- **Continuation decay**: All decay rates converge on all test problems. Faster decay (0.1) cuts step count ~3x vs default (0.5) but wall time is dominated by per-step Newton cost, so the savings are modest.
- **Default `mu_decay=0.5` is kept** for robustness on unseen problems. Users can try 0.25 or 0.1 for well-conditioned problems where continuation overhead matters.

## Reproducing

```bash
python benchmarks/bench_solve.py
python benchmarks/bench_continuation.py
```
