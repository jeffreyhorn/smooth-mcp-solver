# Benchmark results — 2026-04-14

## Environment

- Platform: macOS-15.7.3-x86_64-i386-64bit
- Processor: i386 (Intel)
- JAX: 0.4.38
- Device: CpuDevice
- Dtype: float64 (jax_enable_x64=True)

## Solver performance (`benchmarks/bench_solve.py`)

Last updated: 2026-04-17. 2D LCP: M = [[2,-1],[-1,3]], q = [1,-2], bounds [0, inf).

| Benchmark                                | ms/call | notes                        |
|------------------------------------------|---------|------------------------------|
| solve_mcp (eager)                        | 1652.3  | non-differentiable           |
| make_mcp_solver (eager)                  | 1645.4  | forward-only factory         |
| make_mcp_solver (jit, warm)              |    1.7  | ~970x speedup over eager     |
| make_mcp_solver_diff (eager forward)     | 1754.3  | differentiable, no grad      |
| make_mcp_solver_diff (jit forward, warm) |    1.4  |                              |
| jax.grad (eager)                         | 3588.0  | forward + backward, no JIT   |
| jax.jit(jax.grad) — first call           | 8088.4  | includes JAX tracing         |
| jax.jit(jax.grad) — cached              |    1.7  | ~2074x speedup over eager    |

Note: absolute wall times vary between runs due to machine load. The relative
ordering and speedup ratios are consistent.

Command: `python benchmarks/bench_solve.py`

## Continuation settings (`benchmarks/bench_continuation.py`)

Last updated: 2026-04-17. Six problems, three `mu_decay` values.

| Problem          | n   | solver | mu_decay | steps | residual   | converged | ms/call    |
|------------------|-----|--------|----------|-------|------------|-----------|------------|
| lcp_2d           |   2 | dense  | 0.50     |    35 | 4.29e-11   | Y         |      952   |
| lcp_2d           |   2 | dense  | 0.25     |    18 | 4.29e-11   | Y         |      936   |
| lcp_2d           |   2 | dense  | 0.10     |    11 | 7.42e-11   | Y         |      975   |
| ncp_2d           |   2 | dense  | 0.50     |    34 | 1.44e-11   | Y         |     1089   |
| ncp_2d           |   2 | dense  | 0.25     |    18 | 7.15e-12   | Y         |     1151   |
| ncp_2d           |   2 | dense  | 0.10     |    11 | 1.24e-11   | Y         |     1069   |
| spatial_eq       |   2 | dense  | 0.50     |    32 | 4.41e-11   | Y         |     1187   |
| spatial_eq       |   2 | dense  | 0.25     |    16 | 8.84e-11   | Y         |     1342   |
| spatial_eq       |   2 | dense  | 0.10     |    10 | 9.49e-11   | Y         |     1121   |
| obstacle_50d     |  50 | dense  | 0.50     |    40 | 6.33e-11   | Y         |     1097   |
| obstacle_50d     |  50 | dense  | 0.25     |    21 | 2.09e-13   | Y         |     1070   |
| obstacle_50d     |  50 | dense  | 0.10     |    13 | 1.82e-13   | Y         |     1136   |
| random_lcp_30d   |  30 | dense  | 0.50     |    40 | 8.94e-11   | Y         |     1144   |
| random_lcp_30d   |  30 | dense  | 0.25     |    21 | 2.14e-15   | Y         |     1128   |
| random_lcp_30d   |  30 | dense  | 0.10     |    13 | 1.60e-15   | Y         |     1032   |
| obstacle_100d    | 100 | gmres  | 0.50     |    41 | 8.72e-13   | Y         |    54854   |
| obstacle_100d    | 100 | gmres  | 0.25     |    21 | 1.09e-12   | Y         |   122090   |
| obstacle_100d    | 100 | gmres  | 0.10     |    13 | 8.03e-13   | Y         |   183654   |

GMRES settings for `obstacle_100d`: `krylov_tol=1e-10`, `krylov_maxiter=1000`,
`krylov_restart=50`. Default `krylov_tol=1e-6` is too loose to converge
`newton_tol=1e-10` — the residual stalls at ~1e-6 regardless of step count.

Command: `python benchmarks/bench_continuation.py`

## Observations

- **JIT speedup**: `jax.jit(jax.grad(...))` is ~1800x faster than eager `jax.grad` after the initial trace. Always JIT in training loops.
- **Continuation decay on dense problems**: All decay rates converge. Faster decay (0.1) cuts step count ~3x vs default (0.5) but wall time is flat or marginally slower on most problems. The exception is `random_lcp_30d` where `mu_decay=0.1` is ~10% faster — this is a well-conditioned SPD system where per-step Newton cost barely changes.
- **Continuation decay on GMRES problems**: The "fewer steps ≠ faster" effect is dramatically amplified. On `obstacle_100d`, `mu_decay=0.1` takes 3x fewer steps but is **3.3x slower** because each Newton step requires many more GMRES iterations when the mu jump is larger. `mu_decay=0.25` is 2x fewer steps and 2.2x slower.
- **Default `mu_decay=0.5` is kept** for robustness. On the full benchmark matrix it is never beaten on wall time except marginally on one well-conditioned dense problem.
- **GMRES `krylov_tol` must be ≤ `newton_tol`** for the outer solve to converge. The default `krylov_tol=1e-6` causes the residual to stall when `newton_tol=1e-10`. Users switching to GMRES should tighten `krylov_tol` accordingly.

## Reproducing

```bash
python benchmarks/bench_solve.py
python benchmarks/bench_continuation.py
```
