# Solver tuning and diagnostics

## Defaults

The defaults work well for most problems. Only adjust when you hit a specific issue.

## Evidence basis

Guidance in this document falls into two categories:

- **Benchmarked** — backed by measured numbers from `benchmarks/bench_continuation.py` on the in-repo problem matrix (six problems, three `mu_decay` values). These claims are scoped to the benchmark matrix and may not generalize to all problems. Benchmarked sections cite the table explicitly.
- **Heuristic** — standard numerical-methods advice or reasonable defaults not directly measured on this repo's benchmark suite. Heuristic guidance is marked as such where it appears.

## Three dimensions to think about

When tuning the solver, keep three different quantities separate — they are
not the same, and they often move in opposite directions:

1. **Continuation step count** — how many outer mu-reduction steps the
   solver takes (`info.num_steps`, `result.num_steps`). Cheap to reason
   about; easy to count.
2. **Newton difficulty per step** — how hard each Newton subproblem is.
   This is mostly invisible as a single number, but it shows up as line
   search activity, iteration counts inside Newton, and the conditioning
   of the linearized system. Aggressive continuation schedules (small
   `mu_decay`) make each step harder because the jump between smoothed
   subproblems is larger.
3. **Total runtime** — wall-clock time per solve. This is what users
   actually care about, and it is the *product* of step count and
   per-step cost, not just the step count.

The folklore "smaller `mu_decay` is faster" assumes per-step cost stays
roughly constant. On the benchmark matrix in this repo it does not:
fewer steps at `mu_decay=0.1` are often offset by a larger per-step
cost, and wall time can be flat or *worse* than the `0.5` default. See
[Continuation schedule (`mu_decay`)](#continuation-schedule-mu_decay)
below for concrete numbers, and the
[profiling recipe](#how-to-profile-mu_decay-on-your-problem)
for measuring this on your own problem.

## Continuation schedule (`mu_decay`)

Controls how aggressively the smoothing parameter is reduced. Smaller
values mean fewer continuation steps but larger per-step difficulty.

### Step count vs. wall time (benchmarked)

Numbers from `benchmarks/bench_continuation.py` on the in-repo problem
matrix (macOS Intel CPU, JAX 0.4.38, float64; see
`docs/internal/benchmarks/` for the raw record):

| Problem          | n   | Solver | `mu_decay` | Steps | Wall time (ms/call) | Notes |
|------------------|----|--------|----------:|------:|---------------------:|-------|
| `lcp_2d`         |  2 | dense  | `0.50`    |    35 |                 952  | baseline |
| `lcp_2d`         |  2 | dense  | `0.25`    |    18 |                 936  | flat |
| `lcp_2d`         |  2 | dense  | `0.10`    |    11 |                 975  | flat |
| `ncp_2d`         |  2 | dense  | `0.50`    |    34 |                1089  | baseline |
| `ncp_2d`         |  2 | dense  | `0.25`    |    18 |                1151  | flat |
| `ncp_2d`         |  2 | dense  | `0.10`    |    11 |                1069  | flat |
| `spatial_eq`     |  2 | dense  | `0.50`    |    32 |                1187  | baseline |
| `spatial_eq`     |  2 | dense  | `0.25`    |    16 |                1342  | ~13% slower |
| `spatial_eq`     |  2 | dense  | `0.10`    |    10 |                1121  | flat |
| `obstacle_50d`   | 50 | dense  | `0.50`    |    40 |                1097  | baseline |
| `obstacle_50d`   | 50 | dense  | `0.25`    |    21 |                1070  | flat |
| `obstacle_50d`   | 50 | dense  | `0.10`    |    13 |                1136  | flat |
| `random_lcp_30d` | 30 | dense  | `0.50`    |    40 |                1144  | baseline |
| `random_lcp_30d` | 30 | dense  | `0.25`    |    21 |                1128  | flat |
| `random_lcp_30d` | 30 | dense  | `0.10`    |    13 |                1032  | ~10% faster |
| `obstacle_100d`  |100 | gmres  | `0.50`    |    41 |               54854  | baseline |
| `obstacle_100d`  |100 | gmres  | `0.25`    |    21 |              122090  | 2x fewer steps, **2.2x slower** |
| `obstacle_100d`  |100 | gmres  | `0.10`    |    13 |              183654  | 3x fewer steps, **3.3x slower** |

Reading this table:
- **Step count always falls** as `mu_decay` shrinks. The step-count
  column is the one you would expect from the continuation schedule.
- **Wall time rarely falls** and sometimes rises sharply. On
  `obstacle_100d` (GMRES), `mu_decay=0.1` takes 3x fewer steps but
  runs 3.3x slower because each Newton step requires far more GMRES
  iterations when the mu jump is larger.
- **One problem gets marginally faster from more aggressive decay.**
  `random_lcp_30d` is a well-conditioned SPD system where per-step
  Newton cost barely changes, so the step-count saving translates to a
  ~10% wall-time saving. This is the exception, not the rule.
- **GMRES amplifies the effect.** On dense problems the cost increase
  per step is modest (Newton converges quickly regardless of mu jump).
  On GMRES problems the inner linear-solve cost grows dramatically
  with larger mu jumps, making aggressive decay counterproductive.

This is exactly the "fewer steps ≠ faster runtime" point. The
benchmarks on this repo's problem matrix do not support the intuition
that smaller `mu_decay` is a general-purpose speed optimization.

### When to reach for a different value

| `mu_decay` | Steps to `mu_min=1e-12` | When to use |
|---|---|---|
| `0.5` (default) | ~35–41 | First choice for any problem. Robust and, on the six-problem benchmark matrix, beaten on wall time only once (marginally, on a well-conditioned dense SPD system). |
| `0.25` | ~18–21 | Try if per-step cost is very low (small `n`, cheap `F`, no GMRES) and step count is a bottleneck. On the benchmark matrix this is flat or slightly slower than `0.5`; on the GMRES problem it is 2.2x slower. |
| `0.1` | ~11–13 | Experimental. Don't assume it's faster — measure. On dense problems it is flat or marginally faster; on GMRES problems it is dramatically slower (3.3x on `obstacle_100d`). |
| `0.7` | ~80 | Conservative fallback (heuristic — not benchmarked). Try if `0.5` diverges or the Newton steps are struggling. More steps, each easier. |

Rule of thumb: **profile before changing `mu_decay`**. The step-count
numbers above are deterministic properties of the schedule; the
wall-time numbers are problem-dependent and must be measured on your
problem.

### How to profile `mu_decay` on your problem

```python
import time
import jax
from smooth_mcp import make_mcp_solver

for decay in [0.5, 0.25, 0.1]:
    solver = jax.jit(make_mcp_solver(F, mu_decay=decay, return_aux=True))
    x, info = solver(l, u, x0, theta)  # first call traces
    x.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(10):
        x, info = solver(l, u, x0, theta)
        x.block_until_ready()
    ms = (time.perf_counter() - t0) / 10 * 1000

    print(f"mu_decay={decay}: {int(info.num_steps)} steps, "
          f"{ms:.1f} ms/call, converged={bool(info.converged)}")
```

Pick the `mu_decay` with the lowest wall time *that still converges*.
Don't assume fewer steps is faster — measure.

## Iteration budget (`max_mu_steps`)

Safety limit on continuation steps. If `converged=False`:

1. Check progress with `verbose=True` (for `solve_mcp`) or `return_aux=True` (for `make_mcp_solver` or `make_mcp_solver_diff`).
2. If the residual is decreasing but hasn't reached tolerance, increase `max_mu_steps`.
3. If the residual is stuck or growing, the problem may need different settings (see below).

For intentionally truncated solves (coarse approximations in inner loops), decrease `max_mu_steps`. Gradients from truncated solves are consistent with the smoothed system actually solved.

## Linear solver (`linear_solver`)

*Heuristic guidance. The n ≈ 100 crossover is a rough rule of thumb, not a
benchmarked threshold — the actual crossover depends on Jacobian sparsity,
hardware, and problem structure.*

| Solver | Complexity | When to use |
|---|---|---|
| `"dense"` (default) | O(n^3) per Newton step | n < ~100. Forms full Jacobian via `jax.jacfwd`. |
| `"gmres"` | O(n * k) per Newton step | n > ~100 or sparse Jacobians. Matrix-free via JVPs. |

If using `"gmres"` and Newton convergence is slow, tune:
- `krylov_tol`: Loosen (e.g., `1e-4`) if inner solves are too expensive, tighten if Newton steps are inaccurate. `krylov_tol` is the GMRES linear-solve tolerance, while `newton_tol` is a nonlinear residual threshold, so they are not directly comparable. In practice, if you target a very small `newton_tol`, you may need to tighten `krylov_tol` as well; on `obstacle_100d`, the default `krylov_tol=1e-6` was too loose when `newton_tol=1e-10`, and the residual stalled (benchmarked).
- `krylov_maxiter`: Increase for hard linear systems.
- `krylov_restart`: Increase (e.g., `50`, `100`) if GMRES stalls.

## Regularization (`regularize`)

*Heuristic guidance. The threshold values below are standard numerical-methods
rules of thumb, not benchmarked on this repo's problem matrix.*

Tikhonov regularization added to the Newton Jacobian: `J + reg * I`.

| Value | Effect |
|---|---|
| `1e-12` (default) | Prevents singular-Jacobian failures. Negligible accuracy impact. |
| `1e-8` to `1e-6` | Use if you see NaN or divergence, especially with symmetric initial guesses. |
| `0` | No regularization. Only if you're sure the Jacobian is always well-conditioned. |

## Adjoint solver settings (differentiable solver only)

*Heuristic guidance except where noted. The GMRES-vs-CG distinction is a
mathematical property (CG requires SPD), not a heuristic.*

These control the backward pass linear solve, not the forward solve.

**`adjoint_method`**: `"gmres"` (default) works for any problem. Use `"cg"` only when the Jacobian `dH/dx` is symmetric positive-definite. CG is faster per iteration but gives wrong gradients on non-SPD systems.

**Adjoint convergence issues**: If gradients are NaN but the forward solve converges:
1. Increase `gmres_maxiter` (e.g., `1000`).
2. Loosen `gmres_tol` (e.g., `1e-6`).
3. Supply a `precond` callable if the adjoint system is poorly conditioned.

## Diagnostics

### `solve_mcp` with `verbose=True`

Prints per-step mu values and lets you see convergence progress:

```python
result = solve_mcp(F, l, u, x0, verbose=True)
# Step  0 | mu = 1.00e+00
# Step  1 | mu = 5.00e-01
# ...
# Finished. Final residual norm ~ 4.29e-11
```

### Factory solvers with `return_aux=True`

Returns a `SolveInfo` alongside the solution, compatible with JIT. Works
on both the forward-only factory and the differentiable factory:

```python
# Forward-only:
solver = make_mcp_solver(F, return_aux=True)
x, info = solver(l, u, x0, theta)

# Differentiable:
solver = make_mcp_solver_diff(F, return_aux=True)
x, info = solver(l, u, x0, theta)

print(info.num_steps, info.residual_norm, info.converged, info.mu_used)
```

`SolveInfo` fields are JAX arrays (not Python scalars). On the
differentiable factory, the fields do not carry gradients.
