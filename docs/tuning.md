# Solver tuning and diagnostics

## Defaults

The defaults work well for most problems. Only adjust when you hit a specific issue.

## Continuation schedule (`mu_decay`)

Controls how aggressively the smoothing parameter is reduced. Smaller values mean fewer continuation steps but larger jumps between subproblems.

| `mu_decay` | Steps to `mu_min=1e-12` | When to use |
|---|---|---|
| `0.5` (default) | ~40 | Conservative. Best for unknown or difficult problems. |
| `0.25` | ~18 | Good for well-conditioned problems. No accuracy loss in benchmarks. |
| `0.1` | ~12 | Aggressive. Use when profiling shows continuation overhead dominates. |
| `0.7` | ~80 | Very conservative. Try if `0.5` diverges on a tough problem. |

## Iteration budget (`max_mu_steps`)

Safety limit on continuation steps. If `converged=False`:

1. Check progress with `verbose=True` (for `solve_mcp`) or `return_aux=True` (for `make_mcp_solver_diff`).
2. If the residual is decreasing but hasn't reached tolerance, increase `max_mu_steps`.
3. If the residual is stuck or growing, the problem may need different settings (see below).

For intentionally truncated solves (coarse approximations in inner loops), decrease `max_mu_steps`. Gradients from truncated solves are consistent with the smoothed system actually solved.

## Linear solver (`linear_solver`)

| Solver | Complexity | When to use |
|---|---|---|
| `"dense"` (default) | O(n^3) per Newton step | n < ~100. Forms full Jacobian via `jax.jacfwd`. |
| `"gmres"` | O(n * k) per Newton step | n > ~100 or sparse Jacobians. Matrix-free via JVPs. |

If using `"gmres"` and Newton convergence is slow, tune:
- `krylov_tol`: Loosen (e.g., `1e-4`) if inner solves are too expensive, tighten if Newton steps are inaccurate.
- `krylov_maxiter`: Increase for hard linear systems.
- `krylov_restart`: Increase (e.g., `50`, `100`) if GMRES stalls.

## Regularization (`regularize`)

Tikhonov regularization added to the Newton Jacobian: `J + reg * I`.

| Value | Effect |
|---|---|
| `1e-12` (default) | Prevents singular-Jacobian failures. Negligible accuracy impact. |
| `1e-8` to `1e-6` | Use if you see NaN or divergence, especially with symmetric initial guesses. |
| `0` | No regularization. Only if you're sure the Jacobian is always well-conditioned. |

## Adjoint solver settings (differentiable solver only)

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

### `make_mcp_solver_diff` with `return_aux=True`

Returns a `SolveInfo` alongside the solution, compatible with JIT:

```python
solver = make_mcp_solver_diff(F, return_aux=True)
x, info = solver(l, u, x0, theta)
print(info.num_steps, info.residual_norm, info.converged)
```

`SolveInfo` fields are JAX arrays (not Python scalars) and do not carry gradients.
