# Troubleshooting

## Solver does not converge (`converged=False`)

1. **Check progress**: Use `verbose=True` or `return_aux=True` to see if the residual is decreasing.
2. **Increase budget**: Try `max_mu_steps=100` or higher.
3. **Slow down continuation**: Try `mu_decay=0.7` for a more gradual schedule.
4. **Better initial guess**: `x0` closer to the solution reduces Newton steps needed per mu level.
5. **Check problem formulation**: Verify that `F`, `l`, `u` define a well-posed MCP with a solution.

## NaN in the solution

- **Singular Jacobian**: Increase `regularize` (e.g., `1e-8`). This is common with symmetric initial guesses on symmetric problems.
- **GMRES failure**: If using `linear_solver="gmres"`, a failed GMRES solve propagates NaN. Increase `krylov_maxiter` or loosen `krylov_tol`.
- **NaN in bounds**: `l` and `u` must not contain NaN. Use `jnp.inf` / `-jnp.inf` for unbounded components.
- **`strict_validation=True` poisoning the output**: If you built the solver with `strict_validation=True`, `NaN` output means the solver detected invalid bounds under tracing (`l > u` or NaN in `l`/`u`). Check `SolveInfo.converged` (it will be `False` for poisoned rows) or inspect the inputs.

## Invalid bounds not rejected under `jax.jit` / `jax.grad` / `jax.vmap`

Symptom: you pass `l > u` or NaN bounds into a jitted or vmapped call and the solver returns a finite number instead of raising.

Cause: by default, value checks are skipped under tracing because the concrete values are not available. Only shape checks survive.

Fix — pick whichever matches your pipeline (see [Input validation](api.md#input-validation) in `docs/api.md`):

- Bounds are static across the training loop: call `preflight_validate(l, u, x0)` once before entering the loop.
- Bounds are traced, batched, or learned: build the solver with `strict_validation=True` (on `make_mcp_solver` or `make_mcp_solver_diff`). Invalid rows produce `NaN` output and `SolveInfo.converged=False`.
- You want raised exceptions: use `strict_validation="checkify"` (on `make_mcp_solver` or `make_mcp_solver_diff`) and call `err.throw()`.

## NaN in gradients (forward solve is fine)

The adjoint linear solve may be failing:

1. Increase `gmres_maxiter` (e.g., `1000`).
2. Loosen `gmres_tol` (e.g., `1e-6`).
3. Try `adjoint_method="cg"` if your system is symmetric positive-definite.
4. Supply a `precond` callable.

## `TracerBoolConversionError` inside `jax.jit`

If you see this error when JIT-compiling code that uses `solve_mcp`:

`solve_mcp` is not JIT-compatible. Use `make_mcp_solver` (forward-only) or `make_mcp_solver_diff` (differentiable) instead:

```python
# Instead of:
jax.jit(lambda: solve_mcp(F, l, u, x0))  # fails

# Forward-only:
solver = jax.jit(make_mcp_solver(F))
solver(l, u, x0, theta)                   # works

# Differentiable (supports jax.grad):
solver = jax.jit(make_mcp_solver_diff(F))
solver(l, u, x0, theta)                   # works
```

## Slow performance

- **Always JIT in loops**: `jax.jit(jax.grad(loss))` is often orders of magnitude faster than eager `jax.grad` after the initial trace.
- **Continuation tuning**: Smaller `mu_decay` (e.g., `0.25`, `0.1`) reduces step count but increases per-step cost. Profile before assuming fewer steps is faster — on the benchmark matrix, aggressive decay is flat or slower for most problems, and dramatically slower with GMRES. See [Continuation schedule](tuning.md#continuation-schedule-mu_decay) in `docs/tuning.md`.
- **Large problems**: Switch to `linear_solver="gmres"` for n > ~100 to avoid O(n^3) Jacobian construction.
- **First call is slow**: The first JIT call includes tracing. Subsequent calls with the same input shapes are fast.

## Shape mismatch errors

`l`, `u`, and `x0` must all have the same shape. Common mistakes:
- Passing a scalar where an array is expected: use `jnp.array([0.0])` not `0.0`.
- Mismatched dimensions between bounds and initial guess.

## Float32 warnings or poor accuracy

This solver requires float64. Add `jax.config.update("jax_enable_x64", True)` at the top of your script, before any JAX array operations. See [Float64](installation.md#float64) in `docs/installation.md` for details.
