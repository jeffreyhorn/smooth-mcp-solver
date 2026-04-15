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

## NaN in gradients (forward solve is fine)

The adjoint linear solve may be failing:

1. Increase `gmres_maxiter` (e.g., `1000`).
2. Loosen `gmres_tol` (e.g., `1e-6`).
3. Try `adjoint_method="cg"` if your system is symmetric positive-definite.
4. Supply a `precond` callable.

## `TracerBoolConversionError` inside `jax.jit`

If you see this error when JIT-compiling code that uses `solve_mcp`:

`solve_mcp` is not JIT-compatible. Use `make_mcp_solver_diff` instead:

```python
# Instead of:
jax.jit(lambda: solve_mcp(F, l, u, x0))  # fails

# Use:
solver = make_mcp_solver_diff(F)
jax.jit(solver)(l, u, x0, theta)          # works
```

## Slow performance

- **Always JIT in loops**: `jax.jit(jax.grad(loss))` is often orders of magnitude faster than eager `jax.grad` after the initial trace.
- **Faster continuation**: Try `mu_decay=0.25` or `0.1` to reduce step count.
- **Large problems**: Switch to `linear_solver="gmres"` for n > ~100 to avoid O(n^3) Jacobian construction.
- **First call is slow**: The first JIT call includes tracing. Subsequent calls with the same input shapes are fast.

## Shape mismatch errors

`l`, `u`, and `x0` must all have the same shape. Common mistakes:
- Passing a scalar where an array is expected: use `jnp.array([0.0])` not `0.0`.
- Mismatched dimensions between bounds and initial guess.

## Float32 warnings or poor accuracy

This solver requires float64. Ensure you set this before any JAX operations:

```python
import jax
jax.config.update("jax_enable_x64", True)
```
