# API reference

## Solvers

### `solve_mcp(F_fn, l, u, x0, ...)`

Solve a Mixed Complementarity Problem. Returns an `MCPResult`.

```python
from smooth_mcp import solve_mcp

result = solve_mcp(F, l, u, x0)
# result.x            — solution array
# result.converged    — True if residual < newton_tol
# result.residual_norm — max |H(x, mu_min)|
# result.num_steps    — continuation steps taken
```

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `F_fn` | callable | `F(x)` or `F(x, theta)` |
| `l` | array | Lower bounds |
| `u` | array | Upper bounds (`jnp.inf` for unbounded) |
| `x0` | array | Initial guess (same shape as `l`) |
| `theta` | array, optional | Parameters for `F(x, theta)`. Optional if `F` takes only `x`. |
| `verbose` | bool | Print per-step progress (default `False`) |

Plus all [common solver options](#common-solver-options) and [forward linear solver options](#forward-linear-solver-options).

### `make_mcp_solver_diff(F_fn, ...)`

Factory that returns a differentiable MCP solver with `custom_vjp`. The returned function supports `jax.grad`, `jax.jit`, and `jax.vmap`.

```python
import jax
import jax.numpy as jnp
from smooth_mcp import make_mcp_solver_diff

def F(x, theta):
    return x - theta

l = jnp.array([0.0, 0.0])
u = jnp.full(2, jnp.inf)
x0 = jnp.array([0.5, 0.5])
theta = jnp.array([1.0, 2.0])

solver = make_mcp_solver_diff(F)
x_star = solver(l, u, x0, theta)         # forward solve

def loss(theta):
    return jnp.sum(solver(l, u, x0, theta) ** 2)

grad = jax.grad(loss)(theta)              # implicit differentiation
```

**Arguments:** All [common solver options](#common-solver-options), [forward linear solver options](#forward-linear-solver-options), plus:

| Argument | Type | Default | Description |
|---|---|---|---|
| `adjoint_method` | str | `"gmres"` | Adjoint solver: `"gmres"` (general) or `"cg"` (SPD only) |
| `gmres_tol` | float | `1e-8` | Adjoint GMRES tolerance |
| `gmres_restart` | int | `30` | Adjoint GMRES restart parameter |
| `gmres_maxiter` | int | `500` | Adjoint GMRES max iterations |
| `cg_tol` | float | `1e-8` | Adjoint CG tolerance |
| `cg_maxiter` | int | `1000` | Adjoint CG max iterations |
| `precond` | callable or None | `None` | Preconditioner `M(v) -> v` for adjoint solve |
| `differentiate_through_x0` | bool | `False` | Enable straight-through gradients for `x0` |
| `return_aux` | bool | `False` | Return `(x_star, SolveInfo)` instead of just `x_star` |

**Returns:** A function `solve(l, u, x0, theta) -> x_star` (or `-> (x_star, SolveInfo)` if `return_aux=True`).

**Note:** `solve_mcp` cannot be JIT-compiled because it returns Python scalars. Use `make_mcp_solver_diff` for JIT-compatible code.

## Data types

### `MCPResult`

NamedTuple returned by `solve_mcp`.

| Field | Type | Description |
|---|---|---|
| `x` | array | Solution |
| `residual_norm` | float | Max absolute smoothed residual at `mu_min` |
| `num_steps` | int | Continuation steps taken |
| `converged` | bool | `residual_norm < newton_tol` |

### `SolveInfo`

NamedTuple returned alongside `x_star` when `return_aux=True`.

| Field | Type | Description |
|---|---|---|
| `mu_used` | array | Terminal smoothing parameter reached |
| `num_steps` | array | Continuation steps taken |
| `residual_norm` | array | Max absolute smoothed residual at `mu_min` |
| `converged` | array | `residual_norm < newton_tol` |

`SolveInfo` fields are JAX arrays (not Python scalars) and are stop-gradiented — gradients flow through `x_star` only.

## Common solver options

Accepted by both `solve_mcp` and `make_mcp_solver_diff`:

| Parameter | Default | Description |
|---|---|---|
| `mu_init` | `1.0` | Initial smoothing parameter |
| `mu_min` | `1e-12` | Terminal smoothing parameter |
| `mu_decay` | `0.5` | Multiplicative decay per step |
| `newton_tol` | `1e-10` | Newton convergence tolerance |
| `max_mu_steps` | `50` | Maximum continuation steps |
| `armijo_c` | `1e-4` | Armijo sufficient decrease parameter |
| `backtrack_rho` | `0.5` | Line search contraction factor |
| `max_ls_steps` | `20` | Maximum line search steps |

## Forward linear solver options

Accepted by both `solve_mcp` and `make_mcp_solver_diff`:

| Parameter | Default | Description |
|---|---|---|
| `linear_solver` | `"dense"` | `"dense"` (jacfwd + linalg.solve) or `"gmres"` (matrix-free) |
| `krylov_tol` | `1e-6` | GMRES tolerance (only when `linear_solver="gmres"`) |
| `krylov_maxiter` | `500` | GMRES max iterations |
| `krylov_restart` | `30` | GMRES restart parameter |
| `regularize` | `1e-12` | Tikhonov regularization on Newton Jacobian (J + reg*I) |

## Low-level building blocks

These are also exported from `smooth_mcp`:

| Function | Description |
|---|---|
| `smooth_max(a, b, mu)` | Smooth approximation to `max(a, b)` |
| `smooth_min(a, b, mu)` | Smooth approximation to `min(a, b)` |
| `smooth_proj(z, l, u, mu)` | Smooth approximation to `clip(z, l, u)` |
| `smoothed_residual(x, F_fn, l, u, mu, theta)` | Smoothed MCP residual `H(x, mu)` |

**Note:** `smoothed_residual` requires the two-argument form `F(x, theta)`. It does not auto-normalize single-argument functions like the solver APIs do.

## Function signatures

Both solvers accept `F_fn` in two forms:

- **`F(x)`** — for problems with no parameters. With `solve_mcp`, `theta` is optional. With `make_mcp_solver_diff`, pass a dummy `theta` (e.g., `jnp.zeros(0)`) since the returned function always requires four arguments for JAX tracing.
- **`F(x, theta)`** — for parametrized problems.

## Input validation

Both APIs validate inputs eagerly during non-traced execution:
- Shape checks: `l`, `u`, and `x0` must have the same shape.
- NaN checks: `l` and `u` must not contain NaN.
- Bound ordering: `l <= u` element-wise.

Under JAX tracing transforms (for example `jax.jit`, `jax.grad`, or `jax.vmap`), shape checks still run when shapes are available, but value checks are skipped.

## Comparing `solve_mcp` and `make_mcp_solver_diff`

| | `solve_mcp` | `make_mcp_solver_diff` |
|---|---|---|
| **Returns** | `MCPResult` (Python scalars) | JAX array `x_star` (or `(x_star, SolveInfo)`) |
| **Gradients** | No | Yes (`jax.grad`, `jax.vjp`) |
| **JIT-compatible** | No | Yes |
| **Input validation** | Full (shapes, NaN, bounds) | Full when eager; shape-only under JIT |
| **Diagnostics** | `MCPResult.converged`, `.residual_norm`, `.num_steps` | `SolveInfo` via `return_aux=True` |
| **Verbose output** | `verbose=True` prints per-step progress | Not available (use `solve_mcp` for debugging) |
| **`theta` argument** | Optional (can omit if `F` takes only `x`) | Required (pass `jnp.zeros(0)` as dummy) |

Use `solve_mcp` for one-off solves, debugging, and exploration. Use `make_mcp_solver_diff` for gradient-based optimization, JIT compilation, and production training loops.
