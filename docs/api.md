# API reference

## Solvers

### Choosing an entry point

The package exposes three solver entry points. Pick one based on what you need:

| Use case | Entry point | Why |
|---|---|---|
| One-off solve, debugging, exploration | `solve_mcp` | Eager; returns Python-scalar diagnostics; supports `verbose=True` per-step prints |
| Repeated forward solves (no gradients) | `make_mcp_solver` wrapped in `jax.jit` | Reusable JIT-compatible forward factory; no `custom_vjp` overhead |
| Differentiable / gradient-based training | `make_mcp_solver_diff` wrapped in `jax.jit` | Implicit differentiation with `custom_vjp`; supports `jax.grad`, `jax.vmap`, `jax.jit` |

The two factories share option names with `solve_mcp`. Moving from `solve_mcp` to `make_mcp_solver` is mostly mechanical (`make_mcp_solver` requires `theta`; pass `jnp.zeros(0)` for single-argument `F(x)`). Moving from `make_mcp_solver` to `make_mcp_solver_diff` is the same call shape plus the differentiability; no forward-path changes are needed.

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
| `l` | 1D array | Lower bounds |
| `u` | 1D array | Upper bounds (`jnp.inf` for unbounded) |
| `x0` | 1D array | Initial guess (same shape as `l`) |
| `theta` | array, optional | Parameters for `F(x, theta)`. Optional if `F` takes only `x`. |
| `verbose` | bool | Print per-step progress (default `False`) |

Plus all [common solver options](#common-solver-options) and [forward linear solver options](#forward-linear-solver-options).

### `make_mcp_solver(F_fn, ...)`

Factory that returns a reusable forward-only MCP solver. The returned function is JIT-compatible but does not install a `custom_vjp` — use `make_mcp_solver_diff` if you need `jax.grad`.

```python
import jax
import jax.numpy as jnp
from smooth_mcp import make_mcp_solver

def F(x, theta):
    return x - theta

l = jnp.array([0.0, 0.0])
u = jnp.full(2, jnp.inf)
x0 = jnp.array([0.5, 0.5])
theta = jnp.array([1.0, 2.0])

solver = jax.jit(make_mcp_solver(F))        # wrap in jax.jit for repeated solves
x_star = solver(l, u, x0, theta)             # first call traces
x_star = solver(l, u, x0, theta + 0.1)       # subsequent calls reuse compiled graph
```

For diagnostics, build the factory with `return_aux=True`:

```python
solver = jax.jit(make_mcp_solver(F, return_aux=True))
x_star, info = solver(l, u, x0, theta)
# info.converged, info.residual_norm, info.num_steps, info.mu_used are JAX arrays
```

**Arguments:** All [common solver options](#common-solver-options), [forward linear solver options](#forward-linear-solver-options), plus:

| Argument | Type | Default | Description |
|---|---|---|---|
| `return_aux` | bool | `False` | Return `(x_star, SolveInfo)` instead of just `x_star` |
| `strict_validation` | bool or str | `False` | Opt-in traced validation — see [Input validation](#input-validation) |

**Returns:** A function `solve(l, u, x0, theta) -> x_star` (or `-> (x_star, SolveInfo)` if `return_aux=True`). If `strict_validation="checkify"`, the signature is wrapped to return `(Error, ...)` per `jax.experimental.checkify` conventions.

**Note:** The returned function is not auto-JIT-wrapped. Wrap it in `jax.jit(...)` yourself for repeated fast forward solves. Without `jax.jit`, each call rebuilds the Newton and continuation kernels.

### `make_mcp_solver_diff(F_fn, ...)`

Factory that returns a differentiable MCP solver with `custom_vjp`. The returned function supports `jax.grad` / `jax.value_and_grad`, `jax.jit`, and `jax.vmap`.

Gradients flow w.r.t. `theta`, `l`, and `u` by default. To also differentiate w.r.t. `x0`, set `differentiate_through_x0=True`.

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
| `gmres_maxiter` | int | `500` | Adjoint GMRES max iterations |
| `gmres_restart` | int | `30` | Adjoint GMRES restart parameter |
| `cg_tol` | float | `1e-8` | Adjoint CG tolerance |
| `cg_maxiter` | int | `1000` | Adjoint CG max iterations |
| `precond` | callable or None | `None` | Preconditioner `M(v) -> v` for adjoint solve |
| `differentiate_through_x0` | bool | `False` | Enable straight-through gradients for `x0` |
| `return_aux` | bool | `False` | Return `(x_star, SolveInfo)` instead of just `x_star` |

**Returns:** A function `solve(l, u, x0, theta) -> x_star` (or `-> (x_star, SolveInfo)` if `return_aux=True`).

**Note:** `solve_mcp` cannot be JIT-compiled because it returns Python scalars. Use `make_mcp_solver` (forward-only) or `make_mcp_solver_diff` (differentiable) for JIT-compatible code.

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
| `mu_used` | array | Last smoothing parameter at which the Newton solve ran. May be `> mu_min` when the residual at `mu_min` passed the tolerance before continuation reached `mu_min` — gradients in `make_mcp_solver_diff` are taken at this `mu_used`. |
| `num_steps` | array | Continuation steps taken |
| `residual_norm` | array | Max absolute smoothed residual at `mu_min` (the limiting system, not `mu_used`) |
| `converged` | array | `residual_norm < newton_tol` |

`SolveInfo` fields are JAX arrays (not Python scalars) and are stop-gradiented — gradients flow through `x_star` only.

## Common solver options

Accepted by `solve_mcp`, `make_mcp_solver`, and `make_mcp_solver_diff`:

| Parameter | Default | Description |
|---|---|---|
| `mu_init` | `1.0` | Initial smoothing parameter |
| `mu_min` | `1e-12` | Terminal smoothing parameter |
| `mu_decay` | `0.5` | Multiplicative decay per step |
| `newton_tol` | `1e-10` | Newton convergence tolerance |
| `max_mu_steps` | `50` | Maximum continuation steps |
| `armijo_c` | `1e-4` | Armijo sufficient decrease parameter |
| `backtrack_rho` | `0.5` | Line search contraction factor |
| `max_ls_steps` | `20` | Maximum backtracking steps in the Armijo line search. The search always evaluates `alpha=1` first (Armijo-checked) and then backtracks up to `max_ls_steps` times. If no `alpha` passes Armijo, the step is rejected (`alpha=0`, iterate unchanged) — the merit function is never increased. `max_ls_steps=0` means "try `alpha=1` only, reject if it fails." |

## Forward linear solver options

Accepted by `solve_mcp`, `make_mcp_solver`, and `make_mcp_solver_diff`:

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

All three entry points accept `F_fn` in two forms:

- **`F(x)`** — for problems with no parameters. With `solve_mcp`, `theta` is optional. With `make_mcp_solver` and `make_mcp_solver_diff`, pass a dummy `theta` (e.g., `jnp.zeros(0)`) since the returned function always requires four arguments for JAX tracing.
- **`F(x, theta)`** — for parametrized problems.

## Input validation

> **Migration note (2026-04-18):** the factory default for `strict_validation` changed from `False` to `True`. Invalid traced inputs now produce `NaN` output and `SolveInfo.converged=False` instead of silent finite results. If you relied on the old default and your bounds are always valid, pass `strict_validation=False` explicitly to restore the previous fast path.

All three entry points enforce these checks on `l`, `u`, and `x0`:
- Rank check: `l`, `u`, and `x0` must each be 1D (`ndim == 1`). The solver supports only 1D vector state; higher-rank inputs raise `ValueError` at the public boundary. For problems with multidimensional state, flatten with `.ravel()` before calling the solver and reshape the result.
- Shape checks: all three arrays must have the same shape.
- NaN checks: `l`, `u`, and `x0` must not contain NaN.
- Bound ordering: `l <= u` element-wise.

Rank and shape checks run unconditionally because shapes are available even under tracing. Value checks (NaN, ordering) behave differently depending on execution context:

- **Eager execution** — full value checks run. Invalid input raises `ValueError`.
- **Traced execution** (`jax.jit`, `jax.grad`, `jax.vmap`) — the factory default is now **safe**: `make_mcp_solver` and `make_mcp_solver_diff` both construct with `strict_validation=True` (NaN-poisoning), so invalid traced bounds or `x0` produce `NaN` output (and `SolveInfo.converged=False` with `return_aux=True`) instead of silently flowing through. Pass `strict_validation=False` to opt out of the check — see mode 4 below.

Four mechanisms cover the traced case. The factories use mode 2 by default; pick another if it matches your pipeline better:

### 1. `preflight_validate(l, u, x0)` — cheapest, for static bounds

If your bounds do not change across a training loop, call `preflight_validate` once before entering the loop. Zero overhead inside the jitted region.

```python
from smooth_mcp import make_mcp_solver_diff, preflight_validate
import jax

preflight_validate(l, u, x0)  # raises ValueError on invalid input
solver = jax.jit(make_mcp_solver_diff(F))
for step in range(n_steps):
    x_star = solver(l, u, x0, thetas[step])
```

Use this whenever `l`, `u`, and `x0` are known up front and stay constant.

### 2. `strict_validation=True` — NaN-poisoning (factory default)

When bounds or `x0` are themselves traced (learned, batched, or swept with `vmap`), the factory default catches invalid inputs via NaN-poisoning. The factory sanitizes `l`, `u`, and `x0` so the inner solve is well-defined, then replaces the output with NaN when any of those inputs was invalid (NaN in `l`, `u`, or `x0`, or `l > u`). With `return_aux=True`, `SolveInfo.converged` is forced to `False` and `residual_norm` to `NaN` on bad rows.

Both `make_mcp_solver` and `make_mcp_solver_diff` construct with `strict_validation=True` by default, so this mode is on unless you opt out:

```python
# Forward-only:
solver = make_mcp_solver(F, return_aux=True)  # strict_validation=True by default
x, info = jax.jit(solver)(l, u, x0, theta)

# Differentiable:
solver = make_mcp_solver_diff(F, return_aux=True)  # strict_validation=True by default
x, info = jax.jit(solver)(l, u, x0, theta)
# info.converged == False for any invalid row under vmap
```

Composes with `jit`, `grad`, `vmap`, and their combinations at near-zero overhead. Failure surfaces as `NaN` output and `SolveInfo.converged=False`, not as an exception — callers must inspect the result.

### 3. `strict_validation="checkify"` — exception-style, with a `vmap` caveat

For users who want a raised exception rather than a silent NaN, use checkify mode. The factory returns a function whose signature is wrapped per `jax.experimental.checkify` conventions: `(l, u, x0, theta) -> (Error, x_star)` (or `(Error, (x_star, SolveInfo))` when `return_aux=True`).

```python
solver = make_mcp_solver_diff(F, strict_validation="checkify")
err, x = solver(l, u, x0, theta)
err.throw()  # raises JaxRuntimeError on invalid input
```

Composes with `jit` and `grad`.

**Known upstream JAX bug with `vmap` (as of JAX 0.10):** the composition `vmap(solver)` used to work on JAX ≤ 0.6 and reported per-row errors (`"at mapped index N: ..."`), but regressed starting in JAX 0.7 and is still broken in 0.10 — `jax.vmap(checkify_solver)(...)` raises `ValueError: foreach() argument 2 is longer than argument 1` deep inside JAX's jaxpr evaluator. The continuation kernel uses `lax.while_loop`, which is the trigger. The wrong ordering (`checkify(vmap(...))`) has always raised, and still does:

```python
# WRONG (always) — checkify(vmap(...)) rejected at trace time:
#   ValueError: Checkify does not support batched while-loops

# INTENDED — vmap(solver):
#   Works on JAX <= 0.6; broken on JAX 0.7+ (upstream regression).
```

**Recommendation for batched validation:** use `strict_validation=True` (NaN-poisoning) with `vmap`. NaN-poisoning composes cleanly with `vmap`/`jit`/`grad` on every JAX version we test and reports per-row failure via `SolveInfo.converged=False` and `NaN` in the output row. See mode 2 above. Reserve `strict_validation="checkify"` for non-batched `jit`/`grad` workflows.

Per-call overhead on a small 1D problem is about 16% under warm `jit`; negligible for real workloads.

### 4. `strict_validation=False` — explicit opt-out of traced validation

For the rare case where traced-validation overhead matters and you can guarantee every call sees valid inputs (for example, right after `preflight_validate` on static bounds inside a tight inner loop), pass `strict_validation=False` to opt out:

```python
# You've guaranteed l, u, x0 are valid — skip the traced check.
preflight_validate(l, u, x0)
solver = jax.jit(make_mcp_solver(F, strict_validation=False))
for theta in theta_sweep:
    x_star = solver(l, u, x0, theta)  # no per-call traced validation
```

When `strict_validation=False`, invalid traced inputs produce silent finite output (or NaN propagation, depending on the problem) — the factory does not check them. This used to be the default; flipping it to `True` makes the safe path the default and the fast path explicit.

### Which to use

| Situation | Mechanism |
|---|---|
| Factory default (safe) | `strict_validation=True` (no argument needed) |
| Bounds static across a training loop | `preflight_validate` before the loop |
| Want raised exceptions on invalid input | `strict_validation="checkify"` |
| Tight inner loop with guaranteed-valid inputs | `strict_validation=False` (explicit opt-out) |
| Debugging / one-off solves | `solve_mcp` (always eager, always checks) |

## Comparing the three entry points

| | `solve_mcp` | `make_mcp_solver` | `make_mcp_solver_diff` |
|---|---|---|---|
| **Returns** | `MCPResult` (Python scalars) | JAX array `x_star` (or `(x_star, SolveInfo)`) | JAX array `x_star` (or `(x_star, SolveInfo)`) |
| **Gradients** | No | No | Yes (`jax.grad`, `jax.value_and_grad`, `jax.vjp`) |
| **JIT-compatible** | No | Yes (wrap in `jax.jit`) | Yes (wrap in `jax.jit`) |
| **Input validation** | Full (shapes, NaN, bounds) | Full when eager; shape-only under JIT by default, opt-in strict modes | Full when eager; shape-only under JIT by default, opt-in strict modes |
| **Diagnostics** | `MCPResult.converged`, `.residual_norm`, `.num_steps` | `SolveInfo` via `return_aux=True` | `SolveInfo` via `return_aux=True` |
| **Verbose output** | `verbose=True` prints per-step progress | Not available | Not available |
| **`theta` argument** | Optional (can omit if `F` takes only `x`) | Required (pass `jnp.zeros(0)` as dummy) | Required (pass `jnp.zeros(0)` as dummy) |
| **Reusable across calls** | No (rebuilds every call) | Yes (under `jax.jit`, compiled graph is reused) | Yes (under `jax.jit`, compiled graph is reused) |
| **`custom_vjp` overhead** | n/a | No | Yes (negligible on the forward path) |

**When to reach for which:**
- One-off solve, debugging, interactive exploration: **`solve_mcp`**.
- Repeated forward solves with no gradients (e.g. a search sweep, batched evaluation, Monte Carlo, or a training loop where only `F`'s parameters change): **`make_mcp_solver` wrapped in `jax.jit`**.
- Gradient-based training, implicit differentiation, or any workflow that uses `jax.grad` through the MCP solution: **`make_mcp_solver_diff` wrapped in `jax.jit`**.

The two factories produce the same forward solution as `solve_mcp` on the same inputs. See `tests/test_forward_factory.py` for the field-by-field parity contract.
