# smooth-mcp

A JAX library for solving Mixed Complementarity Problems (MCPs) using smoothing methods, with support for implicit differentiation via `jax.grad`.

## What is an MCP?

A Mixed Complementarity Problem finds a vector **x** such that for each component *i*:

- If `x_i = l_i` (at lower bound), then `F_i(x) >= 0`
- If `l_i < x_i < u_i` (interior), then `F_i(x) = 0`
- If `x_i = u_i` (at upper bound), then `F_i(x) <= 0`

MCPs arise in optimization (KKT conditions), game theory (Nash equilibria), economics (spatial price equilibria), and engineering (contact mechanics).

## How it works

### Smoothing approach

The non-smooth MCP complementarity conditions are replaced by a smooth approximation. The key idea is to replace `max(a, b)` and `min(a, b)` with smooth counterparts parameterized by `mu > 0`:

- `smooth_max(a, b, mu) = (a + b + sqrt((a - b)^2 + mu)) / 2`
- `smooth_min(a, b, mu)` — algebraically equivalent to `(a + b - sqrt((a - b)^2 + mu)) / 2`, computed via a [numerically stable reformulation](smooth_mcp/smoothing.py)

These converge to the exact `max`/`min` as `mu -> 0`. The smooth projection `smooth_proj(z, l, u, mu)` composes these to approximate `clip(z, l, u)`, and the **smoothed MCP residual** is:

```
H(x, mu) = x - smooth_proj(x - F(x), l, u, mu)
```

At a solution, `H(x*, mu) ≈ 0` (exactly zero as `mu -> 0`).

### Continuation solver

Rather than solving the smoothed system at a single small `mu` (which would be nearly non-smooth and hard to solve), the solver uses a **mu-continuation** strategy:

1. Start with a large `mu` (default `mu_init = 1.0`) where the problem is well-conditioned
2. Solve `H(x, mu) = 0` using Newton's method with Armijo backtracking line search
3. Reduce `mu` by a factor (default `mu_decay = 0.5`), using the previous solution as the initial guess
4. Repeat until `mu` reaches `mu_min` (default `1e-12`)

Each Newton step computes the Jacobian via `jax.jacfwd` and solves a dense linear system. The line search ensures global convergence by backtracking when the full Newton step doesn't sufficiently decrease the merit function `||H(x)||^2 / 2`.

### Implicit differentiation

The differentiable solver (`make_mcp_solver_diff`) computes the same forward solution but attaches a `custom_vjp` for the backward pass. Instead of differentiating through all the Newton iterations (expensive and numerically unstable), it uses the **implicit function theorem**:

At the solution `H(x*, theta) = 0`, differentiating gives `(dH/dx)(dx*/dtheta) + dH/dtheta = 0`, so:

```
dx*/dtheta = -(dH/dx)^{-1} (dH/dtheta)
```

The backward pass solves the adjoint system `(dH/dx)^T lambda = g` (where `g` is the incoming cotangent) using GMRES, then computes `dtheta = -(dH/dtheta)^T lambda`. This is Jacobian-free — only matrix-vector products via `jax.vjp` are needed, avoiding explicit Jacobian construction in the backward pass.

Gradients are computed at the actual terminal smoothing parameter from the forward solve — not necessarily `mu_min`. This means truncated solves (small `max_mu_steps`) produce gradients consistent with the smoothed system that was actually solved. The differentiable solver is fully compatible with `jax.jit`.

## Installation

```bash
pip install .
```

This installs the `smooth_mcp` package along with its dependencies (`jax`, `jaxlib`).

For development (editable install with test dependencies):

```bash
pip install -e ".[dev]"
```

### JAX platform notes

JAX installs CPU support by default. For GPU acceleration, install the appropriate JAX variant *before* installing this package — see [JAX installation](https://jax.readthedocs.io/en/latest/installation.html).

### Float64

JAX defaults to float32. This solver requires float64 for numerical stability. Enable it before importing JAX arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Usage

### Function signatures

Both `solve_mcp` and `make_mcp_solver_diff` accept `F_fn` in two forms:

- **`F(x)`** — for problems with no parameters. With `solve_mcp`, `theta` is optional and ignored. With `make_mcp_solver_diff`, the returned `solve(l, u, x0, theta)` still requires a `theta` argument for JAX tracing — pass a dummy like `jnp.zeros(0)` if `F` does not use it.
- **`F(x, theta)`** — for parametrized problems. Pass `theta` to differentiate through.

The low-level `smoothed_residual` function requires the two-argument form `F(x, theta)`.

### Solving an MCP

Define your function `F(x)` and call `solve_mcp`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from smooth_mcp import solve_mcp

def F(x):
    M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
    return M @ x + jnp.array([1.0, -2.0])

l = jnp.zeros(2)                # lower bounds
u = jnp.full(2, jnp.inf)       # upper bounds (inf = unbounded)
x0 = jnp.zeros(2)              # initial guess

result = solve_mcp(F, l, u, x0)
print(result.x)                  # ≈ [0, 0.6667]
print(result.converged)          # True
print(result.residual_norm)      # ≈ 4e-11
```

`solve_mcp` returns an `MCPResult` with fields `x`, `converged`, `residual_norm`, and `num_steps`.

If your `F` depends on parameters you want to differentiate through, use `F(x, theta)` and pass `theta`:

```python
def F(x, theta):
    M = theta.reshape(2, 2)
    return M @ x + jnp.array([1.0, -2.0])

result = solve_mcp(F, l, u, x0, theta=theta_init)
```

### Differentiable solving

To compute gradients of a loss through the MCP solution, use `make_mcp_solver_diff`:

```python
from smooth_mcp import make_mcp_solver_diff

def F(x, theta):
    M = theta.reshape(2, 2)
    q = jnp.array([1.0, -2.0])
    return M @ x + q

diff_solver = make_mcp_solver_diff(F)

def loss(theta):
    sol = diff_solver(l, u, x0, theta)
    return jnp.sum(sol ** 2)

grad = jax.grad(loss)(theta_init)
```

`make_mcp_solver_diff` returns a function `solve(l, u, x0, theta)` that produces the same solution as `solve_mcp` but supports `jax.grad` via implicit differentiation.

### JAX integration

The differentiable solver is fully compatible with JAX transformations:

```python
diff_solver = make_mcp_solver_diff(F)

# jax.grad — compute gradients through the MCP solution
grad_fn = jax.grad(lambda th: jnp.sum(diff_solver(l, u, x0, th) ** 2))
g = grad_fn(theta)

# jax.jit — compile the gradient computation (typically orders of magnitude faster)
jit_grad = jax.jit(grad_fn)
g_fast = jit_grad(theta)  # first call traces (~seconds), subsequent calls ~milliseconds
```

Supported JAX transformations:
- `jax.grad` / `jax.value_and_grad` — gradients w.r.t. `theta`, `x0` (with `differentiate_through_x0=True`), `l`, and `u`
- `jax.jit` — full JIT compilation of forward solve, backward pass, or both
- `jax.vmap` — batching over different parameter values (via standard JAX patterns)

Caveats:
- `solve_mcp` cannot be JIT-compiled regardless of `verbose`, because it performs Python-side scalar conversions and returns an `MCPResult` (a Python NamedTuple with eager `float`/`int` fields). Use `make_mcp_solver_diff` for JIT-compatible code.

### Input validation

Both `solve_mcp` and `make_mcp_solver_diff` validate inputs eagerly (outside JIT):
- Shape checks: `l`, `u`, and `x0` must all have the same shape.
- NaN checks: `l` and `u` must not contain NaN.
- Bound ordering: `l <= u` element-wise.

When the differentiable solver is called inside `jax.jit`, shape checks still run (shapes are static in JAX), but value-dependent checks (NaN, bound ordering) are skipped because the values are not available during tracing.

### Solver options

Both `solve_mcp` and `make_mcp_solver_diff` accept these **common parameters**:

| Parameter | Default | Description |
|---|---|---|
| `mu_init` | `1.0` | Initial smoothing parameter |
| `mu_min` | `1e-12` | Terminal smoothing parameter |
| `mu_decay` | `0.5` | Multiplicative decay per step |
| `newton_tol` | `1e-10` | Newton convergence tolerance |
| `max_mu_steps` | `50` | Maximum smoothing reduction steps |
| `armijo_c` | `1e-4` | Armijo sufficient decrease parameter |
| `backtrack_rho` | `0.5` | Line search contraction factor |
| `max_ls_steps` | `20` | Maximum line search steps |

Both also accept these **forward Newton linear solver** parameters:

| Parameter | Default | Description |
|---|---|---|
| `linear_solver` | `"dense"` | `"dense"` (jacfwd + linalg.solve) or `"gmres"` (matrix-free, better for large problems) |
| `krylov_tol` | `1e-6` | Forward GMRES tolerance (only when `linear_solver="gmres"`) |
| `krylov_maxiter` | `500` | Forward GMRES max iterations (only when `linear_solver="gmres"`) |
| `krylov_restart` | `30` | Forward GMRES restart (only when `linear_solver="gmres"`) |
| `regularize` | `1e-12` | Tikhonov regularization on Newton Jacobian (J + reg*I). Set to 0 to disable |

`solve_mcp` additionally accepts:

| Parameter | Default | Description |
|---|---|---|
| `verbose` | `False` | Print progress during solving |

`make_mcp_solver_diff` additionally accepts these **backward (adjoint) solver** parameters:

| Parameter | Default | Description |
|---|---|---|
| `adjoint_method` | `"gmres"` | Adjoint solver: `"gmres"` (general) or `"cg"` (SPD systems only) |
| `gmres_tol` | `1e-8` | Adjoint GMRES tolerance |
| `gmres_restart` | `30` | Adjoint GMRES restart parameter |
| `gmres_maxiter` | `500` | Adjoint GMRES maximum iterations |
| `cg_tol` | `1e-8` | Adjoint CG tolerance (only when `adjoint_method="cg"`) |
| `cg_maxiter` | `1000` | Adjoint CG max iterations (only when `adjoint_method="cg"`) |
| `precond` | `None` | Preconditioner callable for adjoint solve |
| `differentiate_through_x0` | `False` | Enable straight-through gradients for `x0` |
| `return_aux` | `False` | Return `(x_star, SolveInfo)` instead of just `x_star` |

Note: The forward solver parameters (`linear_solver`, `krylov_*`, `regularize`) control how Newton steps are computed during the solve. The adjoint parameters (`adjoint_method`, `gmres_*`, `cg_*`, `precond`) control the implicit differentiation linear solve in the backward pass only.

### Tuning guide

The defaults work well for most problems. Adjust these settings when you hit specific issues:

**`mu_decay`** — Controls how aggressively the smoothing parameter is reduced. Smaller values mean fewer continuation steps but larger jumps between subproblems.

- Default `0.5` is conservative and robust. Each step halves mu, taking ~40 steps to reach `mu_min=1e-12`.
- `0.25` cuts steps roughly in half with no accuracy loss on well-conditioned problems.
- `0.1` is aggressive (~12 steps), good for easy problems or when profiling shows continuation overhead dominates.
- If the solver diverges or returns NaN, try a *larger* (slower) decay like `0.7`.

**`max_mu_steps`** — Safety limit on continuation steps. Increase if `converged=False` but the solver is making progress (check with `verbose=True` or `return_aux=True`). Decrease for intentionally truncated solves where you want a coarse approximation quickly.

**`linear_solver`** — Choose based on problem dimension:

- `"dense"` (default): Forms the full Jacobian via `jax.jacfwd` and solves with `jnp.linalg.solve`. Fast for n < ~100, O(n^3) per Newton step.
- `"gmres"`: Matrix-free Newton-Krylov. Uses only Jacobian-vector products (JVPs), never forms the Jacobian. Better for n > ~100 or sparse problems. Tune `krylov_tol`, `krylov_maxiter`, and `krylov_restart` if GMRES convergence is slow.

**`regularize`** — Tikhonov regularization added to the Newton Jacobian (J + reg*I). The default `1e-12` prevents singular-Jacobian failures with negligible effect on accuracy. Increase to `1e-8`–`1e-6` if you see NaN or non-convergence on problems with near-singular Jacobians (e.g., symmetric initial guesses). Set to `0` only if you're sure your Jacobian is always well-conditioned.

**Adjoint solver settings** (differentiable solver only):

- `adjoint_method="gmres"` (default) is correct for any problem. Use `"cg"` only when the Jacobian dH/dx is symmetric positive-definite — CG is faster per iteration but will give wrong gradients on non-SPD systems.
- If gradients are NaN but the forward solve converges, the adjoint linear solve may be failing. Try increasing `gmres_maxiter` or loosening `gmres_tol`.
- `precond`: Supply a preconditioner callable `M(v) -> v` for the adjoint system if GMRES convergence is slow. A diagonal or incomplete-Cholesky preconditioner can help on stiff problems.

## API

| Function | Description |
|---|---|
| `solve_mcp(F_fn, l, u, x0, ...)` | Solve an MCP, returns `MCPResult` (theta optional) |
| `make_mcp_solver_diff(F_fn, ...)` | Create a differentiable solver (supports `jax.grad`) |
| `MCPResult` | NamedTuple: `x`, `residual_norm`, `num_steps`, `converged` |
| `SolveInfo` | NamedTuple: `mu_used`, `num_steps`, `residual_norm`, `converged` (from `return_aux=True`) |

Lower-level building blocks (`smooth_max`, `smooth_min`, `smooth_proj`, `smoothed_residual`) are also exported. Note: `smoothed_residual` is a low-level function that requires `F_fn(x, theta)` — it does not auto-normalize single-argument functions like the solver APIs do.

## Demos

The `demos/` directory contains runnable examples:

| Script | Problem |
|---|---|
| `lcp_as_mcp.py` | Linear complementarity problem |
| `differentiable_lcp.py` | LCP with gradients through the solution |
| `nonlinear_1d_mcp.py` | 1D nonlinear MCP with finite bounds |
| `2d_nonlinear_complementarity_problem.py` | 2D nonlinear complementarity problem |
| `kkt_conditions.py` | KKT conditions from bound-constrained QP |
| `bound_optimization.py` | Optimizing bounds via `jax.grad` through the MCP solution |
| `obstacle_problem.py` | 50D discretized obstacle problem (showcases `linear_solver="gmres"`) |
| `spatial_price_equilibrium.py` | Spatial price equilibrium network model |
| `traffic_route_choice.py` | Traffic route choice equilibrium |

Run any demo with:

```bash
python demos/lcp_as_mcp.py
```
