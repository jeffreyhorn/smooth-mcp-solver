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

Each Newton step computes the Jacobian via `jax.jacfwd` and solves a dense linear system. The line search enforces the Armijo sufficient-decrease condition on the merit function `||H(x)||^2 / 2`: it backtracks up to `max_ls_steps` times and, if no step passes Armijo, rejects the step (applies `alpha=0`, iterate unchanged). Newton then stalls at this `mu` and the continuation kernel advances to a smaller `mu` where the system is less nonlinear. The merit function is therefore non-increasing across Newton steps.

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

For development (editable install with test dependencies):

```bash
pip install -e ".[dev]"
```

JAX defaults to float32. This solver requires float64 — enable it before importing JAX arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

See [`docs/installation.md`](docs/installation.md) for GPU/TPU setup and platform details.

## Quickstart

### Choosing an entry point

| Use case | Entry point |
|---|---|
| One-off solve, debugging, exploration | `solve_mcp` |
| Repeated forward solves, no gradients | `make_mcp_solver` wrapped in `jax.jit` |
| Differentiable / gradient-based training | `make_mcp_solver_diff` wrapped in `jax.jit` |

All three produce the same forward solution. See [`docs/api.md`](docs/api.md#choosing-an-entry-point) for the full comparison, parameter tables, and function signatures.

### Solving an MCP

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
```

### Repeated forward solves

For parameter sweeps or batched evaluation, use `make_mcp_solver` wrapped in `jax.jit`. The compiled graph is reused across calls:

```python
from smooth_mcp import make_mcp_solver

def F(x, theta):
    M = theta.reshape(2, 2)
    return M @ x + jnp.array([1.0, -2.0])

solver = jax.jit(make_mcp_solver(F))

for theta in theta_sweep:
    x_star = solver(l, u, x0, theta)
```

### Differentiable solving

To compute gradients through the MCP solution:

```python
from smooth_mcp import make_mcp_solver_diff

diff_solver = make_mcp_solver_diff(F)

def loss(theta):
    return jnp.sum(diff_solver(l, u, x0, theta) ** 2)

grad = jax.grad(loss)(theta_init)

# JIT for speed — first call traces, subsequent calls are fast
jit_grad = jax.jit(jax.grad(loss))
grad_fast = jit_grad(theta_init)
```

The differentiable solver supports `jax.grad`, `jax.jit`, and `jax.vmap`. See [`docs/api.md`](docs/api.md) for supported transformations and caveats.

## Documentation

| Document | Contents |
|---|---|
| [`docs/api.md`](docs/api.md) | Full API reference: all parameters, return types, input validation, entry-point comparison |
| [`docs/tuning.md`](docs/tuning.md) | Solver tuning guide: `mu_decay` benchmarks, linear solver selection, regularization, adjoint settings |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Common issues: NaN solutions, slow performance, shape errors, tracing errors |
| [`docs/installation.md`](docs/installation.md) | Platform-specific install (GPU/TPU), float64 setup |

## API

| Function | Description |
|---|---|
| `solve_mcp(F_fn, l, u, x0, ...)` | Solve an MCP eagerly, returns `MCPResult` (theta optional) |
| `make_mcp_solver(F_fn, ...)` | Create a reusable forward-only solver (JIT-compatible, no gradients) |
| `make_mcp_solver_diff(F_fn, ...)` | Create a reusable differentiable solver (supports `jax.grad`) |
| `preflight_validate(l, u, x0)` | Eager validation helper for static bounds before a jitted loop |

Lower-level building blocks (`smooth_max`, `smooth_min`, `smooth_proj`, `smoothed_residual`) are also exported — see [`docs/api.md`](docs/api.md#low-level-building-blocks).

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
