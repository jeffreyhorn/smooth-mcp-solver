# Initial Code Review: smooth-mcp

## 1. Algorithmic / Approach Weaknesses

### 1a. Potential sign error in backward pass (CRITICAL)

`smooth_mcp/core.py:258` — The implicit function theorem for `H(x*, theta) = 0` gives:

```
dx*/dtheta = -(dH/dx)^{-1} (dH/dtheta)
```

The VJP should be `dL/dtheta = -g^T (dH/dx)^{-1} (dH/dtheta)`. The code solves `(dH/dx)^T lambda = g` correctly, but then computes:

```python
dtheta = vjp_theta(lambda_star)[0]   # = (dH/dtheta)^T lambda
```

This is missing a negation — it should be `dtheta = -vjp_theta(lambda_star)[0]`. The demos do not verify gradient correctness against finite differences, so this has not been caught. **Needs urgent verification.**

### 1b. CG used on a potentially non-symmetric operator

`smooth_mcp/core.py:235-241` — The adjoint solve uses CG, which requires a symmetric positive-definite operator. The operator `JTv` computes `(dH/dx)^T v`, and `(dH/dx)^T` is generally not symmetric for MCPs. CG may "converge" to a wrong answer without any indication. The GMRES fallback only triggers when CG reports failure (`info != 0`), but CG can report success on non-SPD systems while returning an incorrect solution.

### 1c. No convergence failure detection

`smooth_mcp/core.py:97-99` — `_newton_solve_fixed_mu` returns whatever `x` results after `max_iter` with no indication of whether convergence was achieved. The outer `solve_mcp` loop has no way to detect or report divergence. There is no return of convergence info, no warning, and no error.

### 1d. Dense Jacobian via `jacfwd` — O(n^3) per Newton step

`smooth_mcp/core.py:70` — The full dense Jacobian is formed at every Newton step, followed by a dense `linalg.solve`. For problems with hundreds or thousands of variables, this is prohibitively expensive. The backward pass already uses iterative solvers (CG/GMRES), but the forward solve has no matrix-free Newton-Krylov option.

### 1e. `dir_deriv` assumes exact Newton direction

`smooth_mcp/core.py:74`:

```python
dir_deriv = -jnp.sum(H ** 2)
```

This is only correct when `d = -J^{-1} H` is computed exactly. If `J` is singular or ill-conditioned and `linalg.solve` returns a poor result, the directional derivative is wrong and the Armijo condition may accept bad steps or reject good ones. There is no check on the condition number of `J`.

### 1f. `smooth_min` numerical hazard for negative inputs

`smooth_mcp/core.py:21`:

```python
return (4.0 * a * b - mu) / (2.0 * (a + b + s))
```

When both `a` and `b` are negative and close together, the denominator `a + b + s` approaches `2*a + sqrt(mu)`, which can be near zero or negative. This creates a division-by-zero hazard for problems where `z = x - F(x)` goes significantly below the lower bound.

### 1g. `_BIG = 1e15` is arbitrary

`smooth_mcp/core.py:7` — This sentinel value for infinite bounds has no justification for why `1e15` vs `1e10` or `1e20`. It works in float64 but would cause overflow in float32 (`1e15^2 = 1e30`, fine for float64 max ~`1.8e308`, but only marginally safe in float32 max ~`3.4e38`). The code does not enforce or verify float64 usage.


## 2. Performance / Efficiency Issues

### 2a. Redundant `@jax.jit` decorators inside `_newton_solve_fixed_mu`

`smooth_mcp/core.py:61, 67, 91` — The `merit`, `body`, and `cond` functions are decorated with `@jax.jit` but are passed to `lax.while_loop`, which already traces and compiles them. The extra `@jax.jit` wrappers add hash-based cache lookup overhead during tracing and create unnecessary tracing boundaries that prevent JAX from optimizing across calls (e.g., `merit` called inside `body`).

### 2b. Recompilation for every mu step

`_newton_solve_fixed_mu` is called in a Python `for` loop with a different `mu` each time. Since `mu` is captured as a Python float in the closures of `body`/`cond`/`merit`, JAX must retrace and recompile the while_loop for each distinct `mu` value. With `max_mu_steps=50`, this means up to 50 recompilations per `solve_mcp` call.

### 2c. Residual evaluated 3 times per Newton iteration

- `cond` (line 93) evaluates `smoothed_residual` to check convergence
- `body` (line 69) re-evaluates it for the Newton step
- `merit` (line 73) re-evaluates it again via `0.5 * sum(H**2)`

The residual could be computed once and carried in the loop state.

### 2d. `solve_mcp_diff` rebuilds the solver on every call

`smooth_mcp/core.py:268-282` — Each call to `solve_mcp_diff` creates a new closure and `custom_vjp`-decorated function, causing JAX to retrace/recompile. If used in a training loop, this is a significant performance trap. The docstring calls it a "convenience wrapper" but does not warn about this.


## 3. API Design Issues

### 3a. Mandatory `theta` parameter even when unused

Every `F_fn` must accept `(x, theta)` even for problems with no parameters. Six of eight demos create a dummy `theta = jnp.array([0.0])` that is never used. The `theta` parameter should be optional.

### 3b. No convergence information returned

`solve_mcp` returns only the solution array. There is no way to programmatically check if the solver converged, how many iterations were taken, or what the final residual was (unless `verbose=True`, which only prints). A structured result (e.g., a NamedTuple with `x`, `converged`, `residual`, `num_steps`) would be more useful.

### 3c. No input validation

There is no checking that `l <= u` elementwise, that `x0` is within bounds, that array dimensions match, or that `mu_init > 0`. Bad inputs produce cryptic JAX tracing errors.

### 3d. No gradients through bounds

`smooth_mcp/core.py:262` returns `None` for the `dl` and `du` gradients. For use cases like optimizing over constraints, bound gradients are mathematically well-defined and would be valuable.


## 4. Maintainability Issues

### 4a. Duplicate demo files

`demos/nonlinear_1d_mcp.py` and `demos/simple_nonlinear_1d_mcp_with_bounds.py` are functionally identical — same `F`, same bounds `[0, 2]`, same `x0`, same output. One should be removed or differentiated.

### 4b. Copy-pasted boilerplate across all demos

All 8 demos repeat the same imports and `jax_enable_x64` config, followed by the same pattern: define F, define bounds, call `solve_mcp`, call `make_mcp_solver_diff`. A shared utility or parametrized runner would reduce duplication.

### 4c. Variable name `l` for lower bounds

`l` is easily confused with `1` in many fonts. `lb`/`ub` or `lower`/`upper` would be clearer. This is used throughout the library and all demos.


## 5. Documentation Issues

### 5a. Missing docstrings on public API functions

`smooth_max`, `smooth_min`, `smooth_proj`, and `smoothed_residual` are exported in `__init__.py` but have no docstrings. `_newton_solve_fixed_mu` also lacks a docstring.

### 5b. README example requires unexplained dummy `theta`

The README usage example defines `F(x, theta)` that ignores `theta` and creates a dummy `theta = jnp.array([0.0])`. This is confusing for new users encountering the library.

### 5c. Minimal explanation of the mathematical method

The README "How it works" section is 3 sentences for a nontrivial numerical method. There is no explanation of what `mu` means, why the smoothing approximations were chosen, or what convergence properties to expect.

### 5d. No license file

No LICENSE file exists in the repository.


## 6. Testing Issues

### 6a. Zero tests

There is no `tests/` directory, no test files, no test configuration. For a numerical solver with implicit differentiation, this is a serious gap. Critical tests needed:

- **Gradient correctness** via `jax.test_util.check_grads` or finite-difference comparison — would catch the potential sign bug in 1a
- **Known solutions** for LCPs and NCPs against analytical answers
- **Edge cases**: single-variable problems, all variables at bounds, infinite bounds on both sides, singular Jacobians
- **`smooth_min`/`smooth_max`/`smooth_proj` unit tests** with known values and extreme inputs
- **Convergence failure behavior**

### 6b. Demos have no assertions

All demos print results but never assert correctness. They cannot serve as regression tests.


## 7. Packaging / Project Structure Issues

### 7a. Incomplete `pyproject.toml`

Missing `description`, `authors`, `license`, `readme`, and `urls` fields. `pip show smooth-mcp` shows blank metadata.

### 7b. No `__version__` in package

`smooth_mcp/__init__.py` does not expose `__version__`, though version `0.1.0` is defined in `pyproject.toml`.

### 7c. No `.gitignore`

Build artifacts (`*.egg-info/`, `__pycache__/`) are not excluded.

### 7d. No optional dependencies for development

No `[project.optional-dependencies]` section for dev/test tools (pytest, etc.).
