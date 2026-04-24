# Installation

## Supported versions

| Component | Supported |
|---|---|
| Python | 3.11, 3.12, 3.13 (CI matrix) |
| JAX / jaxlib | `>=0.4.38` (pinned lower bound) |

Python 3.10 may work but is not exercised by CI. We officially support
Python 3.11+ because that is the version range covered by the current CI
matrix; older Python versions are not supported.

JAX `>=0.4.38` is required for stable composition of `jax.experimental.checkify`
with `jit`/`grad`/`vmap` and for the current sparse linear solver API
(`jax.scipy.sparse.linalg.gmres`, `cg`). Newer JAX releases are expected to
work; CI does not currently test against a JAX matrix.

## Standard install

```bash
pip install .
```

This installs the `smooth_mcp` package along with its dependencies (`jax>=0.4.38`, `jaxlib>=0.4.38`).

## Development install

For an editable install with test and lint dependencies:

```bash
pip install -e ".[dev]"
```

## JAX platform notes

JAX installs CPU support by default. For GPU acceleration, install the appropriate JAX variant *before* installing this package:

```bash
# Example: CUDA 12
pip install jax[cuda12]
pip install .
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for platform-specific instructions (CUDA, ROCm, TPU).

## Float64

This solver is designed for float64 and is tested only at float64. JAX defaults to float32; enable float64 before importing JAX arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Place this at the top of your script, before any `jnp.array(...)` calls.

### What happens at float32

float32 execution is not runtime-rejected — the solver will still build and run. It is also not validated: the test suite globally enables x64 in `tests/conftest.py`, so no float32 convergence or gradient-accuracy guarantee exists. Expect any of the following when you skip `jax_enable_x64`:

- Silent loss of precision in the Newton step, especially as `mu` approaches `mu_min`. The default `newton_tol=1e-10` and `mu_min=1e-12` are below float32's ~`1e-7` relative precision, so convergence can stall well before those thresholds.
- Adjoint GMRES (`gmres_tol=1e-8` by default) may not reach tolerance in float32, producing NaN gradients after GMRES's `maxiter` cap is hit.
- The solver may report `converged=False` with a finite but too-large residual, or `converged=True` with a residual that is correct in float32 but wrong by float64 standards.

If you hit any of those, enable x64 first — that is the only tested configuration.
