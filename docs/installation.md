# Installation

## Supported versions

| Component | Supported |
|---|---|
| Python | 3.11, 3.12, 3.13 (CI matrix) |
| JAX / jaxlib | `>=0.4.38, <0.5` (0.4.x series) |

Python 3.10 may work but is not exercised by CI. We officially support
Python 3.11+ because that is the version range covered by the current CI
matrix; older Python versions are not supported.

The JAX range is pinned to the 0.4.x series. The lower bound is needed
for stable composition of `jax.experimental.checkify` with
`jit`/`grad`/`vmap` and for the current sparse linear solver API
(`jax.scipy.sparse.linalg.gmres`, `cg`). The upper bound excludes the
0.5+ and 0.10+ series, which regressed the `vmap(checkify(...))`
composition that this library's strict-validation path uses. The range
can be widened after re-testing against newer JAX.

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

This solver requires float64 precision. JAX defaults to float32, so you must enable float64 before importing JAX arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Place this at the top of your script, before any `jnp.array(...)` calls. If you forget, the solver will still run but may produce inaccurate results or fail to converge.
