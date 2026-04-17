"""Benchmark matrix for continuation settings (mu_decay, max_mu_steps).

Measures step count, convergence, and wall time across representative
problems and mu_decay schedules.
"""

import platform
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from smooth_mcp import solve_mcp

# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


def _make_lcp_2d():
    """2D linear complementarity: M = [[2,-1],[-1,3]], q = [1,-2]."""

    def F(x):
        M = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
        return M @ x + jnp.array([1.0, -2.0])

    return "lcp_2d", F, jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(2), {}


def _make_ncp_2d():
    """2D nonlinear complementarity."""

    def F(x):
        x1, x2 = x[0], x[1]
        return jnp.array([
            x1**2 + x1 * x2 - 3 * x1 + 2,
            x2**2 + x1 * x2 - 3 * x2 + 2,
        ])

    return (
        "ncp_2d",
        F,
        jnp.zeros(2),
        jnp.full(2, jnp.inf),
        jnp.array([0.5, 1.5]),
        {},
    )


def _make_spatial_eq():
    """2D spatial price equilibrium with finite bounds."""

    def F(x):
        x1, x2 = x[0], x[1]
        f1 = x1 - 5.0 + 0.2 * x1**2 - 0.1 * x2
        f2 = x2 - 3.0 + 0.3 * x2**2 - 0.05 * x1
        return jnp.array([f1, f2])

    return (
        "spatial_eq",
        F,
        jnp.array([0.0, 0.0]),
        jnp.array([10.0, 10.0]),
        jnp.array([4.0, 3.0]),
        {},
    )


def _make_obstacle(n=50, solver_kwargs=None):
    """n-D discretized obstacle problem."""
    h = 1.0 / (n + 1)
    x_grid = jnp.linspace(h, 1.0 - h, n)
    diag = jnp.full(n, -2.0 / h**2)
    off = jnp.full(n - 1, 1.0 / h**2)
    A = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
    f = jnp.full(n, -1.0)
    phi = 0.2 * jnp.exp(-((x_grid - 0.5) ** 2) / 0.005)

    def F(u):
        return -A @ u - f

    return (
        f"obstacle_{n}d",
        F,
        phi,
        jnp.full(n, jnp.inf),
        jnp.zeros(n),
        solver_kwargs or {},
    )


def _make_random_lcp(n=30, seed=0):
    """Medium-size dense LCP: M = A A^T + reg*I (SPD) with random q.

    Exercises the "dense" linear solver on a non-PDE problem where the
    Jacobian is genuinely dense (no tridiagonal structure). This is a
    different regime from the obstacle problem — no spatial structure,
    full-rank coupling across all components.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (n, n))
    M = A @ A.T + 0.1 * jnp.eye(n)
    q = jax.random.normal(k2, (n,))

    def F(x):
        return M @ x + q

    return (
        f"random_lcp_{n}d",
        F,
        jnp.zeros(n),
        jnp.full(n, jnp.inf),
        jnp.zeros(n),
        {},  # default dense
    )


PROBLEMS = [
    _make_lcp_2d(),
    _make_ncp_2d(),
    _make_spatial_eq(),
    _make_obstacle(50),
    _make_random_lcp(30),
    _make_obstacle(
        100,
        solver_kwargs={
            "linear_solver": "gmres",
            "krylov_tol": 1e-10,
            "krylov_maxiter": 1000,
            "krylov_restart": 50,
        },
    ),
]

MU_DECAYS = [0.5, 0.25, 0.1]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_benchmark(n_timing=5):
    print("=" * 72)
    print("Continuation benchmark matrix")
    print("=" * 72)
    print(f"Date:       {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Platform:   {platform.platform()}")
    print(f"Processor:  {platform.processor()}")
    print(f"JAX:        {jax.__version__}")
    print(f"Devices:    {jax.devices()}")
    print("Dtype:      float64 (jax_enable_x64=True)")
    print()

    header = (
        f"{'Problem':<18} {'n':>4} {'solver':>6} {'mu_decay':>8} "
        f"{'steps':>6} {'resid':>10} {'conv':>5} {'ms/call':>8}"
    )
    print(header)
    print("-" * len(header))

    for name, F, l, u, x0, solver_kwargs in PROBLEMS:
        n = int(l.shape[0])
        linear_solver = solver_kwargs.get("linear_solver", "dense")
        for decay in MU_DECAYS:
            result = solve_mcp(F, l, u, x0, mu_decay=decay, **solver_kwargs)

            # Time it
            times = []
            for _ in range(n_timing):
                t0 = time.perf_counter()
                solve_mcp(F, l, u, x0, mu_decay=decay, **solver_kwargs)
                times.append(time.perf_counter() - t0)
            avg_ms = 1000 * sum(times) / len(times)

            print(
                f"{name:<18} {n:>4} {linear_solver:>6} {decay:>8.2f} "
                f"{result.num_steps:>6} {result.residual_norm:>10.2e} "
                f"{'Y' if result.converged else 'N':>5} {avg_ms:>8.1f}"
            )
        print()


if __name__ == "__main__":
    run_benchmark()
