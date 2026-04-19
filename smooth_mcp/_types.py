"""Shared public result types.

Lives in a neutral module so ``forward.py`` and ``diff.py`` can both
import from here without one factory depending on the other. Re-exported
through ``smooth_mcp.__init__`` as the public import path.
"""

from typing import NamedTuple

import jax.numpy as jnp


class MCPResult(NamedTuple):
    """Result from solve_mcp.

    Attributes:
        x: Solution array.
        residual_norm: Max absolute residual at the solution (Python scalar).
        num_steps: Number of mu-continuation steps taken (Python int).
        converged: Whether the solver converged (Python bool).
    """

    x: jnp.ndarray
    residual_norm: float
    num_steps: int
    converged: bool


class SolveInfo(NamedTuple):
    """Auxiliary diagnostics from the factory solvers.

    Returned alongside ``x_star`` when ``return_aux=True`` is passed to
    ``make_mcp_solver`` or ``make_mcp_solver_diff``. Fields are JAX
    array scalars (not Python scalars) so the containing function
    remains JIT-compatible.

    Attributes:
        mu_used: Terminal smoothing parameter actually reached.
        num_steps: Total continuation steps taken.
        residual_norm: Max absolute smoothed residual at ``x_star``
            evaluated at ``mu_min``.
        converged: True if ``residual_norm < newton_tol``.
    """

    mu_used: jnp.ndarray
    num_steps: jnp.ndarray
    residual_norm: jnp.ndarray
    converged: jnp.ndarray
