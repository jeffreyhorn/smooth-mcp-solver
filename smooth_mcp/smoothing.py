"""Smooth approximations for max, min, projection, and MCP residual."""

from typing import Callable

import jax
import jax.numpy as jnp

_BIG = 1e15


@jax.jit
def smooth_max(a: jnp.ndarray, b: jnp.ndarray, mu: float) -> jnp.ndarray:
    """Smooth approximation to elementwise max(a, b).

    Uses the identity smooth_max(a, b, mu) = (a + b + sqrt((a-b)^2 + mu)) / 2.
    Converges to max(a, b) as mu -> 0, and is always >= max(a, b).

    Args:
        a: First input array.
        b: Second input array.
        mu: Smoothing parameter (> 0). Smaller values give a tighter approximation.

    Returns:
        Smooth approximation to max(a, b), same shape as inputs.
    """
    return (a + b + jnp.sqrt((a - b) ** 2 + mu)) / 2.0


@jax.jit
def smooth_min(a: jnp.ndarray, b: jnp.ndarray, mu: float) -> jnp.ndarray:
    """Smooth approximation to elementwise min(a, b).

    Uses a numerically stable reformulation of (a + b - sqrt((a-b)^2 + mu)) / 2
    that avoids catastrophic cancellation when |a - b| >> sqrt(mu).
    Converges to min(a, b) as mu -> 0, and is always <= min(a, b).

    Args:
        a: First input array.
        b: Second input array.
        mu: Smoothing parameter (> 0). Smaller values give a tighter approximation.

    Returns:
        Smooth approximation to min(a, b), same shape as inputs.
    """
    s = jnp.sqrt((a - b) ** 2 + mu)
    denom = a + b + s
    abs_denom = jnp.maximum(jnp.abs(denom), jnp.sqrt(mu))
    safe_denom = jnp.copysign(abs_denom, denom + 1e-300)
    return (4.0 * a * b - mu) / (2.0 * safe_denom)


@jax.jit
def smooth_proj(
    z: jnp.ndarray, l: jnp.ndarray, u: jnp.ndarray, mu: float
) -> jnp.ndarray:
    """Smooth approximation to elementwise clip(z, l, u).

    Composes smooth_max (for the lower bound) and smooth_min (for the upper bound).
    Infinite bounds are replaced with a large finite surrogate to avoid NaN.
    Converges to clip(z, l, u) as mu -> 0.

    Args:
        z: Input array.
        l: Lower bounds (use -jnp.inf for unbounded below).
        u: Upper bounds (use jnp.inf for unbounded above).
        mu: Smoothing parameter (> 0).

    Returns:
        Smooth approximation to clip(z, l, u), same shape as z.
    """
    # Replace only -inf values in l and +inf values in u with _BIG surrogates.
    # NaN bounds are preserved so they propagate as NaN rather than silently
    # becoming large finite values.
    l_safe = jnp.where(jnp.isneginf(l), -_BIG, l)
    u_safe = jnp.where(jnp.isposinf(u), _BIG, u)
    inner = smooth_max(l_safe, z, mu)
    return smooth_min(u_safe, inner, mu)


def smoothed_residual(
    x: jnp.ndarray,
    F_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    l: jnp.ndarray,
    u: jnp.ndarray,
    mu: float,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    """Smoothed MCP residual: H(x) = x - smooth_proj(x - F(x, theta), l, u, mu).

    At a solution x* of the MCP, H(x*) = 0. The smoothing parameter mu
    controls how closely this approximates the exact (non-smooth) MCP residual.

    This is a low-level building block. Unlike ``solve_mcp`` and
    ``make_mcp_solver_diff``, it does **not** auto-normalize ``F_fn``.
    ``F_fn`` must accept exactly two positional arguments ``(x, theta)``.
    If your function takes only x, wrap it: ``lambda x, theta: my_F(x)``.

    Args:
        x: Current iterate.
        F_fn: Residual map with signature F(x, theta) -> array.
        l: Lower bounds.
        u: Upper bounds.
        mu: Smoothing parameter (> 0).
        theta: Parameters passed to F_fn.

    Returns:
        Smoothed residual vector, same shape as x.
    """
    Fx = F_fn(x, theta)
    z = x - Fx
    proj = smooth_proj(z, l, u, mu)
    return x - proj
