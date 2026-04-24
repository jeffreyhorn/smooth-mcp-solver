"""Shared plumbing used by both the forward and differentiable factories.

``make_mcp_solver`` and ``make_mcp_solver_diff`` previously duplicated a
large amount of scaffolding: forward-kernel construction, SolveInfo
packaging, the NaN-poisoning wrapper, and the checkify-mode wrapper.
That duplication repeatedly caused docs/code drift and made shared
semantics (strict_validation, return_aux, diagnostics) harder to keep
consistent across the two paths.

This module centralizes those pieces. Everything VJP-specific
(``custom_vjp`` setup and the adjoint computation) remains in
``diff.py`` — it is genuinely diff-only and does not belong here.
"""

from typing import Callable, Union

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from smooth_mcp._kernel import (
    make_continuation_solver,
    make_newton_solver,
    sanitize_inputs,
    traced_invalid_mask,
    validate_bounds_and_x0,
)
from smooth_mcp._types import SolveInfo
from smooth_mcp.smoothing import smoothed_residual


def validate_strict_validation(strict_validation):
    """Reject unsupported ``strict_validation`` values early.

    Uses ``is`` identity checks for the boolean cases rather than
    membership (``in (False, True, ...)``). Membership tests with
    ``==``, which accepts ``1``, ``0``, ``numpy.bool_(True)``, and
    other values that compare equal to ``True``/``False``; those
    would then slip past the ``build_public_solve`` branch (which is
    ``strict_validation is True``) and silently behave like
    ``strict_validation=False``. Strict identity rejects them up
    front with the same clear error as any other bad value.
    """
    if (
        strict_validation is True
        or strict_validation is False
        or strict_validation == "checkify"
    ):
        return
    raise ValueError(
        f"strict_validation must be False, True, or 'checkify', "
        f"got {strict_validation!r}"
    )


def build_forward_kernel(
    F_fn_normalized: Callable,
    *,
    mu_init: float,
    mu_min: float,
    mu_decay: float,
    newton_tol: float,
    max_mu_steps: int,
    armijo_c: float,
    backtrack_rho: float,
    max_ls_steps: int,
    linear_solver: str,
    krylov_tol: float,
    krylov_maxiter: int,
    krylov_restart: int,
    regularize: float,
) -> Callable:
    """Return a pure-JAX ``(l, u, x0, theta) -> (x_star, mu_used, num_steps)``.

    The Newton and continuation kernels are rebuilt per call because
    ``l``, ``u``, and ``theta`` are closure inputs to those inner kernels.
    Callers should wrap the returned function in ``jax.jit`` (or a
    ``custom_vjp``) for repeated use.
    """

    def _run_forward(l, u, x0, theta):
        newton = make_newton_solver(
            F_fn_normalized,
            l,
            u,
            theta,
            tol=newton_tol,
            armijo_c=armijo_c,
            backtrack_rho=backtrack_rho,
            max_ls_steps=max_ls_steps,
            linear_solver=linear_solver,
            krylov_tol=krylov_tol,
            krylov_maxiter=krylov_maxiter,
            krylov_restart=krylov_restart,
            regularize=regularize,
        )
        continuation = make_continuation_solver(
            newton,
            F_fn_normalized,
            l,
            u,
            theta,
            mu_init,
            mu_min,
            mu_decay,
            newton_tol,
            max_mu_steps,
        )
        return continuation(x0)

    return _run_forward


def build_make_aux(
    F_fn_normalized: Callable,
    *,
    mu_min: float,
    newton_tol: float,
    stop_grad: bool,
) -> Callable:
    """Return a ``_make_aux(x_star, mu_used, num_steps, l, u, theta) -> SolveInfo``.

    The diff factory needs ``stop_gradient`` on every field so aux is
    treated as inert by ``jax.grad``; the forward factory does not
    install a ``custom_vjp`` so the stop_gradient is unnecessary (and
    slightly obscures the forward-only parity story). ``stop_grad``
    toggles which version is produced.
    """
    wrap = jax.lax.stop_gradient if stop_grad else (lambda x: x)

    def _make_aux(x_star, mu_used, num_steps, l, u, theta):
        mu_min_arr = jnp.array(mu_min, dtype=x_star.dtype)
        newton_tol_arr = jnp.array(newton_tol, dtype=x_star.dtype)
        residual = smoothed_residual(x_star, F_fn_normalized, l, u, mu_min_arr, theta)
        residual_norm = jnp.max(jnp.abs(residual))
        converged = residual_norm < newton_tol_arr
        return SolveInfo(
            mu_used=wrap(mu_used),
            num_steps=wrap(num_steps),
            residual_norm=wrap(residual_norm),
            converged=wrap(converged),
        )

    return _make_aux


def _poison_solve_info(aux: SolveInfo, invalid) -> SolveInfo:
    """Mark SolveInfo as failed when traced inputs were invalid."""
    return SolveInfo(
        mu_used=aux.mu_used,
        num_steps=aux.num_steps,
        residual_norm=jnp.where(invalid, jnp.nan, aux.residual_norm),
        converged=jnp.where(invalid, jnp.bool_(False), aux.converged),
    )


def _checkify_value_checks(l, u, x0) -> None:
    """Register checkify value checks shared by both factories."""
    checkify.check(~jnp.any(jnp.isnan(l)), "l contains NaN")
    checkify.check(~jnp.any(jnp.isnan(u)), "u contains NaN")
    checkify.check(~jnp.any(jnp.isnan(x0)), "x0 contains NaN")
    checkify.check(jnp.all(l <= u), "l must be <= u element-wise")


def build_public_solve(
    inner_solve: Callable,
    *,
    strict_validation: Union[bool, str],
    return_aux: bool,
) -> Callable:
    """Wrap an inner ``(l, u, x0, theta) -> x`` (or ``-> (x, SolveInfo)``)
    with the selected ``strict_validation`` mode.

    ``inner_solve`` is the already-packaged solver: for the forward
    factory it is a direct call to the continuation kernel plus aux
    packaging; for the diff factory it wraps the ``custom_vjp`` core.
    The three modes produce the three public contracts documented in
    ``docs/api.md``:

    - ``True`` (default): NaN-poisoning via ``traced_invalid_mask`` +
      ``sanitize_inputs``.
    - ``False``: shape-checked only; traced value checks skipped.
    - ``"checkify"``: value checks reported via a ``jax.experimental.
      checkify`` Error alongside the result.
    """

    def _poisoned(l, u, x0, theta):
        invalid = traced_invalid_mask(l, u, x0)
        safe_l, safe_u, safe_x0 = sanitize_inputs(l, u, x0)
        result = inner_solve(safe_l, safe_u, safe_x0, theta)
        if return_aux:
            x_star, aux = result
            x_star = jnp.where(invalid, jnp.full_like(x_star, jnp.nan), x_star)
            return x_star, _poison_solve_info(aux, invalid)
        return jnp.where(invalid, jnp.full_like(result, jnp.nan), result)

    def _solve_checked(l, u, x0, theta):
        l, u, x0, theta = (
            jnp.asarray(l),
            jnp.asarray(u),
            jnp.asarray(x0),
            jnp.asarray(theta),
        )
        validate_bounds_and_x0(l, u, x0)
        if strict_validation is True:
            return _poisoned(l, u, x0, theta)
        return inner_solve(l, u, x0, theta)

    if strict_validation == "checkify":

        def _checkify_target(l, u, x0, theta):
            l = jnp.asarray(l)
            u = jnp.asarray(u)
            x0 = jnp.asarray(x0)
            theta = jnp.asarray(theta)
            validate_bounds_and_x0(l, u, x0)
            _checkify_value_checks(l, u, x0)
            return inner_solve(l, u, x0, theta)

        return checkify.checkify(_checkify_target)

    return _solve_checked
