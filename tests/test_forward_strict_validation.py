"""Strict-validation contract tests for ``make_mcp_solver`` (forward-only).

Mirrors the coverage in ``tests/test_strict_validation.py`` for
``make_mcp_solver_diff``. The forward factory documents the same
three modes (``False``, ``True``, ``"checkify"``) with the same
safety contract, so every mode needs the same regression coverage on
both APIs.

Forward-factory has no gradient path, so ``grad`` tests are omitted.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import checkify

from smooth_mcp import make_mcp_solver


def _F(x, theta):
    return x - theta


def _make_default():
    return make_mcp_solver(_F)


def _make_strict():
    return make_mcp_solver(_F, strict_validation=True)


def _make_strict_aux():
    return make_mcp_solver(_F, strict_validation=True, return_aux=True)


def _make_checkify():
    return make_mcp_solver(_F, strict_validation="checkify")


# ---------------------------------------------------------------------------
# Default mode (2026-04-18 flip: safe by default)
# ---------------------------------------------------------------------------


class TestForwardDefaultMode:
    """Factory default is ``strict_validation=True`` — safe path."""

    def test_eager_invalid_raises(self):
        solver = _make_default()
        with pytest.raises(ValueError, match="Lower bounds must not exceed"):
            solver(
                jnp.array([5.0]),
                jnp.array([3.0]),
                jnp.array([4.0]),
                jnp.array([2.0]),
            )

    def test_eager_nan_l_raises(self):
        solver = _make_default()
        with pytest.raises(ValueError, match="must not contain NaN"):
            solver(
                jnp.array([jnp.nan]),
                jnp.array([3.0]),
                jnp.array([1.0]),
                jnp.array([2.0]),
            )

    def test_eager_nan_u_raises(self):
        solver = _make_default()
        with pytest.raises(ValueError, match="must not contain NaN"):
            solver(
                jnp.array([0.0]),
                jnp.array([jnp.nan]),
                jnp.array([1.0]),
                jnp.array([2.0]),
            )

    def test_eager_nan_x0_raises(self):
        solver = _make_default()
        with pytest.raises(ValueError, match="x0 must not contain NaN"):
            solver(
                jnp.array([0.0]),
                jnp.array([3.0]),
                jnp.array([jnp.nan]),
                jnp.array([2.0]),
            )

    def test_jit_invalid_poisons_to_nan(self):
        """Safe-by-default: invalid traced inputs produce NaN."""
        solver = _make_default()
        x = jax.jit(solver)(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isnan(x))


# ---------------------------------------------------------------------------
# Explicit opt-out path
# ---------------------------------------------------------------------------


class TestForwardUncheckedMode:
    """Explicit ``strict_validation=False`` opt-out."""

    def _solver(self):
        return make_mcp_solver(_F, strict_validation=False)

    def test_eager_invalid_still_raises(self):
        solver = self._solver()
        with pytest.raises(ValueError, match="Lower bounds must not exceed"):
            solver(
                jnp.array([5.0]),
                jnp.array([3.0]),
                jnp.array([4.0]),
                jnp.array([2.0]),
            )

    def test_jit_invalid_silently_slips_through(self):
        solver = self._solver()
        x = jax.jit(solver)(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isfinite(x))

    def test_jit_valid_matches_strict(self):
        unchecked = jax.jit(self._solver())
        strict = jax.jit(_make_default())
        inputs = (
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        assert jnp.allclose(unchecked(*inputs), strict(*inputs))


# ---------------------------------------------------------------------------
# strict_validation=True (NaN-poisoning)
# ---------------------------------------------------------------------------


class TestForwardStrictNanPoisoning:
    def test_eager_valid_matches_default(self):
        solver = _make_strict()
        x = solver(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        assert jnp.allclose(x, jnp.array([2.0]))

    def test_jit_l_greater_than_u_returns_nan(self):
        solver = _make_strict()
        x = jax.jit(solver)(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isnan(x))

    def test_jit_nan_l_returns_nan(self):
        solver = _make_strict()
        x = jax.jit(solver)(
            jnp.array([jnp.nan]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isnan(x))

    def test_jit_nan_u_returns_nan(self):
        solver = _make_strict()
        x = jax.jit(solver)(
            jnp.array([0.0]),
            jnp.array([jnp.nan]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isnan(x))

    def test_jit_nan_x0_returns_nan(self):
        solver = _make_strict()
        x = jax.jit(solver)(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([jnp.nan]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isnan(x))

    def test_jit_valid_returns_finite(self):
        solver = _make_strict()
        x = jax.jit(solver)(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        assert jnp.allclose(x, jnp.array([2.0]))

    def test_vmap_mixed_batch_per_row_nan(self):
        solver = _make_strict()
        ls = jnp.array([[0.0], [5.0], [jnp.nan]])
        us = jnp.array([[3.0], [3.0], [1.0]])
        x0s = jnp.array([[1.0], [1.0], [1.0]])
        thetas = jnp.array([[2.0], [2.0], [2.0]])
        xs = jax.vmap(solver)(ls, us, x0s, thetas)
        assert jnp.allclose(xs[0], jnp.array([2.0]))
        assert jnp.all(jnp.isnan(xs[1]))
        assert jnp.all(jnp.isnan(xs[2]))

    def test_jit_vmap_mixed_batch_per_row_nan(self):
        solver = _make_strict()
        ls = jnp.array([[0.0], [5.0], [jnp.nan]])
        us = jnp.array([[3.0], [3.0], [1.0]])
        x0s = jnp.array([[1.0], [1.0], [1.0]])
        thetas = jnp.array([[2.0], [2.0], [2.0]])
        xs = jax.jit(jax.vmap(solver))(ls, us, x0s, thetas)
        assert jnp.allclose(xs[0], jnp.array([2.0]))
        assert jnp.all(jnp.isnan(xs[1]))
        assert jnp.all(jnp.isnan(xs[2]))

    def test_return_aux_poisons_converged_and_residual(self):
        solver = _make_strict_aux()
        x, info = jax.jit(solver)(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isnan(x))
        assert bool(info.converged) is False
        assert jnp.isnan(info.residual_norm)

    def test_return_aux_on_valid_reports_converged(self):
        solver = _make_strict_aux()
        x, info = jax.jit(solver)(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        assert jnp.allclose(x, jnp.array([2.0]))
        assert bool(info.converged) is True
        assert jnp.isfinite(info.residual_norm)


# ---------------------------------------------------------------------------
# strict_validation="checkify"
# ---------------------------------------------------------------------------


class TestForwardStrictCheckify:
    def test_eager_valid_no_error(self):
        solver = _make_checkify()
        err, x = solver(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        assert err.get() is None
        assert jnp.allclose(x, jnp.array([2.0]))

    def test_eager_l_greater_than_u_error_set(self):
        solver = _make_checkify()
        err, _x = solver(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        msg = err.get()
        assert msg is not None
        assert "l must be <= u" in msg

    def test_eager_nan_l_error_set(self):
        solver = _make_checkify()
        err, _x = solver(
            jnp.array([jnp.nan]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        msg = err.get()
        assert msg is not None
        assert "l contains NaN" in msg

    def test_eager_nan_x0_error_set(self):
        solver = _make_checkify()
        err, _x = solver(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([jnp.nan]),
            jnp.array([2.0]),
        )
        msg = err.get()
        assert msg is not None
        assert "x0 contains NaN" in msg

    def test_jit_invalid_error_set(self):
        solver = _make_checkify()
        err, _x = jax.jit(solver)(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        msg = err.get()
        assert msg is not None
        assert "l must be <= u" in msg

    def test_jit_valid_no_error(self):
        solver = _make_checkify()
        err, x = jax.jit(solver)(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        assert err.get() is None
        assert jnp.allclose(x, jnp.array([2.0]))

    def test_err_throw_raises_on_invalid(self):
        solver = _make_checkify()
        err, _x = solver(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        with pytest.raises(Exception, match="l must be <= u"):
            err.throw()

    def test_err_throw_noop_on_valid(self):
        solver = _make_checkify()
        err, _x = solver(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        err.throw()

    def test_checkify_of_vmap_raises_at_trace(self):
        """checkify(vmap(...)) is rejected by JAX — documented caveat.

        Users should use the factory's ``strict_validation="checkify"``
        solver directly under ``vmap`` instead.
        """
        raw = make_mcp_solver(_F, strict_validation=False)
        bad = checkify.checkify(jax.vmap(raw))
        with pytest.raises(ValueError):
            bad(
                jnp.array([[0.0]]),
                jnp.array([[3.0]]),
                jnp.array([[1.0]]),
                jnp.array([[2.0]]),
            )
