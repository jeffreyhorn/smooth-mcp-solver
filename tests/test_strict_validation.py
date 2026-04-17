"""Tests for strict validation modes and the preflight helper."""

import jax
import jax.numpy as jnp
import pytest

from smooth_mcp import make_mcp_solver_diff, preflight_validate


def _F(x, theta):
    return x - theta


def _make_default():
    return make_mcp_solver_diff(_F)


def _make_strict():
    return make_mcp_solver_diff(_F, strict_validation=True)


def _make_strict_aux():
    return make_mcp_solver_diff(_F, strict_validation=True, return_aux=True)


def _make_checkify():
    return make_mcp_solver_diff(_F, strict_validation="checkify")


# ---------------------------------------------------------------------------
# preflight_validate
# ---------------------------------------------------------------------------


class TestPreflightValidate:
    def test_accepts_valid_arrays(self):
        preflight_validate(jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.5]))

    def test_accepts_python_lists(self):
        preflight_validate([0.0, -1.0], [1.0, 2.0], [0.5, 0.0])

    def test_rejects_shape_mismatch_lu(self):
        with pytest.raises(ValueError, match="same shape"):
            preflight_validate(jnp.zeros(2), jnp.zeros(3), jnp.zeros(2))

    def test_rejects_shape_mismatch_x0(self):
        with pytest.raises(ValueError, match="same shape as l"):
            preflight_validate(jnp.zeros(2), jnp.full(2, jnp.inf), jnp.zeros(3))

    def test_rejects_l_greater_than_u(self):
        with pytest.raises(ValueError, match="Lower bounds must not exceed"):
            preflight_validate(jnp.array([5.0]), jnp.array([3.0]), jnp.array([4.0]))

    def test_rejects_nan_l(self):
        with pytest.raises(ValueError, match="must not contain NaN"):
            preflight_validate(jnp.array([jnp.nan]), jnp.array([1.0]), jnp.array([0.5]))

    def test_rejects_nan_u(self):
        with pytest.raises(ValueError, match="must not contain NaN"):
            preflight_validate(jnp.array([0.0]), jnp.array([jnp.nan]), jnp.array([0.5]))


# ---------------------------------------------------------------------------
# strict_validation parameter checking
# ---------------------------------------------------------------------------


class TestStrictValidationArg:
    def test_rejects_bad_value(self):
        with pytest.raises(ValueError, match="strict_validation must be"):
            make_mcp_solver_diff(_F, strict_validation="oops")

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="strict_validation must be"):
            make_mcp_solver_diff(_F, strict_validation=None)


# ---------------------------------------------------------------------------
# Default mode preserves existing behavior
# ---------------------------------------------------------------------------


class TestDefaultMode:
    def test_eager_invalid_raises(self):
        solver = _make_default()
        with pytest.raises(ValueError, match="Lower bounds must not exceed"):
            solver(
                jnp.array([5.0]),
                jnp.array([3.0]),
                jnp.array([4.0]),
                jnp.array([2.0]),
            )

    def test_eager_nan_raises(self):
        solver = _make_default()
        with pytest.raises(ValueError, match="must not contain NaN"):
            solver(
                jnp.array([jnp.nan]),
                jnp.array([3.0]),
                jnp.array([1.0]),
                jnp.array([2.0]),
            )

    def test_jit_invalid_silently_slips_through(self):
        """Default mode does not catch invalid bounds under tracing.

        This is the documented limitation that strict mode exists to address.
        Locking it in as a test so regressions in the default mode are visible.
        """
        solver = _make_default()
        x = jax.jit(solver)(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isfinite(x))


# ---------------------------------------------------------------------------
# strict_validation=True (NaN-poisoning)
# ---------------------------------------------------------------------------


class TestStrictNanPoisoning:
    def test_eager_invalid_still_raises(self):
        solver = _make_strict()
        with pytest.raises(ValueError, match="Lower bounds must not exceed"):
            solver(
                jnp.array([5.0]),
                jnp.array([3.0]),
                jnp.array([4.0]),
                jnp.array([2.0]),
            )

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

    def test_grad_on_valid_returns_finite(self):
        solver = _make_strict()

        def loss(theta):
            x = solver(
                jnp.array([0.0]),
                jnp.array([3.0]),
                jnp.array([1.0]),
                theta,
            )
            return jnp.sum(x**2)

        g = jax.grad(loss)(jnp.array([2.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_jit_grad_on_valid_returns_finite(self):
        solver = _make_strict()

        def loss(theta):
            x = solver(
                jnp.array([0.0]),
                jnp.array([3.0]),
                jnp.array([1.0]),
                theta,
            )
            return jnp.sum(x**2)

        g = jax.jit(jax.grad(loss))(jnp.array([2.0]))
        assert jnp.all(jnp.isfinite(g))

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


class TestStrictCheckify:
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

    def test_vmap_mixed_batch_reports_indices(self):
        solver = _make_checkify()
        ls = jnp.array([[0.0], [5.0], [jnp.nan]])
        us = jnp.array([[3.0], [3.0], [1.0]])
        x0s = jnp.array([[1.0], [1.0], [1.0]])
        thetas = jnp.array([[2.0], [2.0], [2.0]])
        errs, _xs = jax.vmap(solver)(ls, us, x0s, thetas)
        msg = errs.get()
        assert msg is not None
        assert "mapped index" in msg

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
        """Documented caveat: checkify(vmap(...)) is rejected by JAX.

        The factory returns the inner-checkify form, so wrapping it again
        in vmap and then checkify must fail at trace time with a clear
        error. Users should use vmap(solver) instead.
        """
        solver = _make_checkify()
        # wrapping checkify around vmap of an already-checkified function
        # and calling it should blow up at trace.
        bad = jax.vmap(solver)
        # Re-checkify-ing is not the pattern users hit; the failure mode
        # is really "user checkifies the solver after vmap". Simulate that
        # by calling vmap on the underlying unchecked path + checkify.
        # Build a solver without checkify and reproduce the bad ordering:
        from jax.experimental import checkify

        raw = make_mcp_solver_diff(_F, strict_validation=True)
        bad = checkify.checkify(jax.vmap(raw))
        with pytest.raises(ValueError):
            bad(
                jnp.array([[0.0]]),
                jnp.array([[3.0]]),
                jnp.array([[1.0]]),
                jnp.array([[2.0]]),
            )
