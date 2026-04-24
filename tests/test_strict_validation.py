"""Tests for strict validation modes and the preflight helper."""

import jax
import jax.numpy as jnp
import pytest
from packaging.version import Version

from smooth_mcp import make_mcp_solver_diff, preflight_validate

# Upstream JAX regression: `vmap(checkify(f_with_lax.while_loop))` — the
# composition JAX's own error message calls "the correct ordering" — was
# broken starting with JAX 0.7.0 and is still broken in 0.10.0. The
# internal `eval_jaxpr` hits a foreach arity mismatch while evaluating
# the checkify-transformed while-loop jaxpr under a BatchTrace. Our
# library's continuation kernel uses ``lax.while_loop``, so any test
# that exercises ``vmap(checkify(solver))`` is blocked on this bug.
#
# NaN-poisoning (``strict_validation=True``) works on every JAX version
# we test and is the recommended traced-validation path under batching
# — see docs/api.md §Input validation.
#
# ``packaging.version.Version`` is used because ``jax.__version__`` may
# include non-numeric suffixes (e.g., ``"0.7.0rc1"``, ``"0.10.0.dev"``)
# that a naïve split-on-dot parse would misread.
_JAX_VMAP_CHECKIFY_BROKEN = Version(jax.__version__) >= Version("0.7")


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

    def test_rejects_nan_x0(self):
        with pytest.raises(ValueError, match="x0 must not contain NaN"):
            preflight_validate(jnp.array([0.0]), jnp.array([1.0]), jnp.array([jnp.nan]))


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

    def test_rejects_bool_equivalent_int(self):
        """Integers like 1 and 0 compare equal to True/False but are not
        the boolean singletons. The downstream strict_validation-is-True
        check would then silently route them through the False branch.
        Identity-based validation rejects them up front."""
        with pytest.raises(ValueError, match="strict_validation must be"):
            make_mcp_solver_diff(_F, strict_validation=1)
        with pytest.raises(ValueError, match="strict_validation must be"):
            make_mcp_solver_diff(_F, strict_validation=0)

    def test_rejects_numpy_bool(self):
        """numpy.bool_ instances compare equal to Python True/False via
        ``==`` but fail ``is True`` / ``is False`` identity checks. They
        must be rejected for the same reason."""
        import numpy as np

        with pytest.raises(ValueError, match="strict_validation must be"):
            make_mcp_solver_diff(_F, strict_validation=np.bool_(True))
        with pytest.raises(ValueError, match="strict_validation must be"):
            make_mcp_solver_diff(_F, strict_validation=np.bool_(False))


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
        """Default factory is safe: invalid traced inputs produce NaN.

        As of 2026-04-18 the factory default is ``strict_validation=True``,
        so invalid traced inputs (l > u, NaN bounds, NaN x0) are poisoned
        to NaN output. This pins that safe-by-default contract so a
        regression to the old silent-finite behavior is visible.
        """
        solver = _make_default()
        x = jax.jit(solver)(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isnan(x))


class TestUncheckedMode:
    """Explicit opt-out path: ``strict_validation=False``.

    Locks in the fast-path behavior for users who explicitly disable
    traced validation (e.g., after ``preflight_validate`` on static
    bounds). This path is unsafe by design — the factory does not check
    value-based inputs inside traced code.
    """

    def _solver(self):
        return make_mcp_solver_diff(_F, strict_validation=False)

    def test_eager_invalid_still_raises(self):
        """Eager validation is unaffected by the traced opt-out."""
        solver = self._solver()
        with pytest.raises(ValueError, match="Lower bounds must not exceed"):
            solver(
                jnp.array([5.0]),
                jnp.array([3.0]),
                jnp.array([4.0]),
                jnp.array([2.0]),
            )

    def test_jit_invalid_silently_slips_through(self):
        """With the explicit opt-out, traced invalid bounds produce a finite
        (but meaningless) result — the documented fast-path behavior."""
        solver = self._solver()
        x = jax.jit(solver)(
            jnp.array([5.0]),
            jnp.array([3.0]),
            jnp.array([4.0]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isfinite(x))

    def test_jit_valid_matches_strict(self):
        """On valid inputs the unchecked path matches the strict default."""
        unchecked = jax.jit(self._solver())
        strict = jax.jit(_make_default())
        inputs = (
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        x_unchecked = unchecked(*inputs)
        x_strict = strict(*inputs)
        assert jnp.allclose(x_unchecked, x_strict)


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

    def test_jit_nan_x0_returns_nan(self):
        solver = _make_strict()
        x = jax.jit(solver)(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([jnp.nan]),
            jnp.array([2.0]),
        )
        assert jnp.all(jnp.isnan(x))

    def test_eager_nan_x0_still_raises(self):
        solver = _make_strict()
        with pytest.raises(ValueError, match="x0 must not contain NaN"):
            solver(
                jnp.array([0.0]),
                jnp.array([3.0]),
                jnp.array([jnp.nan]),
                jnp.array([2.0]),
            )

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

    def test_jit_nan_x0_error_set(self):
        solver = _make_checkify()
        err, _x = jax.jit(solver)(
            jnp.array([0.0]),
            jnp.array([3.0]),
            jnp.array([jnp.nan]),
            jnp.array([2.0]),
        )
        msg = err.get()
        assert msg is not None
        assert "x0 contains NaN" in msg

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

    @pytest.mark.skipif(
        _JAX_VMAP_CHECKIFY_BROKEN,
        reason=(
            "Upstream JAX regression (>=0.7, still in 0.10): "
            "vmap(checkify(f_with_while_loop)) raises "
            "'foreach() argument 2 is longer than argument 1'. "
            "Use strict_validation=True (NaN-poisoning) under vmap instead."
        ),
    )
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

        Simulate the user-facing bad ordering directly: start from a
        non-checkified solver, apply ``vmap``, then wrap the result with
        ``checkify.checkify``. That composition must fail at trace time.
        Users should use the factory's ``strict_validation="checkify"``
        solver directly under ``vmap`` instead.
        """
        from jax.experimental import checkify

        # Build a solver without checkify and reproduce the bad ordering:
        # checkify(vmap(raw)).
        raw = make_mcp_solver_diff(_F, strict_validation=False)
        bad = checkify.checkify(jax.vmap(raw))
        with pytest.raises(ValueError):
            bad(
                jnp.array([[0.0]]),
                jnp.array([[3.0]]),
                jnp.array([[1.0]]),
                jnp.array([[2.0]]),
            )
