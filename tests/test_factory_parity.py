"""Parity tests for make_mcp_solver vs make_mcp_solver_diff on shared
options.

After the #17 refactor, both factories route their shared behavior
(return_aux packaging, strict_validation, diagnostics) through
``smooth_mcp/_factory_common.py``. These tests lock in that those
shared options actually produce identical observable results so the
two factories cannot silently drift again.

Problem-independent contract only — per-problem numeric parity with
``solve_mcp`` is covered separately in test_forward_factory.py
(forward) and test_solver.py::TestDiffSolverMatchesForward (diff).
"""

import jax.numpy as jnp
import pytest

from smooth_mcp import make_mcp_solver, make_mcp_solver_diff


def _F(x, theta):
    return x - theta


_L = jnp.zeros(2)
_U = jnp.full(2, jnp.inf)
_X0 = jnp.zeros(2)
_THETA = jnp.array([1.0, 2.0])


# ---------------------------------------------------------------------------
# return_aux parity
# ---------------------------------------------------------------------------


class TestReturnAuxParity:
    def test_aux_fields_match_field_by_field(self):
        """Both factories populate SolveInfo with matching values for
        mu_used, num_steps, residual_norm, and converged on the same
        problem."""
        fwd = make_mcp_solver(_F, return_aux=True)
        dif = make_mcp_solver_diff(_F, return_aux=True)

        x_fwd, info_fwd = fwd(_L, _U, _X0, _THETA)
        x_dif, info_dif = dif(_L, _U, _X0, _THETA)

        assert jnp.allclose(x_fwd, x_dif, atol=1e-10)
        assert jnp.isclose(info_fwd.mu_used, info_dif.mu_used, rtol=1e-6)
        assert int(info_fwd.num_steps) == int(info_dif.num_steps)
        assert jnp.isclose(info_fwd.residual_norm, info_dif.residual_norm, rtol=1e-6)
        assert bool(info_fwd.converged) == bool(info_dif.converged)

    def test_aux_types_match(self):
        """SolveInfo field dtypes match across the two factories. This
        is what downstream user code keys off of — a dtype skew between
        factories would be a silent backward-incompatible change."""
        fwd = make_mcp_solver(_F, return_aux=True)
        dif = make_mcp_solver_diff(_F, return_aux=True)

        _, info_fwd = fwd(_L, _U, _X0, _THETA)
        _, info_dif = dif(_L, _U, _X0, _THETA)

        assert info_fwd.mu_used.dtype == info_dif.mu_used.dtype
        assert info_fwd.num_steps.dtype == info_dif.num_steps.dtype
        assert info_fwd.residual_norm.dtype == info_dif.residual_norm.dtype
        assert info_fwd.converged.dtype == info_dif.converged.dtype

    def test_return_aux_false_matches_just_x(self):
        """With return_aux=False both factories return a bare array,
        not a tuple."""
        fwd = make_mcp_solver(_F, return_aux=False)
        dif = make_mcp_solver_diff(_F, return_aux=False)

        x_fwd = fwd(_L, _U, _X0, _THETA)
        x_dif = dif(_L, _U, _X0, _THETA)

        assert not isinstance(x_fwd, tuple)
        assert not isinstance(x_dif, tuple)
        assert jnp.allclose(x_fwd, x_dif, atol=1e-10)


# ---------------------------------------------------------------------------
# strict_validation parity
# ---------------------------------------------------------------------------


class TestStrictValidationParity:
    def test_invalid_value_rejected_identically_in_strict_true(self):
        """strict_validation=True poisons the output to NaN in both
        factories when fed an invalid l > u. With return_aux=True,
        SolveInfo.converged must be False and residual_norm NaN in
        both."""
        fwd = make_mcp_solver(_F, return_aux=True, strict_validation=True)
        dif = make_mcp_solver_diff(_F, return_aux=True, strict_validation=True)

        # l > u → invalid. Wrapping in jit so the eager ValueError
        # path does not fire — we want the traced poisoning path that
        # the two factories are supposed to share.
        import jax

        bad_l = jnp.array([5.0, 5.0])
        bad_u = jnp.array([1.0, 1.0])

        fwd_jit = jax.jit(fwd)
        dif_jit = jax.jit(dif)

        x_fwd, info_fwd = fwd_jit(bad_l, bad_u, _X0, _THETA)
        x_dif, info_dif = dif_jit(bad_l, bad_u, _X0, _THETA)

        assert bool(jnp.all(jnp.isnan(x_fwd)))
        assert bool(jnp.all(jnp.isnan(x_dif)))
        assert bool(info_fwd.converged) is False
        assert bool(info_dif.converged) is False
        assert bool(jnp.isnan(info_fwd.residual_norm))
        assert bool(jnp.isnan(info_dif.residual_norm))

    def test_strict_validation_false_path_agrees_on_valid_inputs(self):
        """With strict_validation=False, neither factory does a traced
        value check — both just solve. On valid inputs they must still
        agree."""
        fwd = make_mcp_solver(_F, return_aux=True, strict_validation=False)
        dif = make_mcp_solver_diff(_F, return_aux=True, strict_validation=False)

        x_fwd, info_fwd = fwd(_L, _U, _X0, _THETA)
        x_dif, info_dif = dif(_L, _U, _X0, _THETA)

        assert jnp.allclose(x_fwd, x_dif, atol=1e-10)
        assert bool(info_fwd.converged) == bool(info_dif.converged)

    def test_strict_validation_rejects_bad_enum_identically(self):
        """Both factories reject an invalid strict_validation value
        with the same ValueError message."""
        with pytest.raises(ValueError, match="strict_validation must be"):
            make_mcp_solver(_F, strict_validation="nope")
        with pytest.raises(ValueError, match="strict_validation must be"):
            make_mcp_solver_diff(_F, strict_validation="nope")


# ---------------------------------------------------------------------------
# checkify mode parity
# ---------------------------------------------------------------------------


class TestCheckifyParity:
    def test_checkify_mode_wraps_signature_identically(self):
        """Both factories return an ``(Error, ...)``-wrapping callable
        when strict_validation='checkify' is selected, and both
        produce a clean (empty) Error on valid inputs."""
        fwd = make_mcp_solver(_F, strict_validation="checkify")
        dif = make_mcp_solver_diff(_F, strict_validation="checkify")

        err_fwd, x_fwd = fwd(_L, _U, _X0, _THETA)
        err_dif, x_dif = dif(_L, _U, _X0, _THETA)

        # No error raised on valid inputs.
        err_fwd.throw()
        err_dif.throw()
        assert jnp.allclose(x_fwd, x_dif, atol=1e-10)

    def test_checkify_reports_invalid_input_in_both(self):
        """On invalid input (l > u), both factories surface the error
        via the returned Error object rather than raising eagerly
        inside the traced body."""
        fwd = make_mcp_solver(_F, strict_validation="checkify")
        dif = make_mcp_solver_diff(_F, strict_validation="checkify")

        bad_l = jnp.array([5.0, 5.0])
        bad_u = jnp.array([1.0, 1.0])

        err_fwd, _ = fwd(bad_l, bad_u, _X0, _THETA)
        err_dif, _ = dif(bad_l, bad_u, _X0, _THETA)

        with pytest.raises(Exception):
            err_fwd.throw()
        with pytest.raises(Exception):
            err_dif.throw()
