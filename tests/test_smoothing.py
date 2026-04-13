"""Unit tests for smooth_max, smooth_min, and smooth_proj."""

import jax
import jax.numpy as jnp

from smooth_mcp import smooth_max, smooth_min, smooth_proj


class TestSmoothMax:
    """smooth_max(a, b, mu) should approximate max(a, b) as mu -> 0."""

    def test_converges_to_max(self):
        a = jnp.array([1.0, -2.0, 3.0])
        b = jnp.array([2.0, 1.0, 3.0])
        for mu in [1.0, 0.01, 1e-6, 1e-12]:
            result = smooth_max(a, b, mu)
            expected = jnp.maximum(a, b)
            assert jnp.allclose(
                result, expected, atol=jnp.sqrt(mu)
            ), f"mu={mu}: {result} not close to max {expected}"

    def test_always_greater_than_max(self):
        """smooth_max >= max(a, b) since sqrt((a-b)^2 + mu) > |a - b|."""
        a = jnp.array([-5.0, 0.0, 3.0])
        b = jnp.array([-3.0, 0.0, 1.0])
        for mu in [1e-4, 1e-8, 1e-12]:
            result = smooth_max(a, b, mu)
            assert jnp.all(result >= jnp.maximum(a, b) - 1e-15)

    def test_symmetric(self):
        a = jnp.array([1.0, -2.0])
        b = jnp.array([3.0, 5.0])
        mu = 0.5
        assert jnp.allclose(smooth_max(a, b, mu), smooth_max(b, a, mu))

    def test_scalar(self):
        result = smooth_max(jnp.array(1.0), jnp.array(2.0), 1e-12)
        assert jnp.isclose(result, 2.0, atol=1e-6)

    def test_equal_inputs(self):
        a = jnp.array([5.0, -3.0])
        mu = 0.01
        result = smooth_max(a, a, mu)
        # smooth_max(a, a, mu) = a + sqrt(mu)/2
        expected = a + jnp.sqrt(mu) / 2
        assert jnp.allclose(result, expected)

    def test_gradient_finite(self):
        a = jnp.array(1.0)
        b = jnp.array(2.0)
        for mu in [1.0, 1e-6, 1e-12]:
            ga = jax.grad(lambda x: smooth_max(x, b, mu))(a)
            gb = jax.grad(lambda x: smooth_max(a, x, mu))(b)
            assert jnp.isfinite(ga) and jnp.isfinite(gb), f"mu={mu}: non-finite grad"

    def test_gradient_equal_inputs(self):
        """At a = b, gradient w.r.t. each should be 0.5 (by symmetry)."""
        a = jnp.array(3.0)
        mu = 0.01
        ga = jax.grad(lambda x: smooth_max(x, a, mu))(a)
        assert jnp.isclose(ga, 0.5, atol=1e-6)


class TestSmoothMin:
    """smooth_min(a, b, mu) should approximate min(a, b) as mu -> 0."""

    def test_converges_to_min(self):
        a = jnp.array([1.0, -2.0, 3.0])
        b = jnp.array([2.0, 1.0, 3.0])
        for mu in [1.0, 0.01, 1e-6, 1e-12]:
            result = smooth_min(a, b, mu)
            expected = jnp.minimum(a, b)
            assert jnp.allclose(
                result, expected, atol=jnp.sqrt(mu)
            ), f"mu={mu}: {result} not close to min {expected}"

    def test_always_less_than_min(self):
        """smooth_min <= min(a, b)."""
        a = jnp.array([-5.0, 0.0, 3.0])
        b = jnp.array([-3.0, 0.0, 1.0])
        for mu in [1e-4, 1e-8, 1e-12]:
            result = smooth_min(a, b, mu)
            assert jnp.all(result <= jnp.minimum(a, b) + 1e-15)

    def test_symmetric(self):
        a = jnp.array([1.0, -2.0])
        b = jnp.array([3.0, 5.0])
        mu = 0.5
        assert jnp.allclose(smooth_min(a, b, mu), smooth_min(b, a, mu))

    def test_large_difference(self):
        """Numerically stable when |a - b| >> 0 (the _BIG surrogate case)."""
        a = jnp.array(1e15)
        b = jnp.array(0.5)
        mu = 1e-12
        result = smooth_min(a, b, mu)
        assert jnp.isclose(result, 0.5, atol=1e-6), f"Got {result}, expected ~0.5"

    def test_both_negative(self):
        a = jnp.array(-5.0)
        b = jnp.array(-3.0)
        mu = 1e-12
        result = smooth_min(a, b, mu)
        assert jnp.isclose(result, -5.0, atol=1e-5)

    def test_both_negative_equal(self):
        a = jnp.array(-1.0)
        mu = 1e-12
        result = smooth_min(a, a, mu)
        # Should be close to a = -1.0
        assert jnp.isclose(result, -1.0, atol=jnp.sqrt(mu))

    def test_denom_zero_singularity(self):
        """At a = b = -sqrt(mu)/2, denominator is zero. Should not produce NaN."""
        for mu in [1e-6, 1e-12]:
            a = jnp.array(-jnp.sqrt(mu) / 2)
            result = smooth_min(a, a, mu)
            assert jnp.isfinite(result), f"mu={mu}: got {result}"

    def test_gradient_finite(self):
        a = jnp.array(1.0)
        b = jnp.array(2.0)
        for mu in [1.0, 1e-6, 1e-12]:
            ga = jax.grad(lambda x: smooth_min(x, b, mu))(a)
            gb = jax.grad(lambda x: smooth_min(a, x, mu))(b)
            assert jnp.isfinite(ga) and jnp.isfinite(gb), f"mu={mu}: non-finite grad"

    def test_gradient_large_difference(self):
        """Gradient w.r.t. the smaller input should be ~1 when |a - b| >> 0."""
        a = jnp.array(1e15)
        b = jnp.array(0.5)
        mu = 1e-12
        gb = jax.grad(lambda x: smooth_min(a, x, mu))(b)
        assert jnp.isclose(gb, 1.0, atol=1e-6), f"grad_b={gb}, expected ~1.0"

    def test_gradient_at_singularity(self):
        """Gradient should be finite at the denom=0 singularity."""
        for mu in [1e-6, 1e-12]:
            a = jnp.array(-jnp.sqrt(mu) / 2)
            ga = jax.grad(lambda x: smooth_min(x, a, mu))(a)
            assert jnp.isfinite(ga), f"mu={mu}: grad is {ga}"


class TestSmoothProj:
    """smooth_proj(z, l, u, mu) should approximate clip(z, l, u) as mu -> 0."""

    def test_converges_to_clip(self):
        z = jnp.array([-1.0, 0.5, 3.0])
        l = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([2.0, 2.0, 2.0])
        for mu in [0.01, 1e-6, 1e-12]:
            result = smooth_proj(z, l, u, mu)
            expected = jnp.clip(z, l, u)
            assert jnp.allclose(
                result, expected, atol=jnp.sqrt(mu)
            ), f"mu={mu}: {result} not close to clip {expected}"

    def test_infinite_upper_bound(self):
        z = jnp.array([0.5, 5.0, -1.0])
        l = jnp.array([0.0, 0.0, 0.0])
        u = jnp.full(3, jnp.inf)
        mu = 1e-12
        result = smooth_proj(z, l, u, mu)
        expected = jnp.maximum(z, l)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_infinite_lower_bound(self):
        z = jnp.array([-5.0, 0.5, 3.0])
        l = jnp.full(3, -jnp.inf)
        u = jnp.array([2.0, 2.0, 2.0])
        mu = 1e-12
        result = smooth_proj(z, l, u, mu)
        expected = jnp.minimum(z, u)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_both_bounds_infinite(self):
        z = jnp.array([-100.0, 0.0, 100.0])
        l = jnp.full(3, -jnp.inf)
        u = jnp.full(3, jnp.inf)
        mu = 1e-12
        result = smooth_proj(z, l, u, mu)
        assert jnp.allclose(result, z, atol=1e-5)

    def test_no_nan_with_inf_bounds(self):
        z = jnp.array([1.0])
        l = jnp.array([-jnp.inf])
        u = jnp.array([jnp.inf])
        for mu in [1.0, 1e-6, 1e-12]:
            result = smooth_proj(z, l, u, mu)
            assert jnp.all(jnp.isfinite(result)), f"mu={mu}: got {result}"

    def test_gradient_through_inf_bounds(self):
        """Gradients should be finite even with infinite bounds."""
        z = jnp.array([1.0, 2.0])
        l = jnp.array([0.0, 0.0])
        u = jnp.full(2, jnp.inf)
        mu = 1e-6
        gz = jax.jacfwd(lambda zz: smooth_proj(zz, l, u, mu))(z)
        assert jnp.all(jnp.isfinite(gz)), f"Non-finite gradient: {gz}"

    def test_z_at_bounds(self):
        """When z equals a bound, result should be near that bound."""
        l = jnp.array([0.0])
        u = jnp.array([5.0])
        mu = 1e-12
        assert jnp.isclose(smooth_proj(jnp.array([0.0]), l, u, mu), 0.0, atol=1e-5)
        assert jnp.isclose(smooth_proj(jnp.array([5.0]), l, u, mu), 5.0, atol=1e-5)
