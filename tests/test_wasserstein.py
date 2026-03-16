"""Tests for Wasserstein distance and rank transform."""

import numpy as np
import pytest

from src.conformal.wasserstein import (
    convergence_rate_term,
    fournier_constant,
    rank_transform,
    uniformity_test,
    wasserstein_1_uniform,
)


class TestRankTransform:
    def test_output_range(self):
        """Ranks should be in (0, 1)."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal((100, 4))
        ranks = rank_transform(residuals)
        assert np.all(ranks > 0)
        assert np.all(ranks < 1)

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal((200, 8))
        ranks = rank_transform(residuals)
        assert ranks.shape == (200, 8)

    def test_marginal_uniformity(self):
        """Marginal ranks should be approximately uniform."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal((1000, 4))
        ranks = rank_transform(residuals)
        # Each marginal should have mean ~0.5
        assert np.allclose(ranks.mean(axis=0), 0.5, atol=0.05)


class TestWasserstein:
    def test_identical_distributions(self):
        """W1 between identical distributions should be ~0."""
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 1, size=(500, 4))
        w1 = wasserstein_1_uniform(samples, d=4, method="sliced", seed=42)
        # Should be small but not exactly 0 (finite sample)
        assert w1 < 0.2

    def test_w1_decreases_with_n(self):
        """W1 should decrease as sample size increases."""
        w1_values = []
        for n in [100, 500, 2000]:
            rng = np.random.default_rng(n)  # Independent seed per size
            samples = rng.uniform(0, 1, size=(n, 4))
            w1 = wasserstein_1_uniform(samples, d=4, method="sliced", seed=n)
            w1_values.append(w1)
        # Generally decreasing (allow some noise)
        assert w1_values[0] > w1_values[-1]

    def test_exact_method(self):
        """Exact method should give a result for small d and k."""
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 1, size=(100, 2))
        w1 = wasserstein_1_uniform(samples, d=2, method="exact", seed=42)
        assert w1 >= 0
        assert w1 < 1.0

    def test_auto_method_switches(self):
        """Auto should use exact for small d/k, sliced for large."""
        rng = np.random.default_rng(42)
        # Small: d=2, k=100 -> should use exact
        small = rng.uniform(0, 1, size=(100, 2))
        w1_small = wasserstein_1_uniform(small, d=2, method="auto", seed=42)
        assert w1_small >= 0

        # Large: d=16, k=3000 -> should use sliced
        large = rng.uniform(0, 1, size=(3000, 16))
        w1_large = wasserstein_1_uniform(large, d=16, method="auto", seed=42)
        assert w1_large >= 0


class TestConvergenceRate:
    def test_rate_d1(self):
        assert convergence_rate_term(100, 1) == pytest.approx(0.01)

    def test_rate_d2(self):
        rate = convergence_rate_term(100, 2)
        # k^{-1/2} * (log k)^{1/2}
        expected = (1.0 / 10.0) * np.sqrt(np.log(100))
        assert rate == pytest.approx(expected)

    def test_rate_d8(self):
        rate = convergence_rate_term(1000, 8)
        expected = 1000 ** (-1.0 / 8)
        assert rate == pytest.approx(expected)


class TestUniformityTest:
    def test_uniform_passes(self):
        """Truly uniform samples should pass the KS test."""
        rng = np.random.default_rng(42)
        ranks = rng.uniform(0, 1, size=(1000, 4))
        result = uniformity_test(ranks)
        # All marginal p-values should be > 0.01
        assert result["min_marginal_pvalue"] > 0.01

    def test_non_uniform_fails(self):
        """Non-uniform samples should fail the KS test."""
        rng = np.random.default_rng(42)
        # Beta(0.5, 0.5) is far from uniform
        ranks = rng.beta(0.5, 0.5, size=(1000, 4))
        result = uniformity_test(ranks)
        assert result["min_marginal_pvalue"] < 0.01
