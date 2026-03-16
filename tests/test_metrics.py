"""Tests for coverage, efficiency, and diagnostic metrics."""

import numpy as np
import pytest

from src.conformal.rank_region import ConformalRegion, build_region, unit_ball_volume, log_unit_ball_volume
from src.metrics.coverage import marginal_coverage, coverage_gap, violation_rate, rolling_coverage
from src.metrics.efficiency import region_volume, log_region_volume, winkler_multivariate, mean_diameter
from src.metrics.diagnostics import rank_uniformity_ks, bound_tightness


class TestConformalRegion:
    def test_contains_inside(self):
        region = build_region(np.zeros(4), radius=1.0)
        assert region.contains(np.array([0.1, 0.1, 0.1, 0.1]))

    def test_contains_outside(self):
        region = build_region(np.zeros(4), radius=1.0)
        assert not region.contains(np.array([1.0, 1.0, 1.0, 1.0]))

    def test_volume_d2(self):
        region = build_region(np.zeros(2), radius=1.0)
        assert region.volume() == pytest.approx(np.pi, rel=1e-6)

    def test_volume_d3(self):
        region = build_region(np.zeros(3), radius=2.0)
        expected = (4.0 / 3.0) * np.pi * 8.0  # V_3 * r^3
        assert region.volume() == pytest.approx(expected, rel=1e-4)

    def test_log_volume_high_d(self):
        """Log-volume should not overflow/underflow for high d."""
        region = build_region(np.zeros(64), radius=0.5)
        log_vol = region.log_volume()
        assert np.isfinite(log_vol)

    def test_diameter(self):
        region = build_region(np.zeros(4), radius=3.0)
        assert region.diameter() == 6.0

    def test_sample_boundary(self):
        region = build_region(np.zeros(4), radius=1.0)
        boundary = region.sample_boundary(100)
        assert boundary.shape == (100, 4)
        # All points should be on the boundary (norm = radius from center)
        norms = np.linalg.norm(boundary, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


class TestCoverage:
    def _make_regions(self, centers, radius):
        return [build_region(c, radius) for c in centers]

    def test_perfect_coverage(self):
        centers = np.zeros((10, 4))
        regions = self._make_regions(centers, radius=10.0)
        Y_true = np.random.default_rng(42).standard_normal((10, 4))
        assert marginal_coverage(regions, Y_true) == 1.0

    def test_zero_coverage(self):
        centers = np.zeros((10, 4))
        regions = self._make_regions(centers, radius=0.001)
        Y_true = np.ones((10, 4)) * 100
        assert marginal_coverage(regions, Y_true) == 0.0

    def test_coverage_gap_positive(self):
        assert coverage_gap(0.95, 0.90) == pytest.approx(0.05)

    def test_violation_rate(self):
        centers = np.zeros((100, 2))
        regions = self._make_regions(centers, radius=10.0)
        Y_true = np.random.default_rng(42).standard_normal((100, 2))
        assert violation_rate(regions, Y_true) == 0.0

    def test_rolling_coverage(self):
        centers = np.zeros((100, 2))
        regions = self._make_regions(centers, radius=10.0)
        Y_true = np.random.default_rng(42).standard_normal((100, 2))
        rolling = rolling_coverage(regions, Y_true, n_windows=5)
        assert rolling.shape == (5,)
        assert np.all(rolling == 1.0)


class TestEfficiency:
    def test_winkler_inside(self):
        region = build_region(np.zeros(2), radius=1.0)
        point = np.array([0.1, 0.1])  # Inside
        score = winkler_multivariate(region, point, alpha=0.1)
        assert score == pytest.approx(np.pi, rel=1e-6)

    def test_winkler_outside(self):
        region = build_region(np.zeros(2), radius=1.0)
        point = np.array([5.0, 5.0])  # Outside
        score = winkler_multivariate(region, point, alpha=0.1)
        expected = np.pi + (2.0 / 0.1) * np.pi  # vol + penalty
        assert score == pytest.approx(expected, rel=1e-6)


class TestDiagnostics:
    def test_rank_uniformity_uniform(self):
        rng = np.random.default_rng(42)
        ranks = rng.uniform(0, 1, size=(1000, 4))
        result = rank_uniformity_ks(ranks)
        assert result["min_pvalue"] > 0.01

    def test_bound_tightness(self):
        assert bound_tightness(1.0, 1.5) == 1.5
        assert bound_tightness(1.0, 1.0) == 1.0


class TestUnitBallVolume:
    def test_d1(self):
        assert unit_ball_volume(1) == pytest.approx(2.0, rel=1e-6)

    def test_d2(self):
        assert unit_ball_volume(2) == pytest.approx(np.pi, rel=1e-6)

    def test_d3(self):
        assert unit_ball_volume(3) == pytest.approx(4.0 / 3.0 * np.pi, rel=1e-4)

    def test_log_volume_consistency(self):
        for d in [1, 2, 4, 8, 16]:
            assert log_unit_ball_volume(d) == pytest.approx(
                np.log(unit_ball_volume(d)), rel=1e-6
            )
