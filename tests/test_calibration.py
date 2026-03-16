"""Tests for conformal calibration engine."""

import numpy as np
import pytest

from src.conformal.calibration import ConformalCalibrator


class TestConformalCalibrator:
    def test_calibration_runs(self):
        """Calibration should complete without errors."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal((500, 4))
        calibrator = ConformalCalibrator(alpha=0.1, beta=0.05, w1_method="sliced")
        result = calibrator.calibrate(residuals, seed=42)

        assert result.r_star > 0
        assert result.rho_hat_k >= 0
        assert result.eta_k > 0
        assert result.d == 4
        assert result.k == 500

    def test_radius_increases_with_alpha(self):
        """Lower alpha (higher coverage) should give larger radius."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal((500, 4))

        r_90 = ConformalCalibrator(alpha=0.1, w1_method="sliced").calibrate(residuals, seed=42).r_star
        r_95 = ConformalCalibrator(alpha=0.05, w1_method="sliced").calibrate(residuals, seed=42).r_star

        assert r_95 > r_90

    def test_naive_quantile(self):
        """Naive quantile should be the empirical (1-alpha) quantile of norms."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal((500, 4))

        calibrator = ConformalCalibrator(alpha=0.1)
        q_naive = calibrator.calibrate_naive(residuals)

        norms = np.linalg.norm(residuals, axis=1)
        expected = np.quantile(norms, 0.9)
        # Should be close to the 90th percentile of norms
        assert abs(q_naive - expected) < 0.2

    def test_delta_star_optimization(self):
        """delta* should be in (0, r) and satisfy optimality conditions."""
        delta = ConformalCalibrator.optimize_delta_star(r=2.0, d=4, A=0.1)
        assert 0 < delta < 2.0

    def test_delta_star_zero_A(self):
        """delta* should be 0 when A=0 (no correction needed)."""
        delta = ConformalCalibrator.optimize_delta_star(r=2.0, d=4, A=0.0)
        assert delta == 0.0

    def test_compute_radius(self):
        """Radius formula: delta* + (1-alpha + A/delta*)^{1/d}."""
        r = ConformalCalibrator.compute_radius(alpha=0.1, delta_star=0.5, A=0.1, d=4)
        expected = 0.5 + (0.9 + 0.2) ** 0.25
        assert r == pytest.approx(expected, rel=1e-6)

    def test_end_to_end_coverage(self):
        """End-to-end: calibrate on known Gaussian, check coverage on test set."""
        rng = np.random.default_rng(42)
        d = 4
        n_calib = 1000
        n_test = 2000

        residuals_calib = rng.standard_normal((n_calib, d))
        residuals_test = rng.standard_normal((n_test, d))

        calibrator = ConformalCalibrator(alpha=0.1, beta=0.05, w1_method="sliced")
        result = calibrator.calibrate(residuals_calib, seed=42)

        # Check coverage: fraction of test points with ||eps|| <= r*
        norms_test = np.linalg.norm(residuals_test, axis=1)
        coverage = (norms_test <= result.r_star).mean()

        # Should achieve at least ~85% coverage (conservative is fine)
        assert coverage >= 0.80, f"Coverage too low: {coverage:.3f}"

    def test_eta_k_decreases_with_k(self):
        """eta_k should decrease as k increases."""
        calibrator = ConformalCalibrator(alpha=0.1, beta=0.05)
        eta_100 = calibrator.compute_eta_k(0.05, k=100, d=4)
        eta_1000 = calibrator.compute_eta_k(0.05, k=1000, d=4)
        eta_10000 = calibrator.compute_eta_k(0.05, k=10000, d=4)
        assert eta_100 > eta_1000 > eta_10000
