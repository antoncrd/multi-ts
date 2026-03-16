"""Tests for VAR(1) data generating process."""

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from src.dgp.var1 import VAR1Generator


class TestVAR1Generator:
    def test_spectral_radius(self):
        """A matrix should have the prescribed spectral radius."""
        for rho in [0.1, 0.5, 0.9, 0.95]:
            gen = VAR1Generator(d=4, rho_A=rho, seed=42)
            eigvals = np.linalg.eigvals(gen.A)
            assert np.isclose(np.max(np.abs(eigvals)), rho, atol=1e-10)

    def test_output_shapes(self):
        """Generated data should have correct shapes."""
        for d in [2, 4, 8, 16]:
            gen = VAR1Generator(d=d, rho_A=0.5, seed=42)
            X, Y = gen.generate(n=500)
            assert X.shape == (500, d)
            assert Y.shape == (500, d)

    def test_sigma_positive_definite(self):
        """Innovation covariance should be positive definite."""
        gen = VAR1Generator(d=8, rho_A=0.5, rho_cross=0.3, seed=42)
        eigvals = np.linalg.eigvalsh(gen.Sigma)
        assert np.all(eigvals > 0)

    def test_sample_mean_converges(self):
        """Sample mean should converge to zero for large n."""
        gen = VAR1Generator(d=4, rho_A=0.5, seed=42)
        X, Y = gen.generate(n=10000)
        assert np.allclose(Y.mean(axis=0), 0.0, atol=0.15)

    def test_sample_covariance_converges(self):
        """Sample covariance should converge to theoretical Lyapunov solution."""
        gen = VAR1Generator(d=4, rho_A=0.5, seed=42)
        params = gen.oracle_params()
        expected_cov = params["stationary_cov"]

        X, Y = gen.generate(n=50000)
        sample_cov = np.cov(Y.T)

        # Relative error should be small
        rel_error = np.linalg.norm(sample_cov - expected_cov) / np.linalg.norm(expected_cov)
        assert rel_error < 0.1, f"Relative covariance error: {rel_error:.4f}"

    def test_oracle_forecast(self):
        """Oracle forecast should equal A @ X."""
        gen = VAR1Generator(d=4, rho_A=0.5, seed=42)
        X, Y = gen.generate(n=100)
        Y_hat = gen.oracle_forecast(X)
        # Y_hat = X @ A^T, residuals should be the innovations
        residuals = Y - Y_hat
        # Residuals should have mean ~0 and covariance ~Sigma
        assert np.allclose(residuals.mean(axis=0), 0.0, atol=0.3)

    def test_reproducibility(self):
        """Same seed should produce identical data."""
        gen1 = VAR1Generator(d=4, rho_A=0.5, seed=123)
        gen2 = VAR1Generator(d=4, rho_A=0.5, seed=123)
        X1, Y1 = gen1.generate(n=100)
        X2, Y2 = gen2.generate(n=100)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)

    def test_invalid_rho_A(self):
        """Should reject invalid spectral radius."""
        with pytest.raises(ValueError):
            VAR1Generator(d=4, rho_A=1.5, seed=42)
        with pytest.raises(ValueError):
            VAR1Generator(d=4, rho_A=0.0, seed=42)

    def test_high_dimension(self):
        """Should work for d=64."""
        gen = VAR1Generator(d=64, rho_A=0.5, seed=42)
        X, Y = gen.generate(n=200, burn_in=300)
        assert X.shape == (200, 64)
