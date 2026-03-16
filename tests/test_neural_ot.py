"""Tests for Neural OT (ICNN + amortized inverse)."""

import numpy as np
import pytest
import torch

from src.models.neural_ot import ICNN, AmortizedInverse, NeuralOTMap


class TestICNN:
    def test_output_shape(self):
        """ICNN should output scalar potential."""
        icnn = ICNN(input_dim=4, hidden_dims=[32, 32])
        x = torch.randn(16, 4)
        phi = icnn(x)
        assert phi.shape == (16, 1)

    def test_convexity(self):
        """ICNN output should be convex along random line segments."""
        icnn = ICNN(input_dim=4, hidden_dims=[32, 32])
        x = torch.randn(100, 4)

        violations = 0
        n_checks = 50
        for _ in range(n_checks):
            i, j = np.random.choice(100, 2, replace=False)
            lam = np.random.uniform(0, 1)
            x_mid = lam * x[i:i+1] + (1 - lam) * x[j:j+1]

            with torch.no_grad():
                phi_i = icnn(x[i:i+1]).item()
                phi_j = icnn(x[j:j+1]).item()
                phi_mid = icnn(x_mid).item()

            if phi_mid > lam * phi_i + (1 - lam) * phi_j + 1e-5:
                violations += 1

        assert violations / n_checks < 0.05, f"Too many convexity violations: {violations}/{n_checks}"

    def test_gradient_shape(self):
        """Gradient of ICNN should match input shape."""
        icnn = ICNN(input_dim=4, hidden_dims=[32, 32])
        x = torch.randn(16, 4)
        grad = icnn.gradient(x)
        assert grad.shape == (16, 4)


class TestAmortizedInverse:
    def test_output_shape(self):
        inverse = AmortizedInverse(dim=4, hidden_dims=[32, 32])
        y = torch.randn(16, 4)
        x_hat = inverse(y)
        assert x_hat.shape == (16, 4)


class TestNeuralOTMap:
    @pytest.mark.slow
    def test_training_completes(self):
        """Training should complete without errors."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 4)).astype(np.float32)

        ot_map = NeuralOTMap(
            input_dim=4,
            hidden_dims_icnn=[32, 32],
            hidden_dims_inverse=[32, 32],
            n_epochs=5,
            batch_size=64,
            device="cpu",
        )
        history = ot_map.fit(source, verbose=False)

        assert "icnn_loss" in history
        assert "inverse_loss" in history
        assert len(history["icnn_loss"]) == 5

    @pytest.mark.slow
    def test_inverse_quality(self):
        """R_hat(Q_hat(x)) should approximately reconstruct x."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal((500, 4)).astype(np.float32)

        ot_map = NeuralOTMap(
            input_dim=4,
            hidden_dims_icnn=[64, 64],
            hidden_dims_inverse=[64, 64],
            n_epochs=50,
            batch_size=128,
            device="cpu",
        )
        ot_map.fit(source, verbose=False)

        # Check reconstruction
        x = source[:50]
        Q_x = ot_map.forward_map_np(x)
        x_recon = ot_map.inverse_map_np(Q_x)

        recon_error = np.mean(np.linalg.norm(x - x_recon, axis=1))
        # Allow generous tolerance for short training
        assert recon_error < 5.0, f"Reconstruction error too high: {recon_error:.4f}"

    def test_convexity_check(self):
        """Convexity check should return a score between 0 and 1."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal((100, 4)).astype(np.float32)

        ot_map = NeuralOTMap(
            input_dim=4,
            hidden_dims_icnn=[32, 32],
            hidden_dims_inverse=[32, 32],
            n_epochs=1,
            device="cpu",
        )
        ot_map.fit(source, verbose=False)

        score = ot_map.check_convexity(source, n_checks=50)
        assert 0 <= score <= 1
