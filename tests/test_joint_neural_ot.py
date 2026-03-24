"""Tests for joint Neural OT model (QuantileMap + RankMap)."""

import numpy as np
import pytest
import torch

from src.models.joint_neural_ot import (
    JointNeuralOTMap,
    QuantileMap,
    RankMap,
    compute_lcyc,
    compute_lot_hungarian,
    compute_lunif_mmd,
    project_to_ball,
)
from src.models.neural_ot import sample_uniform_ball


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def d():
    return 4


@pytest.fixture
def batch_size():
    return 32


# ---------------------------------------------------------------------------
# project_to_ball
# ---------------------------------------------------------------------------

class TestProjectToBall:
    def test_strictly_inside_ball(self):
        x = torch.tensor([[0.1, 0.2], [0.3, 0.0]])
        out = project_to_ball(x)
        norms = out.norm(dim=1)
        assert (norms < 1.0).all()
        # Direction preserved
        cos_sim = (x * out).sum(dim=1) / (x.norm(dim=1) * out.norm(dim=1) + 1e-8)
        assert (cos_sim > 0.99).all()

    def test_large_input_still_inside(self):
        x = torch.tensor([[3.0, 4.0]])  # norm = 5
        out = project_to_ball(x)
        assert out.norm(dim=1).item() < 1.0

    def test_batch(self, d, batch_size):
        x = torch.randn(batch_size, d) * 3
        out = project_to_ball(x)
        norms = out.norm(dim=1)
        assert (norms <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# QuantileMap
# ---------------------------------------------------------------------------

class TestQuantileMap:
    def test_output_shape(self, d, batch_size):
        net = QuantileMap(d, hidden_dims=[32, 16])
        u = torch.randn(batch_size, d)
        out = net(u)
        assert out.shape == (batch_size, d)


# ---------------------------------------------------------------------------
# RankMap
# ---------------------------------------------------------------------------

class TestRankMap:
    def test_output_in_ball(self, d, batch_size):
        net = RankMap(d, hidden_dims=[32, 16])
        eps = torch.randn(batch_size, d) * 5
        out = net(eps)
        norms = out.norm(dim=1)
        assert (norms <= 1.0 + 1e-5).all()

    def test_output_shape(self, d, batch_size):
        net = RankMap(d, hidden_dims=[32, 16])
        eps = torch.randn(batch_size, d)
        out = net(eps)
        assert out.shape == (batch_size, d)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestLosses:
    def test_lot_hungarian_gradient(self, d, batch_size):
        q = torch.randn(batch_size, d, requires_grad=True)
        eps = torch.randn(batch_size, d)
        loss = compute_lot_hungarian(q, eps)
        loss.backward()
        assert q.grad is not None
        assert q.grad.shape == (batch_size, d)

    def test_lot_hungarian_zero_perfect_match(self, d, batch_size):
        eps = torch.randn(batch_size, d)
        loss = compute_lot_hungarian(eps.clone().requires_grad_(True), eps)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_lcyc_scalar(self, d, batch_size):
        qmap = QuantileMap(d, hidden_dims=[16, 8])
        rmap = RankMap(d, hidden_dims=[16, 8])
        eps = torch.randn(batch_size, d)
        u = sample_uniform_ball(batch_size, d, torch.device("cpu"))
        loss = compute_lcyc(qmap, rmap, eps, u)
        assert loss.dim() == 0  # scalar

    def test_lunif_mmd_same_dist(self, d):
        """MMD between two U(B_d) samples should be near zero."""
        n = 500
        u1 = sample_uniform_ball(n, d, torch.device("cpu"))
        u2 = sample_uniform_ball(n, d, torch.device("cpu"))
        mmd = compute_lunif_mmd(u1, u2)
        assert mmd.item() < 0.1, f"MMD too large for same distribution: {mmd.item()}"

    def test_lunif_mmd_different_dist(self, d):
        """MMD between U(B_d) and a point mass should be large."""
        n = 200
        u = sample_uniform_ball(n, d, torch.device("cpu"))
        point_mass = torch.zeros(n, d) + 0.5
        mmd = compute_lunif_mmd(point_mass, u)
        assert mmd.item() > 0.01


# ---------------------------------------------------------------------------
# JointNeuralOTMap
# ---------------------------------------------------------------------------

class TestJointNeuralOTMap:
    def test_training_completes(self, d):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, d)).astype(np.float32)

        model = JointNeuralOTMap(
            input_dim=d,
            hidden_dims_q=[32, 16],
            hidden_dims_r=[32, 16],
            n_epochs=10,
            batch_size=32,
            warmup_epochs=2,
        )
        history = model.fit(data, verbose=False)

        assert "total" in history
        assert "lot" in history
        assert "lcyc" in history
        assert "lunif" in history
        assert len(history["total"]) == 10

    def test_loss_decreases(self, d):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, d)).astype(np.float32)

        model = JointNeuralOTMap(
            input_dim=d,
            hidden_dims_q=[64, 32],
            hidden_dims_r=[64, 32],
            n_epochs=30,
            batch_size=64,
            warmup_epochs=3,
        )
        history = model.fit(data, verbose=False)
        # Total loss should decrease from first settled epoch to last
        assert history["total"][-1] < history["total"][5]

    def test_forward_map_shapes(self, d):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, d)).astype(np.float32)

        model = JointNeuralOTMap(
            input_dim=d,
            hidden_dims_q=[32, 16],
            hidden_dims_r=[32, 16],
            n_epochs=5,
            batch_size=32,
            warmup_epochs=1,
        )
        model.fit(data, verbose=False)

        ranks = model.forward_map_np(data)
        assert ranks.shape == (100, d)

        norms = np.linalg.norm(ranks, axis=1)
        assert np.all(norms <= 1.0 + 1e-5)

    def test_inverse_map_shapes(self, d):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, d)).astype(np.float32)

        model = JointNeuralOTMap(
            input_dim=d,
            hidden_dims_q=[32, 16],
            hidden_dims_r=[32, 16],
            n_epochs=5,
            batch_size=32,
            warmup_epochs=1,
        )
        model.fit(data, verbose=False)

        u = np.random.default_rng(0).standard_normal((50, d)).astype(np.float32)
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1.0)
        residuals = model.inverse_map_np(u)
        assert residuals.shape == (50, d)

    def test_ball_coverage_always_one(self, d):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, d)).astype(np.float32)

        model = JointNeuralOTMap(
            input_dim=d,
            hidden_dims_q=[32, 16],
            hidden_dims_r=[32, 16],
            n_epochs=5,
            batch_size=32,
            warmup_epochs=1,
        )
        model.fit(data, verbose=False)
        assert model.ball_coverage(data) == pytest.approx(1.0)

    @pytest.mark.slow
    def test_cycle_consistency_after_training(self):
        d = 4
        rng = np.random.default_rng(42)
        data = rng.standard_normal((500, d)).astype(np.float32)

        model = JointNeuralOTMap(
            input_dim=d,
            hidden_dims_q=[128, 64],
            hidden_dims_r=[128, 64],
            n_epochs=200,
            batch_size=64,
            lambda_cyc=10.0,
        )
        model.fit(data, verbose=False)

        # Check R(eps) -> Q(R(eps)) ≈ eps
        eps_t = torch.tensor(data[:50], dtype=torch.float32)
        eps_norm = model._normalize(eps_t)
        model.rank_map.eval()
        model.quantile_map.eval()
        with torch.no_grad():
            r = model.rank_map(eps_norm)
            recon = model.quantile_map(r)

        reconstruction_error = ((recon - eps_norm) ** 2).mean().item()
        assert reconstruction_error < 1.0, (
            f"Cycle consistency error too large: {reconstruction_error:.4f}"
        )
