"""Compare PICNN vs marginal ranks: W1 distance to U(B_d)."""
import numpy as np
from src.models.neural_ot import NeuralOTMap
from src.conformal.wasserstein import (
    rank_transform, wasserstein_1_uniform, _sample_uniform_ball
)

rng = np.random.default_rng(42)

for d in [2, 4]:
    k = 500
    residuals = rng.standard_normal((k, d)).astype(np.float32)

    # --- Marginal ranks ---
    marginal_ranks = rank_transform(residuals)
    w1_marginal = wasserstein_1_uniform(marginal_ranks, d, method='sliced',
                                        n_projections=500, seed=42)

    # --- Perfect U(B_d) baseline ---
    perfect = _sample_uniform_ball(k, d, rng)
    w1_perfect = wasserstein_1_uniform(perfect, d, method='sliced',
                                       n_projections=500, seed=42)

    # --- PICNN OT ranks (self-conditioning) ---
    ot_map = NeuralOTMap(
        input_dim=d,
        hidden_dims_icnn=[128, 128, 64],
        hidden_dims_inverse=[128, 128, 64],
        context_hidden_dims=[128, 128, 64],
        n_epochs=300,
        batch_size=min(128, k),
        n_projections=200,
        strong_convexity=0.1,
        ball_penalty_weight=20.0,
        warmup_epochs=30,
        device="cpu",
    )
    ot_map.fit(residuals, verbose=False)

    ot_ranks = ot_map.forward_map_np(residuals)
    w1_ot = wasserstein_1_uniform(ot_ranks, d, method='sliced',
                                   n_projections=500, seed=42)
    stats = ot_map.output_norm_stats(residuals)

    ratio = w1_marginal / max(w1_ot, 1e-6)
    print(f"d={d}: Marginal W1={w1_marginal:.4f} | PICNN W1={w1_ot:.4f} | "
          f"Perfect W1={w1_perfect:.4f} | "
          f"norms: mean={stats['mean']:.3f} max={stats['max']:.3f} "
          f"in_ball={stats['frac_in_ball']:.0%} | "
          f"{ratio:.1f}x improvement")
