"""d=2 focused calibration: PICNN vs marginal ranks (proper train/calib split)."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from src.models.neural_ot import NeuralOTMap
from src.conformal.wasserstein import rank_transform, wasserstein_1_uniform
from src.conformal.calibration import ConformalCalibrator

rng = np.random.default_rng(42)
d = 2

print("=== d=2: Proper train/calib split ===\n")

for k in [5000, 10000, 50000, 100000]:
    # n_train samples to train OT map, k samples for calibration (held out)
    n_train = k  # same size for training
    n_total = n_train + k
    all_residuals = rng.standard_normal((n_total, d)).astype(np.float32)
    residuals_train = all_residuals[:n_train]
    residuals_calib = all_residuals[n_train:]

    print(f"--- k={k}: train OT on {n_train}, calibrate on {k} ---")

    # Train PICNN on training residuals only
    ot_map = NeuralOTMap(
        input_dim=d, hidden_dims_icnn=[64, 64],
        hidden_dims_inverse=[64, 64], context_hidden_dims=[64, 64],
        n_epochs=150, batch_size=min(512, n_train), n_projections=100,
        ball_penalty_weight=20.0, warmup_epochs=20, device="cpu",
    )
    ot_map.fit(residuals_train, verbose=True)

    # Apply trained OT map to held-out calibration residuals
    ot_ranks_calib = ot_map.forward_map_np(residuals_calib)

    # Calibrate with OT ranks (on held-out data)
    cal = ConformalCalibrator(alpha=0.1, beta=0.05, w1_method="sliced")
    result_ot = cal.calibrate(residuals_calib, seed=42, ot_ranks=ot_ranks_calib)

    # Calibrate with marginal ranks (on held-out data)
    result_m = cal.calibrate(residuals_calib, seed=42)

    A_ot = result_ot.rho_hat_k + result_ot.eta_k
    A_m = result_m.rho_hat_k + result_m.eta_k

    # Also show in-ball stats on calib set
    norms_calib = np.linalg.norm(ot_ranks_calib, axis=1)
    in_ball = np.mean(norms_calib <= 1.0)
    mean_norm = norms_calib.mean()

    print(f"  Calib OT norms: mean={mean_norm:.3f} in_ball={in_ball:.0%}")
    print(f"  PICNN    rho={result_ot.rho_hat_k:.4f} eta={result_ot.eta_k:.4f} A={A_ot:.4f} r*={result_ot.r_star:.4f}")
    print(f"  Marginal rho={result_m.rho_hat_k:.4f} eta={result_m.eta_k:.4f} A={A_m:.4f} r*={result_m.r_star:.4f}")
    print()
