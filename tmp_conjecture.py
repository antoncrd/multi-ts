"""Test log(1/beta) conjecture for eta_k."""
import numpy as np
from src.conformal.calibration import ConformalCalibrator
from src.conformal.wasserstein import _sample_uniform_ball
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)

print(f"log(1/beta) conjecture: log(1/0.05)={np.log(1/0.05):.4f} vs 1/0.05={1/0.05:.1f}")
print(f"  => factor reduction: {(1/0.05)/np.log(1/0.05):.1f}x\n")

print("=== OT ranks from U(B_d), alpha=0.1, beta=0.05 ===")
for d in [2, 4, 8]:
    print(f"\n  d={d}")
    for k in [500, 1000, 5000, 10000]:
        ot_ranks = _sample_uniform_ball(k, d, rng)
        r = ConformalCalibrator(alpha=0.1, beta=0.05, w1_method="sliced").calibrate(
            np.zeros((k, d)), seed=42, ot_ranks=ot_ranks)
        A = r.rho_hat_k + r.eta_k
        tag = "" if r.r_star < 1.0 else "  VACUOUS"
        print(f"    k={k:>5d}  r*={r.r_star:.6f}  eta_k={r.eta_k:.4f}  rho={r.rho_hat_k:.4f}  A={A:.4f}{tag}")

print("\n\n=== Marginal ranks (Gaussian), alpha=0.1, beta=0.05 ===")
for d in [2, 4]:
    print(f"\n  d={d}")
    for k in [500, 1000, 5000, 10000]:
        res = rng.standard_normal((k, d))
        r = ConformalCalibrator(alpha=0.1, beta=0.05, w1_method="sliced").calibrate(res, seed=42)
        A = r.rho_hat_k + r.eta_k
        tag = "" if r.r_star < 1.0 else "  VACUOUS"
        print(f"    k={k:>5d}  r*={r.r_star:.6f}  eta_k={r.eta_k:.4f}  rho={r.rho_hat_k:.4f}  A={A:.4f}{tag}")
