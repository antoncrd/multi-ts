"""Run the synthetic VAR(1) experiment grid.

Usage:
    # Single run:
    python scripts/run_synthetic.py dgp.d=4 dgp.rho_A=0.5 conformal.k=500

    # Full grid sweep (Hydra multirun):
    python scripts/run_synthetic.py -m \
        dgp.d=2,4,8,16,32,64 \
        dgp.rho_A=0.1,0.3,0.5,0.7,0.9,0.95 \
        conformal.k=50,100,250,500,1000,5000,10000
"""

import json
import logging
import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.conformal.calibration import ConformalCalibrator
from src.conformal.rank_region import build_regions_batch
from src.conformal.wasserstein import rank_transform, uniformity_test
from src.dgp.var1 import VAR1Generator
from src.metrics.coverage import (
    coverage_gap,
    marginal_coverage,
    rolling_coverage,
    violation_rate,
)
from src.metrics.diagnostics import rank_uniformity_ks
from src.metrics.efficiency import mean_diameter, mean_log_volume, mean_winkler_log
from src.models.forecasters import LinearForecaster
from src.models.neural_ot import NeuralOTMap
from src.utils.seeds import set_all_seeds

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    """Run a single synthetic experiment configuration.

    Returns the coverage gap (for Hydra optimization if needed).
    """
    t0 = time.time()
    set_all_seeds(cfg.seed)

    d = cfg.dgp.d
    rho_A = cfg.dgp.rho_A
    rho_cross = cfg.dgp.rho_cross
    k = cfg.conformal.k  # calibration set size
    alpha = cfg.alpha
    beta = cfg.beta

    log.info(f"Config: d={d}, rho_A={rho_A}, k={k}, alpha={alpha}, beta={beta}")

    # --- 1. Generate data ---
    n_train = cfg.n_train
    n_test = cfg.n_test
    n_total = n_train + k + n_test

    dgp = VAR1Generator(d=d, rho_A=rho_A, rho_cross=rho_cross, seed=cfg.seed)
    X, Y = dgp.generate(n=n_total)

    # --- 2. Temporal split (NO shuffle) ---
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_calib, Y_calib = X[n_train:n_train + k], Y[n_train:n_train + k]
    X_test, Y_test = X[n_train + k:], Y[n_train + k:]

    log.info(f"Splits: train={n_train}, calib={k}, test={X_test.shape[0]}")

    # --- 3. Fit forecaster ---
    forecaster = LinearForecaster()
    forecaster.fit(X_train, Y_train)

    # --- 4. Compute calibration residuals ---
    residuals_calib = forecaster.residuals(X_calib, Y_calib)
    residuals_test = forecaster.residuals(X_test, Y_test)

    # --- 5. Train Neural OT map ---
    # Must happen BEFORE calibration: the OT map Q_hat defines the rank
    # transform R_hat(eps) = Q_hat(eps). The calibration step then computes
    # W_1(mu_hat_k, U_d) on these OT-based ranks, not marginal ranks.
    ot_map = None
    ot_ranks = None  # Will hold Q_hat(residuals) normalized to [0,1]^d
    ot_metrics = {}
    device = "cuda" if cfg.device == "cuda" and _cuda_available() else "cpu"

    try:
        ot_map = NeuralOTMap(
            input_dim=d,
            hidden_dims_icnn=list(cfg.model.icnn.hidden_dims),
            hidden_dims_inverse=list(cfg.model.inverse.hidden_dims),
            lr_icnn=cfg.model.icnn.lr,
            lr_inverse=cfg.model.inverse.lr,
            weight_decay=cfg.model.icnn.weight_decay,
            n_epochs=cfg.model.n_epochs,
            batch_size=cfg.model.batch_size,
            monge_gap_weight=cfg.model.monge_gap_weight,
            grad_clip=cfg.model.grad_clip,
            warmup_epochs=cfg.model.warmup_epochs,
            device=device,
        )
        train_hist = ot_map.fit(residuals_calib, verbose=False)
        convexity = ot_map.check_convexity(residuals_calib, n_checks=100)
        ot_metrics = {
            "icnn_final_loss": train_hist["icnn_loss"][-1] if train_hist["icnn_loss"] else None,
            "inverse_final_loss": train_hist["inverse_loss"][-1] if train_hist["inverse_loss"] else None,
            "convexity_score": convexity,
        }
        log.info(f"OT map trained. Convexity: {convexity:.3f}")

        # Compute OT-based ranks: apply Q_hat then normalize to [0,1]^d
        # via marginal rank transform of the OT-mapped residuals.
        ot_mapped = ot_map.forward_map_np(residuals_calib)
        ot_ranks = rank_transform(ot_mapped)

    except Exception as e:
        log.warning(f"Neural OT training failed: {e}. Falling back to marginal ranks.")

    # --- 6. Rank diagnostics ---
    # Use OT ranks if available, otherwise plain marginal ranks
    ranks = ot_ranks if ot_ranks is not None else rank_transform(residuals_calib)
    unif_test = uniformity_test(ranks)
    rank_ks = rank_uniformity_ks(ranks)

    # --- 7. Conformal calibration (uses OT ranks when available) ---
    calibrator = ConformalCalibrator(
        alpha=alpha,
        beta=beta,
        w1_method=cfg.conformal.w1_method,
        w1_n_projections=cfg.conformal.w1_n_projections,
    )
    cal_result = calibrator.calibrate(
        residuals_calib, seed=cfg.seed, ot_ranks=ot_ranks,
    )

    log.info(
        f"Calibration: rho_hat={cal_result.rho_hat_k:.4f}, "
        f"eta_k={cal_result.eta_k:.4f}, "
        f"r*={cal_result.r_star:.4f}"
    )

    # --- 8. Build prediction regions ---
    Y_hat_test = forecaster.predict(X_test)
    ot_inverse_fn = ot_map.inverse_map if ot_map is not None else None
    regions = build_regions_batch(Y_hat_test, cal_result.r_star, ot_inverse_fn)

    # --- 9. Evaluate metrics ---
    cov = marginal_coverage(regions, Y_test)
    cov_gap = coverage_gap(cov, 1.0 - alpha)
    viol = violation_rate(regions, Y_test)
    rolling_covs = rolling_coverage(regions, Y_test, n_windows=5)
    avg_log_vol = mean_log_volume(regions)
    avg_diam = mean_diameter(regions)
    avg_winkler_log = mean_winkler_log(regions, Y_test, alpha)

    # Naive baseline (no OT correction)
    naive_radius = calibrator.calibrate_naive(residuals_calib)
    naive_regions = build_regions_batch(Y_hat_test, naive_radius)
    naive_cov = marginal_coverage(naive_regions, Y_test)

    elapsed = time.time() - t0

    # --- 10. Collect and save results ---
    results = {
        "config": {
            "d": d,
            "rho_A": rho_A,
            "rho_cross": rho_cross,
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "seed": cfg.seed,
            "n_train": n_train,
            "n_test": int(X_test.shape[0]),
        },
        "calibration": {
            "rho_hat_k": float(cal_result.rho_hat_k),
            "eta_k": float(cal_result.eta_k),
            "delta_star": float(cal_result.delta_star),
            "r_star": float(cal_result.r_star),
            "c_rho": float(cal_result.c_rho),
            "naive_radius": float(naive_radius),
        },
        "metrics": {
            "coverage": float(cov),
            "coverage_gap": float(cov_gap),
            "violation_rate": float(viol),
            "rolling_coverages": rolling_covs.tolist(),
            "mean_log_volume": float(avg_log_vol),
            "mean_diameter": float(avg_diam),
            "mean_winkler_log": float(avg_winkler_log),
            "naive_coverage": float(naive_cov),
        },
        "diagnostics": {
            "rank_min_pvalue": float(rank_ks["min_pvalue"]),
            "rank_max_statistic": float(rank_ks["max_statistic"]),
            "norm_pvalue": float(unif_test["ks_norm_pvalue"]),
        },
        "ot_metrics": ot_metrics,
        "elapsed_seconds": elapsed,
    }

    # Save JSON
    output_path = os.path.join(os.getcwd(), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(
        f"Done. Coverage={cov:.3f} (gap={cov_gap:+.3f}), "
        f"log_vol={avg_log_vol:.2f}, time={elapsed:.1f}s"
    )

    return cov_gap


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
