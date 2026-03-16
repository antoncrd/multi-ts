"""Run ablation studies A-E.

Ablation A: OT map quality (vary ICNN depth/width)
Ablation B: Confidence budget beta
Ablation C: Smoothing delta* (compare optimized vs fixed)
Ablation D: Explicit vs non-explicit constants
Ablation E: Model sensitivity (forecaster choice)

Usage:
    python scripts/run_ablations.py --ablation A --d 8 --rho_A 0.5 --k 500
    python scripts/run_ablations.py --ablation all
"""

import argparse
import json
import logging
import os
import time

import numpy as np

from src.conformal.calibration import ConformalCalibrator
from src.conformal.rank_region import build_regions_batch
from src.dgp.var1 import VAR1Generator
from src.metrics.coverage import marginal_coverage, coverage_gap
from src.metrics.efficiency import mean_log_volume
from src.models.forecasters import LinearForecaster, MLPForecaster, OracleForecaster
from src.models.neural_ot import NeuralOTMap
from src.utils.seeds import set_all_seeds

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def ablation_A_ot_quality(d=8, rho_A=0.5, k=500, seed=42, output_dir="results/ablations"):
    """Ablation A: Effect of ICNN architecture on OT map quality."""
    set_all_seeds(seed)
    dgp = VAR1Generator(d=d, rho_A=rho_A, seed=seed)
    X, Y = dgp.generate(n=2000 + k + 500)
    X_train, Y_train = X[:2000], Y[:2000]
    X_calib, Y_calib = X[2000:2000+k], Y[2000:2000+k]
    X_test, Y_test = X[2000+k:], Y[2000+k:]

    forecaster = LinearForecaster()
    forecaster.fit(X_train, Y_train)
    residuals = forecaster.residuals(X_calib, Y_calib)

    architectures = [
        {"name": "small", "hidden_dims_icnn": [32, 32], "hidden_dims_inverse": [32, 32]},
        {"name": "medium", "hidden_dims_icnn": [128, 128, 64], "hidden_dims_inverse": [128, 128]},
        {"name": "large", "hidden_dims_icnn": [256, 256, 128, 64], "hidden_dims_inverse": [256, 256, 128]},
    ]

    results = []
    for arch in architectures:
        ot_map = NeuralOTMap(
            input_dim=d,
            hidden_dims_icnn=arch["hidden_dims_icnn"],
            hidden_dims_inverse=arch["hidden_dims_inverse"],
            n_epochs=100, batch_size=128, device="cpu",
        )
        ot_map.fit(residuals, verbose=False)
        convexity = ot_map.check_convexity(residuals, n_checks=100)

        calibrator = ConformalCalibrator(alpha=0.1, beta=0.05, w1_method="sliced")
        cal = calibrator.calibrate(residuals, seed=seed)

        Y_hat = forecaster.predict(X_test)
        regions = build_regions_batch(Y_hat, cal.r_star)
        cov = marginal_coverage(regions, Y_test)

        results.append({
            "architecture": arch["name"],
            "convexity": convexity,
            "coverage": float(cov),
            "r_star": float(cal.r_star),
            "mean_log_volume": float(mean_log_volume(regions)),
        })
        log.info(f"  Arch {arch['name']}: convexity={convexity:.3f}, coverage={cov:.3f}")

    _save(results, "ablation_A", output_dir)
    return results


def ablation_B_beta_budget(d=8, rho_A=0.5, k=500, seed=42, output_dir="results/ablations"):
    """Ablation B: Effect of confidence budget beta."""
    set_all_seeds(seed)
    dgp = VAR1Generator(d=d, rho_A=rho_A, seed=seed)
    X, Y = dgp.generate(n=2000 + k + 500)
    X_train, Y_train = X[:2000], Y[:2000]
    X_calib, Y_calib = X[2000:2000+k], Y[2000:2000+k]
    X_test, Y_test = X[2000+k:], Y[2000+k:]

    forecaster = LinearForecaster()
    forecaster.fit(X_train, Y_train)
    residuals = forecaster.residuals(X_calib, Y_calib)

    betas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results = []

    for beta in betas:
        calibrator = ConformalCalibrator(alpha=0.1, beta=beta, w1_method="sliced")
        cal = calibrator.calibrate(residuals, seed=seed)

        Y_hat = forecaster.predict(X_test)
        regions = build_regions_batch(Y_hat, cal.r_star)
        cov = marginal_coverage(regions, Y_test)

        results.append({
            "beta": beta,
            "coverage": float(cov),
            "r_star": float(cal.r_star),
            "eta_k": float(cal.eta_k),
            "mean_log_volume": float(mean_log_volume(regions)),
        })
        log.info(f"  beta={beta}: coverage={cov:.3f}, r*={cal.r_star:.4f}")

    _save(results, "ablation_B", output_dir)
    return results


def ablation_C_delta_smoothing(d=8, rho_A=0.5, k=500, seed=42, output_dir="results/ablations"):
    """Ablation C: Optimized delta* vs fixed delta values."""
    set_all_seeds(seed)
    dgp = VAR1Generator(d=d, rho_A=rho_A, seed=seed)
    X, Y = dgp.generate(n=2000 + k + 500)
    X_train, Y_train = X[:2000], Y[:2000]
    X_calib, Y_calib = X[2000:2000+k], Y[2000:2000+k]
    X_test, Y_test = X[2000+k:], Y[2000+k:]

    forecaster = LinearForecaster()
    forecaster.fit(X_train, Y_train)
    residuals = forecaster.residuals(X_calib, Y_calib)

    calibrator = ConformalCalibrator(alpha=0.1, beta=0.05, w1_method="sliced")
    cal = calibrator.calibrate(residuals, seed=seed)

    # Optimized delta
    delta_opt = cal.delta_star
    A = cal.rho_hat_k + cal.eta_k

    # Fixed delta values
    deltas = [0.0, delta_opt * 0.5, delta_opt, delta_opt * 2.0, delta_opt * 5.0]
    results = []

    for delta in deltas:
        r = calibrator.compute_radius(0.1, delta, A, d)
        Y_hat = forecaster.predict(X_test)
        regions = build_regions_batch(Y_hat, r)
        cov = marginal_coverage(regions, Y_test)

        results.append({
            "delta": float(delta),
            "delta_is_optimal": abs(delta - delta_opt) < 1e-10,
            "r_star": float(r),
            "coverage": float(cov),
            "mean_log_volume": float(mean_log_volume(regions)),
        })
        log.info(f"  delta={delta:.4f}: r*={r:.4f}, coverage={cov:.3f}")

    _save(results, "ablation_C", output_dir)
    return results


def ablation_E_model_sensitivity(d=8, rho_A=0.5, k=500, seed=42, output_dir="results/ablations"):
    """Ablation E: Sensitivity to forecaster choice."""
    set_all_seeds(seed)
    dgp = VAR1Generator(d=d, rho_A=rho_A, seed=seed)
    X, Y = dgp.generate(n=2000 + k + 500)
    X_train, Y_train = X[:2000], Y[:2000]
    X_calib, Y_calib = X[2000:2000+k], Y[2000:2000+k]
    X_test, Y_test = X[2000+k:], Y[2000+k:]

    forecasters = {
        "oracle": OracleForecaster(dgp.A),
        "linear": LinearForecaster(),
        "mlp": MLPForecaster(d=d, n_epochs=50, device="cpu"),
    }

    results = []
    for name, forecaster in forecasters.items():
        forecaster.fit(X_train, Y_train)
        residuals = forecaster.residuals(X_calib, Y_calib)

        calibrator = ConformalCalibrator(alpha=0.1, beta=0.05, w1_method="sliced")
        cal = calibrator.calibrate(residuals, seed=seed)

        Y_hat = forecaster.predict(X_test)
        regions = build_regions_batch(Y_hat, cal.r_star)
        cov = marginal_coverage(regions, Y_test)

        # Forecaster quality
        mse = np.mean(np.sum((Y_test - Y_hat) ** 2, axis=1))

        results.append({
            "forecaster": name,
            "coverage": float(cov),
            "r_star": float(cal.r_star),
            "mean_log_volume": float(mean_log_volume(regions)),
            "mse": float(mse),
        })
        log.info(f"  {name}: coverage={cov:.3f}, MSE={mse:.4f}")

    _save(results, "ablation_E", output_dir)
    return results


def _save(results: list, name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {name} to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="all",
                        help="A, B, C, E, or all")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--rho_A", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/ablations")
    args = parser.parse_args()

    kwargs = dict(d=args.d, rho_A=args.rho_A, k=args.k, seed=args.seed,
                  output_dir=args.output_dir)

    ablations = {
        "A": ablation_A_ot_quality,
        "B": ablation_B_beta_budget,
        "C": ablation_C_delta_smoothing,
        "E": ablation_E_model_sensitivity,
    }

    to_run = ablations if args.ablation == "all" else {args.ablation: ablations[args.ablation]}

    for name, fn in to_run.items():
        log.info(f"Running ablation {name}...")
        fn(**kwargs)


if __name__ == "__main__":
    main()
