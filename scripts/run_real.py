"""Run experiments on real-world datasets.

Usage:
    python scripts/run_real.py --dataset wind --d 8 --alpha 0.1
    python scripts/run_real.py --dataset all --d 4,8,16
"""

import argparse
import json
import logging
import os
import time

import numpy as np

from src.conformal.calibration import ConformalCalibrator
from src.conformal.rank_region import build_regions_batch
from src.conformal.wasserstein import rank_transform, uniformity_test
from src.data.loaders import load_dataset
from src.data.preprocessing import (
    StandardScaler,
    rolling_window_splits,
    subsample_dimensions,
)
from src.metrics.coverage import marginal_coverage, coverage_gap, violation_rate
from src.metrics.efficiency import mean_log_volume, mean_diameter, mean_winkler_log
from src.models.forecasters import LinearForecaster, MLPForecaster
from src.models.neural_ot import NeuralOTMap
from src.utils.seeds import set_all_seeds

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ALL_DATASETS = ["wind", "solar", "traffic", "exchange", "electricity", "nyc_taxi"]


def run_single_experiment(
    dataset_name: str,
    d: int,
    alpha: float = 0.1,
    beta: float = 0.05,
    n_windows: int = 5,
    seed: int = 42,
    forecaster_type: str = "linear",
    device: str = "cpu",
    output_dir: str = "results/real",
) -> dict:
    """Run a single real-data experiment."""
    set_all_seeds(seed)
    t0 = time.time()

    # Load and preprocess
    dataset = load_dataset(dataset_name)
    log.info(f"Loaded {dataset_name}: T={dataset.T}, D={dataset.D}")

    # Subsample dimensions
    data_sub, selected_dims = subsample_dimensions(dataset.data, d, seed=seed)
    log.info(f"Subsampled to d={data_sub.shape[1]} dims")

    # Create rolling window splits
    X_full = data_sub[:-1]
    Y_full = data_sub[1:]
    splits = rolling_window_splits(X_full, Y_full, n_windows=n_windows)

    all_results = []

    for w, split in enumerate(splits):
        X_train, Y_train = split["train"]
        X_calib, Y_calib = split["calib"]
        X_test, Y_test = split["test"]

        if X_test.shape[0] == 0:
            log.warning(f"Window {w}: empty test set, skipping")
            continue

        # Normalize
        scaler = StandardScaler().fit(X_train)
        X_train_n = scaler.transform(X_train)
        X_calib_n = scaler.transform(X_calib)
        X_test_n = scaler.transform(X_test)

        scaler_y = StandardScaler().fit(Y_train)
        Y_train_n = scaler_y.transform(Y_train)
        Y_calib_n = scaler_y.transform(Y_calib)
        Y_test_n = scaler_y.transform(Y_test)

        # Fit forecaster
        if forecaster_type == "linear":
            forecaster = LinearForecaster()
        else:
            forecaster = MLPForecaster(d=d, device=device)
        forecaster.fit(X_train_n, Y_train_n)

        # Calibrate
        residuals_calib = forecaster.residuals(X_calib_n, Y_calib_n)
        calibrator = ConformalCalibrator(alpha=alpha, beta=beta, w1_method="auto")
        cal_result = calibrator.calibrate(residuals_calib, seed=seed)

        # Build regions on test
        Y_hat_test = forecaster.predict(X_test_n)
        regions = build_regions_batch(Y_hat_test, cal_result.r_star)

        # Evaluate
        cov = marginal_coverage(regions, Y_test_n)
        gap = coverage_gap(cov, 1.0 - alpha)
        log_vol = mean_log_volume(regions)

        all_results.append({
            "window": w,
            "coverage": float(cov),
            "coverage_gap": float(gap),
            "mean_log_volume": float(log_vol),
            "r_star": float(cal_result.r_star),
            "n_test": int(X_test.shape[0]),
        })
        log.info(f"  Window {w}: coverage={cov:.3f}, gap={gap:+.3f}")

    elapsed = time.time() - t0

    results = {
        "dataset": dataset_name,
        "d": d,
        "alpha": alpha,
        "beta": beta,
        "seed": seed,
        "n_windows": n_windows,
        "windows": all_results,
        "mean_coverage": float(np.mean([r["coverage"] for r in all_results])),
        "mean_coverage_gap": float(np.mean([r["coverage_gap"] for r in all_results])),
        "elapsed_seconds": elapsed,
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset_name}_d{d}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved results to {path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="exchange")
    parser.add_argument("--d", type=str, default="4,8,16")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/real")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    dims = [int(x) for x in args.d.split(",")]

    for ds in datasets:
        for d in dims:
            try:
                run_single_experiment(
                    dataset_name=ds, d=d,
                    alpha=args.alpha, beta=args.beta,
                    seed=args.seed, output_dir=args.output_dir,
                )
            except FileNotFoundError as e:
                log.warning(f"Skipping {ds} d={d}: {e}")
            except Exception as e:
                log.error(f"Error on {ds} d={d}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
