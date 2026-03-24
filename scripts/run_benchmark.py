"""Benchmark: Neural OT method vs all baselines on real-world datasets.

Downloads the Exchange Rate dataset automatically, then runs:
  - Neural OT (ours)
  - Naive Euclidean ball (no OT correction)
  - Coordinate-wise SPCI
  - MultiDimSPCI (ellipsoidal)
  - CopulaCPTS
  - ACI (adaptive)
  - EnbPI (ensemble bootstrap)

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --dataset exchange --d 4,8 --alpha 0.1
    python scripts/run_benchmark.py --dataset all --forecaster mlp
"""

import argparse
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from src.conformal.calibration import ConformalCalibrator
from src.conformal.rank_region import build_region, build_regions_batch
from src.data.loaders import TimeSeriesDataset, load_dataset
from src.data.preprocessing import (
    StandardScaler,
    rolling_window_splits,
    subsample_dimensions,
)
from src.metrics.coverage import marginal_coverage, coverage_gap, violation_rate
from src.metrics.efficiency import mean_log_volume, mean_diameter, mean_winkler_log
from src.models.baselines.aci import ACI
from src.models.baselines.coordinatewise_spci import CoordinateWiseSPCI
from src.models.baselines.copulacpts import CopulaCPTS
from src.models.baselines.enbpi import EnbPI
from src.models.baselines.multidimspci import MultiDimSPCI
from src.models.forecasters import LinearForecaster, MLPForecaster
from src.models.neural_ot import NeuralOTMap
from src.utils.seeds import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data download helpers
# ---------------------------------------------------------------------------

# Primary URLs (Hugging Face THUML mirror)
DATASET_URLS = {
    "exchange": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/exchange_rate/exchange_rate.csv",
    ],
    "electricity": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv",
    ],
    "traffic": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv",
    ],
    "weather": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv",
    ],
}

# Map from dataset name to the CSV filename expected by loaders.py
DATASET_FILES = {
    "exchange": "exchange_rate.csv",
    "electricity": "electricity.csv",
    "traffic": "traffic.csv",
    "weather": "weather.csv",
}


def download_dataset(name: str, data_dir: str = "data/raw") -> Path:
    """Download a dataset if not already present. Tries multiple URLs."""
    if name not in DATASET_URLS:
        raise ValueError(
            f"No download URL for '{name}'. "
            f"Available: {list(DATASET_URLS.keys())}"
        )

    os.makedirs(data_dir, exist_ok=True)
    filename = DATASET_FILES[name]
    filepath = Path(data_dir) / filename

    if filepath.exists():
        log.info(f"Dataset '{name}' already present at {filepath}")
        return filepath

    urls = DATASET_URLS[name]
    for url in urls:
        log.info(f"Trying to download '{name}' from {url} ...")
        try:
            urllib.request.urlretrieve(url, filepath)
            log.info(f"Saved to {filepath}")
            return filepath
        except Exception as e:
            log.warning(f"  Failed: {e}")

    raise RuntimeError(
        f"All download URLs failed for '{name}'.\n"
        f"Please download manually to {filepath}"
    )


def load_real_dataset(name: str, data_dir: str = "data/raw") -> TimeSeriesDataset:
    """Load a real-world CSV dataset."""
    filename = DATASET_FILES.get(name, f"{name}.csv")
    filepath = Path(data_dir) / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    import pandas as pd
    df = pd.read_csv(str(filepath))

    # Drop date column if present
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Handle NaN
    if df.isnull().any().any():
        df = df.ffill().bfill()

    data = df.values.astype(np.float64)
    log.info(f"Loaded '{name}': shape {data.shape}")
    return TimeSeriesDataset(data=data, name=name)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def run_neural_ot(
    residuals_calib: np.ndarray,
    Y_hat_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float,
    beta: float,
    seed: int,
    device: str,
    n_epochs: int = 300,
) -> dict:
    """Run Neural OT method (ours)."""
    k, d = residuals_calib.shape

    # Train Neural OT map on calibration residuals
    ot_map = NeuralOTMap(
        input_dim=d,
        n_epochs=n_epochs,
        batch_size=min(256, k),
        device=device,
        strong_convexity=0.1,
        ball_penalty_weight=10.0,
    )
    ot_map.fit(residuals_calib, verbose=False)

    # Get OT-mapped norms for calibration residuals
    Q_calib = ot_map.forward_map_np(residuals_calib)
    ot_norms = np.linalg.norm(Q_calib, axis=1)

    # Empirical OT quantile
    calibrator = ConformalCalibrator(alpha=alpha, beta=beta)
    r_star_empirical = calibrator.calibrate_empirical_ot(ot_norms)

    # Theoretical W1-based radius
    cal_result = calibrator.calibrate(residuals_calib, seed=seed, ot_ranks=Q_calib)
    r_star_theory = cal_result.r_star

    # Use empirical OT radius (tighter in practice)
    r_star = r_star_empirical

    # Build regions with OT forward/inverse + calib residuals for volume estimation
    regions = [
        build_region(
            Y_hat_test[i], r_star,
            ot_forward=ot_map.forward_map_np,
            ot_inverse=ot_map.inverse_map_np,
            calib_residuals=residuals_calib,
        )
        for i in range(Y_hat_test.shape[0])
    ]

    cov = marginal_coverage(regions, Y_test)
    gap = coverage_gap(cov, 1.0 - alpha)
    log_vol = mean_log_volume(regions)
    diam = mean_diameter(regions)
    winkler = mean_winkler_log(regions, Y_test, alpha)

    return {
        "method": "Neural OT (ours)",
        "coverage": float(cov),
        "coverage_gap": float(gap),
        "mean_log_volume": float(log_vol),
        "mean_diameter": float(diam),
        "mean_winkler_log": float(winkler),
        "r_star": float(r_star),
        "r_star_theory": float(r_star_theory),
        "ball_coverage": float(ot_map.ball_coverage(residuals_calib)),
    }


def run_naive_ball(
    residuals_calib: np.ndarray,
    Y_hat_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float,
) -> dict:
    """Naive Euclidean ball (no OT)."""
    calibrator = ConformalCalibrator(alpha=alpha)
    r_star = calibrator.calibrate_naive(residuals_calib)

    regions = build_regions_batch(Y_hat_test, r_star)

    cov = marginal_coverage(regions, Y_test)
    gap = coverage_gap(cov, 1.0 - alpha)
    log_vol = mean_log_volume(regions)
    diam = mean_diameter(regions)
    winkler = mean_winkler_log(regions, Y_test, alpha)

    return {
        "method": "Naive Ball",
        "coverage": float(cov),
        "coverage_gap": float(gap),
        "mean_log_volume": float(log_vol),
        "mean_diameter": float(diam),
        "mean_winkler_log": float(winkler),
        "r_star": float(r_star),
    }


def run_coordinatewise(
    residuals_calib: np.ndarray,
    Y_hat_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float,
) -> dict:
    """Coordinate-wise SPCI baseline."""
    method = CoordinateWiseSPCI(alpha=alpha)
    method.calibrate(residuals_calib)
    regions = method.predict_regions(Y_hat_test)

    # Also compute true rectangular containment
    residuals_test = Y_test - Y_hat_test
    rect_covered = sum(
        method.contains_rect(residuals_test[t]) for t in range(len(residuals_test))
    )
    rect_coverage = rect_covered / len(residuals_test)

    cov = marginal_coverage(regions, Y_test)
    gap = coverage_gap(cov, 1.0 - alpha)
    log_vol = mean_log_volume(regions)
    diam = mean_diameter(regions)
    winkler = mean_winkler_log(regions, Y_test, alpha)

    return {
        "method": "Coord-wise SPCI",
        "coverage": float(cov),
        "coverage_gap": float(gap),
        "coverage_rect": float(rect_coverage),
        "mean_log_volume": float(log_vol),
        "mean_diameter": float(diam),
        "mean_winkler_log": float(winkler),
    }


def run_multidimspci(
    residuals_calib: np.ndarray,
    Y_hat_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float,
) -> dict:
    """MultiDimSPCI (ellipsoidal) baseline."""
    method = MultiDimSPCI(alpha=alpha)
    method.calibrate(residuals_calib)
    regions = method.predict_regions(Y_hat_test)

    # True ellipsoidal containment
    residuals_test = Y_test - Y_hat_test
    ell_covered = sum(
        method.contains(residuals_test[t]) for t in range(len(residuals_test))
    )
    ell_coverage = ell_covered / len(residuals_test)

    cov = marginal_coverage(regions, Y_test)
    gap = coverage_gap(cov, 1.0 - alpha)
    log_vol = mean_log_volume(regions)
    diam = mean_diameter(regions)
    winkler = mean_winkler_log(regions, Y_test, alpha)

    return {
        "method": "MultiDimSPCI",
        "coverage": float(cov),
        "coverage_gap": float(gap),
        "coverage_ellipsoid": float(ell_coverage),
        "mean_log_volume": float(log_vol),
        "mean_diameter": float(diam),
        "mean_winkler_log": float(winkler),
    }


def run_copulacpts(
    residuals_calib: np.ndarray,
    Y_hat_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float,
) -> dict:
    """CopulaCPTS baseline."""
    method = CopulaCPTS(alpha=alpha)
    method.calibrate(residuals_calib)
    regions = method.predict_regions(Y_hat_test)

    residuals_test = Y_test - Y_hat_test
    rect_covered = sum(
        method.contains_rect(residuals_test[t]) for t in range(len(residuals_test))
    )
    rect_coverage = rect_covered / len(residuals_test)

    cov = marginal_coverage(regions, Y_test)
    gap = coverage_gap(cov, 1.0 - alpha)
    log_vol = mean_log_volume(regions)
    diam = mean_diameter(regions)
    winkler = mean_winkler_log(regions, Y_test, alpha)

    return {
        "method": "CopulaCPTS",
        "coverage": float(cov),
        "coverage_gap": float(gap),
        "coverage_rect": float(rect_coverage),
        "mean_log_volume": float(log_vol),
        "mean_diameter": float(diam),
        "mean_winkler_log": float(winkler),
    }


def run_aci(
    residuals_calib: np.ndarray,
    Y_hat_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float,
) -> dict:
    """ACI (adaptive) baseline."""
    method = ACI(alpha=alpha, gamma=0.01)
    method.calibrate(residuals_calib)
    regions = method.predict_regions_adaptive(Y_hat_test, Y_true=Y_test)

    cov = marginal_coverage(regions, Y_test)
    gap = coverage_gap(cov, 1.0 - alpha)
    log_vol = mean_log_volume(regions)
    diam = mean_diameter(regions)
    winkler = mean_winkler_log(regions, Y_test, alpha)

    return {
        "method": "ACI",
        "coverage": float(cov),
        "coverage_gap": float(gap),
        "mean_log_volume": float(log_vol),
        "mean_diameter": float(diam),
        "mean_winkler_log": float(winkler),
    }


def run_enbpi(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_calib: np.ndarray,
    Y_calib: np.ndarray,
    Y_hat_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float,
    seed: int,
    forecaster_type: str = "linear",
    d: int = 4,
) -> dict:
    """EnbPI (ensemble bootstrap) baseline."""
    if forecaster_type == "linear":
        fclass = LinearForecaster
        fkwargs = {}
    else:
        fclass = MLPForecaster
        fkwargs = {"d": d}

    method = EnbPI(alpha=alpha, n_bootstrap=20, seed=seed)
    method.calibrate(fclass, fkwargs, X_train, Y_train, X_calib, Y_calib)
    regions = method.predict_regions(Y_hat_test)

    cov = marginal_coverage(regions, Y_test)
    gap = coverage_gap(cov, 1.0 - alpha)
    log_vol = mean_log_volume(regions)
    diam = mean_diameter(regions)
    winkler = mean_winkler_log(regions, Y_test, alpha)

    return {
        "method": "EnbPI",
        "coverage": float(cov),
        "coverage_gap": float(gap),
        "mean_log_volume": float(log_vol),
        "mean_diameter": float(diam),
        "mean_winkler_log": float(winkler),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_name: str,
    d: int,
    alpha: float = 0.1,
    beta: float = 0.05,
    n_windows: int = 3,
    seed: int = 42,
    forecaster_type: str = "linear",
    device: str = "cpu",
    ot_epochs: int = 300,
) -> dict:
    """Run full benchmark on a single dataset/dimension combination."""
    set_all_seeds(seed)
    t0 = time.time()

    # Download and load dataset
    download_dataset(dataset_name)
    dataset = load_real_dataset(dataset_name)
    log.info(f"Dataset '{dataset_name}': T={dataset.T}, D={dataset.D}")

    if d > dataset.D:
        log.warning(f"Requested d={d} > D={dataset.D}, using d={dataset.D}")
        d = dataset.D

    # Subsample dimensions
    data_sub, selected_dims = subsample_dimensions(dataset.data, d, seed=seed)
    log.info(f"Subsampled to d={data_sub.shape[1]}")

    # X[t] = data[t], Y[t] = data[t+1]
    X_full = data_sub[:-1]
    Y_full = data_sub[1:]
    splits = rolling_window_splits(X_full, Y_full, n_windows=n_windows)

    # Aggregate results across windows
    all_window_results = []

    for w, split in enumerate(splits):
        X_train, Y_train = split["train"]
        X_calib, Y_calib = split["calib"]
        X_test, Y_test = split["test"]

        if X_test.shape[0] < 10:
            log.warning(f"Window {w}: test set too small ({X_test.shape[0]}), skipping")
            continue

        log.info(
            f"Window {w}: train={X_train.shape[0]}, "
            f"calib={X_calib.shape[0]}, test={X_test.shape[0]}"
        )

        # Normalize
        scaler_x = StandardScaler().fit(X_train)
        X_train_n = scaler_x.transform(X_train)
        X_calib_n = scaler_x.transform(X_calib)
        X_test_n = scaler_x.transform(X_test)

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

        # Shared predictions
        Y_hat_test = forecaster.predict(X_test_n)
        residuals_calib = forecaster.residuals(X_calib_n, Y_calib_n)

        log.info(f"  Residual norms: mean={np.linalg.norm(residuals_calib, axis=1).mean():.4f}")

        # Run all methods
        window_results = {}

        # 1. Neural OT (ours)
        log.info("  Running Neural OT ...")
        try:
            window_results["Neural OT (ours)"] = run_neural_ot(
                residuals_calib, Y_hat_test, Y_test_n,
                alpha, beta, seed, device, n_epochs=ot_epochs,
            )
        except Exception as e:
            log.error(f"  Neural OT failed: {e}")
            window_results["Neural OT (ours)"] = {"method": "Neural OT (ours)", "error": str(e)}

        # 2. Naive ball
        log.info("  Running Naive Ball ...")
        window_results["Naive Ball"] = run_naive_ball(
            residuals_calib, Y_hat_test, Y_test_n, alpha,
        )

        # 3. Coordinate-wise SPCI
        log.info("  Running Coord-wise SPCI ...")
        window_results["Coord-wise SPCI"] = run_coordinatewise(
            residuals_calib, Y_hat_test, Y_test_n, alpha,
        )

        # 4. MultiDimSPCI
        log.info("  Running MultiDimSPCI ...")
        window_results["MultiDimSPCI"] = run_multidimspci(
            residuals_calib, Y_hat_test, Y_test_n, alpha,
        )

        # 5. CopulaCPTS
        log.info("  Running CopulaCPTS ...")
        window_results["CopulaCPTS"] = run_copulacpts(
            residuals_calib, Y_hat_test, Y_test_n, alpha,
        )

        # 6. ACI
        log.info("  Running ACI ...")
        window_results["ACI"] = run_aci(
            residuals_calib, Y_hat_test, Y_test_n, alpha,
        )

        # 7. EnbPI
        log.info("  Running EnbPI ...")
        window_results["EnbPI"] = run_enbpi(
            X_train_n, Y_train_n, X_calib_n, Y_calib_n,
            Y_hat_test, Y_test_n,
            alpha, seed, forecaster_type, d,
        )

        all_window_results.append(window_results)

    # Aggregate across windows
    method_names = [
        "Neural OT (ours)", "Naive Ball", "Coord-wise SPCI",
        "MultiDimSPCI", "CopulaCPTS", "ACI", "EnbPI",
    ]

    aggregated = {}
    for method_name in method_names:
        metrics = {
            "coverage": [],
            "coverage_gap": [],
            "mean_log_volume": [],
            "mean_diameter": [],
            "mean_winkler_log": [],
        }
        for wr in all_window_results:
            if method_name in wr and "error" not in wr[method_name]:
                for key in metrics:
                    if key in wr[method_name]:
                        metrics[key].append(wr[method_name][key])

        if metrics["coverage"]:
            aggregated[method_name] = {
                k: {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                }
                for k, v in metrics.items()
                if v
            }
        else:
            aggregated[method_name] = {"error": "No valid windows"}

    elapsed = time.time() - t0

    # Print results table
    print("\n" + "=" * 90)
    print(f"  BENCHMARK: {dataset_name} | d={d} | alpha={alpha} | "
          f"forecaster={forecaster_type} | {n_windows} windows")
    print("=" * 90)
    print(f"{'Method':<22} {'Coverage':>10} {'Gap':>10} {'Log-Vol':>10} "
          f"{'Diameter':>10} {'Winkler':>10}")
    print("-" * 90)

    for method_name in method_names:
        if method_name in aggregated and "error" not in aggregated[method_name]:
            agg = aggregated[method_name]
            cov = agg.get("coverage", {})
            gap = agg.get("coverage_gap", {})
            vol = agg.get("mean_log_volume", {})
            dia = agg.get("mean_diameter", {})
            win = agg.get("mean_winkler_log", {})

            marker = " *" if method_name == "Neural OT (ours)" else ""
            print(
                f"{method_name:<22}"
                f" {cov.get('mean', 0):>8.3f}±{cov.get('std', 0):.3f}"
                f" {gap.get('mean', 0):>8.4f}"
                f" {vol.get('mean', 0):>9.2f}"
                f" {dia.get('mean', 0):>9.3f}"
                f" {win.get('mean', 0):>9.2f}"
                f"{marker}"
            )
        else:
            print(f"{method_name:<22}  {'FAILED':>8}")

    print("-" * 90)
    print(f"  Target coverage: {1 - alpha:.1%} | Elapsed: {elapsed:.1f}s")
    print(f"  * = our method")
    print("=" * 90)

    results = {
        "dataset": dataset_name,
        "d": d,
        "alpha": alpha,
        "beta": beta,
        "forecaster": forecaster_type,
        "n_windows": n_windows,
        "seed": seed,
        "aggregated": aggregated,
        "per_window": [
            {k: v for k, v in wr.items()} for wr in all_window_results
        ],
        "elapsed_seconds": elapsed,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Neural OT vs baselines")
    parser.add_argument("--dataset", type=str, default="exchange",
                        help="Dataset name or 'all'")
    parser.add_argument("--d", type=str, default="4,8",
                        help="Dimensions to test (comma-separated)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage level")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="Confidence for W1 bound")
    parser.add_argument("--n_windows", type=int, default=3,
                        help="Number of rolling windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forecaster", type=str, default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ot_epochs", type=int, default=300,
                        help="Training epochs for Neural OT")
    parser.add_argument("--output_dir", type=str, default="results/benchmark")
    args = parser.parse_args()

    available = list(DATASET_URLS.keys())
    datasets = available if args.dataset == "all" else [args.dataset]
    dims = [int(x) for x in args.d.split(",")]

    all_results = []

    for ds in datasets:
        for d in dims:
            log.info(f"\n{'='*60}")
            log.info(f"Running benchmark: {ds} d={d}")
            log.info(f"{'='*60}")

            try:
                result = run_benchmark(
                    dataset_name=ds,
                    d=d,
                    alpha=args.alpha,
                    beta=args.beta,
                    n_windows=args.n_windows,
                    seed=args.seed,
                    forecaster_type=args.forecaster,
                    device=args.device,
                    ot_epochs=args.ot_epochs,
                )
                all_results.append(result)

                # Save individual result
                os.makedirs(args.output_dir, exist_ok=True)
                path = os.path.join(args.output_dir, f"{ds}_d{d}.json")
                with open(path, "w") as f:
                    json.dump(result, f, indent=2)
                log.info(f"Saved to {path}")

            except Exception as e:
                log.error(f"Failed on {ds} d={d}: {e}", exc_info=True)

    # Save combined results
    if all_results:
        os.makedirs(args.output_dir, exist_ok=True)
        combined_path = os.path.join(args.output_dir, "benchmark_all.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
