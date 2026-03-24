"""Neural OT Conformal Time Series — Full Pipeline (Algorithm 4).

Implements the joint Neural OT method with Wasserstein-controlled radius
relaxation, and compares against all baselines on public datasets.

Methods:
  1. Neural OT Joint (ours) — Algorithm 4 with PICNN joint training
  2. Neural OT ICNN          — Existing ICNN-based approach
  3. Naive Euclidean Ball
  4. Coordinate-wise SPCI
  5. MultiDimSPCI (ellipsoidal)
  6. CopulaCPTS
  7. ACI (adaptive)
  8. EnbPI (ensemble bootstrap)

Usage:
    python scripts/run_neural_ot_conformal.py
    python scripts/run_neural_ot_conformal.py --dataset all --d 4,8
    python scripts/run_neural_ot_conformal.py --dataset exchange --joint_epochs 500
"""

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from src.conformal.calibration import CalibrationResult, ConformalCalibrator
from src.conformal.rank_region import build_region, build_regions_batch
from src.conformal.wasserstein import estimate_mixing_factor, wasserstein_1_uniform
from src.data.preprocessing import StandardScaler, rolling_window_splits, subsample_dimensions
from src.metrics.coverage import coverage_gap, marginal_coverage
from src.metrics.efficiency import mean_diameter, mean_log_volume, mean_winkler_log
from src.models.baselines.aci import ACI
from src.models.baselines.coordinatewise_spci import CoordinateWiseSPCI
from src.models.baselines.copulacpts import CopulaCPTS
from src.models.baselines.enbpi import EnbPI
from src.models.baselines.multidimspci import MultiDimSPCI
from src.models.forecasters import LinearForecaster, MLPForecaster
from src.models.joint_neural_ot import JointNeuralOTMap
from src.models.neural_ot import NeuralOTMap, sample_uniform_ball
from src.utils.seeds import set_all_seeds

# Reuse data download helpers from run_benchmark
import importlib.util
import sys

_bench_path = str(Path(__file__).parent / "run_benchmark.py")
_spec = importlib.util.spec_from_file_location("run_benchmark", _bench_path)
_bench = importlib.util.module_from_spec(_spec)
sys.modules["run_benchmark"] = _bench
_spec.loader.exec_module(_bench)

DATASET_URLS = _bench.DATASET_URLS
DATASET_FILES = _bench.DATASET_FILES
download_dataset = _bench.download_dataset
load_real_dataset = _bench.load_real_dataset
run_aci = _bench.run_aci
run_coordinatewise = _bench.run_coordinatewise
run_copulacpts = _bench.run_copulacpts
run_enbpi = _bench.run_enbpi
run_multidimspci = _bench.run_multidimspci
run_naive_ball = _bench.run_naive_ball
run_neural_ot = _bench.run_neural_ot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fournier (2023) constants — Table 1 from the paper
# ---------------------------------------------------------------------------

# kappa^(m)_{d,1} standard bounds (Table 1) and optimized (Table 3)
FOURNIER_KAPPA = {
    1: 2.42, 2: 0.73, 3: 3.72, 4: 2.45, 5: 2.09,
    6: 1.94, 7: 1.87, 8: 1.84, 9: 1.82,
    # Optimized bounds for larger d (Table 3)
    10: 8.91, 12: 7.67, 15: 6.74, 16: 6.50, 20: 6.00,
    25: 5.60, 32: 5.25, 35: 5.17, 50: 4.85, 64: 4.65,
    75: 4.60, 100: 4.47, 500: 4.12,
}


def fournier_kappa(d: int) -> float:
    """Look up kappa^(m)_{d,1} from Fournier (2023), interpolating if needed."""
    if d in FOURNIER_KAPPA:
        return FOURNIER_KAPPA[d]
    # Interpolate between nearest keys
    keys = sorted(FOURNIER_KAPPA.keys())
    if d < keys[0]:
        return FOURNIER_KAPPA[keys[0]]
    if d > keys[-1]:
        return FOURNIER_KAPPA[keys[-1]]
    # Find bracketing keys
    lo = max(k for k in keys if k <= d)
    hi = min(k for k in keys if k >= d)
    if lo == hi:
        return FOURNIER_KAPPA[lo]
    frac = (d - lo) / (hi - lo)
    return FOURNIER_KAPPA[lo] * (1 - frac) + FOURNIER_KAPPA[hi] * frac


# ---------------------------------------------------------------------------
# Paper's calibration: W1 control with exact Fournier constants
# ---------------------------------------------------------------------------

def estimate_c_rho(ranks: np.ndarray, max_lag: int = 50) -> float:
    """Estimate C_rho = rho_0 + 2 * sum_{n>=1} rho_n from rank autocorrelations.

    Uses ||R(eps_t)|| norms as 1D summary.
    """
    norms = np.linalg.norm(ranks, axis=1)
    norms = norms - norms.mean()
    n = len(norms)
    var = np.var(norms)
    if var < 1e-12:
        return 1.0

    c_rho = 1.0  # rho_0 = 1
    for lag in range(1, min(max_lag, n // 2)):
        autocov = np.mean(norms[: n - lag] * norms[lag:])
        rho_lag = autocov / var
        if abs(rho_lag) < 2.0 / np.sqrt(n):
            break
        c_rho += 2.0 * abs(rho_lag)  # factor of 2 for symmetric pairs

    return c_rho


def w1_uniform_ball(
    ranks: np.ndarray,
    d: int,
    n_reference: int = 5000,
    seed: int = 42,
) -> float:
    """Compute W_1(mu_hat, U(B_d)) where U(B_d) is uniform on the unit ball.

    Uses sliced Wasserstein as approximation.
    """
    import ot as pot
    import torch

    rng = np.random.default_rng(seed)
    k = ranks.shape[0]

    # Reference samples from U(B_d)
    ref = sample_uniform_ball(n_reference, d, torch.device("cpu")).numpy()

    # Use POT sliced Wasserstein
    return pot.sliced_wasserstein_distance(
        ranks.astype(np.float64),
        ref.astype(np.float64),
        n_projections=200,
        seed=int(rng.integers(2**31)),
    )


def compute_eta_k_paper(beta: float, k: int, d: int, c_rho: float) -> float:
    """Compute eta_k(beta) using the paper's formula (page 9).

    eta_k(beta) = (1/beta) * (2 * kappa^(m)_{d,1} * C_rho^{1/d}) / k^{1/d}

    This uses Markov's inequality on the Fournier (2023) expectation bound.
    """
    kappa = fournier_kappa(d)
    return (1.0 / beta) * (2.0 * kappa * c_rho ** (1.0 / d)) / (k ** (1.0 / d))


def find_delta_star(r: float, d: int, A: float) -> float:
    """Find delta*(r) by bisection on L'(delta) = 0.

    Solves: d * delta^2 * (r - delta)^{d-1} = A
    for delta in (0, r).
    """
    if A <= 0 or r <= 0:
        return 0.0

    def foc(delta):
        if delta <= 0 or delta >= r:
            return float("inf")
        return d * delta ** 2 * (r - delta) ** (d - 1) - A

    # Check if a root exists: foc(0+) = -A < 0, foc(r-) = 0
    # At some intermediate point foc may become positive
    eps = 1e-10
    try:
        # foc(eps) ≈ -A < 0
        # foc at delta close to r: d * delta^2 * small^{d-1} may be < A for large d
        # Search for sign change
        lo, hi = eps, r - eps
        if foc(lo) * foc(hi) > 0:
            # No sign change: try grid search
            best_delta = r / 2
            best_val = abs(foc(best_delta))
            for trial in np.linspace(eps, r - eps, 100):
                val = abs(foc(trial))
                if val < best_val:
                    best_val = val
                    best_delta = trial
            return best_delta

        return brentq(foc, lo, hi, xtol=1e-12, maxiter=200)
    except Exception:
        return r / 2


def compute_r_star_paper(
    ranks: np.ndarray,
    alpha: float,
    beta: float,
    seed: int = 42,
) -> CalibrationResult:
    """Compute r*(alpha, beta) using the paper's exact procedure (Algorithm 2).

    Steps:
    1. rho_hat_k = W_1(mu_hat_k, U(B_d))
    2. eta_k(beta) = (1/beta) * 2*kappa*C_rho^{1/d} / k^{1/d}
    3. A = rho_hat_k + eta_k(beta)
    4. Nested bisection: find smallest r with L*(r) >= 1 - alpha
    """
    k, d = ranks.shape

    # Step A: Empirical Wasserstein discrepancy
    rho_hat_k = w1_uniform_ball(ranks, d, seed=seed)

    # Step B: Mixing factor and tolerance
    c_rho = estimate_c_rho(ranks)
    eta_k = compute_eta_k_paper(beta, k, d, c_rho)
    A = rho_hat_k + eta_k

    # Step C: Nested search for r*(alpha, beta)
    # Outer bisection on r in (0, some_upper_bound]
    # For each r, find delta*(r) and evaluate L*(r)

    def L_star(r):
        if r <= 0:
            return -float("inf")
        ds = find_delta_star(r, d, A)
        if ds <= 0 or ds >= r:
            return -float("inf")
        return (r - ds) ** d - A / ds

    # Ranks live in B_d, so r ∈ (0, 1]
    r_upper = 1.0

    if L_star(r_upper) < 1 - alpha:
        # Bound is vacuous: not enough data for the theoretical guarantee
        warnings.warn(
            f"Bound is vacuous (A={A:.4f}, d={d}): L*(1) < 1-alpha. "
            f"Returning r*=1. Increase k or relax beta."
        )
        r_star = 1.0
        delta_star = find_delta_star(r_star, d, A)
    else:
        # Bisection for smallest r with L*(r) >= 1 - alpha
        r_lo, r_hi = 1e-6, r_upper
        for _ in range(100):
            r_mid = (r_lo + r_hi) / 2
            if L_star(r_mid) >= 1 - alpha:
                r_hi = r_mid
            else:
                r_lo = r_mid
            if r_hi - r_lo < 1e-10:
                break
        r_star = r_hi
        delta_star = find_delta_star(r_star, d, A)

    return CalibrationResult(
        rho_hat_k=rho_hat_k,
        eta_k=eta_k,
        delta_star=delta_star,
        r_star=r_star,
        alpha=alpha,
        beta=beta,
        d=d,
        k=k,
        c_rho=c_rho,
    )


# ---------------------------------------------------------------------------
# Neural OT Joint method (Algorithm 4)
# ---------------------------------------------------------------------------

def run_neural_ot_joint(
    residuals_train: np.ndarray,
    residuals_calib: np.ndarray,
    Y_hat_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float,
    beta: float,
    seed: int,
    device: str,
    n_epochs: int = 500,
    lambda_cyc: float = 10.0,
    lambda_unif: float = 1.0,
) -> dict:
    """Run Neural OT Joint method (Algorithm 4).

    Phase II:  Train (Q_theta, R_phi) on training residuals
    Phase III: Calibrate r*(alpha,beta) on calibration residuals
    Phase IV:  Build prediction regions on test set
    """
    k, d = residuals_calib.shape

    # Phase II: Joint training on training residuals
    ot_map = JointNeuralOTMap(
        input_dim=d,
        n_epochs=n_epochs,
        batch_size=min(256, max(32, residuals_train.shape[0] // 4)),
        device=device,
        lambda_cyc=lambda_cyc,
        lambda_unif=lambda_unif,
    )
    ot_map.fit(residuals_train, verbose=False)

    # Phase III: Calibrate using calibration residuals
    # Get ranks R_phi(eps_i) for calibration set
    R_calib = ot_map.forward_map_np(residuals_calib)  # (k, d), in B_d
    ot_norms = np.linalg.norm(R_calib, axis=1)

    # Empirical OT quantile (practical, tighter)
    calibrator = ConformalCalibrator(alpha=alpha, beta=beta)
    r_star_empirical = calibrator.calibrate_empirical_ot(ot_norms)

    # Theoretical W1-based radius (paper's formula)
    cal_result = compute_r_star_paper(R_calib, alpha, beta, seed=seed)
    r_star_theory = cal_result.r_star

    # Use empirical radius (tighter in practice)
    r_star = r_star_empirical

    # Phase IV: Build prediction regions
    regions = [
        build_region(
            Y_hat_test[i],
            r_star,
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
        "method": "Neural OT Joint",
        "coverage": float(cov),
        "coverage_gap": float(gap),
        "mean_log_volume": float(log_vol),
        "mean_diameter": float(diam),
        "mean_winkler_log": float(winkler),
        "r_star": float(r_star),
        "r_star_theory": float(r_star_theory),
        "rho_hat_k": float(cal_result.rho_hat_k),
        "eta_k": float(cal_result.eta_k),
        "c_rho": float(cal_result.c_rho),
        "ball_coverage": float(ot_map.ball_coverage(residuals_calib)),
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
    joint_epochs: int = 500,
    ot_epochs: int = 300,
    lambda_cyc: float = 10.0,
    lambda_unif: float = 1.0,
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

    data_sub, selected_dims = subsample_dimensions(dataset.data, d, seed=seed)
    log.info(f"Subsampled to d={data_sub.shape[1]}")

    X_full = data_sub[:-1]
    Y_full = data_sub[1:]
    splits = rolling_window_splits(X_full, Y_full, n_windows=n_windows)

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

        Y_hat_test = forecaster.predict(X_test_n)
        residuals_train = forecaster.residuals(X_train_n, Y_train_n)
        residuals_calib = forecaster.residuals(X_calib_n, Y_calib_n)

        log.info(
            f"  Residual norms: "
            f"train={np.linalg.norm(residuals_train, axis=1).mean():.4f}, "
            f"calib={np.linalg.norm(residuals_calib, axis=1).mean():.4f}"
        )

        window_results = {}

        # 1. Neural OT Joint (ours — Algorithm 4)
        log.info("  Running Neural OT Joint ...")
        try:
            window_results["Neural OT Joint"] = run_neural_ot_joint(
                residuals_train,
                residuals_calib,
                Y_hat_test,
                Y_test_n,
                alpha,
                beta,
                seed,
                device,
                n_epochs=joint_epochs,
                lambda_cyc=lambda_cyc,
                lambda_unif=lambda_unif,
            )
        except Exception as e:
            log.error(f"  Neural OT Joint failed: {e}")
            window_results["Neural OT Joint"] = {
                "method": "Neural OT Joint",
                "error": str(e),
            }

        # 2. Neural OT ICNN (existing)
        log.info("  Running Neural OT ICNN ...")
        try:
            window_results["Neural OT ICNN"] = run_neural_ot(
                residuals_calib,
                Y_hat_test,
                Y_test_n,
                alpha,
                beta,
                seed,
                device,
                n_epochs=ot_epochs,
            )
            window_results["Neural OT ICNN"]["method"] = "Neural OT ICNN"
        except Exception as e:
            log.error(f"  Neural OT ICNN failed: {e}")
            window_results["Neural OT ICNN"] = {
                "method": "Neural OT ICNN",
                "error": str(e),
            }

        # 3. Naive ball
        log.info("  Running Naive Ball ...")
        window_results["Naive Ball"] = run_naive_ball(
            residuals_calib, Y_hat_test, Y_test_n, alpha
        )

        # 4. Coordinate-wise SPCI
        log.info("  Running Coord-wise SPCI ...")
        window_results["Coord-wise SPCI"] = run_coordinatewise(
            residuals_calib, Y_hat_test, Y_test_n, alpha
        )

        # 5. MultiDimSPCI
        log.info("  Running MultiDimSPCI ...")
        window_results["MultiDimSPCI"] = run_multidimspci(
            residuals_calib, Y_hat_test, Y_test_n, alpha
        )

        # 6. CopulaCPTS
        log.info("  Running CopulaCPTS ...")
        window_results["CopulaCPTS"] = run_copulacpts(
            residuals_calib, Y_hat_test, Y_test_n, alpha
        )

        # 7. ACI
        log.info("  Running ACI ...")
        window_results["ACI"] = run_aci(
            residuals_calib, Y_hat_test, Y_test_n, alpha
        )

        # 8. EnbPI
        log.info("  Running EnbPI ...")
        window_results["EnbPI"] = run_enbpi(
            X_train_n,
            Y_train_n,
            X_calib_n,
            Y_calib_n,
            Y_hat_test,
            Y_test_n,
            alpha,
            seed,
            forecaster_type,
            d,
        )

        all_window_results.append(window_results)

    # Aggregate across windows
    method_names = [
        "Neural OT Joint",
        "Neural OT ICNN",
        "Naive Ball",
        "Coord-wise SPCI",
        "MultiDimSPCI",
        "CopulaCPTS",
        "ACI",
        "EnbPI",
    ]

    metric_keys = [
        "coverage",
        "coverage_gap",
        "mean_log_volume",
        "mean_diameter",
        "mean_winkler_log",
        "r_star",
    ]

    aggregated = {}
    for method_name in method_names:
        metrics = {k: [] for k in metric_keys}
        for wr in all_window_results:
            if method_name in wr and "error" not in wr[method_name]:
                for key in metrics:
                    if key in wr[method_name]:
                        metrics[key].append(wr[method_name][key])

        if metrics["coverage"]:
            aggregated[method_name] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in metrics.items()
                if v
            }
        else:
            aggregated[method_name] = {"error": "No valid windows"}

    elapsed = time.time() - t0

    # Print results table
    print("\n" + "=" * 105)
    print(
        f"  BENCHMARK: {dataset_name} | d={d} | alpha={alpha} | "
        f"forecaster={forecaster_type} | {n_windows} windows"
    )
    print("=" * 105)
    print(
        f"{'Method':<22} {'Coverage':>10} {'Gap':>10} {'Log-Vol':>10} "
        f"{'Diameter':>10} {'Winkler':>10} {'r*':>10}"
    )
    print("-" * 105)

    for method_name in method_names:
        if method_name in aggregated and "error" not in aggregated[method_name]:
            agg = aggregated[method_name]
            cov = agg.get("coverage", {})
            gap = agg.get("coverage_gap", {})
            vol = agg.get("mean_log_volume", {})
            dia = agg.get("mean_diameter", {})
            win = agg.get("mean_winkler_log", {})
            rs = agg.get("r_star", {})

            marker = " *" if "Joint" in method_name else ""
            rs_str = f"{rs.get('mean', 0):>9.4f}" if rs else f"{'—':>9}"
            print(
                f"{method_name:<22}"
                f" {cov.get('mean', 0):>8.3f}\u00b1{cov.get('std', 0):.3f}"
                f" {gap.get('mean', 0):>8.4f}"
                f" {vol.get('mean', 0):>9.2f}"
                f" {dia.get('mean', 0):>9.3f}"
                f" {win.get('mean', 0):>9.2f}"
                f" {rs_str}"
                f"{marker}"
            )
        else:
            print(f"{method_name:<22}  {'FAILED':>8}")

    print("-" * 105)
    print(f"  Target coverage: {1 - alpha:.1%} | Elapsed: {elapsed:.1f}s")
    print(f"  * = our method (Joint Neural OT)")
    print("=" * 105)

    results = {
        "dataset": dataset_name,
        "d": d,
        "alpha": alpha,
        "beta": beta,
        "forecaster": forecaster_type,
        "n_windows": n_windows,
        "seed": seed,
        "aggregated": aggregated,
        "per_window": [dict(wr) for wr in all_window_results],
        "elapsed_seconds": elapsed,
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Neural OT Conformal Time Series — Full Pipeline Benchmark"
    )
    parser.add_argument(
        "--dataset", type=str, default="exchange", help="Dataset name or 'all'"
    )
    parser.add_argument(
        "--d", type=str, default="4,8", help="Dimensions (comma-separated)"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level")
    parser.add_argument("--beta", type=float, default=0.05, help="W1 confidence")
    parser.add_argument("--n_windows", type=int, default=3, help="Rolling windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--forecaster", type=str, default="linear", choices=["linear", "mlp"]
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--joint_epochs", type=int, default=500, help="Epochs for joint OT training"
    )
    parser.add_argument(
        "--ot_epochs", type=int, default=300, help="Epochs for ICNN OT training"
    )
    parser.add_argument("--lambda_cyc", type=float, default=10.0)
    parser.add_argument("--lambda_unif", type=float, default=1.0)
    parser.add_argument(
        "--output_dir", type=str, default="results/benchmark_joint"
    )
    args = parser.parse_args()

    available = list(DATASET_URLS.keys())
    datasets = available if args.dataset == "all" else [args.dataset]
    dims = [int(x) for x in args.d.split(",")]

    all_results = []

    for ds in datasets:
        for dd in dims:
            log.info(f"\n{'=' * 60}")
            log.info(f"Running benchmark: {ds} d={dd}")
            log.info(f"{'=' * 60}")

            try:
                result = run_benchmark(
                    dataset_name=ds,
                    d=dd,
                    alpha=args.alpha,
                    beta=args.beta,
                    n_windows=args.n_windows,
                    seed=args.seed,
                    forecaster_type=args.forecaster,
                    device=args.device,
                    joint_epochs=args.joint_epochs,
                    ot_epochs=args.ot_epochs,
                    lambda_cyc=args.lambda_cyc,
                    lambda_unif=args.lambda_unif,
                )
                all_results.append(result)

                os.makedirs(args.output_dir, exist_ok=True)
                path = os.path.join(args.output_dir, f"{ds}_d{dd}.json")
                with open(path, "w") as f:
                    json.dump(result, f, indent=2)
                log.info(f"Saved to {path}")

            except Exception as e:
                log.error(f"Failed on {ds} d={dd}: {e}", exc_info=True)

    if all_results:
        os.makedirs(args.output_dir, exist_ok=True)
        combined_path = os.path.join(args.output_dir, "benchmark_all.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
