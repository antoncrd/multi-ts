"""Gamma interpolation: r(gamma) = (1-gamma)*(1-alpha)^{1/d} + gamma*r*_uncapped.

Evaluates coverage and region volume on held-out test data for different gamma.
Uses real datasets with proper train/calib/test split.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from src.conformal.calibration import ConformalCalibrator
from src.conformal.rank_region import ConformalRegion, unit_ball_volume
from src.data.loaders import load_dataset
from src.data.preprocessing import (
    StandardScaler,
    subsample_dimensions,
    temporal_train_calib_test_split,
)
from src.models.forecasters import LinearForecaster
from src.models.neural_ot import NeuralOTMap

seed = 42
alpha = 0.1
beta = 0.1
gammas = [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1]


def mc_volume_y_space(r, ot_map, center, d, residuals_calib, n_mc=20000, seed=0):
    """Monte Carlo hit-or-miss volume of {y : ||Q_hat(y - center)|| <= r} in Y-space.

    Samples uniformly in a bounding box derived from calibration residuals,
    then counts what fraction falls inside the OT-ball.
    Volume = fraction_in * volume_of_box.
    """
    rng = np.random.default_rng(seed)

    # Bounding box: use calibration residuals to estimate a sensible range
    # Expand by a safety margin so covered region fits inside
    res_min = residuals_calib.min(axis=0)
    res_max = residuals_calib.max(axis=0)
    margin = (res_max - res_min) * 0.3
    box_lo = center + res_min - margin
    box_hi = center + res_max + margin

    # Sample uniformly in the bounding box
    y_samples = rng.uniform(box_lo, box_hi, size=(n_mc, d)).astype(np.float32)
    residuals_mc = y_samples - center[np.newaxis, :]

    # Map through Q_hat and check containment
    ranks_mc = ot_map.forward_map_np(residuals_mc)
    norms_mc = np.linalg.norm(ranks_mc, axis=1)
    frac_in = np.mean(norms_mc <= r)

    box_volume = np.prod(box_hi - box_lo)
    vol = frac_in * box_volume
    return vol


def evaluate_radius(r, ot_map, Y_hat_test, Y_test, d, residuals_calib, n_mc=20000):
    """Compute coverage and mean MC volume for a given radius r in OT space."""
    # Coverage: check ||Q_hat(Y_test - Y_hat_test)|| <= r
    residuals_test = Y_test - Y_hat_test
    ot_ranks_test = ot_map.forward_map_np(residuals_test.astype(np.float32))
    norms_test = np.linalg.norm(ot_ranks_test, axis=1)
    coverage = np.mean(norms_test <= r)

    # MC volume: average over a few test centers for speed
    n_centers = min(20, Y_hat_test.shape[0])
    rng_idx = np.random.default_rng(seed)
    idx = rng_idx.choice(Y_hat_test.shape[0], n_centers, replace=False)
    vols = []
    for i in idx:
        v = mc_volume_y_space(r, ot_map, Y_hat_test[i], d, residuals_calib,
                              n_mc=n_mc, seed=i)
        vols.append(v)
    mean_vol = np.mean(vols)

    return coverage, mean_vol, norms_test


for dataset_name, d_list in [("exchange", [2, 4]), ("electricity", [2, 4])]:
    print(f"\n{'='*65}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*65}")

    try:
        dataset = load_dataset(dataset_name)
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")
        continue

    for d in d_list:
        print(f"\n--- {dataset_name} d={d} ---")

        data_sub, sel_dims = subsample_dimensions(dataset.data, d, seed=seed)
        X_full = data_sub[:-1]
        Y_full = data_sub[1:]

        # 3-way split: 40% train, 30% calib, 30% test
        split = temporal_train_calib_test_split(
            X_full, Y_full, train_frac=0.4, calib_frac=0.3
        )
        X_train, Y_train = split["train"]
        X_calib, Y_calib = split["calib"]
        X_test, Y_test = split["test"]

        scaler_x = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(Y_train)
        X_train_n = scaler_x.transform(X_train)
        X_calib_n = scaler_x.transform(X_calib)
        X_test_n = scaler_x.transform(X_test)
        Y_train_n = scaler_y.transform(Y_train)
        Y_calib_n = scaler_y.transform(Y_calib)
        Y_test_n = scaler_y.transform(Y_test)

        forecaster = LinearForecaster()
        forecaster.fit(X_train_n, Y_train_n)

        residuals_train = forecaster.residuals(X_train_n, Y_train_n)
        residuals_calib = forecaster.residuals(X_calib_n, Y_calib_n)
        Y_hat_test_n = forecaster.predict(X_test_n)

        n_train = residuals_train.shape[0]
        k_calib = residuals_calib.shape[0]
        n_test = X_test_n.shape[0]
        print(f"  n_train={n_train}, k_calib={k_calib}, n_test={n_test}")

        # Train PICNN on training residuals
        ot_map = NeuralOTMap(
            input_dim=d,
            hidden_dims_icnn=[64, 64],
            hidden_dims_inverse=[64, 64],
            context_hidden_dims=[64, 64],
            n_epochs=150,
            batch_size=min(512, n_train),
            n_projections=100,
            ball_penalty_weight=20.0,
            warmup_epochs=20,
            device="cpu",
        )
        ot_map.fit(residuals_train.astype(np.float32), verbose=True)

        # Get OT ranks on calibration set
        ot_ranks_calib = ot_map.forward_map_np(residuals_calib.astype(np.float32))
        norms_calib = np.linalg.norm(ot_ranks_calib, axis=1)
        print(f"  Calib OT norms: mean={norms_calib.mean():.3f} max={norms_calib.max():.3f} in_ball={np.mean(norms_calib<=1):.0%}")

        # Get r*_uncapped from calibration
        cal = ConformalCalibrator(alpha=alpha, beta=beta, w1_method="sliced")
        result_ot = cal.calibrate(residuals_calib, seed=seed, ot_ranks=ot_ranks_calib)
        r_uncapped = result_ot.r_star_uncapped

        # Naive radius: (1-alpha)^{1/d}
        r_naive = (1.0 - alpha) ** (1.0 / d)

        # Empirical OT radius: (1-alpha)-quantile of ||Q(calib residuals)||
        r_empirical = cal.calibrate_empirical_ot(norms_calib)

        print(f"\n  r_naive = (1-alpha)^(1/d) = {r_naive:.4f}")
        print(f"  r*_uncapped (theory)      = {r_uncapped:.4f}")
        print(f"  r*_empirical (OT quantile)= {r_empirical:.4f}")
        print(f"  Target coverage = {1-alpha:.0%}")

        # Evaluate for each gamma
        res_calib_np = residuals_calib.astype(np.float32)

        print(f"\n  {'gamma':>6s} | {'r(gamma)':>9s} | {'coverage':>9s} | {'vol_MC':>12s} | {'status'}")
        print(f"  {'-'*6} | {'-'*9} | {'-'*9} | {'-'*12} | {'-'*15}")

        for gamma in gammas:
            r_gamma = (1.0 - gamma) * r_naive + gamma * r_uncapped
            cov, vol, _ = evaluate_radius(r_gamma, ot_map, Y_hat_test_n, Y_test_n, d, res_calib_np)
            status = "OK" if cov >= 1 - alpha else "UNDER-COVERAGE"
            print(f"  {gamma:>6.2f} | {r_gamma:>9.4f} | {cov:>9.1%} | {vol:>12.6f} | {status}")

        # Also show empirical OT radius
        cov_emp, vol_emp, _ = evaluate_radius(r_empirical, ot_map, Y_hat_test_n, Y_test_n, d, res_calib_np)
        status_emp = "OK" if cov_emp >= 1 - alpha else "UNDER-COVERAGE"
        print(f"  {'emp':>6s} | {r_empirical:>9.4f} | {cov_emp:>9.1%} | {vol_emp:>12.6f} | {status_emp} (empirical OT)")
        print()
