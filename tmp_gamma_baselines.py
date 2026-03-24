"""Gamma interpolation with baseline comparisons.

r(gamma) = (1-gamma)*(1-alpha)^{1/d} + gamma*r*_uncapped

Compares Neural OT (ours) at different gamma values against:
  - Naive Euclidean Ball
  - Coordinate-wise SPCI (Bonferroni)
  - MultiDimSPCI (Mahalanobis ellipsoid)
  - CopulaCPTS (copula hyperrectangle)

All volumes estimated via MC hit-or-miss in Y-space.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from src.conformal.calibration import ConformalCalibrator
from src.data.loaders import load_dataset
from src.data.preprocessing import (
    StandardScaler,
    subsample_dimensions,
    temporal_train_calib_test_split,
)
from src.models.baselines.coordinatewise_spci import CoordinateWiseSPCI
from src.models.baselines.copulacpts import CopulaCPTS
from src.models.baselines.multidimspci import MultiDimSPCI
from src.models.forecasters import LinearForecaster
from src.models.neural_ot import NeuralOTMap

seed = 42
alpha = 0.1
beta = 0.1
gammas = [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1]
N_MC = 20000
N_CENTERS = 20


# ---------------------------------------------------------------------------
# MC volume estimation
# ---------------------------------------------------------------------------

def mc_volume_y_space(containment_fn, center, d, residuals_calib, n_mc=N_MC, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    res_min = residuals_calib.min(axis=0)
    res_max = residuals_calib.max(axis=0)
    margin = (res_max - res_min) * 0.3
    box_lo = center + res_min - margin
    box_hi = center + res_max + margin

    y_samples = rng.uniform(box_lo, box_hi, size=(n_mc, d)).astype(np.float64)
    residuals_mc = y_samples - center[np.newaxis, :]

    inside = containment_fn(residuals_mc)
    frac_in = np.mean(inside)
    box_volume = np.prod(box_hi - box_lo)
    if verbose:
        print(f"    [MC baseline] box_lo={box_lo[:3]}... box_hi={box_hi[:3]}... "
              f"box_vol={box_volume:.6f} frac_in={frac_in:.4f} n_mc={n_mc} vol={frac_in*box_volume:.6f}")
    return frac_in * box_volume


def mean_mc_volume(containment_fn, Y_hat_test, d, residuals_calib, n_mc=N_MC, verbose=False):
    n_centers = min(N_CENTERS, Y_hat_test.shape[0])
    rng_idx = np.random.default_rng(seed)
    idx = rng_idx.choice(Y_hat_test.shape[0], n_centers, replace=False)
    vols = []
    for i in idx:
        v = mc_volume_y_space(containment_fn, Y_hat_test[i], d, residuals_calib, n_mc=n_mc, seed=i,
                              verbose=(verbose and len(vols) == 0))  # verbose only for 1st center
        vols.append(v)
    if verbose:
        print(f"    [MC baseline] {len(vols)} centers, vols: min={min(vols):.6f} max={max(vols):.6f} mean={np.mean(vols):.6f}")
    return np.mean(vols)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_ot_radius(r, ot_map, Y_hat_test, Y_test, d, residuals_calib, verbose=False):
    residuals_test = Y_test - Y_hat_test
    ot_ranks_test = ot_map.forward_map_np(residuals_test.astype(np.float32))
    norms_test = np.linalg.norm(ot_ranks_test, axis=1)
    coverage = np.mean(norms_test <= r)

    def containment_fn(res):
        ranks = ot_map.forward_map_np(res.astype(np.float32))
        return np.linalg.norm(ranks, axis=1) <= r

    vol = mean_mc_volume(containment_fn, Y_hat_test, d, residuals_calib, verbose=verbose)
    return coverage, vol


def eval_naive_ball(radius, Y_hat_test, Y_test, d, residuals_calib, verbose=False):
    residuals_test = Y_test - Y_hat_test
    norms = np.linalg.norm(residuals_test, axis=1)
    coverage = np.mean(norms <= radius)

    def containment_fn(res):
        return np.linalg.norm(res, axis=1) <= radius

    vol = mean_mc_volume(containment_fn, Y_hat_test, d, residuals_calib, verbose=verbose)
    return coverage, vol


def eval_coordwise(half_widths, Y_hat_test, Y_test, d, residuals_calib):
    residuals_test = Y_test - Y_hat_test
    coverage = np.mean(np.all(np.abs(residuals_test) <= half_widths, axis=1))

    def containment_fn(res):
        return np.all(np.abs(res) <= half_widths, axis=1)

    vol = mean_mc_volume(containment_fn, Y_hat_test, d, residuals_calib)
    return coverage, vol


def eval_mahalanobis(Sigma_inv, quantile, Y_hat_test, Y_test, d, residuals_calib):
    residuals_test = Y_test - Y_hat_test
    scores = np.sqrt(np.einsum('ij,jk,ik->i', residuals_test, Sigma_inv, residuals_test))
    coverage = np.mean(scores <= quantile)

    def containment_fn(res):
        s = np.sqrt(np.einsum('ij,jk,ik->i', res, Sigma_inv, res))
        return s <= quantile

    vol = mean_mc_volume(containment_fn, Y_hat_test, d, residuals_calib)
    return coverage, vol


def eval_copula_rect(marginal_quantiles, Y_hat_test, Y_test, d, residuals_calib):
    residuals_test = Y_test - Y_hat_test
    coverage = np.mean(np.all(np.abs(residuals_test) <= marginal_quantiles, axis=1))

    def containment_fn(res):
        return np.all(np.abs(res) <= marginal_quantiles, axis=1)

    vol = mean_mc_volume(containment_fn, Y_hat_test, d, residuals_calib)
    return coverage, vol


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for dataset_name, d_list in [("exchange", [2, 4]), ("electricity", [2, 4])]:
    print(f"\n{'='*75}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*75}")

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

        res_calib_f32 = residuals_calib.astype(np.float32)

        # ===== 1. Train PICNN (Neural OT) =====
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

        ot_ranks_calib = ot_map.forward_map_np(res_calib_f32)
        norms_calib = np.linalg.norm(ot_ranks_calib, axis=1)
        print(f"  OT calib norms: mean={norms_calib.mean():.3f} max={norms_calib.max():.3f} in_ball={np.mean(norms_calib<=1):.0%}")

        cal = ConformalCalibrator(alpha=alpha, beta=beta, w1_method="sliced")
        result_ot = cal.calibrate(residuals_calib, seed=seed, ot_ranks=ot_ranks_calib)
        r_uncapped = result_ot.r_star_uncapped
        r_naive_theory = (1.0 - alpha) ** (1.0 / d)
        r_empirical = cal.calibrate_empirical_ot(norms_calib)

        # ===== 2. Baselines: calibrate =====
        naive_radius = cal.calibrate_naive(residuals_calib)

        cw = CoordinateWiseSPCI(alpha=alpha)
        cw.calibrate(residuals_calib)

        mdspci = MultiDimSPCI(alpha=alpha)
        mdspci.calibrate(residuals_calib)

        copula = CopulaCPTS(alpha=alpha)
        copula.calibrate(residuals_calib)

        # ===== 3. Print reference radii =====
        print(f"\n  r_naive_theory = (1-alpha)^(1/d) = {r_naive_theory:.4f}")
        print(f"  r*_uncapped (W1 bound)          = {r_uncapped:.4f}")
        print(f"  r*_empirical (OT quantile)      = {r_empirical:.4f}")
        print(f"  r_naive (Euclid quantile)       = {naive_radius:.4f}")
        print(f"  Target coverage = {1-alpha:.0%}")

        # ===== 4. Evaluate all methods =====
        hdr = f"  {'method':>25s} | {'radius/param':>12s} | {'coverage':>9s} | {'vol_MC':>12s} | {'status'}"
        sep = f"  {'-'*25} | {'-'*12} | {'-'*9} | {'-'*12} | {'-'*15}"
        print(f"\n{hdr}\n{sep}")

        # --- Baselines ---
        cov_n, vol_n = eval_naive_ball(naive_radius, Y_hat_test_n, Y_test_n, d, residuals_calib, verbose=True)
        st_n = "OK" if cov_n >= 1 - alpha else "UNDER"
        print(f"  {'Naive Ball':>25s} | {naive_radius:>12.4f} | {cov_n:>9.1%} | {vol_n:>12.6f} | {st_n}")

        cov_cw, vol_cw = eval_coordwise(cw._half_widths, Y_hat_test_n, Y_test_n, d, residuals_calib)
        st_cw = "OK" if cov_cw >= 1 - alpha else "UNDER"
        print(f"  {'Coord-wise SPCI':>25s} | {'rect':>12s} | {cov_cw:>9.1%} | {vol_cw:>12.6f} | {st_cw}")

        cov_md, vol_md = eval_mahalanobis(mdspci._Sigma_inv, mdspci._quantile, Y_hat_test_n, Y_test_n, d, residuals_calib)
        st_md = "OK" if cov_md >= 1 - alpha else "UNDER"
        print(f"  {'MultiDimSPCI':>25s} | {mdspci._quantile:>12.4f} | {cov_md:>9.1%} | {vol_md:>12.6f} | {st_md}")

        cov_cp, vol_cp = eval_copula_rect(copula._marginal_quantiles, Y_hat_test_n, Y_test_n, d, residuals_calib)
        st_cp = "OK" if cov_cp >= 1 - alpha else "UNDER"
        print(f"  {'CopulaCPTS':>25s} | {'rect':>12s} | {cov_cp:>9.1%} | {vol_cp:>12.6f} | {st_cp}")

        print(sep)

        # --- Neural OT: empirical radius ---
        cov_emp, vol_emp = eval_ot_radius(r_empirical, ot_map, Y_hat_test_n, Y_test_n, d, residuals_calib, verbose=True)
        st_emp = "OK" if cov_emp >= 1 - alpha else "UNDER"
        print(f"  {'OT empirical':>25s} | {r_empirical:>12.4f} | {cov_emp:>9.1%} | {vol_emp:>12.6f} | {st_emp}")

        # --- Neural OT: gamma interpolation ---
        for gamma in gammas:
            r_gamma = (1.0 - gamma) * r_naive_theory + gamma * r_uncapped
            cov_g, vol_g = eval_ot_radius(r_gamma, ot_map, Y_hat_test_n, Y_test_n, d, residuals_calib)
            st_g = "OK" if cov_g >= 1 - alpha else "UNDER"
            label = f"OT gamma={gamma:.2f}"
            print(f"  {label:>25s} | {r_gamma:>12.4f} | {cov_g:>9.1%} | {vol_g:>12.6f} | {st_g}")

        print()