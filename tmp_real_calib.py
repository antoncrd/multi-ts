"""Real data calibration diagnostic: PICNN vs marginal ranks (proper split)."""
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
from src.models.forecasters import LinearForecaster
from src.models.neural_ot import NeuralOTMap

seed = 42
alpha = 0.1
beta = 0.9

# ---- Exchange Rate (T=7588, D=8) ----
for dataset_name, d_list in [("exchange", [2]), ("electricity", [2])]:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    try:
        dataset = load_dataset(dataset_name)
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")
        continue

    for d in d_list:
        print(f"\n--- {dataset_name} d={d} ---")

        # Subsample dimensions
        data_sub, sel_dims = subsample_dimensions(dataset.data, d, seed=seed)

        # X[t] = data[t], Y[t] = data[t+1]
        X_full = data_sub[:-1]
        Y_full = data_sub[1:]

        # Temporal split: 60% train, 20% calib, 20% test
        split = temporal_train_calib_test_split(X_full, Y_full, train_frac=0.4, calib_frac=0.4)
        X_train, Y_train = split["train"]
        X_calib, Y_calib = split["calib"]

        # Normalize
        scaler_x = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(Y_train)
        X_train_n = scaler_x.transform(X_train)
        X_calib_n = scaler_x.transform(X_calib)
        Y_train_n = scaler_y.transform(Y_train)
        Y_calib_n = scaler_y.transform(Y_calib)

        # Fit forecaster on train
        forecaster = LinearForecaster()
        forecaster.fit(X_train_n, Y_train_n)

        # Get residuals
        residuals_train = forecaster.residuals(X_train_n, Y_train_n)
        residuals_calib = forecaster.residuals(X_calib_n, Y_calib_n)
        k_train = residuals_train.shape[0]
        k_calib = residuals_calib.shape[0]

        print(f"  n_train={k_train}, k_calib={k_calib}")

        # Train PICNN on training residuals
        ot_map = NeuralOTMap(
            input_dim=d,
            hidden_dims_icnn=[64, 64],
            hidden_dims_inverse=[64, 64],
            context_hidden_dims=[64, 64],
            n_epochs=150,
            batch_size=min(512, k_train),
            n_projections=100,
            ball_penalty_weight=20.0,
            warmup_epochs=20,
            device="cpu",
        )
        ot_map.fit(residuals_train.astype(np.float32), verbose=True)

        # Apply to held-out calibration residuals
        ot_ranks_calib = ot_map.forward_map_np(residuals_calib.astype(np.float32))

        norms_calib = np.linalg.norm(ot_ranks_calib, axis=1)
        print(f"  Calib OT norms: mean={norms_calib.mean():.3f} max={norms_calib.max():.3f} in_ball={np.mean(norms_calib<=1):.0%}")

        # Calibrate with OT ranks
        cal = ConformalCalibrator(alpha=alpha, beta=beta, w1_method="sliced")
        result_ot = cal.calibrate(residuals_calib, seed=seed, ot_ranks=ot_ranks_calib)

        # Calibrate with marginal ranks
        result_m = cal.calibrate(residuals_calib, seed=seed)

        A_ot = result_ot.rho_hat_k + result_ot.eta_k
        A_m = result_m.rho_hat_k + result_m.eta_k

        print(f"  PICNN    rho={result_ot.rho_hat_k:.4f} eta={result_ot.eta_k:.4f} A={A_ot:.4f} r*={result_ot.r_star:.4f}")
        print(f"  Marginal rho={result_m.rho_hat_k:.4f} eta={result_m.eta_k:.4f} A={A_m:.4f} r*={result_m.r_star:.4f}")
