"""Preprocessing utilities: temporal splits, normalization, subsampling."""

from typing import Dict, Optional, Tuple

import numpy as np

from src.data.loaders import TimeSeriesDataset


def temporal_train_calib_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    train_frac: float = 0.6,
    calib_frac: float = 0.2,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split (X, Y) into train/calibration/test preserving temporal order.

    Args:
        X: shape (T, d).
        Y: shape (T, d).
        train_frac: Fraction for training.
        calib_frac: Fraction for calibration.

    Returns:
        Dict with 'train', 'calib', 'test' keys, each a (X, Y) tuple.
    """
    T = X.shape[0]
    n_train = int(T * train_frac)
    n_calib = int(T * calib_frac)

    return {
        "train": (X[:n_train], Y[:n_train]),
        "calib": (X[n_train:n_train + n_calib], Y[n_train:n_train + n_calib]),
        "test": (X[n_train + n_calib:], Y[n_train + n_calib:]),
    }


def rolling_window_splits(
    X: np.ndarray,
    Y: np.ndarray,
    n_windows: int = 5,
    train_frac: float = 0.6,
    calib_frac: float = 0.2,
) -> list:
    """Create n_windows rolling train/calib/test splits.

    Each window starts at a different offset, ensuring that the test
    sets cover different parts of the time series.

    Args:
        X: shape (T, d).
        Y: shape (T, d).
        n_windows: Number of rolling windows.
        train_frac: Fraction for training in each window.
        calib_frac: Fraction for calibration in each window.

    Returns:
        List of dicts, each with 'train', 'calib', 'test' keys.
    """
    T = X.shape[0]
    test_frac = 1.0 - train_frac - calib_frac
    window_size = int(T / (1 + (n_windows - 1) * test_frac))

    splits = []
    for w in range(n_windows):
        offset = int(w * window_size * test_frac)
        end = min(offset + window_size, T)

        X_window = X[offset:end]
        Y_window = Y[offset:end]

        split = temporal_train_calib_test_split(
            X_window, Y_window, train_frac=train_frac, calib_frac=calib_frac
        )
        splits.append(split)

    return splits


class StandardScaler:
    """Z-score normalization preserving temporal structure."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and std from training data.

        Args:
            X: shape (n, d).
        """
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # Avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted.")
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted.")
        return X * self.std_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def subsample_dimensions(
    data: np.ndarray,
    d_target: int,
    seed: int = 42,
    method: str = "random",
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample dimensions from a high-dimensional dataset.

    Args:
        data: shape (T, D) with D >= d_target.
        d_target: Target number of dimensions.
        seed: Random seed.
        method: "random" or "variance" (select highest-variance dimensions).

    Returns:
        Subsampled data of shape (T, d_target) and selected indices.
    """
    D = data.shape[1]
    if d_target >= D:
        return data, np.arange(D)

    if method == "random":
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(D, size=d_target, replace=False))
    elif method == "variance":
        variances = np.var(data, axis=0)
        selected = np.argsort(variances)[-d_target:]
        selected = np.sort(selected)
    else:
        raise ValueError(f"Unknown method: {method}")

    return data[:, selected], selected
