"""Data loaders for real-world multivariate time series datasets.

Supported datasets:
- Wind (NREL)
- Solar Energy
- Traffic (PeMS)
- Exchange Rate
- Electricity
- NYC Taxi
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class TimeSeriesDataset:
    """Container for a multivariate time series dataset."""

    def __init__(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        name: str = "unnamed",
    ):
        self.data = data            # shape (T, D)
        self.timestamps = timestamps
        self.feature_names = feature_names or [f"dim_{i}" for i in range(data.shape[1])]
        self.name = name
        self.T, self.D = data.shape

    def subsample_dims(self, d: int, seed: int = 42) -> "TimeSeriesDataset":
        """Subsample d dimensions from the full dataset.

        Args:
            d: Target dimension.
            seed: Seed for random dimension selection.

        Returns:
            New TimeSeriesDataset with d dimensions.
        """
        if d >= self.D:
            return self

        rng = np.random.default_rng(seed)
        selected = rng.choice(self.D, size=d, replace=False)
        selected = np.sort(selected)

        return TimeSeriesDataset(
            data=self.data[:, selected],
            timestamps=self.timestamps,
            feature_names=[self.feature_names[i] for i in selected],
            name=f"{self.name}_d{d}",
        )

    def to_xy(self, lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to (X, Y) pairs for forecasting.

        X[t] = data[t], Y[t] = data[t + lag].

        Args:
            lag: Forecast horizon.

        Returns:
            X: shape (T - lag, D), Y: shape (T - lag, D).
        """
        X = self.data[:-lag]
        Y = self.data[lag:]
        return X, Y


def load_dataset(
    name: str,
    data_dir: str = "data/raw",
    **kwargs,
) -> TimeSeriesDataset:
    """Load a dataset by name.

    Args:
        name: One of 'wind', 'solar', 'traffic', 'exchange', 'electricity', 'nyc_taxi'.
        data_dir: Directory containing raw data files.
        **kwargs: Additional arguments passed to the specific loader.

    Returns:
        TimeSeriesDataset instance.
    """
    loaders = {
        "wind": _load_wind,
        "solar": _load_solar,
        "traffic": _load_traffic,
        "exchange": _load_exchange,
        "electricity": _load_electricity,
        "nyc_taxi": _load_nyc_taxi,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    return loaders[name](data_dir, **kwargs)


def _load_csv_generic(
    filepath: str,
    name: str,
    date_col: Optional[str] = None,
    value_cols: Optional[list] = None,
) -> TimeSeriesDataset:
    """Generic CSV loader."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}. "
            f"Please download the dataset and place it in the data/raw/ directory."
        )

    df = pd.read_csv(filepath)

    timestamps = None
    if date_col and date_col in df.columns:
        timestamps = pd.to_datetime(df[date_col]).values
        df = df.drop(columns=[date_col])

    if value_cols:
        df = df[value_cols]

    # Drop non-numeric columns
    df = df.select_dtypes(include=[np.number])

    data = df.values.astype(np.float64)

    # Handle NaN
    if np.any(np.isnan(data)):
        # Forward fill then backward fill
        df_filled = pd.DataFrame(data).ffill().bfill()
        data = df_filled.values

    return TimeSeriesDataset(
        data=data,
        timestamps=timestamps,
        feature_names=list(df.columns),
        name=name,
    )


def _load_wind(data_dir: str, **kwargs) -> TimeSeriesDataset:
    """Load NREL Wind dataset."""
    return _load_csv_generic(
        f"{data_dir}/wind.csv", name="wind", date_col="date"
    )


def _load_solar(data_dir: str, **kwargs) -> TimeSeriesDataset:
    """Load Solar Energy dataset."""
    return _load_csv_generic(
        f"{data_dir}/solar_AL.csv", name="solar"
    )


def _load_traffic(data_dir: str, **kwargs) -> TimeSeriesDataset:
    """Load PeMS Traffic dataset."""
    return _load_csv_generic(
        f"{data_dir}/traffic.csv", name="traffic"
    )


def _load_exchange(data_dir: str, **kwargs) -> TimeSeriesDataset:
    """Load Exchange Rate dataset."""
    return _load_csv_generic(
        f"{data_dir}/exchange_rate.csv", name="exchange"
    )


def _load_electricity(data_dir: str, **kwargs) -> TimeSeriesDataset:
    """Load Electricity dataset."""
    return _load_csv_generic(
        f"{data_dir}/electricity.csv", name="electricity", date_col="date"
    )


def _load_nyc_taxi(data_dir: str, **kwargs) -> TimeSeriesDataset:
    """Load NYC Taxi dataset."""
    return _load_csv_generic(
        f"{data_dir}/nyc_taxi.csv", name="nyc_taxi", date_col="pickup_datetime"
    )
