"""EnbPI baseline: Ensemble Batch Prediction Intervals.

Reference: Xu & Xie (2021), "Conformal prediction interval for
    dynamic time-series"
"""
# NOTE: re-implementation, official code unavailable

from typing import List, Optional

import numpy as np

from src.conformal.rank_region import ConformalRegion, build_region
from src.models.base import BaseForecaster


class EnbPI:
    """Ensemble Batch Prediction Intervals.

    Uses bootstrap aggregation of the forecaster to estimate
    prediction uncertainty, then applies conformal calibration
    to the aggregated residuals.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        n_bootstrap: int = 20,
        seed: Optional[int] = None,
    ):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self._rng = np.random.default_rng(seed)
        self._quantile: Optional[float] = None
        self._bootstrap_residuals: Optional[np.ndarray] = None

    def calibrate(
        self,
        forecaster_class: type,
        forecaster_kwargs: dict,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_calib: np.ndarray,
        Y_calib: np.ndarray,
    ) -> None:
        """Calibrate using bootstrap ensemble.

        Fits n_bootstrap forecasters on bootstrap samples of the training data,
        aggregates their predictions on the calibration set, and computes
        conformal quantiles on the aggregated residuals.

        Args:
            forecaster_class: Class to instantiate (e.g., LinearForecaster).
            forecaster_kwargs: Keyword args for the forecaster constructor.
            X_train, Y_train: Training data.
            X_calib, Y_calib: Calibration data.
        """
        n_train = X_train.shape[0]
        k = X_calib.shape[0]
        d = X_calib.shape[1]

        # Bootstrap predictions on calibration set
        predictions = np.zeros((self.n_bootstrap, k, d))

        for b in range(self.n_bootstrap):
            # Bootstrap sample
            idx = self._rng.choice(n_train, size=n_train, replace=True)
            X_boot, Y_boot = X_train[idx], Y_train[idx]

            # Fit and predict
            forecaster = forecaster_class(**forecaster_kwargs)
            forecaster.fit(X_boot, Y_boot)
            predictions[b] = forecaster.predict(X_calib)

        # Aggregated prediction (mean of bootstrap ensemble)
        Y_hat_agg = predictions.mean(axis=0)

        # Residuals from aggregated prediction
        residuals = Y_calib - Y_hat_agg
        norms = np.linalg.norm(residuals, axis=1)
        norms_sorted = np.sort(norms)

        # Conformal quantile
        idx = int(np.ceil((1 - self.alpha) * (k + 1))) - 1
        idx = min(idx, k - 1)
        self._quantile = norms_sorted[idx]
        self._bootstrap_residuals = residuals

    def predict_regions(
        self, Y_hat: np.ndarray
    ) -> List[ConformalRegion]:
        """Build prediction regions using the ensemble quantile.

        Args:
            Y_hat: Point forecasts, shape (n, d).

        Returns:
            List of ConformalRegion instances.
        """
        if self._quantile is None:
            raise RuntimeError("Not calibrated.")

        return [
            build_region(Y_hat[i], self._quantile)
            for i in range(Y_hat.shape[0])
        ]
