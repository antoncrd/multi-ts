"""MultiDimSPCI baseline: Multidimensional Split Conformal Prediction Intervals.

Reference: Xu & Xie (2023), "Conformal prediction for multi-dimensional time series"
Official repo: https://github.com/hamrel-cxu/MultiDimSPCI
"""
# NOTE: re-implementation, integrate official code when available

from typing import List, Optional

import numpy as np

from src.conformal.rank_region import ConformalRegion, build_region
from src.models.base import BaseForecaster


class MultiDimSPCI:
    """Multi-dimensional Split Conformal Prediction Intervals.

    Uses an ellipsoidal prediction region based on the empirical covariance
    of calibration residuals, scaled by a conformal quantile.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._Sigma_inv: Optional[np.ndarray] = None
        self._quantile: Optional[float] = None

    def calibrate(
        self,
        residuals: np.ndarray,
    ) -> None:
        """Calibrate on residuals from the calibration set.

        Args:
            residuals: shape (k, d), calibration residuals.
        """
        k, d = residuals.shape

        # Empirical covariance of residuals
        Sigma = np.cov(residuals.T) + 1e-6 * np.eye(d)
        self._Sigma_inv = np.linalg.inv(Sigma)

        # Mahalanobis distances as nonconformity scores
        scores = np.array([
            np.sqrt(r @ self._Sigma_inv @ r) for r in residuals
        ])
        scores_sorted = np.sort(scores)

        # Conformal quantile
        idx = int(np.ceil((1 - self.alpha) * (k + 1))) - 1
        idx = min(idx, k - 1)
        self._quantile = scores_sorted[idx]

    def predict_regions(
        self,
        Y_hat: np.ndarray,
    ) -> List[ConformalRegion]:
        """Build ellipsoidal prediction regions.

        For MultiDimSPCI the region is an ellipsoid defined by the
        Mahalanobis distance. We approximate it as a ball with effective
        radius for compatibility with the ConformalRegion interface.

        Args:
            Y_hat: Point forecasts, shape (n, d).

        Returns:
            List of ConformalRegion instances.
        """
        if self._quantile is None:
            raise RuntimeError("Not calibrated. Call calibrate() first.")

        # Effective radius: the Mahalanobis quantile scaled back to Euclidean
        # For a ball approximation, use the quantile as-is
        regions = []
        for i in range(Y_hat.shape[0]):
            regions.append(build_region(Y_hat[i], self._quantile))
        return regions

    def contains(self, residual: np.ndarray) -> bool:
        """Check if a residual is inside the ellipsoidal region."""
        if self._Sigma_inv is None or self._quantile is None:
            raise RuntimeError("Not calibrated.")
        score = np.sqrt(residual @ self._Sigma_inv @ residual)
        return score <= self._quantile
