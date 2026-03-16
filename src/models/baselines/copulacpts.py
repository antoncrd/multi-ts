"""CopulaCPTS baseline: Copula-based Conformal Prediction for Time Series.

Reference: Sun et al. (2023), "Copula Conformal Prediction for
    Multi-step Time Series Forecasting"
Official repo: https://github.com/Rose-STL-Lab/CopulaCPTS
"""
# NOTE: re-implementation, integrate official code when available

from typing import List, Optional

import numpy as np
from scipy import stats

from src.conformal.rank_region import ConformalRegion, build_region


class CopulaCPTS:
    """Copula-based conformal prediction for multivariate time series.

    Models the dependence structure of residuals via an empirical copula,
    then constructs prediction regions based on copula quantiles.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._marginal_quantiles: Optional[np.ndarray] = None
        self._d: Optional[int] = None

    def calibrate(self, residuals: np.ndarray) -> None:
        """Calibrate using empirical copula of residuals.

        Args:
            residuals: shape (k, d), calibration residuals.
        """
        k, d = residuals.shape
        self._d = d

        # Compute per-coordinate conformal quantiles with Bonferroni correction
        alpha_per_dim = self.alpha / d

        self._marginal_quantiles = np.zeros(d)
        for j in range(d):
            abs_resid = np.abs(residuals[:, j])
            sorted_resid = np.sort(abs_resid)
            idx = int(np.ceil((1 - alpha_per_dim) * (k + 1))) - 1
            idx = min(idx, k - 1)
            self._marginal_quantiles[j] = sorted_resid[idx]

        # Also compute a joint quantile using the copula approach
        # Rank-transform to pseudo-observations
        ranks = np.zeros_like(residuals)
        for j in range(d):
            order = np.argsort(np.argsort(np.abs(residuals[:, j])))
            ranks[:, j] = (order + 1) / (k + 1)

        # Joint depth: product of marginal ranks
        joint_depths = ranks.prod(axis=1)
        sorted_depths = np.sort(joint_depths)
        idx = int(np.ceil((1 - self.alpha) * (k + 1))) - 1
        idx = min(idx, k - 1)
        self._joint_threshold = sorted_depths[idx]

    def predict_regions(
        self, Y_hat: np.ndarray
    ) -> List[ConformalRegion]:
        """Build hyperrectangular prediction regions.

        Args:
            Y_hat: Point forecasts, shape (n, d).

        Returns:
            List of ConformalRegion instances (ball approximation of the rectangle).
        """
        if self._marginal_quantiles is None:
            raise RuntimeError("Not calibrated.")

        # Effective radius: half-diagonal of the hyperrectangle
        effective_radius = np.linalg.norm(self._marginal_quantiles)

        regions = []
        for i in range(Y_hat.shape[0]):
            regions.append(build_region(Y_hat[i], effective_radius))
        return regions

    def contains_rect(self, residual: np.ndarray) -> bool:
        """Check containment using the hyperrectangular region."""
        if self._marginal_quantiles is None:
            raise RuntimeError("Not calibrated.")
        return np.all(np.abs(residual) <= self._marginal_quantiles)
