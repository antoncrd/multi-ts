"""Coordinate-wise SPCI baseline.

Applies standard 1D split conformal prediction independently per coordinate,
then takes the Cartesian product to form a hyperrectangular prediction region.
"""

from typing import List, Optional

import numpy as np

from src.conformal.rank_region import ConformalRegion, build_region


class CoordinateWiseSPCI:
    """Coordinate-wise Split Conformal Prediction Intervals.

    Applies Bonferroni-corrected 1D conformal quantiles per coordinate.
    The resulting region is a hyperrectangle (axis-aligned box).
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._half_widths: Optional[np.ndarray] = None

    def calibrate(self, residuals: np.ndarray) -> None:
        """Calibrate per-coordinate quantiles.

        Args:
            residuals: shape (k, d).
        """
        k, d = residuals.shape

        # Bonferroni correction: each coordinate uses alpha/d
        alpha_j = self.alpha / d

        self._half_widths = np.zeros(d)
        for j in range(d):
            abs_resid = np.abs(residuals[:, j])
            sorted_resid = np.sort(abs_resid)
            idx = int(np.ceil((1 - alpha_j) * (k + 1))) - 1
            idx = min(idx, k - 1)
            self._half_widths[j] = sorted_resid[idx]

    def predict_regions(
        self, Y_hat: np.ndarray
    ) -> List[ConformalRegion]:
        """Build hyperrectangular regions (approximated as balls).

        Args:
            Y_hat: shape (n, d).

        Returns:
            List of ConformalRegion. Radius = half-diagonal of hyperrectangle.
        """
        if self._half_widths is None:
            raise RuntimeError("Not calibrated.")

        effective_radius = np.linalg.norm(self._half_widths)
        return [build_region(Y_hat[i], effective_radius) for i in range(Y_hat.shape[0])]

    def contains_rect(self, residual: np.ndarray) -> bool:
        """True containment check (hyperrectangle, not ball)."""
        if self._half_widths is None:
            raise RuntimeError("Not calibrated.")
        return np.all(np.abs(residual) <= self._half_widths)
