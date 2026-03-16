"""ACI baseline: Adaptive Conformal Inference.

Reference: Gibbs & Candes (2021), "Adaptive Conformal Inference Under
    Distribution Shift"
"""
# NOTE: re-implementation, official code unavailable

from typing import List, Optional

import numpy as np

from src.conformal.rank_region import ConformalRegion, build_region


class ACI:
    """Adaptive Conformal Inference.

    Adjusts the miscoverage level alpha_t online based on whether
    previous predictions covered the true value. This adapts to
    distribution shift and serial dependence.

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    where err_t = 1{Y_t not in C_t}.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.01,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self._calibration_scores: Optional[np.ndarray] = None

    def calibrate(self, residuals: np.ndarray) -> None:
        """Store calibration nonconformity scores.

        Args:
            residuals: shape (k, d).
        """
        self._calibration_scores = np.linalg.norm(residuals, axis=1)

    def predict_regions_adaptive(
        self,
        Y_hat: np.ndarray,
        Y_true: Optional[np.ndarray] = None,
    ) -> List[ConformalRegion]:
        """Build prediction regions with adaptive alpha.

        If Y_true is provided, adapts alpha online. Otherwise uses fixed alpha.

        Args:
            Y_hat: Point forecasts, shape (T, d).
            Y_true: True values for online adaptation, shape (T, d). Optional.

        Returns:
            List of ConformalRegion instances.
        """
        if self._calibration_scores is None:
            raise RuntimeError("Not calibrated.")

        T = Y_hat.shape[0]
        scores_sorted = np.sort(self._calibration_scores)
        k = len(scores_sorted)

        regions = []
        alpha_t = self.alpha

        for t in range(T):
            # Compute quantile with current alpha_t
            alpha_clamped = np.clip(alpha_t, 0.001, 0.999)
            idx = int(np.ceil((1 - alpha_clamped) * (k + 1))) - 1
            idx = min(max(idx, 0), k - 1)
            radius = scores_sorted[idx]

            regions.append(build_region(Y_hat[t], radius))

            # Online update if true values available
            if Y_true is not None:
                err_t = 1.0 if not regions[-1].contains(Y_true[t]) else 0.0
                alpha_t = alpha_t + self.gamma * (self.alpha - err_t)

        return regions
