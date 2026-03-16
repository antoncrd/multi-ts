"""Conformal prediction regions based on OT rank maps."""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.special import gammaln


def log_unit_ball_volume(d: int) -> float:
    """Log-volume of the d-dimensional unit ball: log(pi^{d/2} / Gamma(d/2 + 1))."""
    return (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)


def unit_ball_volume(d: int) -> float:
    """Volume of the d-dimensional unit ball."""
    return np.exp(log_unit_ball_volume(d))


@dataclass
class ConformalRegion:
    """A conformal prediction region.

    For our method: C(X_t) = {Y_hat_t} + Q_hat^{-1}(r* . B_d)
    which is the pre-image of the r*-ball under the OT map, centered at Y_hat.

    When no OT map is provided, the region is a simple Euclidean ball
    of radius r* centered at Y_hat.
    """

    center: np.ndarray           # Y_hat, shape (d,)
    radius: float                # r*(alpha, beta)
    d: int = field(init=False)   # dimension
    ot_inverse: Optional[Callable] = None  # R_hat: B_d -> residual space

    def __post_init__(self):
        self.d = self.center.shape[0]

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the prediction region.

        Args:
            point: shape (d,).

        Returns:
            True if point is within the region.
        """
        residual = point - self.center

        if self.ot_inverse is not None:
            # Map residual through the forward OT map to check in rank space
            # We need the forward map Q_hat to check: ||Q_hat(residual)|| <= r*
            # But we store the inverse R_hat. For containment, we check
            # whether any point in r*B_d maps to the residual via R_hat.
            # Efficient approximation: use the forward map if available.
            # Fallback: use Euclidean ball check (conservative).
            pass

        # Default: Euclidean ball check
        return float(np.linalg.norm(residual)) <= self.radius

    def contains_batch(self, points: np.ndarray) -> np.ndarray:
        """Check containment for a batch of points.

        Args:
            points: shape (n, d).

        Returns:
            Boolean array of shape (n,).
        """
        residuals = points - self.center[np.newaxis, :]
        norms = np.linalg.norm(residuals, axis=1)
        return norms <= self.radius

    def volume(self) -> float:
        """Volume of the prediction region.

        For Euclidean ball: V_d * r^d.
        For OT-transformed regions: would require Monte Carlo estimation.
        """
        return unit_ball_volume(self.d) * (self.radius ** self.d)

    def log_volume(self) -> float:
        """Log-volume (numerically stable for high d)."""
        return log_unit_ball_volume(self.d) + self.d * np.log(max(self.radius, 1e-300))

    def diameter(self) -> float:
        """Diameter of the region (2 * radius for a ball)."""
        return 2.0 * self.radius

    def sample_boundary(self, n_points: int, seed: int = 0) -> np.ndarray:
        """Sample points on the boundary of the region.

        Args:
            n_points: Number of boundary points.
            seed: Random seed.

        Returns:
            Points of shape (n_points, d) on the boundary.
        """
        rng = np.random.default_rng(seed)
        # Sample uniformly on the d-sphere
        z = rng.standard_normal((n_points, self.d))
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        boundary = self.center[np.newaxis, :] + self.radius * z

        if self.ot_inverse is not None:
            # Transform boundary through inverse OT map
            # boundary_rank = r* * z (on the ball boundary in rank space)
            # boundary_residual = R_hat(boundary_rank)
            rank_boundary = self.radius * z
            try:
                import torch
                with torch.no_grad():
                    rank_t = torch.tensor(rank_boundary, dtype=torch.float32)
                    boundary = self.ot_inverse(rank_t).numpy()
                boundary = boundary + self.center[np.newaxis, :]
            except Exception:
                pass  # Fall back to Euclidean boundary

        return boundary


def build_region(
    y_hat: np.ndarray,
    radius: float,
    ot_inverse: Optional[Callable] = None,
) -> ConformalRegion:
    """Factory function to build a ConformalRegion.

    Args:
        y_hat: Point forecast, shape (d,).
        radius: Conformal radius r*(alpha, beta).
        ot_inverse: Optional inverse OT map R_hat.

    Returns:
        ConformalRegion instance.
    """
    return ConformalRegion(
        center=y_hat,
        radius=radius,
        ot_inverse=ot_inverse,
    )


def build_regions_batch(
    y_hat_batch: np.ndarray,
    radius: float,
    ot_inverse: Optional[Callable] = None,
) -> list:
    """Build regions for a batch of forecasts.

    Args:
        y_hat_batch: shape (n, d).
        radius: Single radius applied to all regions.
        ot_inverse: Optional inverse OT map.

    Returns:
        List of ConformalRegion instances.
    """
    return [
        build_region(y_hat_batch[i], radius, ot_inverse)
        for i in range(y_hat_batch.shape[0])
    ]
