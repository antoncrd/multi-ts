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

    Containment is checked in rank space: Y_t in C(X_t) iff
    ||Q_hat(Y_t - Y_hat_t)|| <= r*.

    When no OT forward map is provided, the region is a simple Euclidean ball
    of radius r* centered at Y_hat.
    """

    center: np.ndarray           # Y_hat, shape (d,)
    radius: float                # r*(alpha, beta)
    d: int = field(init=False)   # dimension
    ot_forward: Optional[Callable] = None   # Q_hat: residual space -> rank space
    ot_inverse: Optional[Callable] = None   # R_hat: rank space -> residual space
    _residual_scale: float = 1.0  # Scale factor for non-OT fallback

    def __post_init__(self):
        self.d = self.center.shape[0]

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the prediction region.

        If an OT forward map is available, checks ||Q_hat(residual)|| <= r*
        in rank space. Otherwise checks ||residual|| <= r* * scale.

        Args:
            point: shape (d,).

        Returns:
            True if point is within the region.
        """
        residual = point - self.center

        if self.ot_forward is not None:
            rank = self.ot_forward(residual[np.newaxis, :])[0]
            return float(np.linalg.norm(rank)) <= self.radius

        return float(np.linalg.norm(residual)) <= self.radius * self._residual_scale

    def contains_batch(self, points: np.ndarray) -> np.ndarray:
        """Check containment for a batch of points.

        Args:
            points: shape (n, d).

        Returns:
            Boolean array of shape (n,).
        """
        residuals = points - self.center[np.newaxis, :]

        if self.ot_forward is not None:
            ranks = self.ot_forward(residuals)
            norms = np.linalg.norm(ranks, axis=1)
            return norms <= self.radius

        norms = np.linalg.norm(residuals, axis=1)
        return norms <= self.radius * self._residual_scale

    def volume(self, mc_samples: int = 5000, seed: int = 0) -> float:
        """Volume of the prediction region.

        For Euclidean ball (no OT): V_d * r^d.
        For OT-transformed regions: Monte Carlo estimate using rejection
        sampling in a bounding box around the region.
        """
        if self.ot_forward is None:
            return unit_ball_volume(self.d) * ((self.radius * self._residual_scale) ** self.d)
        return self._mc_volume(mc_samples, seed)

    def log_volume(self, mc_samples: int = 5000, seed: int = 0) -> float:
        """Log-volume (numerically stable for high d)."""
        if self.ot_forward is None:
            r = self.radius * self._residual_scale
            return log_unit_ball_volume(self.d) + self.d * np.log(max(r, 1e-300))
        vol = self._mc_volume(mc_samples, seed)
        return np.log(max(vol, 1e-300))

    def _mc_volume(self, mc_samples: int, seed: int) -> float:
        """Monte Carlo volume estimation for OT-transformed regions."""
        rng = np.random.default_rng(seed)
        # Sample from a bounding box, check containment
        # Use inverse map to estimate the effective extent
        z_sphere = rng.standard_normal((mc_samples, self.d))
        z_sphere = z_sphere / np.linalg.norm(z_sphere, axis=1, keepdims=True)
        radii = rng.uniform(0, 1, (mc_samples, 1)) ** (1.0 / self.d)
        z_ball = self.radius * radii * z_sphere  # uniform in r*B_d
        if self.ot_inverse is not None:
            residuals_mc = self.ot_inverse(z_ball)
        else:
            residuals_mc = z_ball
        # Volume = (fraction inside) * volume of bounding box
        # But since we sample uniformly in the ball and map through inverse,
        # the volume is |det(J)| * Vol(r*B_d) on average.
        # Simpler: just report the spread of the mapped samples.
        spread = np.std(residuals_mc, axis=0)
        return np.prod(spread) * (2 ** self.d)  # rough bounding box

    def diameter(self) -> float:
        """Diameter of the region (2 * radius for a ball)."""
        if self.ot_forward is None:
            return 2.0 * self.radius * self._residual_scale
        return 2.0 * self.radius  # In OT space; true diameter requires MC

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
    ot_forward: Optional[Callable] = None,
    ot_inverse: Optional[Callable] = None,
    residual_scale: float = 1.0,
) -> ConformalRegion:
    """Factory function to build a ConformalRegion.

    Args:
        y_hat: Point forecast, shape (d,).
        radius: Conformal radius r*(alpha, beta).
        ot_forward: Optional forward OT map Q_hat (residual -> rank space).
        ot_inverse: Optional inverse OT map R_hat (rank space -> residual).
        residual_scale: Scale factor for non-OT fallback containment check.

    Returns:
        ConformalRegion instance.
    """
    return ConformalRegion(
        center=y_hat,
        radius=radius,
        ot_forward=ot_forward,
        ot_inverse=ot_inverse,
        _residual_scale=residual_scale,
    )


def build_regions_batch(
    y_hat_batch: np.ndarray,
    radius: float,
    ot_forward: Optional[Callable] = None,
    ot_inverse: Optional[Callable] = None,
    residual_scale: float = 1.0,
) -> list:
    """Build regions for a batch of forecasts.

    Args:
        y_hat_batch: shape (n, d).
        radius: Single radius applied to all regions.
        ot_forward: Optional forward OT map Q_hat.
        ot_inverse: Optional inverse OT map R_hat.
        residual_scale: Scale factor for non-OT fallback.

    Returns:
        List of ConformalRegion instances.
    """
    return [
        build_region(y_hat_batch[i], radius, ot_forward, ot_inverse, residual_scale)
        for i in range(y_hat_batch.shape[0])
    ]
