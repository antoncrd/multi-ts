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


def _sample_uniform_ball_np(
    n: int, d: int, radius: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample n points uniformly from the d-dimensional ball of given radius."""
    z = rng.standard_normal((n, d))
    z = z / np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-12)
    u = rng.uniform(0, 1, (n, 1))
    r = u ** (1.0 / d)
    return radius * r * z


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
    calib_residuals: Optional[np.ndarray] = None  # (k, d) calibration residuals for bounding box

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

    def volume(
        self, mc_samples: int = 10000, seed: int = 0, method: str = "hit_or_miss",
    ) -> float:
        """Volume of the prediction region.

        For Euclidean ball (no OT): V_d * r^d (exact).
        For OT-transformed regions: MC estimate in residual (Y) space.

        Args:
            mc_samples: Number of MC samples.
            seed: Random seed.
            method: "hit_or_miss" (default, samples in Y-space bounding box
                and counts fraction with ||Q(res)|| <= r*) or "jacobian"
                (change-of-variable via finite-difference Jacobian of R).
        """
        if self.ot_forward is None:
            return unit_ball_volume(self.d) * ((self.radius * self._residual_scale) ** self.d)
        if method == "jacobian":
            return self._mc_volume_jacobian(mc_samples, seed)
        return self._mc_volume_hit_or_miss(mc_samples, seed)

    def log_volume(
        self, mc_samples: int = 10000, seed: int = 0, method: str = "hit_or_miss",
    ) -> float:
        """Log-volume (numerically stable for high d).

        For OT regions, defaults to hit-or-miss in Y-space.
        """
        if self.ot_forward is None:
            r = self.radius * self._residual_scale
            return log_unit_ball_volume(self.d) + self.d * np.log(max(r, 1e-300))
        if method == "jacobian":
            return self._mc_log_volume_jacobian(mc_samples, seed)
        return self._mc_log_volume_hit_or_miss(mc_samples, seed)

    # ------------------------------------------------------------------
    # Jacobian-based MC volume (change of variable)
    # ------------------------------------------------------------------

    def _jacobian_logabsdet(
        self, z: np.ndarray, eps: float = 1e-4
    ) -> np.ndarray:
        """Estimate log|det J_R(z_i)| for each row of z via finite differences.

        J_R is the Jacobian of the inverse map R = ot_inverse.
        For each sample z_i we perturb each coordinate j by +-eps
        and build the d x d Jacobian column by column, then compute
        the log absolute determinant.

        Args:
            z: shape (n, d), points in rank space.
            eps: step size for central differences.

        Returns:
            Array of shape (n,) with log|det J_R(z_i)|.
        """
        n, d = z.shape
        log_dets = np.empty(n)

        # Process in batches to limit memory: each sample needs 2d
        # forward passes through the inverse map, but we can batch
        # all perturbations together.
        # Build perturbed points: for each sample i and coord j,
        # z_plus[i,j] = z[i] + eps*e_j  and  z_minus[i,j] = z[i] - eps*e_j
        # Total: 2*n*d points of shape (d,).
        eye = np.eye(d) * eps  # (d, d)

        # z_plus_all[i*d + j] = z[i] + eps*e_j
        z_expanded = np.repeat(z, d, axis=0)  # (n*d, d)
        eye_tiled = np.tile(eye, (n, 1))      # (n*d, d)
        z_plus = z_expanded + eye_tiled
        z_minus = z_expanded - eye_tiled

        # Evaluate inverse map on all perturbed points
        R_plus = self.ot_inverse(z_plus)    # (n*d, d)
        R_minus = self.ot_inverse(z_minus)  # (n*d, d)

        # Central difference: dR/dz_j ~ (R(z+eps*e_j) - R(z-eps*e_j)) / (2*eps)
        dR = (R_plus - R_minus) / (2.0 * eps)  # (n*d, d)

        # Reshape to (n, d, d): for sample i, row j of the Jacobian is dR[i*d+j]
        J = dR.reshape(n, d, d)  # J[i, j, k] = dR_k / dz_j at z_i

        # log|det J| via slogdet
        signs, logdets = np.linalg.slogdet(J)
        log_dets = logdets  # sign doesn't matter for volume

        return log_dets

    def _mc_volume_jacobian(self, mc_samples: int, seed: int) -> float:
        """Volume via change-of-variable: Vol(C) = Vol(r*B_d) * E[|det J_R|].

        Samples z ~ Uniform(r*B_d), estimates |det J_R(z)| by finite
        differences, and returns Vol(r*B_d) * mean(|det J_R|).
        """
        if self.ot_inverse is None:
            # No inverse map: fall back to hit-or-miss
            return self._mc_volume_hit_or_miss(mc_samples, seed)

        rng = np.random.default_rng(seed)

        # Sample uniformly in r*B_d
        z = _sample_uniform_ball_np(mc_samples, self.d, self.radius, rng)

        # Jacobian log-abs-dets
        log_abs_dets = self._jacobian_logabsdet(z)

        # Vol(r*B_d) * mean(|det J|) = exp(log_vol_ball) * mean(exp(log_abs_dets))
        log_vol_ball = log_unit_ball_volume(self.d) + self.d * np.log(self.radius)
        # Use log-sum-exp for stability
        max_ld = np.max(log_abs_dets)
        log_mean_det = max_ld + np.log(np.mean(np.exp(log_abs_dets - max_ld)))

        log_vol = log_vol_ball + log_mean_det
        return np.exp(log_vol)

    def _mc_log_volume_jacobian(self, mc_samples: int, seed: int) -> float:
        """Log-volume via change-of-variable (numerically stable).

        log Vol(C) = log Vol(r*B_d) + log E[|det J_R(z)|]
        """
        if self.ot_inverse is None:
            return self._mc_log_volume_hit_or_miss(mc_samples, seed)

        rng = np.random.default_rng(seed)
        z = _sample_uniform_ball_np(mc_samples, self.d, self.radius, rng)
        log_abs_dets = self._jacobian_logabsdet(z)

        log_vol_ball = log_unit_ball_volume(self.d) + self.d * np.log(max(self.radius, 1e-300))
        # log E[|det J|] via log-sum-exp
        max_ld = np.max(log_abs_dets)
        log_mean_det = max_ld + np.log(np.mean(np.exp(log_abs_dets - max_ld)))

        return log_vol_ball + log_mean_det

    # ------------------------------------------------------------------
    # Hit-or-miss MC volume (validation / fallback)
    # ------------------------------------------------------------------

    def _mc_volume_hit_or_miss(self, mc_samples: int, seed: int) -> float:
        """Hit-or-miss MC volume in residual (Y) space.

        Samples uniformly in a tight bounding box around the region and
        counts the fraction satisfying ||Q(res)|| <= r*.

        The bounding box is derived from calibration residuals that are
        inside the region (i.e., those with ||Q(res)|| <= r*). If no
        calibration residuals are available, falls back to the inverse
        map for probing.
        """
        rng = np.random.default_rng(seed)

        # --- Build tight bounding box from calibration residuals ---
        if self.calib_residuals is not None and self.ot_forward is not None:
            # Filter to residuals inside the region
            Q_calib = self.ot_forward(self.calib_residuals)
            calib_norms = np.linalg.norm(Q_calib, axis=1)
            inside_mask = calib_norms <= self.radius
            if inside_mask.sum() >= 2:
                inside_res = self.calib_residuals[inside_mask]
                lo = inside_res.min(axis=0)
                hi = inside_res.max(axis=0)
                # Add margin to ensure we cover the full region boundary
                margin = 0.3 * (hi - lo + 1e-10)
                lo -= margin
                hi += margin
            else:
                # Very few points inside: use all residuals with generous margin
                lo = self.calib_residuals.min(axis=0)
                hi = self.calib_residuals.max(axis=0)
                margin = 0.5 * (hi - lo + 1e-10)
                lo -= margin
                hi += margin
        elif self.ot_inverse is not None:
            # Fallback: probe via inverse map
            n_probe = min(5000, mc_samples)
            z_probe = _sample_uniform_ball_np(n_probe, self.d, self.radius, rng)
            res_probe = self.ot_inverse(z_probe)
            lo = res_probe.min(axis=0)
            hi = res_probe.max(axis=0)
            margin = 0.3 * (hi - lo + 1e-10)
            lo -= margin
            hi += margin
        else:
            # No OT map at all: simple ball
            r = self.radius * self._residual_scale
            lo = -r * np.ones(self.d)
            hi = r * np.ones(self.d)

        # Sample uniformly in bounding box
        samples = rng.uniform(lo, hi, size=(mc_samples, self.d))

        # Check containment via forward map
        if self.ot_forward is not None:
            ranks = self.ot_forward(samples)
            norms = np.linalg.norm(ranks, axis=1)
            inside = norms <= self.radius
        else:
            norms = np.linalg.norm(samples, axis=1)
            inside = norms <= self.radius * self._residual_scale

        frac = inside.mean()
        box_volume = np.prod(hi - lo)
        return box_volume * frac

    def _mc_log_volume_hit_or_miss(self, mc_samples: int, seed: int) -> float:
        """Log of hit-or-miss volume estimate."""
        vol = self._mc_volume_hit_or_miss(mc_samples, seed)
        return np.log(max(vol, 1e-300))

    def diameter(self, mc_samples: int = 2000, seed: int = 0) -> float:
        """Diameter of the region.

        For Euclidean ball: 2*r.
        For OT regions: estimated from calibration residuals inside the
        region (max pairwise extent), or via inverse-mapped boundary.
        """
        if self.ot_forward is None:
            return 2.0 * self.radius * self._residual_scale

        # Use calibration residuals if available
        if self.calib_residuals is not None and self.ot_forward is not None:
            Q_calib = self.ot_forward(self.calib_residuals)
            norms = np.linalg.norm(Q_calib, axis=1)
            inside = self.calib_residuals[norms <= self.radius]
            if len(inside) >= 2:
                # Diameter = max extent across all coordinates
                spread = inside.max(axis=0) - inside.min(axis=0)
                return float(np.linalg.norm(spread))

        if self.ot_inverse is None:
            return 2.0 * self.radius

        rng = np.random.default_rng(seed)
        z = rng.standard_normal((mc_samples, self.d))
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        z = self.radius * z
        res = self.ot_inverse(z)
        centroid = res.mean(axis=0)
        dists = np.linalg.norm(res - centroid, axis=1)
        return 2.0 * float(dists.max())

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
    calib_residuals: Optional[np.ndarray] = None,
) -> ConformalRegion:
    """Factory function to build a ConformalRegion.

    Args:
        y_hat: Point forecast, shape (d,).
        radius: Conformal radius r*(alpha, beta).
        ot_forward: Optional forward OT map Q_hat (residual -> rank space).
        ot_inverse: Optional inverse OT map R_hat (rank space -> residual).
        residual_scale: Scale factor for non-OT fallback containment check.
        calib_residuals: Optional (k, d) calibration residuals used to
            build a tight bounding box for hit-or-miss volume estimation.

    Returns:
        ConformalRegion instance.
    """
    return ConformalRegion(
        center=y_hat,
        radius=radius,
        ot_forward=ot_forward,
        ot_inverse=ot_inverse,
        _residual_scale=residual_scale,
        calib_residuals=calib_residuals,
    )


def build_regions_batch(
    y_hat_batch: np.ndarray,
    radius: float,
    ot_forward: Optional[Callable] = None,
    ot_inverse: Optional[Callable] = None,
    residual_scale: float = 1.0,
    calib_residuals: Optional[np.ndarray] = None,
) -> list:
    """Build regions for a batch of forecasts.

    Args:
        y_hat_batch: shape (n, d).
        radius: Single radius applied to all regions.
        ot_forward: Optional forward OT map Q_hat.
        ot_inverse: Optional inverse OT map R_hat.
        residual_scale: Scale factor for non-OT fallback.
        calib_residuals: Optional (k, d) calibration residuals for volume estimation.

    Returns:
        List of ConformalRegion instances.
    """
    return [
        build_region(
            y_hat_batch[i], radius, ot_forward, ot_inverse,
            residual_scale, calib_residuals,
        )
        for i in range(y_hat_batch.shape[0])
    ]
