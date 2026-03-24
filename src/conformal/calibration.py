"""Conformal calibration engine.

Implements the full calibration pipeline:
- rho_hat_k = W_1(mu_hat_k, U(B_d)) from rank-transformed residuals
- eta_k(beta) = (1/beta) * (2 * kappa_{d,1} * C_rho^{1/d}) / k^{1/d}
- delta*(r) via FOC bisection: d * delta^2 * (r - delta)^{d-1} = A
- r*(alpha, beta) via nested bisection on L*(r) >= 1 - alpha
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from src.conformal.wasserstein import (
    estimate_mixing_factor,
    fournier_constant,
    rank_transform,
    wasserstein_1_uniform,
)


@dataclass
class CalibrationResult:
    """Results from conformal calibration."""

    rho_hat_k: float       # Empirical W_1(mu_hat_k, U_d)
    eta_k: float           # Concentration bound eta_k(beta)
    delta_star: float      # Optimal smoothing parameter
    r_star: float          # Final conformal radius r*(alpha, beta)
    r_star_uncapped: float # r* before clipping to 1.0
    alpha: float           # Target miscoverage level
    beta: float            # Confidence parameter for W_1 bound
    d: int                 # Dimension
    k: int                 # Calibration set size
    c_rho: float           # Mixing factor estimate


class ConformalCalibrator:
    """Conformal calibration for multidimensional prediction regions.

    Implements the calibration procedure from the paper:
    1. Rank-transform calibration residuals
    2. Compute W_1 between empirical ranks and uniform
    3. Compute concentration bound eta_k(beta)
    4. Optimize smoothing parameter delta*
    5. Compute conformal radius r*(alpha, beta)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.05,
        w1_method: str = "auto",
        w1_n_projections: int = 100,
    ):
        """
        Args:
            alpha: Target miscoverage level (e.g. 0.1 for 90% coverage).
            beta: Confidence parameter for Wasserstein bound.
            w1_method: Method for W_1 computation ("exact", "sliced", "auto").
            w1_n_projections: Number of projections for sliced W_1.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")

        self.alpha = alpha
        self.beta = beta
        self.w1_method = w1_method
        self.w1_n_projections = w1_n_projections

    def calibrate(
        self,
        residuals: np.ndarray,
        seed: Optional[int] = None,
        ot_ranks: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """Run the full calibration pipeline.

        Args:
            residuals: shape (k, d), calibration residuals Y - Yhat.
            seed: Random seed for W_1 computation.
            ot_ranks: Optional pre-computed OT ranks of shape (k, d).
                If provided, these are used instead of the marginal rank
                transform. Typically obtained via Q_hat(residuals) from a
                trained Neural OT map, normalized to [0,1]^d.

        Returns:
            CalibrationResult with all intermediate and final quantities.
        """
        k, d = residuals.shape

        # Step 1: Rank transform — use OT ranks if available, else marginal
        if ot_ranks is not None:
            ranks = ot_ranks
        else:
            ranks = rank_transform(residuals)

        # Step 2: Compute W_1(mu_hat_k, U(B_d))
        rho_hat_k = wasserstein_1_uniform(
            ranks, d,
            method=self.w1_method,
            n_projections=self.w1_n_projections,
            seed=seed,
        )

        # Step 3: Estimate mixing factor
        c_rho = estimate_mixing_factor(ranks)

        # Step 4: Compute eta_k(beta)
        eta_k = self.compute_eta_k(self.beta, k, d, c_rho)

        # Step 5: Effective Wasserstein bound
        A = rho_hat_k + eta_k

        # Step 6: Nested bisection for r*(alpha, beta)
        r_star, delta_star, r_star_uncapped = self._nested_bisection(self.alpha, d, A)

        return CalibrationResult(
            rho_hat_k=rho_hat_k,
            eta_k=eta_k,
            delta_star=delta_star,
            r_star=r_star,
            r_star_uncapped=r_star_uncapped,
            alpha=self.alpha,
            beta=self.beta,
            d=d,
            k=k,
            c_rho=c_rho,
        )

    @staticmethod
    def _nested_bisection(alpha: float, d: int, A: float) -> tuple:
        """Find r*(alpha, beta) via nested bisection over r in (0, 1].

        Outer bisection on r: find smallest r with L*(r) >= 1 - alpha,
        where L*(r) = (r - delta*(r))^d - A / delta*(r).
        Inner bisection: solve FOC for delta*(r).

        Since ranks live in B_d, r is restricted to (0, 1].
        If no valid r exists (bound is vacuous), returns r=1.

        Returns:
            (r_star, delta_star) tuple.
        """
        def L_star(r):
            if r <= 0:
                return -float("inf")
            ds = ConformalCalibrator.optimize_delta_star(r, d, A)
            if ds <= 0 or ds >= r:
                return -float("inf")
            return (r - ds) ** d - A / ds

        # Ranks live in B_d, so r ∈ (0, 1]
        r_upper = 1.0

        # First, find the uncapped r* (no upper-bound restriction)
        # to report the true theoretical radius before clipping.
        r_uncapped_hi = max(r_upper, 10.0)
        while L_star(r_uncapped_hi) < 1 - alpha and r_uncapped_hi < 1e6:
            r_uncapped_hi *= 2.0
        if L_star(r_uncapped_hi) >= 1 - alpha:
            r_lo_u, r_hi_u = 1e-6, r_uncapped_hi
            for _ in range(100):
                r_mid = (r_lo_u + r_hi_u) / 2
                if L_star(r_mid) >= 1 - alpha:
                    r_hi_u = r_mid
                else:
                    r_lo_u = r_mid
                if r_hi_u - r_lo_u < 1e-10:
                    break
            r_star_uncapped = r_hi_u
        else:
            r_star_uncapped = float("inf")
        print(f"  [calibration] r*_uncapped = {r_star_uncapped:.6f}  (before clipping to 1.0)")

        if L_star(r_upper) < 1 - alpha:
            # Bound is vacuous: not enough data for the theoretical
            # guarantee at this (alpha, beta, k, d).
            warnings.warn(
                f"Bound is vacuous (A={A:.4f}, d={d}): L*(1) < 1-alpha. "
                f"Returning r*=1. Increase k or relax beta."
            )
            r_star = 1.0
            delta_star = ConformalCalibrator.optimize_delta_star(r_star, d, A)
        else:
            r_lo, r_hi = 1e-6, r_upper
            for _ in range(100):
                r_mid = (r_lo + r_hi) / 2
                if L_star(r_mid) >= 1 - alpha:
                    r_hi = r_mid
                else:
                    r_lo = r_mid
                if r_hi - r_lo < 1e-10:
                    break
            r_star = r_hi
            delta_star = ConformalCalibrator.optimize_delta_star(r_star, d, A)

        return r_star, delta_star, r_star_uncapped

    def compute_eta_k(
        self, beta: float, k: int, d: int, c_rho: float = 1.0
    ) -> float:
        """Compute concentration bound eta_k(beta).

        eta_k(beta) = log(1/beta) * (2 * kappa_{d,1} * C_rho^{1/d}) / k^{1/d}

        Uses the Fournier (2023) explicit expectation bound with a
        log(1/beta) concentration factor (conjectured tighter than Markov).
        """
        kappa = fournier_constant(d)
        return np.log(1.0 / beta) * (2.0 * kappa * c_rho ** (1.0 / d)) / (k ** (1.0 / d))

    @staticmethod
    def optimize_delta_star(r: float, d: int, A: float) -> float:
        """Find delta*(r) by solving the FOC via bisection.

        Solves: d * delta^2 * (r - delta)^{d-1} = A
        for the LEFT root in (0, 2r/(d+1)), which maximizes L(delta).

        Args:
            r: Radius.
            d: Dimension.
            A: Wasserstein upper bound (rho_hat + eta).

        Returns:
            Optimal delta*.
        """
        if A <= 0 or r <= 0:
            return 0.0

        eps = 1e-10

        def foc(delta):
            return d * delta ** 2 * (r - delta) ** (d - 1) - A

        # g(delta) = d*delta^2*(r-delta)^{d-1} peaks at delta = 2r/(d+1)
        delta_peak = 2.0 * r / (d + 1)

        if foc(delta_peak) < 0:
            # g never reaches A — no root exists; return peak as best approx
            return delta_peak

        # Left root in (eps, delta_peak) is the L-maximizer
        try:
            return brentq(foc, eps, delta_peak, xtol=1e-12, maxiter=200)
        except Exception:
            warnings.warn(
                f"delta* optimization failed (r={r:.4f}, d={d}, A={A:.6f}). "
                f"Using fallback delta* = 2r/(d+1)."
            )
            return delta_peak

    @staticmethod
    def compute_radius(
        alpha: float, delta_star: float, A: float, d: int
    ) -> float:
        """Compute conservative upper bound on conformal radius.

        r_upper(alpha, beta) = delta* + (1 - alpha + A/delta*)^{1/d}

        This is a closed-form conservative bound. The primary method
        uses nested bisection in calibrate().

        Args:
            alpha: Miscoverage level.
            delta_star: Optimal smoothing parameter.
            A: Wasserstein upper bound.
            d: Dimension.

        Returns:
            Conservative conformal radius.
        """
        if delta_star <= 0:
            # No OT correction: standard conformal quantile
            return (1 - alpha) ** (1.0 / d)

        inner = 1 - alpha + A / delta_star
        if inner <= 0:
            warnings.warn(
                f"Negative inner term in radius computation "
                f"(1-alpha + A/delta* = {inner:.6f}). Clamping to epsilon."
            )
            inner = 1e-10

        return min(delta_star + inner ** (1.0 / d), 1.0)

    def calibrate_naive(self, residuals: np.ndarray) -> float:
        """Naive conformal quantile (no OT correction) for comparison.

        Returns the ceil((1-alpha)(k+1))/k -th quantile of ||residuals||.
        """
        k = residuals.shape[0]
        norms = np.linalg.norm(residuals, axis=1)
        norms_sorted = np.sort(norms)
        idx = int(np.ceil((1 - self.alpha) * (k + 1))) - 1
        idx = min(idx, k - 1)
        return norms_sorted[idx]

    def calibrate_empirical_ot(self, ot_norms: np.ndarray) -> float:
        """Empirical conformal quantile in OT rank space.

        Given ||Q_hat(residual_i)|| for i in calibration set,
        returns the (1-alpha)-quantile. This is used at test time with
        ||Q_hat(Y_t - Y_hat_t)|| <= r* for containment.

        This bypasses the theoretical W_1 bound and directly uses the
        empirical quantile of OT-mapped norms.

        Args:
            ot_norms: shape (k,), norms of Q_hat(residuals) in rank space.

        Returns:
            Empirical conformal radius in OT space.
        """
        k = len(ot_norms)
        norms_sorted = np.sort(ot_norms)
        idx = int(np.ceil((1 - self.alpha) * (k + 1))) - 1
        idx = min(idx, k - 1)
        return norms_sorted[idx]
