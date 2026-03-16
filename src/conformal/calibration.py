"""Conformal calibration engine.

Implements the full calibration pipeline:
- rho_hat_k = W_1(mu_hat_k, U_d) from rank-transformed residuals
- eta_k(beta) = (C/beta) * (a_d(k) + k^{-(q-1)/q}) with explicit Fournier constants
- delta* optimization via 1D search
- r*(alpha, beta) = delta* + (1 - alpha + A/delta*)^{1/d}
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from src.conformal.wasserstein import (
    convergence_rate_term,
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
        q: float = 2.0,
    ):
        """
        Args:
            alpha: Target miscoverage level (e.g. 0.1 for 90% coverage).
            beta: Confidence parameter for Wasserstein bound.
            w1_method: Method for W_1 computation ("exact", "sliced", "auto").
            w1_n_projections: Number of projections for sliced W_1.
            q: Moment parameter for concentration inequality (q > 1).
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")

        self.alpha = alpha
        self.beta = beta
        self.w1_method = w1_method
        self.w1_n_projections = w1_n_projections
        self.q = q

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

        # Step 2: Compute W_1(mu_hat_k, U_d)
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
        # A = rho_hat_k + eta_k is the upper bound on the true W_1
        A = rho_hat_k + eta_k

        # Step 6: Optimize delta*
        # We need a preliminary radius estimate to set the search range
        r_prelim = (1 - self.alpha) ** (1.0 / d) + A
        delta_star = self.optimize_delta_star(r_prelim, d, A)

        # Step 7: Compute r*(alpha, beta)
        r_star = self.compute_radius(self.alpha, delta_star, A, d)

        return CalibrationResult(
            rho_hat_k=rho_hat_k,
            eta_k=eta_k,
            delta_star=delta_star,
            r_star=r_star,
            alpha=self.alpha,
            beta=self.beta,
            d=d,
            k=k,
            c_rho=c_rho,
        )

    def compute_eta_k(
        self, beta: float, k: int, d: int, c_rho: float = 1.0
    ) -> float:
        """Compute concentration bound eta_k(beta).

        Uses a two-part bound combining:
        1. Expected rate: kappa * a_d(k), the Fournier-Guillin rate term
        2. Concentration: McDiarmid's inequality applied to W_1.

        Since W_1 has bounded differences (changing one sample out of k
        changes W_1 by at most diam(support)/k = sqrt(d)/k on [0,1]^d),
        McDiarmid gives:
            P(W_1 > E[W_1] + t) <= exp(-2 k t^2 / d)

        Setting this to beta and solving for t:
            t = sqrt(d * log(1/beta) / (2k))

        The full bound is:
            eta_k(beta) = c_rho * (kappa * a_d(k) + sqrt(d * log(1/beta) / (2k)))

        where c_rho accounts for temporal dependence (rho-mixing).
        """
        kappa = fournier_constant(d)
        a_d_k = convergence_rate_term(k, d)

        # Part 1: Expected rate (Fournier-Guillin)
        expected_rate = kappa * a_d_k

        # Part 2: McDiarmid concentration tail
        concentration = np.sqrt(d * np.log(1.0 / beta) / (2.0 * k))

        return c_rho * (expected_rate + concentration)

    @staticmethod
    def optimize_delta_star(r: float, d: int, A: float) -> float:
        """Optimize delta* via 1D search.

        Minimizes L(delta) = (r - delta)^d - A / delta on (0, r).
        The optimal delta balances the volume loss from shrinking the ball
        against the Wasserstein correction.

        Args:
            r: Preliminary radius.
            d: Dimension.
            A: Wasserstein upper bound (rho_hat + eta).

        Returns:
            Optimal delta*.
        """
        if A <= 0:
            return 0.0

        eps = 1e-10

        def objective(delta):
            if delta <= eps or delta >= r - eps:
                return np.inf
            return (r - delta) ** d - A / delta

        # Check if a valid interior minimum exists
        # At delta -> 0+: L -> -inf (from -A/delta term)
        # At delta -> r-: L -> -A/r (from (r-delta)^d -> 0)
        # We want the minimum, which may be at the boundary
        # The derivative: -d*(r-delta)^{d-1} + A/delta^2 = 0
        # => A/delta^2 = d*(r-delta)^{d-1}

        try:
            result = minimize_scalar(
                objective,
                bounds=(eps, r - eps),
                method="bounded",
                options={"xatol": 1e-12, "maxiter": 1000},
            )
            if result.success and eps < result.x < r - eps:
                return result.x
        except Exception:
            pass

        # Fallback: delta = r/2 with warning
        warnings.warn(
            f"delta* optimization failed (r={r:.4f}, d={d}, A={A:.6f}). "
            f"Using fallback delta* = r/2."
        )
        return r / 2.0

    @staticmethod
    def compute_radius(
        alpha: float, delta_star: float, A: float, d: int
    ) -> float:
        """Compute conformal radius r*(alpha, beta).

        r*(alpha, beta) = delta* + (1 - alpha + A/delta*)^{1/d}

        Args:
            alpha: Miscoverage level.
            delta_star: Optimal smoothing parameter.
            A: Wasserstein upper bound.
            d: Dimension.

        Returns:
            Conformal radius r*.
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

        return delta_star + inner ** (1.0 / d)

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
