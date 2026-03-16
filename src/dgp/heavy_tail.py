"""Heavy-tail VAR(1) Data Generating Process with Student-t innovations."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from src.dgp.base import BaseDGP
from src.dgp.var1 import VAR1Generator


class HeavyTailGenerator(BaseDGP):
    """VAR(1) with multivariate Student-t innovations.

    Y_t = A @ Y_{t-1} + eps_t where eps_t ~ t_nu(0, Sigma).

    Uses the same A construction as VAR1Generator (prescribed spectral radius),
    but replaces Gaussian innovations with Student-t via the scale mixture
    representation: eps = sqrt(nu / chi2(nu)) * N(0, Sigma).
    """

    def __init__(
        self,
        d: int = 8,
        rho_A: float = 0.5,
        rho_cross: float = 0.3,
        nu: float = 5.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            d: Dimension.
            rho_A: Spectral radius of A.
            rho_cross: Cross-correlation parameter for Sigma.
            nu: Degrees of freedom for Student-t. Use np.inf for Gaussian.
            seed: Random seed.
        """
        if nu <= 2:
            raise ValueError(f"nu must be > 2 for finite variance, got {nu}")

        self._d = d
        self._rho_A = rho_A
        self._rho_cross = rho_cross
        self._nu = nu
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Reuse VAR1's matrix construction
        self._var1 = VAR1Generator(d=d, rho_A=rho_A, rho_cross=rho_cross, seed=seed)
        self._A = self._var1.A
        self._Sigma = self._var1.Sigma
        self._L = np.linalg.cholesky(self._Sigma)

    def _sample_student_t(self, n: int) -> np.ndarray:
        """Sample n vectors from multivariate Student-t(nu, 0, Sigma).

        Uses the scale mixture: eps = sqrt(nu / chi2(nu)) * N(0, Sigma).
        For nu = inf, returns Gaussian.
        """
        d = self._d

        # Gaussian part
        z = self._rng.standard_normal((n, d))
        gaussian = z @ self._L.T

        if np.isinf(self._nu):
            return gaussian

        # Chi-squared scaling
        chi2_samples = self._rng.chisquare(df=self._nu, size=n)
        scale = np.sqrt(self._nu / chi2_samples)
        return gaussian * scale[:, np.newaxis]

    def generate(
        self, n: int, burn_in: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n (X, Y) pairs from heavy-tail VAR(1).

        Args:
            n: Number of (X, Y) pairs.
            burn_in: Samples to discard. If None, uses adaptive formula.

        Returns:
            X: shape (n, d), Y: shape (n, d).
        """
        if burn_in is None:
            burn_in = max(200, int(500 / (1 - self._rho_A)))

        total = n + 1 + burn_in
        d = self._d

        # Generate Student-t innovations
        eps = self._sample_student_t(total)

        # Simulate VAR(1)
        series = np.zeros((total, d))
        for t in range(1, total):
            series[t] = self._A @ series[t - 1] + eps[t]

        # Discard burn-in
        series = series[burn_in:]
        X = series[:-1]
        Y = series[1:]

        return X, Y

    @property
    def dimension(self) -> int:
        return self._d

    def oracle_params(self) -> Dict:
        # For Student-t with nu > 2, the stationary covariance is
        # Cov(Y) = (nu / (nu - 2)) * solve_discrete_lyapunov(A, Sigma)
        base_cov = solve_discrete_lyapunov(self._A, self._Sigma)
        scale = self._nu / (self._nu - 2) if not np.isinf(self._nu) else 1.0
        return {
            "A": self._A.copy(),
            "Sigma": self._Sigma.copy(),
            "nu": self._nu,
            "d": self._d,
            "rho_A": self._rho_A,
            "rho_cross": self._rho_cross,
            "stationary_cov": scale * base_cov,
        }
