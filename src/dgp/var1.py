"""VAR(1) Data Generating Process with tunable spectral radius and cross-correlation."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from src.dgp.base import BaseDGP


class VAR1Generator(BaseDGP):
    """VAR(1) process: Y_t = A @ Y_{t-1} + eps_t.

    A is constructed to have a prescribed spectral radius rho_A via
    A = rho_A * Q @ D @ Q^T where Q is random orthogonal and D is diagonal
    with entries in [0, 1] normalized so the largest equals 1.

    eps_t ~ N(0, Sigma) with Sigma = (1 - rho_cross) * I + rho_cross * 1@1^T.
    """

    def __init__(
        self,
        d: int,
        rho_A: float,
        rho_cross: float = 0.3,
        seed: Optional[int] = None,
    ):
        if not 0 < rho_A < 1:
            raise ValueError(f"rho_A must be in (0, 1), got {rho_A}")
        if rho_cross < -1.0 / (d - 1) if d > 1 else rho_cross < -1:
            raise ValueError(
                f"rho_cross={rho_cross} yields non-PD Sigma for d={d}"
            )

        self._d = d
        self._rho_A = rho_A
        self._rho_cross = rho_cross
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._A = self._build_A()
        self._Sigma = self._build_Sigma()
        self._L = np.linalg.cholesky(self._Sigma)

    def _build_A(self) -> np.ndarray:
        """Construct transition matrix with prescribed spectral radius."""
        d = self._d
        # Random orthogonal matrix via QR of Gaussian
        Z = self._rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(Z)

        # Diagonal with entries in [0, 1], normalized so max = 1
        raw = self._rng.uniform(0.1, 1.0, size=d)
        raw /= np.max(np.abs(raw))
        D = np.diag(raw)

        A = self._rho_A * (Q @ D @ Q.T)

        # Verify spectral radius
        eigvals = np.linalg.eigvals(A)
        actual_rho = np.max(np.abs(eigvals))
        assert np.isclose(actual_rho, self._rho_A, atol=1e-10), (
            f"Spectral radius mismatch: expected {self._rho_A}, got {actual_rho}"
        )
        return A

    def _build_Sigma(self) -> np.ndarray:
        """Build innovation covariance: (1 - rho_cross) * I + rho_cross * 11^T."""
        d = self._d
        ones = np.ones((d, d))
        Sigma = (1 - self._rho_cross) * np.eye(d) + self._rho_cross * ones

        # Verify positive definiteness
        eigvals = np.linalg.eigvalsh(Sigma)
        if eigvals[0] <= 0:
            raise ValueError(
                f"Sigma is not positive definite (min eigenvalue={eigvals[0]:.6f})"
            )
        return Sigma

    def generate(
        self, n: int, burn_in: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n time-aligned (X, Y) pairs from the VAR(1) process.

        Args:
            n: Number of (X, Y) pairs.
            burn_in: Samples to discard. If None, uses adaptive formula.

        Returns:
            X: shape (n, d), Y: shape (n, d) where Y[t] = A @ X[t] + eps[t+1].
        """
        if burn_in is None:
            burn_in = max(200, int(500 / (1 - self._rho_A)))

        total = n + 1 + burn_in  # +1 because we need n+1 points for n pairs
        d = self._d

        # Pre-generate all innovations
        z = self._rng.standard_normal((total, d))
        eps = z @ self._L.T  # shape (total, d), each row ~ N(0, Sigma)

        # Simulate VAR(1)
        series = np.zeros((total, d))
        for t in range(1, total):
            series[t] = self._A @ series[t - 1] + eps[t]

        # Discard burn-in, create (X, Y) pairs
        series = series[burn_in:]
        X = series[:-1]  # shape (n, d)
        Y = series[1:]   # shape (n, d)

        return X, Y

    @property
    def dimension(self) -> int:
        return self._d

    @property
    def A(self) -> np.ndarray:
        return self._A.copy()

    @property
    def Sigma(self) -> np.ndarray:
        return self._Sigma.copy()

    def oracle_params(self) -> Dict:
        """Return ground-truth parameters."""
        stationary_cov = solve_discrete_lyapunov(self._A, self._Sigma)
        return {
            "A": self._A.copy(),
            "Sigma": self._Sigma.copy(),
            "rho_A": self._rho_A,
            "rho_cross": self._rho_cross,
            "d": self._d,
            "stationary_cov": stationary_cov,
        }

    def oracle_forecast(self, X: np.ndarray) -> np.ndarray:
        """Oracle point forecast: E[Y_t | X_t] = A @ X_t."""
        return X @ self._A.T
