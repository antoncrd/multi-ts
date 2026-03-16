"""Lorenz-96 Data Generating Process.

System: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
with cyclic boundary conditions.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from src.dgp.base import BaseDGP


class Lorenz96Generator(BaseDGP):
    """Lorenz-96 chaotic dynamical system.

    Parameters control the forcing F (higher = more chaotic) and
    the dimension d (number of variables with cyclic coupling).
    """

    def __init__(
        self,
        d: int = 8,
        F: float = 8.0,
        dt: float = 0.01,
        subsample_dt: float = 0.05,
        seed: Optional[int] = None,
    ):
        self._d = d
        self._F = F
        self._dt = dt
        self._subsample_dt = subsample_dt
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._subsample_factor = max(1, int(round(subsample_dt / dt)))

    def _lorenz96_rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """Right-hand side of the Lorenz-96 system."""
        d = self._d
        dxdt = np.zeros(d)
        for i in range(d):
            ip1 = (i + 1) % d
            im1 = (i - 1) % d
            im2 = (i - 2) % d
            dxdt[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + self._F
        return dxdt

    def generate(
        self, n: int, burn_in: int = 2000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n (X, Y) pairs from the Lorenz-96 system.

        The system is integrated with an adaptive RK45 solver, then
        subsampled at intervals of subsample_dt.

        Args:
            n: Number of (X, Y) pairs.
            burn_in: Number of subsample steps to discard as burn-in.

        Returns:
            X: shape (n, d), Y: shape (n, d).
        """
        total_steps = n + 1 + burn_in
        t_end = total_steps * self._subsample_dt

        # Initial condition: small perturbation around F
        x0 = self._F * np.ones(self._d)
        x0 += self._rng.uniform(-0.01, 0.01, size=self._d)

        # Integrate
        t_eval = np.arange(0, t_end, self._subsample_dt)
        sol = solve_ivp(
            self._lorenz96_rhs,
            t_span=(0, t_end),
            y0=x0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            max_step=self._dt,
        )

        if not sol.success:
            raise RuntimeError(f"Lorenz-96 integration failed: {sol.message}")

        # sol.y has shape (d, n_timepoints)
        series = sol.y.T  # shape (n_timepoints, d)

        # Discard burn-in
        series = series[burn_in:]

        if series.shape[0] < n + 1:
            raise RuntimeError(
                f"Not enough data after burn-in: got {series.shape[0]}, need {n + 1}"
            )

        X = series[:n]       # shape (n, d)
        Y = series[1:n + 1]  # shape (n, d)

        return X, Y

    @property
    def dimension(self) -> int:
        return self._d

    def oracle_params(self) -> Dict:
        return {
            "d": self._d,
            "F": self._F,
            "dt": self._dt,
            "subsample_dt": self._subsample_dt,
        }
