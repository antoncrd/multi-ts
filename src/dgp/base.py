"""Abstract base class for Data Generating Processes."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class BaseDGP(ABC):
    """Base class for synthetic data generating processes.

    All DGPs produce time-aligned (X, Y) pairs where X[t] is used to predict Y[t].
    For autoregressive processes, Y[t] = X[t+1] in the original series.
    """

    @abstractmethod
    def generate(
        self, n: int, burn_in: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n time-aligned (X, Y) pairs.

        Args:
            n: Number of (X, Y) pairs to return.
            burn_in: Number of initial samples to discard.

        Returns:
            X: Predictor array of shape (n, d).
            Y: Response array of shape (n, d).
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality d of the process."""
        ...

    @abstractmethod
    def oracle_params(self) -> Dict:
        """Return ground-truth parameters for diagnostic comparison."""
        ...
