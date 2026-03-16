"""Abstract base classes for forecasters and OT maps."""

from abc import ABC, abstractmethod

import torch
import numpy as np


class BaseForecaster(ABC):
    """Base class for point forecasters."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the forecaster on training data.

        Args:
            X: Predictor array of shape (n, d).
            Y: Response array of shape (n, d).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Produce point forecasts.

        Args:
            X: Input array of shape (n, d).

        Returns:
            Predictions of shape (n, d).
        """
        ...

    def residuals(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute calibration residuals Y - predict(X).

        Args:
            X: Input array of shape (n, d).
            Y: True values of shape (n, d).

        Returns:
            Residuals of shape (n, d).
        """
        return Y - self.predict(X)


class BaseOTMap(ABC):
    """Base class for optimal transport maps."""

    @abstractmethod
    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """Forward OT map Q_hat: source -> target.

        Args:
            x: Input tensor of shape (n, d).

        Returns:
            Mapped tensor of shape (n, d).
        """
        ...

    @abstractmethod
    def inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse OT map R_hat: target -> source.

        Args:
            y: Input tensor of shape (n, d).

        Returns:
            Mapped tensor of shape (n, d).
        """
        ...

    @abstractmethod
    def fit(
        self, source_samples: np.ndarray, target_samples: np.ndarray
    ) -> dict:
        """Train the OT map.

        Args:
            source_samples: Samples from source distribution, shape (n, d).
            target_samples: Samples from target distribution, shape (n, d).

        Returns:
            Training metrics dict.
        """
        ...
