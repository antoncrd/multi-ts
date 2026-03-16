"""Point forecasters: Oracle, Linear (OLS), and MLP."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseForecaster


class OracleForecaster(BaseForecaster):
    """Oracle forecaster using the true VAR(1) transition matrix A.

    E[Y_t | X_t] = A @ X_t.
    """

    def __init__(self, A: np.ndarray):
        self._A = A.copy()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass  # No training needed

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self._A.T


class LinearForecaster(BaseForecaster):
    """Linear forecaster via OLS: Y = X @ A^T + noise.

    Estimates A_hat = (X^T X)^{-1} X^T Y.
    """

    def __init__(self, ridge: float = 1e-6):
        self._A_hat: Optional[np.ndarray] = None
        self._ridge = ridge

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        n, d = X.shape
        # Ridge regression: A_hat = Y^T X (X^T X + lambda I)^{-1}
        XtX = X.T @ X + self._ridge * np.eye(d)
        XtY = X.T @ Y
        self._A_hat = np.linalg.solve(XtX, XtY).T  # shape (d, d)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._A_hat is None:
            raise RuntimeError("Forecaster not fitted. Call fit() first.")
        return X @ self._A_hat.T

    @property
    def A_hat(self) -> np.ndarray:
        if self._A_hat is None:
            raise RuntimeError("Forecaster not fitted.")
        return self._A_hat.copy()


class MLPForecaster(BaseForecaster):
    """MLP forecaster for nonlinear dynamics (e.g., Lorenz-96).

    2-layer MLP: input -> hidden -> hidden -> output.
    """

    def __init__(
        self,
        d: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        n_epochs: int = 200,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
        ).to(self.device)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.net.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = ((pred - yb) ** 2).mean()
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.net(X_t)
        return pred.cpu().numpy()
