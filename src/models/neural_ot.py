"""Neural Optimal Transport via Input-Convex Neural Networks (ICNN).

Implements:
- ICNN: Input-Convex Neural Network for the OT potential
- Forward map Q_hat: gradient of the ICNN potential
- AmortizedInverse: MLP trained to invert Q_hat
- NeuralOTMap: combines forward + inverse, with Monge gap training
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class ICNN(nn.Module):
    """Input-Convex Neural Network.

    Architecture: z_{l+1} = softplus(W_z^l @ z_l + W_x^l @ x + b^l)
    with W_z^l >= 0 enforced via softplus reparameterization.

    The network outputs a scalar convex potential phi(x).
    The OT map is Q_hat(x) = nabla_x phi(x).
    """

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 64] if input_dim <= 16 else [256, 256, 128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        n_layers = len(hidden_dims)

        # W_x layers (unconstrained): input skip connections
        self.wx_layers = nn.ModuleList()
        # W_z layers (non-negative via softplus): propagation layers
        self.wz_raw = nn.ParameterList()
        # Biases
        self.biases = nn.ParameterList()

        # First hidden layer: only W_x (no z input yet)
        self.wx_layers.append(nn.Linear(input_dim, hidden_dims[0], bias=False))
        self.biases.append(nn.Parameter(torch.zeros(hidden_dims[0])))

        # Subsequent hidden layers: W_z (non-negative) + W_x (skip)
        for i in range(1, n_layers):
            self.wz_raw.append(
                nn.Parameter(torch.randn(hidden_dims[i], hidden_dims[i - 1]) * 0.01)
            )
            self.wx_layers.append(nn.Linear(input_dim, hidden_dims[i], bias=False))
            self.biases.append(nn.Parameter(torch.zeros(hidden_dims[i])))

        # Output layer: maps to scalar
        self.wz_raw.append(
            nn.Parameter(torch.randn(1, hidden_dims[-1]) * 0.01)
        )
        self.wx_out = nn.Linear(input_dim, 1, bias=False)
        self.bias_out = nn.Parameter(torch.zeros(1))

        # Add a quadratic term for strong convexity: (epsilon/2)||x||^2
        self.strong_convexity = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the convex potential phi(x).

        Args:
            x: shape (batch, input_dim).

        Returns:
            Scalar potential, shape (batch, 1).
        """
        # First hidden layer
        z = F.softplus(self.wx_layers[0](x) + self.biases[0])

        # Subsequent hidden layers with non-negative W_z
        for i, wz_raw in enumerate(self.wz_raw[:-1]):
            wz = F.softplus(wz_raw)  # Enforce non-negativity
            z = F.softplus(
                F.linear(z, wz) + self.wx_layers[i + 1](x) + self.biases[i + 1]
            )

        # Output layer
        wz_out = F.softplus(self.wz_raw[-1])
        phi = F.linear(z, wz_out) + self.wx_out(x) + self.bias_out

        # Add strong convexity term
        phi = phi + 0.5 * self.strong_convexity * (x ** 2).sum(dim=1, keepdim=True)

        return phi

    def gradient(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        """Compute nabla_x phi(x) = the OT map Q_hat(x).

        Args:
            x: shape (batch, input_dim).
            create_graph: If True, the graph of the gradient is constructed,
                allowing higher-order gradients and backprop through the map.

        Returns:
            Gradient, shape (batch, input_dim).
        """
        x_in = x.detach().clone().requires_grad_(True)
        phi = self.forward(x_in)
        grad = torch.autograd.grad(
            outputs=phi.sum(),
            inputs=x_in,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return grad


class AmortizedInverse(nn.Module):
    """MLP trained to approximate the inverse of the ICNN gradient map.

    R_hat(y) ≈ Q_hat^{-1}(y) such that R_hat(Q_hat(x)) ≈ x.
    """

    def __init__(self, dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers = []
        prev = dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class NeuralOTMap:
    """Neural OT map combining ICNN forward map and amortized inverse.

    Training procedure:
    1. Train ICNN potential with Monge gap regularization
    2. Train inverse MLP to invert the forward map
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims_icnn: Optional[List[int]] = None,
        hidden_dims_inverse: Optional[List[int]] = None,
        lr_icnn: float = 1e-3,
        lr_inverse: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 500,
        batch_size: int = 256,
        monge_gap_weight: float = 0.1,
        grad_clip: float = 1.0,
        warmup_epochs: int = 50,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.input_dim = input_dim

        self.icnn = ICNN(input_dim, hidden_dims_icnn).to(self.device)
        self.inverse = AmortizedInverse(input_dim, hidden_dims_inverse).to(self.device)

        self.lr_icnn = lr_icnn
        self.lr_inverse = lr_inverse
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.monge_gap_weight = monge_gap_weight
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs

    def fit(
        self,
        source_samples: np.ndarray,
        target_samples: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Train the neural OT map.

        If target_samples is None, uses standard normal as target.

        Args:
            source_samples: shape (n, d), source distribution samples (residuals).
            target_samples: shape (n, d), target distribution samples.
            verbose: Show progress bar.

        Returns:
            Dict with training loss histories.
        """
        d = source_samples.shape[1]

        source_t = torch.tensor(source_samples, dtype=torch.float32, device=self.device)

        if target_samples is not None:
            target_t = torch.tensor(
                target_samples, dtype=torch.float32, device=self.device
            )
        else:
            target_t = None

        # Phase 1: Train ICNN with Monge gap
        history_icnn = self._train_icnn(source_t, target_t, verbose)

        # Phase 2: Train amortized inverse
        history_inverse = self._train_inverse(source_t, verbose)

        return {"icnn_loss": history_icnn, "inverse_loss": history_inverse}

    def _train_icnn(
        self,
        source: torch.Tensor,
        target: Optional[torch.Tensor],
        verbose: bool,
    ) -> list:
        """Train the ICNN potential."""
        optimizer = torch.optim.Adam(
            self.icnn.parameters(),
            lr=self.lr_icnn,
            weight_decay=self.weight_decay,
        )

        # Linear warmup scheduler
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / max(self.warmup_epochs, 1)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        dataset = TensorDataset(source)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        losses = []
        rng = range(self.n_epochs)
        if verbose:
            rng = trange(self.n_epochs, desc="ICNN training")

        for epoch in rng:
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()

                # Forward map (create_graph=True for backprop to ICNN params)
                Q_x = self.icnn.gradient(batch, create_graph=True)

                # Monge cost: E[||x - Q(x)||^2]
                monge_cost = ((batch - Q_x) ** 2).sum(dim=1).mean()

                # Regularizer: push Q(x) toward target distribution
                if target is not None:
                    idx = torch.randint(0, target.shape[0], (batch.shape[0],))
                    target_batch = target[idx]
                    target_loss = ((Q_x.mean(0) - target_batch.mean(0)) ** 2).sum()
                else:
                    # Push toward standard normal
                    target_loss = (
                        ((Q_x.mean(0)) ** 2).sum()
                        + ((Q_x.var(0) - 1.0) ** 2).sum()
                    )

                loss = monge_cost + self.monge_gap_weight * target_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.icnn.parameters(), self.grad_clip
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * batch.shape[0]

            epoch_loss /= source.shape[0]
            losses.append(epoch_loss)

            if verbose and isinstance(rng, trange):
                rng.set_postfix(loss=f"{epoch_loss:.4f}")

        return losses

    def _train_inverse(
        self, source: torch.Tensor, verbose: bool
    ) -> list:
        """Train the amortized inverse map."""
        optimizer = torch.optim.Adam(
            self.inverse.parameters(), lr=self.lr_inverse
        )

        dataset = TensorDataset(source)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        losses = []
        n_inverse_epochs = min(self.n_epochs, 200)
        rng = range(n_inverse_epochs)
        if verbose:
            rng = trange(n_inverse_epochs, desc="Inverse training")

        self.icnn.eval()
        for epoch in rng:
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()

                # Use create_graph=False (no backprop through ICNN needed)
                # Cannot use torch.no_grad() because gradient() needs autograd
                Q_x = self.icnn.gradient(batch, create_graph=False).detach()

                # Reconstruct: R(Q(x)) should be ≈ x
                x_reconstructed = self.inverse(Q_x)
                loss = ((x_reconstructed - batch) ** 2).sum(dim=1).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.shape[0]

            epoch_loss /= source.shape[0]
            losses.append(epoch_loss)

            if verbose and isinstance(rng, trange):
                rng.set_postfix(loss=f"{epoch_loss:.6f}")

        self.icnn.train()
        return losses

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward OT map Q_hat."""
        self.icnn.eval()
        result = self.icnn.gradient(x, create_graph=False).detach()
        return result

    def inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the inverse OT map R_hat."""
        self.inverse.eval()
        with torch.no_grad():
            result = self.inverse(y)
        return result

    def forward_map_np(self, x: np.ndarray) -> np.ndarray:
        """Numpy wrapper for forward map."""
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self.forward_map(x_t).cpu().numpy()

    def inverse_map_np(self, y: np.ndarray) -> np.ndarray:
        """Numpy wrapper for inverse map."""
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        return self.inverse_map(y_t).cpu().numpy()

    def check_convexity(self, x: np.ndarray, n_checks: int = 100) -> float:
        """Verify convexity of the ICNN potential along random line segments.

        Returns the fraction of checks that satisfy convexity.
        """
        self.icnn.eval()
        n = x.shape[0]
        violations = 0

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for _ in range(n_checks):
                i, j = np.random.choice(n, 2, replace=False)
                lam = np.random.uniform(0, 1)
                x_mid = lam * x_t[i:i+1] + (1 - lam) * x_t[j:j+1]

                phi_i = self.icnn(x_t[i:i+1]).item()
                phi_j = self.icnn(x_t[j:j+1]).item()
                phi_mid = self.icnn(x_mid).item()

                if phi_mid > lam * phi_i + (1 - lam) * phi_j + 1e-6:
                    violations += 1

        return 1.0 - violations / n_checks
