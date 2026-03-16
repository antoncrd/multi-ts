"""Neural Optimal Transport via Input-Convex Neural Networks (ICNN).

Implements:
- ICNN: Input-Convex Neural Network for the OT potential phi
- Forward map Q_hat = nabla phi: pushes residuals -> B_d(0,1)
- AmortizedInverse: MLP trained to invert Q_hat
- NeuralOTMap: trains via W2 semi-dual with target U(B_d),
  with strong convexity and ball confinement penalties.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_uniform_ball(n: int, d: int, device: torch.device) -> torch.Tensor:
    """Sample n points uniformly from the d-dimensional unit ball B_d(0,1).

    Method: sample direction uniformly on S^{d-1}, then radius r ~ U^{1/d}.
    """
    z = torch.randn(n, d, device=device)
    z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)
    u = torch.rand(n, 1, device=device)
    r = u.pow(1.0 / d)
    return r * z


def differentiable_sliced_wasserstein(
    X: torch.Tensor, Y: torch.Tensor, n_projections: int = 50
) -> torch.Tensor:
    """Differentiable Sliced Wasserstein distance (squared) between two sets.

    SW_2^2(X, Y) = E_theta[ W_2^2(X.theta, Y.theta) ]

    where theta is a random direction on S^{d-1} and W_2^2 on 1D
    sorted samples is ||sort(X.theta) - sort(Y.theta)||^2.

    Both X and Y must have the same number of samples.
    torch.sort is differentiable, so this is end-to-end differentiable.

    Args:
        X: (n, d) source samples (output of Q_hat, requires grad).
        Y: (n, d) target samples (from U(B_d), no grad needed).
        n_projections: number of random directions.

    Returns:
        Scalar SW_2^2 loss.
    """
    d = X.shape[1]
    device = X.device

    # Random directions on S^{d-1}
    theta = torch.randn(n_projections, d, device=device)
    theta = theta / theta.norm(dim=1, keepdim=True)

    # Project: (n_projections, n)
    proj_X = X @ theta.T  # (n, n_proj) -> transpose to (n_proj, n) below
    proj_Y = Y @ theta.T

    # Sort along sample dimension
    proj_X_sorted = torch.sort(proj_X, dim=0).values  # (n, n_proj)
    proj_Y_sorted = torch.sort(proj_Y, dim=0).values

    # W_2^2 per projection: mean of squared differences
    sw2 = ((proj_X_sorted - proj_Y_sorted) ** 2).mean()
    return sw2


# ---------------------------------------------------------------------------
# ICNN
# ---------------------------------------------------------------------------

class ICNN(nn.Module):
    """Input-Convex Neural Network.

    Architecture: z_{l+1} = softplus(W_z^l @ z_l + W_x^l @ x + b^l)
    with W_z^l >= 0 enforced via softplus reparameterization.

    The network outputs a scalar convex potential phi(x).
    The OT map is Q_hat(x) = nabla_x phi(x).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        strong_convexity: float = 0.1,
    ):
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

        # Strong convexity: phi(x) += (epsilon/2)||x||^2
        # ensures nabla^2 phi >= epsilon I, making Q_hat a diffeomorphism
        self.strong_convexity = strong_convexity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the convex potential phi(x). Shape (batch, 1)."""
        z = F.softplus(self.wx_layers[0](x) + self.biases[0])

        for i, wz_raw in enumerate(self.wz_raw[:-1]):
            wz = F.softplus(wz_raw)
            z = F.softplus(
                F.linear(z, wz) + self.wx_layers[i + 1](x) + self.biases[i + 1]
            )

        wz_out = F.softplus(self.wz_raw[-1])
        phi = F.linear(z, wz_out) + self.wx_out(x) + self.bias_out
        phi = phi + 0.5 * self.strong_convexity * (x ** 2).sum(dim=1, keepdim=True)
        return phi

    def gradient(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        """Compute nabla_x phi(x) = Q_hat(x). Shape (batch, input_dim)."""
        x_in = x.detach().clone().requires_grad_(True)
        phi = self.forward(x_in)
        grad = torch.autograd.grad(
            outputs=phi.sum(),
            inputs=x_in,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return grad


# ---------------------------------------------------------------------------
# Amortized inverse
# ---------------------------------------------------------------------------

class AmortizedInverse(nn.Module):
    """MLP trained to approximate Q_hat^{-1}: B_d -> residual space."""

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


# ---------------------------------------------------------------------------
# NeuralOTMap
# ---------------------------------------------------------------------------

class NeuralOTMap:
    """Neural OT map: residuals -> B_d(0,1) via ICNN Brenier potential.

    Training uses the W2 semi-dual of Kantorovich:
        max_{phi convex} E_mu[phi(x)] + E_nu[phi*(y)]
    where nu = U(B_d) and phi*(y) = sup_z {<y,z> - phi(z)}.

    The gradient nabla phi is the OT map Q_hat pushing mu to nu.

    Additional losses:
    - Ball confinement: lambda_ball * E[ReLU(||Q(x)|| - 1)^2]
    - Monge cost monitor: E[||x - Q(x)||^2] (not optimized, just logged)
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
        strong_convexity: float = 0.1,
        ball_penalty_weight: float = 10.0,
        n_conjugate_candidates: int = 512,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.input_dim = input_dim

        self.icnn = ICNN(input_dim, hidden_dims_icnn, strong_convexity).to(self.device)
        self.inverse = AmortizedInverse(input_dim, hidden_dims_inverse).to(self.device)

        self.lr_icnn = lr_icnn
        self.lr_inverse = lr_inverse
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.monge_gap_weight = monge_gap_weight
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.ball_penalty_weight = ball_penalty_weight
        self.n_conjugate_candidates = n_conjugate_candidates

        # Normalization parameters (set during fit)
        self._source_mean: Optional[torch.Tensor] = None
        self._source_std: Optional[torch.Tensor] = None
        # Post-hoc output normalization: Q_norm(x) = Q(x) / _output_scale
        # so that max(||Q_norm(x)||) <= 1 over training data
        self._output_scale: float = 1.0

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using stored mean/std."""
        return (x - self._source_mean) / self._source_std

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize."""
        return x_norm * self._source_std + self._source_mean

    def fit(
        self,
        source_samples: np.ndarray,
        target_samples: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Train the neural OT map.

        If target_samples is None, target is U(B_d).

        Residuals are z-score normalized internally so that the ICNN
        operates on O(1)-magnitude inputs, making it feasible for
        nabla phi to output vectors within B_d.

        Args:
            source_samples: shape (n, d), source distribution (residuals).
            target_samples: shape (n, d), optional explicit target samples.
            verbose: Show progress bar.

        Returns:
            Dict with training loss histories.
        """
        source_t = torch.tensor(source_samples, dtype=torch.float32, device=self.device)

        # Compute and store normalization parameters
        self._source_mean = source_t.mean(dim=0, keepdim=True)
        self._source_std = source_t.std(dim=0, keepdim=True).clamp(min=1e-6)

        # Normalize source
        source_norm = self._normalize(source_t)

        if target_samples is not None:
            target_t = torch.tensor(
                target_samples, dtype=torch.float32, device=self.device
            )
        else:
            target_t = None  # Will sample U(B_d) on the fly

        history_icnn = self._train_icnn(source_norm, target_t, verbose)

        # Post-hoc output normalization: find the scale so Q maps into B_d
        # Use a margin (1.05) to ensure all training points are strictly inside
        self.icnn.eval()
        Q_all = self.icnn.gradient(source_norm, create_graph=False).detach()
        max_norm = Q_all.norm(dim=1).max().item()
        self._output_scale = max(max_norm * 1.05, 1e-6)  # small margin
        self.icnn.train()

        history_inverse = self._train_inverse(source_norm, verbose)

        return {"icnn_loss": history_icnn, "inverse_loss": history_inverse}

    def _train_icnn(
        self,
        source: torch.Tensor,
        target: Optional[torch.Tensor],
        verbose: bool,
    ) -> list:
        """Train ICNN so that Q = nabla phi pushes source into U(B_d).

        Loss = SW_2^2(Q(x_batch), y_batch)                   [distribution match]
             + ball_penalty_weight * E[ReLU(||Q(x)||-1)^2]    [confine to B_d]

        where y_batch ~ U(B_d) and SW_2^2 is the differentiable
        Sliced Wasserstein distance.
        """
        d = source.shape[1]
        optimizer = torch.optim.Adam(
            self.icnn.parameters(),
            lr=self.lr_icnn,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / max(self.warmup_epochs, 1)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        dataset = TensorDataset(source)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        losses = []
        rng_iter = range(self.n_epochs)
        if verbose:
            rng_iter = trange(self.n_epochs, desc="ICNN (SW -> B_d)")

        for epoch in rng_iter:
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                bs = batch.shape[0]

                # Forward OT map: Q(x) = nabla phi(x)
                Q_x = self.icnn.gradient(batch, create_graph=True)  # (bs, d)

                # Target samples from U(B_d) or provided
                if target is not None:
                    idx_t = torch.randint(0, target.shape[0], (bs,), device=self.device)
                    y_batch = target[idx_t]
                else:
                    y_batch = sample_uniform_ball(bs, d, self.device)

                # Sliced Wasserstein loss: match Q#mu to nu
                sw_loss = differentiable_sliced_wasserstein(
                    Q_x, y_batch, n_projections=50
                )

                # Ball confinement penalty
                norms_Q = Q_x.norm(dim=1)
                ball_violation = F.relu(norms_Q - 1.0)
                ball_loss = (ball_violation ** 2).mean()

                loss = sw_loss + self.ball_penalty_weight * ball_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.icnn.parameters(), self.grad_clip
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * bs

            epoch_loss /= source.shape[0]
            losses.append(epoch_loss)

            if verbose and hasattr(rng_iter, 'set_postfix'):
                with torch.no_grad():
                    idx_s = torch.randint(0, source.shape[0], (min(200, source.shape[0]),))
                    Q_s = self.icnn.gradient(source[idx_s])
                    in_ball = (Q_s.norm(dim=1) <= 1.0).float().mean().item()
                    mean_norm = Q_s.norm(dim=1).mean().item()
                rng_iter.set_postfix(
                    loss=f"{epoch_loss:.4f}",
                    in_ball=f"{in_ball:.0%}",
                    norm=f"{mean_norm:.2f}",
                )

        return losses

    def _train_inverse(
        self, source: torch.Tensor, verbose: bool
    ) -> list:
        """Train amortized inverse R_hat: B_d -> normalized residual space.

        The inverse is trained in the post-hoc normalized space:
        Q_scaled(x) = nabla phi(x) / output_scale, so that
        R_hat(Q_scaled(x)) should reconstruct x.

        Loss: E[||R_hat(Q_scaled(x)) - x||^2].
        """
        optimizer = torch.optim.Adam(
            self.inverse.parameters(), lr=self.lr_inverse
        )

        dataset = TensorDataset(source)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        losses = []
        n_inverse_epochs = min(self.n_epochs, 200)
        rng_iter = range(n_inverse_epochs)
        if verbose:
            rng_iter = trange(n_inverse_epochs, desc="Inverse training")

        self.icnn.eval()
        for epoch in rng_iter:
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()

                # Q_scaled in B_d
                Q_x = self.icnn.gradient(batch, create_graph=False).detach()
                Q_scaled = Q_x / self._output_scale

                # Inverse maps from B_d back to normalized residual space
                x_reconstructed = self.inverse(Q_scaled)
                loss = ((x_reconstructed - batch) ** 2).sum(dim=1).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.shape[0]

            epoch_loss /= source.shape[0]
            losses.append(epoch_loss)

            if verbose and hasattr(rng_iter, 'set_postfix'):
                rng_iter.set_postfix(loss=f"{epoch_loss:.6f}")

        self.icnn.train()
        return losses

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Q_hat = nabla phi(normalize(x)) / scale. Output in B_d."""
        self.icnn.eval()
        x_norm = self._normalize(x)
        result = self.icnn.gradient(x_norm, create_graph=False).detach()
        return result / self._output_scale

    def inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        """Apply R_hat = denormalize(inverse(y * scale))."""
        self.inverse.eval()
        with torch.no_grad():
            x_norm = self.inverse(y * self._output_scale)
        return self._denormalize(x_norm)

    def forward_map_np(self, x: np.ndarray) -> np.ndarray:
        """Numpy wrapper for forward map."""
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self.forward_map(x_t).cpu().numpy()

    def inverse_map_np(self, y: np.ndarray) -> np.ndarray:
        """Numpy wrapper for inverse map."""
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        return self.inverse_map(y_t).cpu().numpy()

    def check_convexity(self, x: np.ndarray, n_checks: int = 100) -> float:
        """Fraction of random line-segment checks satisfying convexity of phi."""
        self.icnn.eval()
        n = x.shape[0]
        violations = 0

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        x_norm = self._normalize(x_t)

        with torch.no_grad():
            for _ in range(n_checks):
                i, j = np.random.choice(n, 2, replace=False)
                lam = np.random.uniform(0, 1)
                x_mid = lam * x_norm[i:i+1] + (1 - lam) * x_norm[j:j+1]

                phi_i = self.icnn(x_norm[i:i+1]).item()
                phi_j = self.icnn(x_norm[j:j+1]).item()
                phi_mid = self.icnn(x_mid).item()

                if phi_mid > lam * phi_i + (1 - lam) * phi_j + 1e-6:
                    violations += 1

        return 1.0 - violations / n_checks

    def ball_coverage(self, x: np.ndarray) -> float:
        """Fraction of Q_hat(x) that lands inside B_d."""
        Q = self.forward_map_np(x)
        norms = np.linalg.norm(Q, axis=1)
        return float(np.mean(norms <= 1.0))

    def output_norm_stats(self, x: np.ndarray) -> Dict[str, float]:
        """Statistics of ||Q_hat(x)|| for diagnostics."""
        Q = self.forward_map_np(x)
        norms = np.linalg.norm(Q, axis=1)
        return {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "max": float(norms.max()),
            "median": float(np.median(norms)),
            "frac_in_ball": float(np.mean(norms <= 1.0)),
        }
