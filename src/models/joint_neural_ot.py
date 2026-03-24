"""Joint Neural OT for Conformal Time Series (Algorithm 1 from the paper).

Implements joint training of:
- Q_theta (quantile map): B_d -> R^d, MLP
- R_phi (rank map): R^d -> B_d, MLP with ball projection

Three-loss objective:
- L_OT: transport fidelity via mini-batch optimal assignment
- L_cyc: cycle consistency ||Q(R(eps)) - eps||^2 + ||R(Q(u)) - u||^2
- L_unif: rank uniformity via MMD against U(B_d)
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from src.models.neural_ot import sample_uniform_ball


# ---------------------------------------------------------------------------
# Ball projection
# ---------------------------------------------------------------------------

def project_to_ball(x: torch.Tensor) -> torch.Tensor:
    """Map points strictly inside the open unit ball via tanh rescaling.

    r -> tanh(r) * (x / ||x||), so ||output|| = tanh(||x||) < 1 strictly.
    """
    norms = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return x / norms * torch.tanh(norms)


# ---------------------------------------------------------------------------
# Quantile map Q_theta: B_d -> R^d
# ---------------------------------------------------------------------------

class QuantileMap(nn.Module):
    """MLP mapping points in B_d to residual space R^d."""

    def __init__(self, dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers: list = []
        prev = dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Map B_d -> R^d."""
        return self.net(u)


# ---------------------------------------------------------------------------
# Rank map R_phi: R^d -> B_d
# ---------------------------------------------------------------------------

class RankMap(nn.Module):
    """MLP mapping residuals to the unit ball B_d.

    Output is projected onto B_d via x / max(||x||, 1).
    """

    def __init__(self, dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers: list = []
        prev = dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        """Map residuals to B_d."""
        return project_to_ball(self.net(eps))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_lot_hungarian(
    q_batch: torch.Tensor, eps_batch: torch.Tensor
) -> torch.Tensor:
    """Transport fidelity loss via Hungarian optimal assignment.

    L_OT = (1/(B*d)) sum_i ||Q(u_sigma(i)) - eps_i||^2
    where sigma is the optimal assignment minimizing total cost.

    The assignment is computed on detached tensors; the matched cost
    retains gradients through Q.
    """
    with torch.no_grad():
        C = torch.cdist(q_batch, eps_batch, p=2).pow(2)
        row_idx, col_idx = linear_sum_assignment(C.cpu().numpy())

    row_idx = torch.tensor(row_idx, device=q_batch.device, dtype=torch.long)
    col_idx = torch.tensor(col_idx, device=q_batch.device, dtype=torch.long)
    matched_q = q_batch[row_idx]
    matched_eps = eps_batch[col_idx]
    return ((matched_q - matched_eps) ** 2).mean()


def compute_lcyc(
    quantile_map: QuantileMap,
    rank_map: RankMap,
    eps_batch: torch.Tensor,
    u_batch: torch.Tensor,
) -> torch.Tensor:
    """Cycle consistency loss.

    L_cyc = ||Q(R(eps)) - eps||^2 + ||R(Q(u)) - u||^2
    """
    # Forward cycle: eps -> R -> Q -> should reconstruct eps
    r_eps = rank_map(eps_batch)
    q_r_eps = quantile_map(r_eps)
    fwd_loss = ((q_r_eps - eps_batch) ** 2).mean()

    # Backward cycle: u -> Q -> R -> should reconstruct u
    q_u = quantile_map(u_batch)
    r_q_u = rank_map(q_u)
    bwd_loss = ((r_q_u - u_batch) ** 2).mean()

    return fwd_loss + bwd_loss


def _gaussian_kernel_matrix(
    X: torch.Tensor, Y: torch.Tensor, bandwidth: float
) -> torch.Tensor:
    """Gaussian kernel k(x,y) = exp(-||x-y||^2 / (2*bw^2))."""
    dists_sq = torch.cdist(X, Y, p=2).pow(2)
    return torch.exp(-dists_sq / (2.0 * bandwidth ** 2))


def compute_lunif_mmd(
    r_batch: torch.Tensor, u_batch: torch.Tensor
) -> torch.Tensor:
    """Uniformity loss via MMD^2 with multi-bandwidth Gaussian kernel.

    MMD^2 = E[k(r,r')] + E[k(u,u')] - 2 E[k(r,u)]
    """
    with torch.no_grad():
        all_pts = torch.cat([r_batch, u_batch], dim=0)
        pairwise = torch.cdist(all_pts, all_pts, p=2)
        median_dist = pairwise.median().clamp(min=1e-4).item()

    bandwidths = [0.5 * median_dist, median_dist, 2.0 * median_dist]

    mmd_sq = torch.tensor(0.0, device=r_batch.device)
    for bw in bandwidths:
        K_rr = _gaussian_kernel_matrix(r_batch, r_batch, bw)
        K_uu = _gaussian_kernel_matrix(u_batch, u_batch, bw)
        K_ru = _gaussian_kernel_matrix(r_batch, u_batch, bw)
        mmd_sq = mmd_sq + K_rr.mean() + K_uu.mean() - 2.0 * K_ru.mean()

    return mmd_sq / len(bandwidths)


# ---------------------------------------------------------------------------
# JointNeuralOTMap
# ---------------------------------------------------------------------------

class JointNeuralOTMap:
    """Joint Neural OT map implementing Algorithm 1.

    Trains Q_theta (quantile map) and R_phi (rank map) simultaneously
    with transport fidelity, cycle consistency, and uniformity losses.

    Interface:
        forward_map_np(residuals) -> ranks in B_d  (R_phi)
        inverse_map_np(u)         -> residuals      (Q_theta)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims_q: Optional[List[int]] = None,
        hidden_dims_r: Optional[List[int]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 500,
        batch_size: int = 256,
        lambda_cyc: float = 10.0,
        lambda_unif: float = 1.0,
        grad_clip: float = 1.0,
        warmup_epochs: int = 50,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.input_dim = input_dim

        self.quantile_map = QuantileMap(
            input_dim, hidden_dims_q
        ).to(self.device)
        self.rank_map = RankMap(input_dim, hidden_dims_r).to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda_cyc = lambda_cyc
        self.lambda_unif = lambda_unif
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs

        # Normalization parameters (set during fit)
        self._source_mean: Optional[torch.Tensor] = None
        self._source_std: Optional[torch.Tensor] = None

    # -------------------------------------------------------------------
    # Normalization (same pattern as NeuralOTMap)
    # -------------------------------------------------------------------

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._source_mean) / self._source_std

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self._source_std + self._source_mean

    # -------------------------------------------------------------------
    # Training — Algorithm 1
    # -------------------------------------------------------------------

    def fit(
        self, source_samples: np.ndarray, verbose: bool = True
    ) -> Dict[str, list]:
        """Train (Q_theta, R_phi) jointly on training residuals.

        Args:
            source_samples: shape (n, d), training residuals.
            verbose: Show progress bar.

        Returns:
            Dict with loss histories: total, lot, lcyc, lunif.
        """
        d = self.input_dim
        source_t = torch.tensor(
            source_samples, dtype=torch.float32, device=self.device
        )

        # Store normalization params
        self._source_mean = source_t.mean(dim=0, keepdim=True)
        self._source_std = source_t.std(dim=0, keepdim=True).clamp(min=1e-6)

        source_norm = self._normalize(source_t)

        optimizer = torch.optim.Adam(
            list(self.quantile_map.parameters())
            + list(self.rank_map.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / max(self.warmup_epochs, 1)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        dataset = TensorDataset(source_norm)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        history = {"total": [], "lot": [], "lcyc": [], "lunif": []}

        rng_iter = range(self.n_epochs)
        if verbose:
            rng_iter = trange(self.n_epochs, desc="Joint OT (Q,R)")

        for epoch in rng_iter:
            ep_total, ep_lot, ep_cyc, ep_unif, ep_n = 0.0, 0.0, 0.0, 0.0, 0

            for (eps_batch,) in loader:
                optimizer.zero_grad()
                bs = eps_batch.shape[0]

                # Sample u ~ U(B_d)
                u_batch = sample_uniform_ball(bs, d, self.device)

                # --- L_OT: transport fidelity ---
                q_u = self.quantile_map(u_batch)
                loss_ot = compute_lot_hungarian(q_u, eps_batch)

                # --- L_cyc: cycle consistency ---
                loss_cyc = compute_lcyc(
                    self.quantile_map, self.rank_map, eps_batch, u_batch
                )

                # --- L_unif: rank uniformity ---
                r_eps = self.rank_map(eps_batch)
                u_ref = sample_uniform_ball(bs, d, self.device)
                loss_unif = compute_lunif_mmd(r_eps, u_ref)

                # Total loss
                loss = (
                    loss_ot
                    + self.lambda_cyc * loss_cyc
                    + self.lambda_unif * loss_unif
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.quantile_map.parameters())
                    + list(self.rank_map.parameters()),
                    self.grad_clip,
                )
                optimizer.step()
                scheduler.step()

                ep_total += loss.item() * bs
                ep_lot += loss_ot.item() * bs
                ep_cyc += loss_cyc.item() * bs
                ep_unif += loss_unif.item() * bs
                ep_n += bs

            if ep_n > 0:
                history["total"].append(ep_total / ep_n)
                history["lot"].append(ep_lot / ep_n)
                history["lcyc"].append(ep_cyc / ep_n)
                history["lunif"].append(ep_unif / ep_n)

            if verbose and hasattr(rng_iter, "set_postfix"):
                rng_iter.set_postfix(
                    L=f"{history['total'][-1]:.4f}",
                    OT=f"{history['lot'][-1]:.4f}",
                    cyc=f"{history['lcyc'][-1]:.4f}",
                    unif=f"{history['lunif'][-1]:.4f}",
                )

        return history

    # -------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """R_phi(normalize(x)): residuals -> ranks in B_d."""
        self.rank_map.eval()
        x_norm = self._normalize(x)
        with torch.no_grad():
            result = self.rank_map(x_norm)
        self.rank_map.train()
        return result

    def inverse_map(self, u: torch.Tensor) -> torch.Tensor:
        """Q_theta(u) denormalized: B_d -> residual space."""
        self.quantile_map.eval()
        with torch.no_grad():
            q_u = self.quantile_map(u)
        self.quantile_map.train()
        return self._denormalize(q_u)

    def forward_map_np(self, x: np.ndarray) -> np.ndarray:
        """Numpy wrapper: residuals -> ranks in B_d."""
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self.forward_map(x_t).cpu().numpy()

    def inverse_map_np(self, y: np.ndarray) -> np.ndarray:
        """Numpy wrapper: B_d -> residual space."""
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        return self.inverse_map(y_t).cpu().numpy()

    def ball_coverage(self, x: np.ndarray) -> float:
        """Fraction of R_phi(x) inside B_d (always 1.0 by construction)."""
        R = self.forward_map_np(x)
        norms = np.linalg.norm(R, axis=1)
        return float(np.mean(norms <= 1.0))

    def output_norm_stats(self, x: np.ndarray) -> Dict[str, float]:
        """Statistics of ||R_phi(x)|| for diagnostics."""
        R = self.forward_map_np(x)
        norms = np.linalg.norm(R, axis=1)
        return {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "max": float(norms.max()),
            "median": float(np.median(norms)),
            "frac_in_ball": float(np.mean(norms <= 1.0)),
        }
