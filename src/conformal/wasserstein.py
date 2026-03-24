"""Wasserstein distance computation, rank transforms, and uniformity tests.

Supports both exact W1 (via POT emd2) and sliced Wasserstein,
switchable via the `method` parameter.
"""

import warnings
from typing import Dict, Optional

import numpy as np
import ot
from scipy import stats
from scipy.spatial.distance import cdist


def rank_transform(residuals: np.ndarray) -> np.ndarray:
    """Transform residuals to marginal ranks in (0, 1).

    For each coordinate j, rank_j(x) = rank(x_j) / (n + 1).

    Args:
        residuals: shape (n, d).

    Returns:
        Ranks in (0, 1), shape (n, d).
    """
    n, d = residuals.shape
    ranks = np.zeros_like(residuals)
    for j in range(d):
        order = np.argsort(np.argsort(residuals[:, j]))
        ranks[:, j] = (order + 1) / (n + 1)
    return ranks


def _sample_uniform_ball(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly from the d-dimensional unit ball B_d(0,1).

    Method: sample direction uniformly on S^{d-1}, then radius r ~ U^{1/d}.
    """
    z = rng.standard_normal((n, d))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    z = z / norms
    u = rng.uniform(0, 1, size=(n, 1))
    r = u ** (1.0 / d)
    return r * z


def wasserstein_1_uniform(
    samples: np.ndarray,
    d: int,
    method: str = "auto",
    n_projections: int = 100,
    n_reference: int = 5000,
    seed: Optional[int] = None,
) -> float:
    """Compute W_1(mu_hat, U(B_d)) between empirical measure and uniform on the unit ball.

    Args:
        samples: shape (k, d), empirical samples (typically rank-transformed).
        d: dimension (must match samples.shape[1]).
        method: "exact", "sliced", or "auto".
            - "exact": POT emd2 against reference samples from U(B_d).
            - "sliced": sliced Wasserstein distance with random projections.
            - "auto": exact if d <= 8 and k <= 2000, sliced otherwise.
        n_projections: Number of projections for sliced method.
        n_reference: Number of reference samples from U(B_d) for exact method.
        seed: Random seed for reference sampling.

    Returns:
        Estimated W_1 distance.
    """
    k = samples.shape[0]
    assert samples.shape[1] == d

    if method == "auto":
        method = "exact" if (d <= 8 and k <= 2000) else "sliced"

    rng = np.random.default_rng(seed)

    if method == "exact":
        return _w1_exact(samples, d, n_reference, rng)
    elif method == "sliced":
        return _w1_sliced(samples, d, n_projections, rng)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact', 'sliced', or 'auto'.")


def _w1_exact(
    samples: np.ndarray,
    d: int,
    n_reference: int,
    rng: np.random.Generator,
) -> float:
    """Exact W1 via POT emd2 against uniform reference samples on B_d."""
    k = samples.shape[0]
    reference = _sample_uniform_ball(n_reference, d, rng)

    # Uniform weights
    a = np.ones(k) / k
    b = np.ones(n_reference) / n_reference

    # Cost matrix (Euclidean distance)
    M = cdist(samples, reference, metric="euclidean")

    return ot.emd2(a, b, M)


def _w1_sliced(
    samples: np.ndarray,
    d: int,
    n_projections: int,
    rng: np.random.Generator,
) -> float:
    """Sliced Wasserstein distance against uniform on B_d."""
    k = samples.shape[0]
    n_ref = max(k, 2000)  # Use a large reference set for stable estimates
    reference = _sample_uniform_ball(n_ref, d, rng)

    return ot.sliced_wasserstein_distance(
        samples, reference, n_projections=n_projections, seed=int(rng.integers(2**31))
    )


def fournier_constant(d: int, m: int = 1) -> float:
    """Fournier (2023) constant kappa^(m)_{d,1} for W_1 convergence.

    Values from Fournier (2023) Table 1 (d <= 9) and Table 3 (d >= 10).
    For dimensions not in the table, values are linearly interpolated
    between the nearest available entries.

    Args:
        d: dimension.
        m: moment order (default 1 for W_1).

    Returns:
        The constant kappa^(m)_{d,1}.
    """
    table = {
        1: 2.42, 2: 0.73, 3: 3.72, 4: 2.45, 5: 2.09,
        6: 1.94, 7: 1.87, 8: 1.84, 9: 1.82,
        # Optimized bounds for larger d (Table 3)
        10: 8.91, 12: 7.67, 15: 6.74, 16: 6.50, 20: 6.00,
        25: 5.60, 32: 5.25, 35: 5.17, 50: 4.85, 64: 4.65,
        75: 4.60, 100: 4.47, 500: 4.12,
    }
    if d in table:
        return table[d]

    # Interpolate between nearest keys
    keys = sorted(table.keys())
    if d < keys[0]:
        return table[keys[0]]
    if d > keys[-1]:
        return table[keys[-1]]
    lo = max(k for k in keys if k <= d)
    hi = min(k for k in keys if k >= d)
    if lo == hi:
        return table[lo]
    frac = (d - lo) / (hi - lo)
    return table[lo] * (1 - frac) + table[hi] * frac


def convergence_rate_term(k: int, d: int) -> float:
    """Compute a_d(k): the dimension-dependent rate term.

    a_d(k) = k^{-1/d} for d >= 3
    a_d(k) = k^{-1/2} * (log k)^{1/2} for d = 2
    a_d(k) = k^{-1} for d = 1 (not used in practice, but included for completeness)
    """
    if d == 1:
        return 1.0 / k
    elif d == 2:
        return (1.0 / np.sqrt(k)) * np.sqrt(np.log(max(k, 2)))
    else:
        return k ** (-1.0 / d)


def estimate_mixing_factor(
    ranks: np.ndarray, max_lag: int = 50
) -> float:
    """Estimate C_rho = sum_{n>=0} rho_n from autocorrelations of rank norms.

    Uses the norm of rank vectors ||R(eps_t)|| as a 1D summary,
    then sums the absolute autocorrelations up to max_lag.
    C_rho = rho_0 + 2 * sum_{n>=1} |rho_n|.

    Args:
        ranks: shape (n, d), rank-transformed residuals.
        max_lag: Maximum lag to consider.

    Returns:
        Estimated C_rho factor.
    """
    norms = np.linalg.norm(ranks, axis=1)
    norms = norms - norms.mean()
    n = len(norms)
    var = np.var(norms)
    if var < 1e-12:
        return 1.0

    c_rho = 1.0  # lag 0 contributes 1
    for lag in range(1, min(max_lag, n // 2)):
        autocov = np.mean(norms[:n - lag] * norms[lag:])
        rho_lag = autocov / var
        if abs(rho_lag) < 2.0 / np.sqrt(n):
            break  # Insignificant autocorrelation
        c_rho += 2.0 * abs(rho_lag)  # factor of 2 for symmetric pairs (i,s) and (s,i)

    return c_rho


def uniformity_test(ranks: np.ndarray) -> Dict[str, float]:
    """Test uniformity of rank-transformed residuals.

    Performs:
    - Per-dimension KS test against U[0,1]
    - Test on ||R||: KS against the CDF of ||U||
      where U ~ Uniform([0,1]^d)

    Args:
        ranks: shape (n, d), values in (0, 1).

    Returns:
        Dict with 'ks_per_dim' (list of p-values), 'ks_norm_pvalue',
        'ks_norm_statistic', 'min_marginal_pvalue'.
    """
    n, d = ranks.shape
    marginal_pvalues = []

    for j in range(d):
        stat, pval = stats.kstest(ranks[:, j], "uniform")
        marginal_pvalues.append(pval)

    # Test on norms: ||U||^2 where U ~ Uniform([0,1]^d)
    # For d-dim uniform on [0,1]^d, ||U||^2 has a known distribution
    # but it's easier to use a Monte Carlo reference
    norms = np.linalg.norm(ranks, axis=1)
    ref_norms = np.linalg.norm(
        np.random.default_rng(0).uniform(0, 1, size=(10000, d)), axis=1
    )
    stat_norm, pval_norm = stats.ks_2samp(norms, ref_norms)

    return {
        "ks_per_dim": marginal_pvalues,
        "min_marginal_pvalue": min(marginal_pvalues),
        "ks_norm_statistic": stat_norm,
        "ks_norm_pvalue": pval_norm,
    }
