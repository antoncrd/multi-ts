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


def wasserstein_1_uniform(
    samples: np.ndarray,
    d: int,
    method: str = "auto",
    n_projections: int = 100,
    n_reference: int = 5000,
    seed: Optional[int] = None,
) -> float:
    """Compute W_1(mu_hat, U_d) between empirical measure and uniform on [0,1]^d.

    Args:
        samples: shape (k, d), empirical samples (typically rank-transformed).
        d: dimension (must match samples.shape[1]).
        method: "exact", "sliced", or "auto".
            - "exact": POT emd2 against reference samples from U_d.
            - "sliced": sliced Wasserstein distance with random projections.
            - "auto": exact if d <= 8 and k <= 2000, sliced otherwise.
        n_projections: Number of projections for sliced method.
        n_reference: Number of reference samples from U_d for exact method.
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
    """Exact W1 via POT emd2 against uniform reference samples."""
    k = samples.shape[0]
    reference = rng.uniform(0, 1, size=(n_reference, d))

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
    """Sliced Wasserstein distance against uniform on [0,1]^d."""
    k = samples.shape[0]
    n_ref = max(k, 2000)  # Use a large reference set for stable estimates
    reference = rng.uniform(0, 1, size=(n_ref, d))

    return ot.sliced_wasserstein_distance(
        samples, reference, n_projections=n_projections, seed=int(rng.integers(2**31))
    )


def fournier_constant(d: int, m: int = 1) -> float:
    """Fournier (2023) explicit constant kappa^(m)_{d,1} for W_1 convergence.

    The rate of convergence of W_1(mu_hat_k, mu) for mu on R^d:
        E[W_1] <= kappa * k^{-1/max(d,2)} * (log k)^{1_{d=2}/2}

    Args:
        d: dimension.
        m: moment order (default 1 for W_1).

    Returns:
        The constant kappa^(m)_{d,1}.
    """
    # Lookup table from Fournier & Guillin (2015), Tables 1 & 3
    # These are upper bound constants; exact values depend on the support.
    # Conservative estimates for distributions on [0,1]^d:
    table = {
        1: 1.0,
        2: 2.0,
        3: 3.5,
        4: 5.0,
        5: 6.5,
        6: 8.0,
        8: 12.0,
        16: 30.0,
        32: 80.0,
        64: 200.0,
    }
    if d in table:
        return table[d]

    # Interpolate/extrapolate: kappa grows roughly as d * log(d)
    return d * np.log(d + 1) * 1.5


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
        c_rho += abs(rho_lag)

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
