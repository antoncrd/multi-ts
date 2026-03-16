"""Diagnostic metrics: rank uniformity, bound tightness, convergence analysis."""

from typing import Dict

import numpy as np
from scipy import stats


def rank_uniformity_ks(ranks: np.ndarray) -> Dict[str, float]:
    """Per-dimension KS test of rank uniformity against U[0,1].

    Args:
        ranks: shape (n, d), rank-transformed values in (0, 1).

    Returns:
        Dict with 'per_dim_stats', 'per_dim_pvalues', 'min_pvalue', 'max_statistic'.
    """
    n, d = ranks.shape
    ks_stats = []
    ks_pvalues = []

    for j in range(d):
        stat, pval = stats.kstest(ranks[:, j], "uniform")
        ks_stats.append(stat)
        ks_pvalues.append(pval)

    return {
        "per_dim_stats": np.array(ks_stats),
        "per_dim_pvalues": np.array(ks_pvalues),
        "min_pvalue": min(ks_pvalues),
        "max_statistic": max(ks_stats),
    }


def angular_uniformity_test(
    ranks: np.ndarray, n_reference: int = 10000, seed: int = 0
) -> Dict[str, float]:
    """Test angular uniformity of rank vectors.

    Projects rank vectors onto the unit sphere and tests angular distribution.

    Args:
        ranks: shape (n, d), rank-transformed values in (0, 1).
        n_reference: Number of reference samples.
        seed: Random seed.

    Returns:
        Dict with KS statistic and p-value for angular distribution.
    """
    # Center ranks around 0.5, project to sphere
    centered = ranks - 0.5
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    directions = centered / norms

    # Reference: uniform directions from centered uniform cube
    rng = np.random.default_rng(seed)
    ref = rng.uniform(-0.5, 0.5, size=(n_reference, ranks.shape[1]))
    ref_norms = np.linalg.norm(ref, axis=1, keepdims=True)
    ref_norms = np.maximum(ref_norms, 1e-10)
    ref_directions = ref / ref_norms

    # Compare distributions of first coordinate (projection onto e_1)
    stat, pval = stats.ks_2samp(directions[:, 0], ref_directions[:, 0])

    return {
        "angular_ks_statistic": stat,
        "angular_ks_pvalue": pval,
    }


def norm_distribution_test(ranks: np.ndarray) -> Dict[str, float]:
    """Test that ||R(eps)||^2 follows expected distribution.

    For uniform ranks on [0,1]^d, ||R||^2 = sum of Uniform(0,1)^2.

    Args:
        ranks: shape (n, d).

    Returns:
        Dict with KS test results.
    """
    norms_sq = (ranks ** 2).sum(axis=1)
    d = ranks.shape[1]

    # Expected distribution: sum of d Uniform(0,1)^2
    # Mean = d/3, Var = d * (1/5 - 1/9) = d * 4/45
    # Use MC reference
    rng = np.random.default_rng(42)
    ref = rng.uniform(0, 1, size=(10000, d))
    ref_norms_sq = (ref ** 2).sum(axis=1)

    stat, pval = stats.ks_2samp(norms_sq, ref_norms_sq)

    return {
        "norm_sq_ks_statistic": stat,
        "norm_sq_ks_pvalue": pval,
        "mean_norm_sq": norms_sq.mean(),
        "expected_mean_norm_sq": d / 3.0,
    }


def bound_tightness(
    empirical_radius: float, theoretical_radius: float
) -> float:
    """Ratio of empirical to theoretical radius.

    Closer to 1 means the theoretical bound is tight.
    Values >> 1 mean the bound is loose (over-conservative).

    Args:
        empirical_radius: Radius that achieves desired coverage empirically.
        theoretical_radius: Radius from the theoretical calibration.

    Returns:
        Tightness ratio.
    """
    if theoretical_radius <= 0:
        return float("inf")
    return theoretical_radius / max(empirical_radius, 1e-12)


def wasserstein_convergence_diagnostics(
    rho_hat_values: np.ndarray,
    k_values: np.ndarray,
    d: int,
) -> Dict[str, float]:
    """Analyze convergence of rho_hat_k vs k.

    Fits a log-log regression to estimate the empirical convergence rate
    and compares with the theoretical rate k^{-1/d}.

    Args:
        rho_hat_values: W_1 values for different k.
        k_values: Corresponding calibration set sizes.
        d: Dimension.

    Returns:
        Dict with fitted slope, theoretical slope, and R^2.
    """
    # Log-log regression: log(rho_hat) = a + b * log(k)
    log_k = np.log(k_values.astype(float))
    log_rho = np.log(np.maximum(rho_hat_values, 1e-12))

    # OLS
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_rho)

    theoretical_slope = -1.0 / d

    return {
        "empirical_slope": slope,
        "theoretical_slope": theoretical_slope,
        "r_squared": r_value ** 2,
        "intercept": intercept,
        "slope_ratio": slope / theoretical_slope if theoretical_slope != 0 else float("inf"),
    }
