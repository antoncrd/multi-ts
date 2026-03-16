"""Coverage metrics for conformal prediction regions."""

from typing import List, Optional

import numpy as np

from src.conformal.rank_region import ConformalRegion


def marginal_coverage(
    regions: List[ConformalRegion],
    Y_true: np.ndarray,
) -> float:
    """Marginal coverage: fraction of test points contained in their region.

    Args:
        regions: List of T ConformalRegion instances.
        Y_true: True values, shape (T, d).

    Returns:
        Coverage rate in [0, 1].
    """
    T = len(regions)
    assert Y_true.shape[0] == T
    covered = sum(
        regions[t].contains(Y_true[t]) for t in range(T)
    )
    return covered / T


def coverage_gap(achieved: float, target: float) -> float:
    """Coverage gap: achieved - (1 - alpha).

    Positive = over-coverage (conservative), negative = under-coverage.
    """
    return achieved - target


def violation_rate(
    regions: List[ConformalRegion],
    Y_true: np.ndarray,
) -> float:
    """Violation rate: 1 - marginal_coverage. Should be <= alpha."""
    return 1.0 - marginal_coverage(regions, Y_true)


def conditional_coverage(
    regions: List[ConformalRegion],
    Y_true: np.ndarray,
    conditioning_values: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """Coverage conditional on binned values of a conditioning variable.

    Bins the conditioning variable into quantiles and computes coverage per bin.

    Args:
        regions: List of T ConformalRegion instances.
        Y_true: True values, shape (T, d).
        conditioning_values: 1D array of shape (T,) to bin on.
            Typically ||Y_hat|| or estimated volatility.
        n_bins: Number of quantile bins.

    Returns:
        Dict with 'bin_edges', 'bin_coverages', 'bin_counts'.
    """
    T = len(regions)
    assert Y_true.shape[0] == T
    assert conditioning_values.shape[0] == T

    # Compute quantile bin edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(conditioning_values, quantiles)

    bin_coverages = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (conditioning_values >= lo) & (conditioning_values < hi)
        else:
            mask = (conditioning_values >= lo) & (conditioning_values <= hi)

        count = mask.sum()
        if count == 0:
            bin_coverages.append(np.nan)
        else:
            covered = sum(
                regions[t].contains(Y_true[t])
                for t in range(T)
                if mask[t]
            )
            bin_coverages.append(covered / count)
        bin_counts.append(count)

    return {
        "bin_edges": bin_edges,
        "bin_coverages": np.array(bin_coverages),
        "bin_counts": np.array(bin_counts),
    }


def rolling_coverage(
    regions: List[ConformalRegion],
    Y_true: np.ndarray,
    n_windows: int = 5,
) -> np.ndarray:
    """Coverage computed over rolling windows.

    Splits the test set into n_windows consecutive windows and
    computes marginal coverage for each.

    Args:
        regions: List of T ConformalRegion instances.
        Y_true: True values, shape (T, d).
        n_windows: Number of rolling windows.

    Returns:
        Array of coverages, shape (n_windows,).
    """
    T = len(regions)
    window_size = T // n_windows
    coverages = []

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else T
        window_regions = regions[start:end]
        window_Y = Y_true[start:end]
        cov = marginal_coverage(window_regions, window_Y)
        coverages.append(cov)

    return np.array(coverages)
