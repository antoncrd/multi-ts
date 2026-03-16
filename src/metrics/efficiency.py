"""Efficiency metrics for conformal prediction regions."""

from typing import List

import numpy as np

from src.conformal.rank_region import ConformalRegion


def region_volume(region: ConformalRegion) -> float:
    """Volume of a prediction region."""
    return region.volume()


def log_region_volume(region: ConformalRegion) -> float:
    """Log-volume (numerically stable for high d)."""
    return region.log_volume()


def mean_volume(regions: List[ConformalRegion]) -> float:
    """Average volume across regions."""
    return np.mean([r.volume() for r in regions])


def mean_log_volume(regions: List[ConformalRegion]) -> float:
    """Average log-volume across regions."""
    return np.mean([r.log_volume() for r in regions])


def region_diameter(region: ConformalRegion) -> float:
    """Diameter of a prediction region."""
    return region.diameter()


def mean_diameter(regions: List[ConformalRegion]) -> float:
    """Average diameter across regions."""
    return np.mean([r.diameter() for r in regions])


def winkler_multivariate(
    region: ConformalRegion,
    y_true: np.ndarray,
    alpha: float,
) -> float:
    """Multivariate Winkler score.

    S_W = Vol(C) + (2/alpha) * Vol(C) * 1{Y not in C}

    Lower is better: small volume + coverage.

    Args:
        region: Prediction region.
        y_true: True value, shape (d,).
        alpha: Miscoverage level.

    Returns:
        Winkler score.
    """
    vol = region.volume()
    if not region.contains(y_true):
        return vol + (2.0 / alpha) * vol
    return vol


def winkler_multivariate_log(
    region: ConformalRegion,
    y_true: np.ndarray,
    alpha: float,
) -> float:
    """Log-scale Winkler score (stable for high d).

    Uses log-volume to avoid underflow/overflow.
    """
    log_vol = region.log_volume()
    if not region.contains(y_true):
        # log(vol + 2/alpha * vol) = log(vol * (1 + 2/alpha)) = log_vol + log(1 + 2/alpha)
        return log_vol + np.log(1.0 + 2.0 / alpha)
    return log_vol


def mean_winkler(
    regions: List[ConformalRegion],
    Y_true: np.ndarray,
    alpha: float,
) -> float:
    """Average Winkler score across test set."""
    T = len(regions)
    scores = [
        winkler_multivariate(regions[t], Y_true[t], alpha)
        for t in range(T)
    ]
    return np.mean(scores)


def mean_winkler_log(
    regions: List[ConformalRegion],
    Y_true: np.ndarray,
    alpha: float,
) -> float:
    """Average log-Winkler score (stable for high d)."""
    T = len(regions)
    scores = [
        winkler_multivariate_log(regions[t], Y_true[t], alpha)
        for t in range(T)
    ]
    return np.mean(scores)


def size_stratified_efficiency(
    regions: List[ConformalRegion],
    Y_true: np.ndarray,
    alpha: float,
    n_bins: int = 5,
) -> dict:
    """Efficiency stratified by region size (volume).

    Args:
        regions: List of prediction regions.
        Y_true: True values, shape (T, d).
        alpha: Miscoverage level.
        n_bins: Number of size bins.

    Returns:
        Dict with bin edges, mean volume per bin, coverage per bin.
    """
    volumes = np.array([r.volume() for r in regions])
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(volumes, quantiles)

    bin_vols = []
    bin_covs = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (volumes >= lo) & (volumes < hi)
        else:
            mask = (volumes >= lo) & (volumes <= hi)

        if mask.sum() == 0:
            bin_vols.append(np.nan)
            bin_covs.append(np.nan)
        else:
            bin_vols.append(volumes[mask].mean())
            covered = sum(
                regions[t].contains(Y_true[t])
                for t in range(len(regions))
                if mask[t]
            )
            bin_covs.append(covered / mask.sum())

    return {
        "bin_edges": bin_edges,
        "mean_volumes": np.array(bin_vols),
        "coverages": np.array(bin_covs),
    }
