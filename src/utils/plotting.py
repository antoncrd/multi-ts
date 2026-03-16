"""Plotting utilities for the 5 hero plots + ablation figures.

NeurIPS style: white background, Computer Modern font,
single-col 3.25 inches, double-col 6.75 inches.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_style():
    """Configure matplotlib for NeurIPS paper figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
        "text.usetex": False,  # Set True if LaTeX is available
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# Standard figure sizes (inches)
SINGLE_COL = (3.25, 2.5)
DOUBLE_COL = (6.75, 3.5)


# ─── Hero Plot 1: Heatmap d × k ──────────────────────────────────────────────

def plot_coverage_heatmap(
    results_df: pd.DataFrame,
    rho_A_fixed: float,
    alpha: float = 0.1,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Hero Plot 1: coverage gap heatmap over d and k.

    Args:
        results_df: DataFrame with columns 'd', 'k', 'rho_A', 'coverage_gap'.
        rho_A_fixed: Fixed rho_A value to filter on.
        alpha: Target miscoverage level.
        output_path: Path to save figure (optional).
        title: Custom title.

    Returns:
        matplotlib Figure.
    """
    setup_style()

    df = results_df[results_df["rho_A"] == rho_A_fixed].copy()
    pivot = df.pivot_table(values="coverage_gap", index="d", columns="k", aggfunc="mean")

    # Sort axes
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    # Diverging colormap: blue=under, white=exact, red=over
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.05)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "Coverage gap"},
        linewidths=0.5,
    )

    ax.set_xlabel("Calibration set size $k$")
    ax.set_ylabel("Dimension $d$")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Coverage gap ($\\rho(A)={rho_A_fixed}$, $\\alpha={alpha}$)"
        )

    if output_path:
        fig.savefig(output_path)
    return fig


# ─── Hero Plot 2: Coverage vs ρ(A) ───────────────────────────────────────────

def plot_coverage_vs_rho(
    results_df: pd.DataFrame,
    d_fixed: int,
    alpha: float = 0.1,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Hero Plot 2: coverage vs mixing strength rho(A) for various k.

    Args:
        results_df: DataFrame with columns 'rho_A', 'k', 'coverage'.
        d_fixed: Fixed dimension to filter on.
        alpha: Target miscoverage level.
        output_path: Path to save figure.
    """
    setup_style()
    df = results_df[results_df["d"] == d_fixed].copy()

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    k_values = sorted(df["k"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_values)))

    for k_val, color in zip(k_values, colors):
        sub = df[df["k"] == k_val].sort_values("rho_A")
        ax.plot(
            sub["rho_A"], sub["coverage"],
            marker="o", markersize=3, label=f"$k={k_val}$", color=color,
        )

    # Target band
    target = 1 - alpha
    ax.axhspan(target - 0.02, target + 0.02, alpha=0.2, color="gray")
    ax.axhline(target, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Spectral radius $\\rho(A)$")
    ax.set_ylabel("Coverage")
    ax.set_title(f"Coverage vs. mixing ($d={d_fixed}$)")
    ax.legend(fontsize=6, ncol=2, loc="lower left")

    if output_path:
        fig.savefig(output_path)
    return fig


# ─── Hero Plot 3: Region geometry (d=2) ──────────────────────────────────────

def plot_region_geometry_2d(
    residuals: np.ndarray,
    regions_dict: dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Hero Plot 3: 2D scatter of residuals with region contours.

    Args:
        residuals: Test residuals, shape (n, 2).
        regions_dict: {method_name: ConformalRegion} for d=2.
        output_path: Path to save figure.
    """
    setup_style()
    assert residuals.shape[1] == 2

    n_methods = len(regions_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(n_methods * 2.5, 2.5))
    if n_methods == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, (name, region), color in zip(axes, regions_dict.items(), colors):
        ax.scatter(
            residuals[:, 0], residuals[:, 1],
            s=1, alpha=0.3, color="gray", rasterized=True,
        )

        # Draw region boundary
        boundary = region.sample_boundary(500)
        # Sort by angle for a clean contour
        angles = np.arctan2(
            boundary[:, 1] - region.center[1],
            boundary[:, 0] - region.center[0],
        )
        order = np.argsort(angles)
        boundary = boundary[order]
        ax.plot(
            np.append(boundary[:, 0], boundary[0, 0]),
            np.append(boundary[:, 1], boundary[0, 1]),
            color=color, linewidth=1.5, label=name,
        )
        ax.scatter(*region.center, color=color, marker="+", s=50, zorder=5)

        ax.set_title(name, fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


# ─── Hero Plot 4: Log-log ρ̂_k vs k ─────────────────────────────────────────

def plot_wasserstein_convergence(
    results_df: pd.DataFrame,
    rho_A_fixed: float = 0.5,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Hero Plot 4: log-log plot of W_1 convergence.

    Args:
        results_df: DataFrame with columns 'd', 'k', 'rho_hat_k'.
        rho_A_fixed: Fixed rho_A.
        output_path: Path to save figure.
    """
    setup_style()
    df = results_df[results_df["rho_A"] == rho_A_fixed].copy()

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    d_values = sorted(df["d"].unique())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(d_values)))

    for d_val, color in zip(d_values, colors):
        sub = df[df["d"] == d_val].sort_values("k")
        ax.loglog(
            sub["k"], sub["rho_hat_k"],
            marker="o", markersize=3, label=f"$d={d_val}$", color=color,
        )

        # Reference line: k^{-1/d}
        k_ref = np.array(sorted(sub["k"].unique()), dtype=float)
        rate = k_ref ** (-1.0 / d_val)
        # Scale to match first point
        if len(sub) > 0:
            scale = sub["rho_hat_k"].iloc[0] / rate[0]
            ax.loglog(
                k_ref, scale * rate,
                linestyle="--", color=color, alpha=0.5, linewidth=0.8,
            )

    ax.set_xlabel("Calibration set size $k$")
    ax.set_ylabel("$\\hat{\\rho}_k = W_1(\\hat{\\mu}_k, U_d)$")
    ax.set_title(f"Wasserstein convergence ($\\rho(A)={rho_A_fixed}$)")
    ax.legend(fontsize=6, ncol=2)

    if output_path:
        fig.savefig(output_path)
    return fig


# ─── Hero Plot 5: Pareto front (Volume vs Coverage) ─────────────────────────

def plot_pareto_front(
    results_df: pd.DataFrame,
    d_fixed: int,
    k_fixed: int,
    method_col: str = "method",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Hero Plot 5: Pareto front of volume vs coverage.

    Args:
        results_df: DataFrame with columns 'method', 'coverage', 'mean_log_volume'.
        d_fixed: Fixed dimension.
        k_fixed: Fixed calibration size.
        output_path: Path to save figure.
    """
    setup_style()
    df = results_df[
        (results_df["d"] == d_fixed) & (results_df["k"] == k_fixed)
    ].copy()

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    markers = ["o", "s", "^", "D", "v", "P", "*"]
    methods = df[method_col].unique()

    for method, marker in zip(methods, markers):
        sub = df[df[method_col] == method]
        ax.scatter(
            sub["coverage"], sub["mean_log_volume"],
            marker=marker, s=40, label=method, zorder=5,
        )

    # Pareto front
    if len(df) > 1:
        coverages = df["coverage"].values
        volumes = df["mean_log_volume"].values
        # Pareto-optimal: higher coverage AND lower volume
        pareto_mask = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j:
                    if coverages[j] >= coverages[i] and volumes[j] <= volumes[i]:
                        if coverages[j] > coverages[i] or volumes[j] < volumes[i]:
                            pareto_mask[i] = False
                            break
        if pareto_mask.sum() > 1:
            pareto_pts = df[pareto_mask].sort_values("coverage")
            ax.plot(
                pareto_pts["coverage"], pareto_pts["mean_log_volume"],
                "k--", alpha=0.5, linewidth=0.8, label="Pareto front",
            )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Mean log-volume")
    ax.set_title(f"Efficiency–validity tradeoff ($d={d_fixed}$, $k={k_fixed}$)")
    ax.legend(fontsize=6)

    if output_path:
        fig.savefig(output_path)
    return fig
