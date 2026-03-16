"""Generate all paper figures from experiment result CSVs.

Usage:
    python scripts/generate_figures.py --results_dir results/sweep/2026-01-01 --output_dir figures/
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.plotting import (
    plot_coverage_heatmap,
    plot_coverage_vs_rho,
    plot_pareto_front,
    plot_wasserstein_convergence,
    setup_style,
)


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all results.json files from a Hydra sweep directory into a DataFrame."""
    results_dir = Path(results_dir)
    records = []

    for json_path in results_dir.rglob("results.json"):
        with open(json_path) as f:
            data = json.load(f)

        record = {
            **data["config"],
            **data["calibration"],
            **data["metrics"],
            **data.get("ot_metrics", {}),
            "elapsed_seconds": data.get("elapsed_seconds", None),
        }
        records.append(record)

    if not records:
        raise FileNotFoundError(f"No results.json found in {results_dir}")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} experiment results")
    print(f"Dimensions: {sorted(df['d'].unique())}")
    print(f"rho_A values: {sorted(df['rho_A'].unique())}")
    print(f"k values: {sorted(df['k'].unique())}")

    setup_style()

    # Hero Plot 1: Coverage heatmap for each rho_A
    for rho_A in sorted(df["rho_A"].unique()):
        fig = plot_coverage_heatmap(
            df, rho_A_fixed=rho_A, alpha=args.alpha,
            output_path=str(output_dir / f"heatmap_rhoA_{rho_A}.pdf"),
        )
        print(f"  Saved heatmap for rho_A={rho_A}")

    # Hero Plot 2: Coverage vs rho(A) for select dimensions
    for d in [4, 8, 16]:
        if d in df["d"].values:
            fig = plot_coverage_vs_rho(
                df, d_fixed=d, alpha=args.alpha,
                output_path=str(output_dir / f"coverage_vs_rho_d{d}.pdf"),
            )
            print(f"  Saved coverage vs rho for d={d}")

    # Hero Plot 4: Wasserstein convergence
    for rho_A in [0.5]:
        if "rho_hat_k" in df.columns:
            fig = plot_wasserstein_convergence(
                df, rho_A_fixed=rho_A,
                output_path=str(output_dir / f"wasserstein_convergence_rhoA_{rho_A}.pdf"),
            )
            print(f"  Saved Wasserstein convergence for rho_A={rho_A}")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
