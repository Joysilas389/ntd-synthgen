"""
Aggregate raw benchmark cells into publication-ready summary tables.

Produces:
  study/results/summary_by_backend.csv       - mean over all cells per backend
  study/results/summary_by_backend_size.csv  - split by dataset size
  study/results/summary_per_dataset.csv      - per (dataset, n_rows, backend)
  study/results/headline_table.md            - markdown table for the paper
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "study" / "results"


METRIC_COLS = [
    "ks_mean", "wasserstein_mean", "tvd_mean", "corr_distance",
    "tstr_r2_ratio", "tstr_f1_ratio",
    "fit_seconds", "sample_seconds", "disk_bytes",
]

PRETTY = {
    "ours_copula": "Ours (Gaussian Copula)",
    "sdv_copula":  "SDV Gaussian Copula",
    "ctgan":       "CTGAN",
    "tvae":        "TVAE",
}

BACKEND_ORDER = ["ours_copula", "sdv_copula", "ctgan", "tvae"]


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "benchmark.csv")
    return df[df["status"] == "ok"].copy()


def aggregate_by_backend(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("backend")[METRIC_COLS].agg(["mean", "std"]).round(4)
    # Reorder rows
    g = g.reindex([b for b in BACKEND_ORDER if b in g.index])
    return g


def aggregate_by_backend_and_size(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["backend", "n_rows"])[METRIC_COLS].mean().round(4)
    return g


def aggregate_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "n_rows", "backend"])[METRIC_COLS].mean().round(4)
    return g


def headline_markdown(df: pd.DataFrame) -> str:
    """The single most-important table for the paper."""
    means = df.groupby("backend")[METRIC_COLS].mean()
    means = means.reindex([b for b in BACKEND_ORDER if b in means.index])

    lines = []
    lines.append("# Headline benchmark results")
    lines.append("")
    lines.append("Mean across all 18 evaluation cells (3 datasets × 2 sizes × 3 seeds).")
    lines.append("Lower is better for fidelity columns; higher is better for utility ratios.")
    lines.append("")
    lines.append("| Backend | KS↓ | W₁↓ | TVD↓ | CorrΔ↓ | R² ratio↑ | F1 ratio↑ | Fit (s)↓ | Sample (s)↓ | Disk (KB)↓ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for backend in means.index:
        row = means.loc[backend]
        lines.append(
            f"| **{PRETTY[backend]}** "
            f"| {row['ks_mean']:.3f} "
            f"| {row['wasserstein_mean']:.3f} "
            f"| {row['tvd_mean']:.3f} "
            f"| {row['corr_distance']:.3f} "
            f"| {row['tstr_r2_ratio']:.3f} "
            f"| {row['tstr_f1_ratio']:.3f} "
            f"| {row['fit_seconds']:.2f} "
            f"| {row['sample_seconds']:.2f} "
            f"| {row['disk_bytes']/1024:.0f} |"
        )
    lines.append("")
    lines.append("KS = Kolmogorov-Smirnov statistic (mean across numeric columns). ")
    lines.append("W₁ = standardised Wasserstein-1 distance. ")
    lines.append("TVD = total variation distance (categorical columns). ")
    lines.append("CorrΔ = normalised Frobenius distance between Pearson correlation matrices. ")
    lines.append("R² ratio = Train-on-Synthetic-Test-on-Real R² ÷ real-on-real R². ")
    lines.append("F1 ratio = same idea for outbreak classification (cases > 75th percentile).")
    return "\n".join(lines)


def per_size_markdown(df: pd.DataFrame) -> str:
    """Show the data-size dependence — does CTGAN benefit from more rows?"""
    g = df.groupby(["backend", "n_rows"])[METRIC_COLS].mean().round(3)
    lines = []
    lines.append("\n# Effect of training-set size")
    lines.append("")
    lines.append("Does the picture change between n=300 and n=1000?")
    lines.append("")
    lines.append("| Backend | n | KS↓ | W₁↓ | CorrΔ↓ | R² ratio↑ | F1 ratio↑ | Fit (s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for backend in BACKEND_ORDER:
        for n in [300, 1000]:
            if (backend, n) in g.index:
                row = g.loc[(backend, n)]
                lines.append(
                    f"| {PRETTY[backend]} | {n} "
                    f"| {row['ks_mean']:.3f} "
                    f"| {row['wasserstein_mean']:.3f} "
                    f"| {row['corr_distance']:.3f} "
                    f"| {row['tstr_r2_ratio']:.3f} "
                    f"| {row['tstr_f1_ratio']:.3f} "
                    f"| {row['fit_seconds']:.2f} |"
                )
    return "\n".join(lines)


def per_dataset_markdown(df: pd.DataFrame) -> str:
    g = df.groupby(["dataset", "backend"])[METRIC_COLS].mean().round(3)
    lines = []
    lines.append("\n# Per-dataset breakdown (averaged over both sizes and seeds)")
    lines.append("")
    for dataset in sorted(df["dataset"].unique()):
        lines.append(f"\n## {dataset}\n")
        lines.append("| Backend | KS↓ | W₁↓ | TVD↓ | CorrΔ↓ | R² ratio↑ | F1 ratio↑ |")
        lines.append("|---|---|---|---|---|---|---|")
        for backend in BACKEND_ORDER:
            if (dataset, backend) in g.index:
                row = g.loc[(dataset, backend)]
                lines.append(
                    f"| {PRETTY[backend]} "
                    f"| {row['ks_mean']:.3f} "
                    f"| {row['wasserstein_mean']:.3f} "
                    f"| {row['tvd_mean']:.3f} "
                    f"| {row['corr_distance']:.3f} "
                    f"| {row['tstr_r2_ratio']:.3f} "
                    f"| {row['tstr_f1_ratio']:.3f} |"
                )
    return "\n".join(lines)


def main():
    df = load()
    print(f"Loaded {len(df)} successful benchmark cells")

    by_backend = aggregate_by_backend(df)
    by_backend.to_csv(RESULTS / "summary_by_backend.csv")

    by_size = aggregate_by_backend_and_size(df)
    by_size.to_csv(RESULTS / "summary_by_backend_size.csv")

    per_ds = aggregate_per_dataset(df)
    per_ds.to_csv(RESULTS / "summary_per_dataset.csv")

    md = headline_markdown(df) + per_size_markdown(df) + per_dataset_markdown(df)
    (RESULTS / "headline_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
