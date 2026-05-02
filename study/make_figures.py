"""
Generate publication figures from the benchmark results.

Outputs (study/figures/):
  fig1_fidelity_comparison.png    - bar chart: KS / W1 / TVD by backend
  fig2_utility_vs_cost.png        - scatter: TSTR ratio vs fit time, log scale
  fig3_size_effect.png            - line: TSTR ratio vs n_rows for each backend
  fig4_distribution_overlay.png   - real vs synthetic histogram for one column
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "study" / "results"
FIGURES = ROOT / "study" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Editorial palette matching the frontend.
COLORS = {
    "ours_copula": "#8b1e1e",   # oxblood
    "sdv_copula":  "#3b6e8f",   # slate blue
    "ctgan":       "#c97c2b",   # ochre
    "tvae":        "#5a6b4a",   # olive
}
PRETTY = {
    "ours_copula": "Ours (Copula)",
    "sdv_copula":  "SDV Copula",
    "ctgan":       "CTGAN",
    "tvae":        "TVAE",
}
ORDER = ["ours_copula", "sdv_copula", "ctgan", "tvae"]


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })


def fig1_fidelity(df: pd.DataFrame):
    means = df.groupby("backend")[["ks_mean", "wasserstein_mean", "tvd_mean", "corr_distance"]].mean()
    means = means.reindex(ORDER)

    metrics = [
        ("ks_mean", "KS statistic"),
        ("wasserstein_mean", "Std. Wasserstein"),
        ("tvd_mean", "TVD (categorical)"),
        ("corr_distance", "Correlation Δ"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, (col, title) in zip(axes, metrics):
        vals = means[col].values
        bars = ax.bar(
            range(len(ORDER)),
            vals,
            color=[COLORS[b] for b in ORDER],
            edgecolor="#222",
            linewidth=0.8,
        )
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([PRETTY[b] for b in ORDER], rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(vals) * 0.02,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0, max(vals) * 1.20)

    fig.suptitle("Figure 1 — Fidelity (lower is better)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIGURES / "fig1_fidelity_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fig2_utility_vs_cost(df: pd.DataFrame):
    g = df.groupby("backend")[["fit_seconds", "tstr_r2_ratio", "tstr_f1_ratio"]].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    for backend in ORDER:
        if backend not in g.index:
            continue
        x = max(g.loc[backend, "fit_seconds"], 0.001)  # log scale needs > 0
        y = g.loc[backend, "tstr_r2_ratio"]
        ax.scatter(x, y, s=200, color=COLORS[backend], edgecolor="#222",
                   linewidth=1.2, zorder=3, label=PRETTY[backend])
        ax.annotate(
            PRETTY[backend],
            (x, y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Fit time (seconds, log scale)")
    ax.set_ylabel("TSTR R² ratio   (synthetic ÷ real baseline)")
    ax.axhline(1.0, linestyle="--", color="#888", linewidth=0.8, label="Perfect parity")
    ax.axhline(0.0, linestyle=":", color="#888", linewidth=0.6)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Figure 2 — Downstream utility vs computational cost")

    out = FIGURES / "fig2_utility_vs_cost.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fig3_size_effect(df: pd.DataFrame):
    g = df.groupby(["backend", "n_rows"])[["tstr_r2_ratio", "tstr_f1_ratio"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, col, title in zip(axes,
                              ["tstr_r2_ratio", "tstr_f1_ratio"],
                              ["TSTR R² ratio (regression)", "TSTR F1 ratio (outbreak detection)"]):
        for backend in ORDER:
            ys = []
            xs = []
            for n in [300, 1000]:
                if (backend, n) in g.index:
                    xs.append(n)
                    ys.append(g.loc[(backend, n), col])
            if xs:
                ax.plot(xs, ys, marker="o", markersize=8, linewidth=2,
                        color=COLORS[backend], label=PRETTY[backend])
        ax.axhline(1.0, linestyle="--", color="#888", linewidth=0.8)
        ax.axhline(0.0, linestyle=":", color="#888", linewidth=0.6)
        ax.set_xlabel("Training set size (rows)")
        ax.set_ylabel(title)
        ax.set_xticks([300, 1000])
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(title)
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)

    fig.suptitle("Figure 3 — Effect of training set size on downstream utility", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIGURES / "fig3_size_effect.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fig4_distribution_overlay():
    """Show real vs synthetic for rainfall_mm under each backend."""
    from core.seed_data import water_borne
    from study.backends import OursBackend, SDVCopulaBackend, CTGANBackend, TVAEBackend

    real = water_borne(n=1000, seed=42)
    n = len(real)

    backends = {
        "ours_copula": OursBackend(),
        "sdv_copula": SDVCopulaBackend(),
        "ctgan": CTGANBackend(epochs=100),
        "tvae": TVAEBackend(epochs=100),
    }

    samples = {}
    for name, b in backends.items():
        b.fit(real)
        samples[name] = b.sample(n)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    col = "rainfall_mm"
    bins = np.linspace(real[col].min(), real[col].max(), 30)

    for ax, (name, syn) in zip(axes, samples.items()):
        ax.hist(real[col], bins=bins, color="#888", alpha=0.5, label="Real", edgecolor="#444", linewidth=0.5)
        ax.hist(syn[col], bins=bins, color=COLORS[name], alpha=0.55, label="Synthetic", edgecolor="#222", linewidth=0.5)
        ax.set_title(PRETTY[name])
        ax.set_xlabel(col)
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Frequency")
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)

    fig.suptitle("Figure 4 — Marginal distribution of rainfall_mm: real vs synthetic", fontsize=12, y=1.03)
    fig.tight_layout()
    out = FIGURES / "fig4_distribution_overlay.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    _setup_style()
    df = pd.read_csv(RESULTS / "benchmark.csv")
    df = df[df["status"] == "ok"].copy()
    fig1_fidelity(df)
    fig2_utility_vs_cost(df)
    fig3_size_effect(df)
    fig4_distribution_overlay()


if __name__ == "__main__":
    main()
