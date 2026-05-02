"""End-to-end benchmark runner.

Runs every backend against every disease module and reports both
fidelity metrics (KS, Wasserstein, TVD, correlation distance) and
utility metrics (TSTR R², TSTR macro-F1).

Usage
-----
    python -m bench.run_benchmark

Outputs:
    bench/results/raw_results.json        — full result dump
    bench/results/fidelity_table.csv      — fidelity metrics, one row per backend×module
    bench/results/utility_table.csv       — TSTR ratios for each task
    bench/results/runtime_table.csv       — fit/sample wall-clock seconds
    bench/results/figures/*.png           — comparison plots

Configuration
-------------
Tweak `CONFIG` at the top of this file. Defaults:
    n_real = 600   (per module — split 50/50 train/test)
    n_synth = 600  (each backend generates this many rows)
    seeds = (1, 2, 3)  (results averaged across seeds for robustness)
    epochs = 100   (CTGAN/TVAE training epochs)

The full default run takes ~5–10 minutes on a CPU.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Quiet the pgmpy / SDV ecosystem.
warnings.filterwarnings("ignore")

from core.seed_data import water_borne, vector_borne, skin_ntd  # noqa: E402
from core.synthesizer import GaussianCopulaSynthesizer  # noqa: E402
from core.bayesnet_synthesizer import BayesianNetworkSynthesizer  # noqa: E402
from core.validator import (  # noqa: E402
    ks_per_column,
    wasserstein_per_column,
    categorical_tvd,
    correlation_distance,
)
from bench.tasks import TASKS, TaskResult  # noqa: E402


CONFIG = {
    "n_real": 600,
    "n_synth": 600,
    "seeds": (1, 2, 3),
    "epochs": 100,
    "test_fraction": 0.5,
    "modules": {
        "water_borne": water_borne,
        "vector_borne": vector_borne,
        "skin_ntd": skin_ntd,
    },
    # Each entry is (display_name, factory). Neural backends are imported
    # lazily so the script can still run on machines without PyTorch
    # (it will just skip them with a warning).
    "backends": [
        (
            "gaussian_copula",
            lambda seed, **kw: GaussianCopulaSynthesizer(seed=seed),
        ),
        (
            "bayesian_network",
            lambda seed, **kw: BayesianNetworkSynthesizer(
                seed=seed, n_bins=8, max_iter=30,
            ),
        ),
        (
            "ctgan",
            lambda seed, epochs=100, **kw: _make_ctgan(seed, epochs),
        ),
        (
            "tvae",
            lambda seed, epochs=100, **kw: _make_tvae(seed, epochs),
        ),
    ],
}


# ---- Lazy neural backend imports -----------------------------------------

def _make_ctgan(seed: int, epochs: int):
    from core.neural_synthesizer import CTGANSynthesizer
    return CTGANSynthesizer(seed=seed, epochs=epochs)


def _make_tvae(seed: int, epochs: int):
    from core.neural_synthesizer import TVAESynthesizer
    return TVAESynthesizer(seed=seed, epochs=epochs)


# ---- Single trial --------------------------------------------------------

def _train_test_split(df: pd.DataFrame, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[idx[:cut]].reset_index(drop=True), df.iloc[idx[cut:]].reset_index(drop=True)


def _run_one_trial(
    module_name: str,
    backend_name: str,
    backend_factory,
    seed: int,
    n_real: int,
    n_synth: int,
    epochs: int,
    test_fraction: float,
) -> dict:
    print(f"  trial: {module_name} | {backend_name} | seed={seed}")

    # 1. Generate ground truth and split.
    seed_fn = CONFIG["modules"][module_name]
    real = seed_fn(n=n_real, seed=seed)
    real_train, real_test = _train_test_split(real, test_fraction, seed)

    # 2. Fit backend on train half.
    synth_obj = backend_factory(seed=seed, epochs=epochs)
    t0 = time.time()
    synth_obj.fit(real_train)
    fit_seconds = time.time() - t0

    # 3. Sample.
    t0 = time.time()
    synthetic = synth_obj.sample(n_synth)
    sample_seconds = time.time() - t0

    # 4. Fidelity metrics — comparing synthetic to real_train (which the
    #    model saw) is the most charitable test; comparing to real_test
    #    is the strict generalisation test. We report the strict version.
    ks = ks_per_column(real_test, synthetic)
    wasserstein = wasserstein_per_column(real_test, synthetic)
    tvd = categorical_tvd(real_test, synthetic)
    corr = correlation_distance(real_test, synthetic)

    # 5. Utility metrics.
    utilities = {}
    for task_name, task_fn in TASKS.items():
        result: TaskResult = task_fn(real_train, real_test, synthetic)
        utilities[task_name] = asdict(result)

    return {
        "module": module_name,
        "backend": backend_name,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "sample_seconds": sample_seconds,
        "fidelity": {
            "ks_mean_d": float(np.mean([v["statistic"] for v in ks.values()])),
            "ks_min_p": float(np.min([v["p_value"] for v in ks.values()])),
            "wasserstein_mean": float(np.mean(list(wasserstein.values()))),
            "tvd_mean": float(np.mean(list(tvd.values()))) if tvd else 0.0,
            "correlation_normalised": float(corr["normalized"]),
        },
        "utility": utilities,
    }


# ---- Aggregation ---------------------------------------------------------

def _aggregate(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (fidelity_df, utility_df, runtime_df) averaged across seeds."""
    rows = []
    for r in results:
        row = {
            "module": r["module"],
            "backend": r["backend"],
            "seed": r["seed"],
            "fit_s": r["fit_seconds"],
            "sample_s": r["sample_seconds"],
            **r["fidelity"],
        }
        for task, u in r["utility"].items():
            row[f"{task}_score_real"] = u["score_real"]
            row[f"{task}_score_synth"] = u["score_synthetic"]
            row[f"{task}_ratio"] = u["ratio"]
        rows.append(row)
    df = pd.DataFrame(rows)

    agg_cols = [c for c in df.columns if c not in ("module", "backend", "seed")]
    agg = (
        df.groupby(["module", "backend"])[agg_cols]
        .agg(["mean", "std"])
        .round(4)
    )

    fidelity_cols = [
        "ks_mean_d", "ks_min_p", "wasserstein_mean", "tvd_mean",
        "correlation_normalised",
    ]
    utility_cols = [
        c for c in agg_cols
        if c.startswith("case_regression_") or c.startswith("outbreak_classification_")
    ]
    runtime_cols = ["fit_s", "sample_s"]

    return agg[fidelity_cols], agg[utility_cols], agg[runtime_cols]


# ---- Main ----------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    results: list[dict] = []
    skipped = []

    seeds = tuple(args.seeds) if args.seeds else CONFIG["seeds"]
    backends = CONFIG["backends"]
    if args.skip_neural:
        backends = [b for b in backends if b[0] not in ("ctgan", "tvae")]
        print("Skipping neural backends (--skip-neural).")

    total = len(CONFIG["modules"]) * len(backends) * len(seeds)
    done = 0
    print(f"Running {total} trials.\n")

    for module_name in CONFIG["modules"]:
        print(f"Module: {module_name}")
        for backend_name, factory in backends:
            for seed in seeds:
                done += 1
                try:
                    r = _run_one_trial(
                        module_name=module_name,
                        backend_name=backend_name,
                        backend_factory=factory,
                        seed=seed,
                        n_real=args.n_real,
                        n_synth=args.n_synth,
                        epochs=args.epochs,
                        test_fraction=CONFIG["test_fraction"],
                    )
                    results.append(r)
                except Exception as e:
                    skipped.append((module_name, backend_name, seed, str(e)))
                    print(f"    SKIPPED: {e!r}")
        print()

    # Persist raw results
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({"config": {**{k: v for k, v in CONFIG.items() if k != "modules" and k != "backends"},
                              "modules": list(CONFIG["modules"].keys()),
                              "backends": [b[0] for b in backends]},
                   "results": results,
                   "skipped": skipped}, f, indent=2, default=str)

    # Aggregate and save tables
    fidelity_df, utility_df, runtime_df = _aggregate(results)
    fidelity_df.to_csv(out_dir / "fidelity_table.csv")
    utility_df.to_csv(out_dir / "utility_table.csv")
    runtime_df.to_csv(out_dir / "runtime_table.csv")

    # Print human-readable summary
    print("=" * 70)
    print("FIDELITY (lower is better, except ks_min_p)")
    print("=" * 70)
    print(fidelity_df.to_string())
    print()
    print("=" * 70)
    print("UTILITY (higher is better; ratio near 1.0 = synth ≈ real)")
    print("=" * 70)
    print(utility_df.to_string())
    print()
    print("=" * 70)
    print("RUNTIME (seconds)")
    print("=" * 70)
    print(runtime_df.to_string())

    # Plots
    try:
        _make_plots(results, fig_dir)
        print(f"\nFigures saved to {fig_dir}")
    except Exception as e:
        print(f"\nPlot generation failed (non-fatal): {e}")

    print(f"\nWrote {out_dir.resolve()}")


def _make_plots(results: list[dict], out_dir: Path) -> None:
    """Generate publication-quality comparison figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    df = pd.DataFrame([
        {
            "module": r["module"],
            "backend": r["backend"],
            "seed": r["seed"],
            **r["fidelity"],
            "case_regression_ratio": r["utility"]["case_regression"]["ratio"],
            "outbreak_classification_ratio": r["utility"]["outbreak_classification"]["ratio"],
            "fit_seconds": r["fit_seconds"],
        }
        for r in results
    ])

    backend_order = ["gaussian_copula", "bayesian_network", "ctgan", "tvae"]
    backend_order = [b for b in backend_order if b in df["backend"].unique()]
    module_order = ["water_borne", "vector_borne", "skin_ntd"]
    module_order = [m for m in module_order if m in df["module"].unique()]
    palette = {"gaussian_copula": "#8b1e1e", "bayesian_network": "#1e4d8b",
               "ctgan": "#1e8b3a", "tvae": "#8b6e1e"}

    # --- Figure 1: utility (TSTR ratios) ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in zip(
        axes,
        ["case_regression_ratio", "outbreak_classification_ratio"],
        ["Case regression (TSTR R² ratio)", "Outbreak classification (TSTR macro-F1 ratio)"],
    ):
        x = np.arange(len(module_order))
        width = 0.2
        for i, b in enumerate(backend_order):
            sub = df[df["backend"] == b]
            means = [sub[sub["module"] == m][metric].mean() for m in module_order]
            stds = [sub[sub["module"] == m][metric].std() for m in module_order]
            ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                   label=b, color=palette.get(b, "gray"))
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x + width * (len(backend_order) - 1) / 2)
        ax.set_xticklabels(module_order)
        ax.set_ylabel("Ratio (synthetic / real)")
        ax.set_title(title)
        ax.set_ylim(0, 1.2)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "utility_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: fidelity heatmap --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title, lower_better in zip(
        axes,
        ["wasserstein_mean", "correlation_normalised"],
        ["Mean Wasserstein-1 (lower = better)",
         "Normalised correlation distance (lower = better)"],
        [True, True],
    ):
        pivot = df.groupby(["module", "backend"])[metric].mean().unstack()
        pivot = pivot.reindex(index=module_order, columns=backend_order)
        im = ax.imshow(pivot.values, cmap="RdYlGn_r" if lower_better else "RdYlGn",
                       aspect="auto")
        ax.set_xticks(range(len(backend_order)))
        ax.set_xticklabels(backend_order, rotation=30, ha="right")
        ax.set_yticks(range(len(module_order)))
        ax.set_yticklabels(module_order)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > pivot.values.mean() else "black",
                        fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "fidelity_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: runtime --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    runtime = df.groupby("backend")["fit_seconds"].mean().reindex(backend_order)
    ax.bar(runtime.index, runtime.values,
           color=[palette.get(b, "gray") for b in runtime.index])
    ax.set_ylabel("Mean fit time (s)")
    ax.set_title("Backend training cost (averaged across modules and seeds)")
    ax.set_yscale("log")
    for i, v in enumerate(runtime.values):
        ax.text(i, v * 1.1, f"{v:.2f}s", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-real", type=int, default=CONFIG["n_real"])
    parser.add_argument("--n-synth", type=int, default=CONFIG["n_synth"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seed list. Default: 1 2 3")
    parser.add_argument("--skip-neural", action="store_true",
                        help="Skip CTGAN and TVAE (useful if PyTorch unavailable)")
    parser.add_argument("--output-dir", type=str, default="bench/results")
    args = parser.parse_args()
    main(args)
