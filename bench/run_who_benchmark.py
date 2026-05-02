"""Benchmark the four backends on the WHO-anchored Sub-Saharan dataset.

Runs alongside the synthetic-seed benchmark in bench/run_benchmark.py.
Produces a separate results file so the paper can report numbers on
both the synthetic (controlled) and real-world (WHO-anchored) data.

Usage:
    python -m bench.run_who_benchmark
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from core.synthesizer import GaussianCopulaSynthesizer  # noqa: E402
from core.bayesnet_synthesizer import BayesianNetworkSynthesizer  # noqa: E402
from core.validator import (  # noqa: E402
    ks_per_column,
    wasserstein_per_column,
    categorical_tvd,
    correlation_distance,
)
from data.who_anchor import schisto_who_anchored  # noqa: E402


def _make_ctgan(seed, epochs):
    from core.neural_synthesizer import CTGANSynthesizer
    return CTGANSynthesizer(seed=seed, epochs=epochs)


def _make_tvae(seed, epochs):
    from core.neural_synthesizer import TVAESynthesizer
    return TVAESynthesizer(seed=seed, epochs=epochs)


BACKENDS = [
    ("gaussian_copula",
     lambda seed, epochs: GaussianCopulaSynthesizer(seed=seed)),
    ("bayesian_network",
     lambda seed, epochs: BayesianNetworkSynthesizer(seed=seed, n_bins=8, max_iter=30)),
    ("ctgan", _make_ctgan),
    ("tvae", _make_tvae),
]


def _train_test_split(df: pd.DataFrame, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - frac))
    return df.iloc[idx[:cut]].reset_index(drop=True), df.iloc[idx[cut:]].reset_index(drop=True)


def _tstr_regression(real_train, real_test, synthetic, target_col):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import OrdinalEncoder

    feature_cols = [c for c in real_train.columns if c != target_col]
    cat_cols = [c for c in feature_cols if real_train[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        enc.fit(real_train[cat_cols])

    def featurise(df):
        if cat_cols:
            cat_arr = enc.transform(df[cat_cols])
            return np.hstack([df[num_cols].to_numpy(), cat_arr])
        return df[num_cols].to_numpy()

    X_rtr, y_rtr = featurise(real_train), real_train[target_col].to_numpy()
    X_rte, y_rte = featurise(real_test), real_test[target_col].to_numpy()
    X_syn, y_syn = featurise(synthetic), synthetic[target_col].to_numpy()

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_rtr, y_rtr)
    score_real = r2_score(y_rte, rf.predict(X_rte))

    rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf2.fit(X_syn, y_syn)
    score_synth = r2_score(y_rte, rf2.predict(X_rte))
    return {
        "score_real": float(score_real),
        "score_synthetic": float(score_synth),
        "ratio": float(score_synth / score_real) if score_real > 0 else float("nan"),
    }


def main():
    out_dir = Path("bench/results_who")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = (1, 2, 3)
    epochs = 100

    print("Generating WHO-anchored Sub-Saharan dataset...")
    df = schisto_who_anchored(seed=2024)
    # Drop columns that are pure identifiers — synthesizers shouldn't
    # learn country-name strings or year as a feature.
    df = df.drop(columns=["country_name"])
    print(f"  {len(df)} rows, {df.shape[1]} columns")
    print(f"  countries: {df['country_code'].nunique()}, years: {df['year'].min()}-{df['year'].max()}")
    print()

    results = []
    for backend_name, factory in BACKENDS:
        for seed in seeds:
            print(f"  trial: {backend_name} | seed={seed}")
            try:
                real_train, real_test = _train_test_split(df, 0.5, seed)
                model = factory(seed, epochs)
                t0 = time.time()
                model.fit(real_train)
                fit_s = time.time() - t0
                t0 = time.time()
                synth = model.sample(len(df))
                sample_s = time.time() - t0

                ks = ks_per_column(real_test, synth)
                wd = wasserstein_per_column(real_test, synth)
                tvd = categorical_tvd(real_test, synth)
                corr = correlation_distance(real_test, synth)
                tstr = _tstr_regression(real_train, real_test, synth,
                                         target_col="pc_coverage_pct")

                results.append({
                    "backend": backend_name,
                    "seed": seed,
                    "fit_seconds": fit_s,
                    "sample_seconds": sample_s,
                    "ks_mean_d": float(np.mean([v["statistic"] for v in ks.values()])),
                    "wasserstein_mean": float(np.mean(list(wd.values()))),
                    "tvd_mean": float(np.mean(list(tvd.values()))) if tvd else 0.0,
                    "correlation_normalised": float(corr["normalized"]),
                    "tstr_r2_real": tstr["score_real"],
                    "tstr_r2_synth": tstr["score_synthetic"],
                    "tstr_ratio": tstr["ratio"],
                })
            except Exception as e:
                print(f"    SKIPPED: {e}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "who_results.csv", index=False)
    with open(out_dir / "who_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Aggregate
    agg = (
        res_df.groupby("backend")
        [["fit_seconds", "ks_mean_d", "wasserstein_mean", "tvd_mean",
          "correlation_normalised", "tstr_ratio"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(out_dir / "who_summary.csv")

    print("\n" + "=" * 70)
    print("WHO-ANCHORED DATASET RESULTS (Sub-Saharan schistosomiasis PC, 25 countries × 13 years)")
    print("=" * 70)
    print(agg.to_string())
    print(f"\nResults saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
