"""Statistical validation of synthetic vs real data.

Implements the metrics reviewers will ask for:

  * Kolmogorov-Smirnov per numeric column (distribution match)
  * Wasserstein-1 (Earth Mover's) distance per numeric column
  * Categorical TVD (Total Variation Distance) per categorical column
  * Correlation Frobenius distance (joint structure)
  * Train-on-Synthetic / Test-on-Real (TSTR) utility score
  * Per-column summary statistics (mean / std / min / max) side-by-side

All metrics return JSON-serializable dicts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from .logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Univariate metrics
# ---------------------------------------------------------------------------

def ks_per_column(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, dict[str, float]]:
    """KS test per numeric column. Lower statistic = closer match."""
    out: dict[str, dict[str, float]] = {}
    for col in real.select_dtypes(include="number").columns:
        if col not in synth.columns:
            continue
        try:
            r = real[col].dropna().to_numpy(dtype=float)
            s = synth[col].dropna().to_numpy(dtype=float)
            if len(r) < 2 or len(s) < 2:
                continue
            res = stats.ks_2samp(r, s)
            out[col] = {"statistic": float(res.statistic), "p_value": float(res.pvalue)}
        except Exception as exc:  # pragma: no cover - safety
            log.warning("KS failed for %s: %s", col, exc)
    return out


def wasserstein_per_column(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    """Wasserstein-1 distance per numeric column."""
    out: dict[str, float] = {}
    for col in real.select_dtypes(include="number").columns:
        if col not in synth.columns:
            continue
        try:
            r = real[col].dropna().to_numpy(dtype=float)
            s = synth[col].dropna().to_numpy(dtype=float)
            if len(r) and len(s):
                out[col] = float(stats.wasserstein_distance(r, s))
        except Exception as exc:  # pragma: no cover
            log.warning("Wasserstein failed for %s: %s", col, exc)
    return out


def categorical_tvd(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    """Total Variation Distance per categorical column. 0 = identical."""
    out: dict[str, float] = {}
    cats = [c for c in real.columns if not pd.api.types.is_numeric_dtype(real[c])]
    for col in cats:
        if col not in synth.columns:
            continue
        r = real[col].astype(str).value_counts(normalize=True)
        s = synth[col].astype(str).value_counts(normalize=True)
        all_cats = set(r.index) | set(s.index)
        tvd = 0.5 * sum(abs(r.get(c, 0.0) - s.get(c, 0.0)) for c in all_cats)
        out[col] = float(tvd)
    return out


# ---------------------------------------------------------------------------
# Joint structure
# ---------------------------------------------------------------------------

def correlation_distance(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, Any]:
    """Frobenius distance between correlation matrices of numeric columns."""
    num_cols = [c for c in real.select_dtypes(include="number").columns if c in synth.columns]
    if len(num_cols) < 2:
        return {"frobenius_distance": None, "columns": num_cols}
    cr = real[num_cols].corr().to_numpy()
    cs = synth[num_cols].corr().to_numpy()
    diff = cr - cs
    frob = float(np.linalg.norm(diff, ord="fro"))
    # Normalize by matrix size so the score is comparable across schemas.
    norm = frob / max(cr.size ** 0.5, 1.0)
    return {
        "frobenius_distance": frob,
        "normalized": norm,
        "columns": num_cols,
        "real_corr": cr.tolist(),
        "synth_corr": cs.tolist(),
    }


def summary_stats(real: pd.DataFrame, synth: pd.DataFrame) -> list[dict[str, Any]]:
    """Side-by-side numeric summary for the results page."""
    rows: list[dict[str, Any]] = []
    for col in real.select_dtypes(include="number").columns:
        if col not in synth.columns:
            continue
        r = real[col].dropna()
        s = synth[col].dropna()
        if r.empty or s.empty:
            continue
        rows.append({
            "column": col,
            "real_mean": round(float(r.mean()), 4),
            "synth_mean": round(float(s.mean()), 4),
            "real_std": round(float(r.std()), 4),
            "synth_std": round(float(s.std()), 4),
            "real_min": round(float(r.min()), 4),
            "synth_min": round(float(s.min()), 4),
            "real_max": round(float(r.max()), 4),
            "synth_max": round(float(s.max()), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# TSTR utility
# ---------------------------------------------------------------------------

def tstr_utility(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Train-on-Synthetic, Test-on-Real utility score.

    The headline number reviewers look for. We pick a reasonable target
    automatically (the last numeric column unless one is provided), train a
    Random Forest on synthetic data, and score it on real data. We also
    train a baseline (Train-on-Real, Test-on-Real) and report the *ratio*.
    """
    common = [c for c in real.columns if c in synth.columns]
    if not common:
        return {"error": "No overlapping columns for TSTR."}

    if target_col is None:
        # Default: last numeric column tends to be the outcome (e.g. cases).
        num_cols = [c for c in real[common].select_dtypes(include="number").columns]
        if not num_cols:
            return {"error": "No numeric column to use as TSTR target."}
        target_col = num_cols[-1]

    if target_col not in common:
        return {"error": f"Target column '{target_col}' not in both datasets."}

    feat_cols = [c for c in common if c != target_col]
    if not feat_cols:
        return {"error": "No feature columns available for TSTR."}

    real_x = pd.get_dummies(real[feat_cols], drop_first=True).fillna(0)
    synth_x = pd.get_dummies(synth[feat_cols], drop_first=True).fillna(0)
    # Align columns between the two.
    synth_x = synth_x.reindex(columns=real_x.columns, fill_value=0)
    real_y = real[target_col]
    synth_y = synth[target_col]

    is_numeric_target = pd.api.types.is_numeric_dtype(real_y)
    try:
        if is_numeric_target:
            # Regression
            model_synth = RandomForestRegressor(n_estimators=80, random_state=0, n_jobs=1)
            model_synth.fit(synth_x, synth_y.astype(float))
            preds = model_synth.predict(real_x)
            tstr_score = float(r2_score(real_y.astype(float), preds))

            # Real baseline (split for fairness).
            r_x_train, r_x_test, r_y_train, r_y_test = train_test_split(
                real_x, real_y.astype(float), test_size=0.3, random_state=0,
            )
            base = RandomForestRegressor(n_estimators=80, random_state=0, n_jobs=1)
            base.fit(r_x_train, r_y_train)
            real_score = float(r2_score(r_y_test, base.predict(r_x_test)))
            metric = "r2"
        else:
            model_synth = RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=1)
            model_synth.fit(synth_x, synth_y.astype(str))
            preds = model_synth.predict(real_x)
            tstr_score = float(accuracy_score(real_y.astype(str), preds))

            r_x_train, r_x_test, r_y_train, r_y_test = train_test_split(
                real_x, real_y.astype(str), test_size=0.3, random_state=0, stratify=real_y.astype(str),
            )
            base = RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=1)
            base.fit(r_x_train, r_y_train)
            real_score = float(accuracy_score(r_y_test, base.predict(r_x_test)))
            metric = "accuracy"
    except Exception as exc:
        log.warning("TSTR failed: %s", exc)
        return {"error": f"TSTR could not be computed: {exc}"}

    return {
        "target": target_col,
        "metric": metric,
        "tstr_score": round(tstr_score, 4),
        "real_baseline": round(real_score, 4),
        "ratio": round(tstr_score / real_score, 4) if real_score not in (0, None) else None,
        "interpretation": (
            "Ratios near 1.0 indicate the synthetic data is nearly as useful "
            "as real data for this downstream task."
        ),
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def full_report(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Run every metric and bundle the results into one JSON-friendly dict."""
    log.info("Running full validation | real=%s synth=%s", real.shape, synth.shape)
    return {
        "shapes": {"real": list(real.shape), "synthetic": list(synth.shape)},
        "summary_stats": summary_stats(real, synth),
        "ks": ks_per_column(real, synth),
        "wasserstein": wasserstein_per_column(real, synth),
        "categorical_tvd": categorical_tvd(real, synth),
        "correlation": correlation_distance(real, synth),
        "tstr": tstr_utility(real, synth, target_col=target_col),
    }
