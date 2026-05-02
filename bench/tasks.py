"""Downstream public health tasks for TSTR (Train-on-Synthetic-Test-on-Real) evaluation.

Each task definition specifies how to derive features and a target from
a disease-module DataFrame, plus the metric used to score predictions.

Two tasks are implemented:

1. CASE_REGRESSION — predict `disease_cases` (integer count). The
   classical fidelity check: does the synthetic data preserve enough
   structure for a regressor to learn from it? Scored with R².

2. OUTBREAK_CLASSIFICATION — binary outbreak label derived by thresholding
   `disease_cases` at the 75th percentile of the real training data. This
   is closer to a real public health decision: "is this district in the
   top quartile of risk?" Scored with macro-F1 to handle the class
   imbalance robustly.

Why these two tasks
-------------------
Regression tests whether the *continuous* signal is preserved. Classification
tests whether the *decision boundary* — the actionable threshold a public
health official would care about — is preserved. A synthesizer can do well
on one and badly on the other (e.g. compressing the right tail destroys
classification but barely moves regression R²), so both are needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import OrdinalEncoder


@dataclass(frozen=True)
class TaskResult:
    task: str
    metric: str
    score_real: float       # train and test on real (baseline)
    score_synthetic: float  # train on synthetic, test on real (TSTR)
    ratio: float            # score_synthetic / score_real (utility ratio)


def _featurise(
    df: pd.DataFrame,
    target_col: str,
    encoder: OrdinalEncoder | None = None,
    fit_encoder: bool = False,
) -> tuple[np.ndarray, np.ndarray, OrdinalEncoder]:
    """Encode categoricals as ordinals; return X, y, fitted encoder."""
    feature_cols = [c for c in df.columns if c != target_col]
    cat_cols = [c for c in feature_cols if df[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    if cat_cols:
        if fit_encoder:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            cat_arr = encoder.fit_transform(df[cat_cols])
        else:
            assert encoder is not None
            cat_arr = encoder.transform(df[cat_cols])
        num_arr = df[num_cols].to_numpy()
        X = np.hstack([num_arr, cat_arr])
    else:
        X = df[num_cols].to_numpy()
        encoder = encoder or OrdinalEncoder()

    y = df[target_col].to_numpy()
    return X, y, encoder


def case_regression(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str = "disease_cases",
    seed: int = 42,
) -> TaskResult:
    """Predict disease_cases as a continuous count. Score with R²."""
    # Real-on-real baseline
    X_rtr, y_rtr, enc = _featurise(real_train, target_col, fit_encoder=True)
    X_rte, y_rte, _ = _featurise(real_test, target_col, encoder=enc)
    rf = RandomForestRegressor(n_estimators=100, random_state=seed)
    rf.fit(X_rtr, y_rtr)
    score_real = r2_score(y_rte, rf.predict(X_rte))

    # Train-on-synthetic, test-on-real
    X_syn, y_syn, _ = _featurise(synthetic, target_col, encoder=enc)
    rf_syn = RandomForestRegressor(n_estimators=100, random_state=seed)
    rf_syn.fit(X_syn, y_syn)
    score_synth = r2_score(y_rte, rf_syn.predict(X_rte))

    # Guard against negative R² baselines making the ratio meaningless
    ratio = score_synth / score_real if score_real > 0 else float("nan")
    return TaskResult(
        task="case_regression",
        metric="r2",
        score_real=float(score_real),
        score_synthetic=float(score_synth),
        ratio=float(ratio),
    )


def outbreak_classification(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str = "disease_cases",
    seed: int = 42,
) -> TaskResult:
    """Predict whether a district is in the top quartile of disease burden.

    Threshold is computed from the real training set (the only data a
    real-world deployment would have at hand) and applied identically to
    test and synthetic targets.
    """
    threshold = float(np.quantile(real_train[target_col], 0.75))

    rt = real_train.copy()
    rt["_outbreak"] = (rt[target_col] > threshold).astype(int)
    re_ = real_test.copy()
    re_["_outbreak"] = (re_[target_col] > threshold).astype(int)
    sy = synthetic.copy()
    sy["_outbreak"] = (sy[target_col] > threshold).astype(int)

    # Drop the original continuous target so the classifier only sees
    # legitimate features (no leakage).
    rt = rt.drop(columns=[target_col])
    re_ = re_.drop(columns=[target_col])
    sy = sy.drop(columns=[target_col])

    X_rtr, y_rtr, enc = _featurise(rt, "_outbreak", fit_encoder=True)
    X_rte, y_rte, _ = _featurise(re_, "_outbreak", encoder=enc)
    X_syn, y_syn, _ = _featurise(sy, "_outbreak", encoder=enc)

    if len(np.unique(y_syn)) < 2:
        # Synthetic target collapsed — can't train a classifier.
        return TaskResult(
            task="outbreak_classification",
            metric="macro_f1",
            score_real=float("nan"),
            score_synthetic=float("nan"),
            ratio=float("nan"),
        )

    clf_real = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf_real.fit(X_rtr, y_rtr)
    score_real = f1_score(y_rte, clf_real.predict(X_rte), average="macro")

    clf_syn = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf_syn.fit(X_syn, y_syn)
    score_synth = f1_score(y_rte, clf_syn.predict(X_rte), average="macro")

    ratio = score_synth / score_real if score_real > 0 else float("nan")
    return TaskResult(
        task="outbreak_classification",
        metric="macro_f1",
        score_real=float(score_real),
        score_synthetic=float(score_synth),
        ratio=float(ratio),
    )


# Registry the benchmark runner iterates over.
TASKS: dict[str, Callable[..., TaskResult]] = {
    "case_regression": case_regression,
    "outbreak_classification": outbreak_classification,
}
