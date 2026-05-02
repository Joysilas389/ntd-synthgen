"""
Unified evaluation across fidelity and utility dimensions.

Returns one flat dict per (backend, dataset, n_rows, seed) combination so
results can be concatenated into a tidy DataFrame.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ---------- fidelity ----------

def mean_ks(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    cols = real.select_dtypes(include=[np.number]).columns
    if len(cols) == 0:
        return float("nan")
    stats = []
    for c in cols:
        if c in synth.columns:
            stats.append(ks_2samp(real[c].dropna(), synth[c].dropna()).statistic)
    return float(np.mean(stats)) if stats else float("nan")


def mean_wasserstein(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Standardised Wasserstein-1 — divided by real std so it's unit-free."""
    cols = real.select_dtypes(include=[np.number]).columns
    if len(cols) == 0:
        return float("nan")
    dists = []
    for c in cols:
        if c not in synth.columns:
            continue
        std = real[c].std()
        if std == 0 or np.isnan(std):
            continue
        d = wasserstein_distance(real[c].dropna(), synth[c].dropna()) / std
        dists.append(d)
    return float(np.mean(dists)) if dists else float("nan")


def mean_tvd(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    cols = real.select_dtypes(include=["object", "category"]).columns
    if len(cols) == 0:
        return float("nan")
    tvds = []
    for c in cols:
        if c not in synth.columns:
            continue
        rp = real[c].value_counts(normalize=True)
        sp = synth[c].value_counts(normalize=True)
        all_cats = set(rp.index) | set(sp.index)
        rp = rp.reindex(all_cats, fill_value=0.0)
        sp = sp.reindex(all_cats, fill_value=0.0)
        tvds.append(0.5 * float(np.abs(rp - sp).sum()))
    return float(np.mean(tvds)) if tvds else float("nan")


def correlation_distance(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Frobenius distance between Pearson correlation matrices, normalised."""
    cols = list(real.select_dtypes(include=[np.number]).columns)
    cols = [c for c in cols if c in synth.columns]
    if len(cols) < 2:
        return float("nan")
    rc = real[cols].corr().values
    sc = synth[cols].corr().values
    frob = float(np.linalg.norm(rc - sc, ord="fro"))
    # Normalise by sqrt(2 * k * (k-1)) — the maximum possible Frobenius distance
    # between two correlation matrices when off-diagonals can each differ by 2.
    k = len(cols)
    max_frob = np.sqrt(2 * k * (k - 1))
    return frob / max_frob if max_frob > 0 else frob


# ---------- utility ----------

def _prep_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    # Label-encode any categoricals in X.
    for c in X.select_dtypes(include=["object", "category"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return X, y


def tstr_regression(
    real: pd.DataFrame, synth: pd.DataFrame, target: str
) -> dict[str, float]:
    """Train-on-Synthetic, Test-on-Real for regression target."""
    real_train, real_test = train_test_split(real, test_size=0.3, random_state=42)
    Xr_train, yr_train = _prep_features(real_train, target)
    Xr_test, yr_test = _prep_features(real_test, target)
    Xs, ys = _prep_features(synth, target)

    # Align columns (in case synth produced a column real didn't, or vice versa).
    common_cols = [c for c in Xr_train.columns if c in Xs.columns]
    Xr_train = Xr_train[common_cols]
    Xr_test = Xr_test[common_cols]
    Xs = Xs[common_cols]

    rf_real = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf_real.fit(Xr_train, yr_train)
    r2_real = r2_score(yr_test, rf_real.predict(Xr_test))

    rf_syn = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf_syn.fit(Xs, ys)
    r2_syn = r2_score(yr_test, rf_syn.predict(Xr_test))

    ratio = r2_syn / r2_real if r2_real > 0 else float("nan")
    return {"tstr_r2_real": r2_real, "tstr_r2_syn": r2_syn, "tstr_r2_ratio": ratio}


def tstr_outbreak(
    real: pd.DataFrame, synth: pd.DataFrame, target: str, threshold_q: float = 0.75
) -> dict[str, float]:
    """
    Outbreak detection task: binarise the case count at the 75th percentile of
    the *real* training data, train a classifier on synthetic, evaluate F1 on
    real held-out. This is the public-health-relevant downstream task.
    """
    real_train, real_test = train_test_split(real, test_size=0.3, random_state=42)
    threshold = real_train[target].quantile(threshold_q)

    real_train = real_train.assign(_outbreak=(real_train[target] > threshold).astype(int))
    real_test = real_test.assign(_outbreak=(real_test[target] > threshold).astype(int))
    synth = synth.assign(_outbreak=(synth[target] > threshold).astype(int))

    if real_test["_outbreak"].nunique() < 2 or synth["_outbreak"].nunique() < 2:
        return {"tstr_f1_real": float("nan"), "tstr_f1_syn": float("nan"), "tstr_f1_ratio": float("nan")}

    drop_cols = [target, "_outbreak"]
    def _xy(df):
        X = df.drop(columns=drop_cols).copy()
        for c in X.select_dtypes(include=["object", "category"]).columns:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        return X, df["_outbreak"]

    Xr_train, yr_train = _xy(real_train)
    Xr_test, yr_test = _xy(real_test)
    Xs, ys = _xy(synth)

    common = [c for c in Xr_train.columns if c in Xs.columns]
    Xr_train = Xr_train[common]; Xr_test = Xr_test[common]; Xs = Xs[common]

    clf_real = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    clf_real.fit(Xr_train, yr_train)
    f1_real = f1_score(yr_test, clf_real.predict(Xr_test))

    clf_syn = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    clf_syn.fit(Xs, ys)
    f1_syn = f1_score(yr_test, clf_syn.predict(Xr_test))

    ratio = f1_syn / f1_real if f1_real > 0 else float("nan")
    return {"tstr_f1_real": f1_real, "tstr_f1_syn": f1_syn, "tstr_f1_ratio": ratio}


# ---------- top-level ----------

def evaluate(
    real: pd.DataFrame, synth: pd.DataFrame, target: str
) -> dict[str, Any]:
    out = {
        "ks_mean": mean_ks(real, synth),
        "wasserstein_mean": mean_wasserstein(real, synth),
        "tvd_mean": mean_tvd(real, synth),
        "corr_distance": correlation_distance(real, synth),
    }
    out.update(tstr_regression(real, synth, target))
    out.update(tstr_outbreak(real, synth, target))
    return out
