"""Tests for the validation metrics."""
from __future__ import annotations

import pandas as pd

from core.synthesizer import GaussianCopulaSynthesizer
from core.validator import (
    categorical_tvd,
    correlation_distance,
    full_report,
    ks_per_column,
    summary_stats,
    tstr_utility,
    wasserstein_per_column,
)


def _fit_and_sample(df: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    m = GaussianCopulaSynthesizer(seed=4)
    m.fit(df, categorical_cols=cats)
    return m.sample(len(df))


def test_ks_returns_per_numeric_column(water_df: pd.DataFrame):
    synth = _fit_and_sample(water_df, ["region", "water_source"])
    ks = ks_per_column(water_df, synth)
    for col in ("rainfall_mm", "temperature_c", "disease_cases"):
        assert col in ks
        assert "statistic" in ks[col]
        assert "p_value" in ks[col]


def test_wasserstein_low_for_well_fitted_model(water_df: pd.DataFrame):
    synth = _fit_and_sample(water_df, ["region", "water_source"])
    ws = wasserstein_per_column(water_df, synth)
    assert ws["sanitation_index"] < 0.05  # tight bound on a [0,1] field


def test_categorical_tvd_low(water_df: pd.DataFrame):
    synth = _fit_and_sample(water_df, ["region", "water_source"])
    tvd = categorical_tvd(water_df, synth)
    assert tvd["region"] < 0.15
    assert tvd["water_source"] < 0.15


def test_correlation_distance(water_df: pd.DataFrame):
    synth = _fit_and_sample(water_df, ["region", "water_source"])
    out = correlation_distance(water_df, synth)
    assert out["frobenius_distance"] is not None
    assert out["normalized"] < 1.0


def test_summary_stats_paired(water_df: pd.DataFrame):
    synth = _fit_and_sample(water_df, ["region", "water_source"])
    rows = summary_stats(water_df, synth)
    assert any(r["column"] == "rainfall_mm" for r in rows)
    for r in rows:
        assert {"real_mean", "synth_mean", "real_std", "synth_std"} <= set(r.keys())


def test_tstr_returns_score(water_df: pd.DataFrame):
    synth = _fit_and_sample(water_df, ["region", "water_source"])
    out = tstr_utility(water_df, synth, target_col="disease_cases")
    assert "tstr_score" in out
    assert "real_baseline" in out
    assert -1.0 <= out["tstr_score"] <= 1.0  # R^2 in (-inf, 1] but reasonable


def test_full_report_keys(skin_df: pd.DataFrame):
    synth = _fit_and_sample(skin_df, ["region", "skin_contact_occupation"])
    rep = full_report(skin_df, synth, target_col="disease_cases")
    assert {"summary_stats", "ks", "wasserstein", "categorical_tvd", "correlation", "tstr"} <= set(rep)
