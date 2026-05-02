"""Tests for core.preprocessing."""
from __future__ import annotations

import pandas as pd

from core.preprocessing import (
    coerce_types,
    normalize_columns,
    prepare,
    validate_against_schema,
)


SCHEMA = {
    "target": "disease_cases",
    "features": [
        {"name": "region", "type": "categorical"},
        {"name": "rainfall_mm", "type": "numeric"},
    ],
}


def test_normalize_columns_lowercases_and_underscores():
    df = pd.DataFrame({" Region ": [1], "Rainfall MM": [2]})
    out = normalize_columns(df)
    assert list(out.columns) == ["region", "rainfall_mm"]


def test_coerce_types_converts_numeric():
    df = pd.DataFrame({"region": ["a"], "rainfall_mm": ["12.5"], "disease_cases": ["5"]})
    out = coerce_types(df, SCHEMA)
    assert pd.api.types.is_numeric_dtype(out["rainfall_mm"])
    assert pd.api.types.is_numeric_dtype(out["disease_cases"])


def test_validate_against_schema_reports_missing():
    df = pd.DataFrame({"region": ["a"]})
    rep = validate_against_schema(df, SCHEMA)
    assert "rainfall_mm" in rep["missing"]
    assert "disease_cases" in rep["missing"]


def test_prepare_full_pipeline():
    df = pd.DataFrame({
        "Region": ["Lagos", "Accra", None, "Lagos"],
        "Rainfall MM": ["10", "20", "30", "40"],
        "disease_cases": [1, 2, 3, 4],
    })
    out, rep = prepare(df, SCHEMA)
    assert "region" in out.columns
    assert "rainfall_mm" in out.columns
    assert pd.api.types.is_numeric_dtype(out["rainfall_mm"])
    # Empty rows are dropped, but partial nulls survive (filled later by synth).
    assert len(out) == 4
    assert isinstance(rep, dict)
