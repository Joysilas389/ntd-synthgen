"""Preprocessing pipeline for incoming CSVs.

We keep this deliberately conservative because health datasets are messy:
* trim / lowercase column names (with caller's choice)
* coerce numeric-looking columns
* drop fully-empty rows
* validate against a module schema (column subset + types)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .logging_config import get_logger

log = get_logger(__name__)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def coerce_types(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Coerce columns to their schema-declared types where possible."""
    df = df.copy()
    for feat in schema.get("features", []):
        col = feat["name"]
        if col not in df.columns:
            continue
        target_type = feat.get("type", "numeric")
        try:
            if target_type == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_type == "categorical":
                df[col] = df[col].astype(str).str.strip()
        except Exception as exc:  # pragma: no cover
            log.warning("Could not coerce %s -> %s: %s", col, target_type, exc)
    target = schema.get("target")
    if target and target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")
    return df


def validate_against_schema(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    """Return a report of missing / extra columns. Does not raise."""
    expected = {f["name"] for f in schema.get("features", [])}
    target = schema.get("target")
    if target:
        expected.add(target)
    actual = set(df.columns)
    return {
        "expected": sorted(expected),
        "received": sorted(actual),
        "missing": sorted(expected - actual),
        "extra": sorted(actual - expected),
    }


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop rows that are entirely null.
    df = df.dropna(how="all")
    # Replace inf with NaN, which we'll then fill column-wise.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def prepare(df: pd.DataFrame, schema: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full pipeline: normalize, coerce, clean, validate."""
    df = normalize_columns(df)
    df = coerce_types(df, schema)
    df = basic_clean(df)
    report = validate_against_schema(df, schema)
    log.info("Preprocessed | shape=%s missing=%s", df.shape, report["missing"])
    return df, report
