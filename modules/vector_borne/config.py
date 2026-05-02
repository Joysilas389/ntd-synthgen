"""Post-generation constraints for vector-borne NTDs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_constraints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "disease_cases" in df.columns:
        df["disease_cases"] = np.maximum(0, df["disease_cases"]).round().astype(int)
    if "humidity_pct" in df.columns:
        df["humidity_pct"] = df["humidity_pct"].clip(0, 100).round(1)
    for col in ("vegetation_index", "bednet_coverage"):
        if col in df.columns:
            df[col] = df[col].clip(0, 1).round(3)
    if "vector_density" in df.columns:
        df["vector_density"] = df["vector_density"].clip(lower=0).round(1)
    if "rainfall_mm" in df.columns:
        df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0).round(1)
    if "temperature_c" in df.columns:
        df["temperature_c"] = df["temperature_c"].clip(10, 45).round(1)
    return df
