"""Post-generation constraints for water-borne NTDs.

These are domain-aware clamps that ensure the synthesizer's output is
epidemiologically plausible (e.g. sanitation_index in [0, 1]).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_constraints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "disease_cases" in df.columns:
        df["disease_cases"] = np.maximum(0, df["disease_cases"]).round().astype(int)
    for col in ("sanitation_index", "open_defecation_rate"):
        if col in df.columns:
            df[col] = df[col].clip(0, 1)
    if "rainfall_mm" in df.columns:
        df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0).round(1)
    if "temperature_c" in df.columns:
        df["temperature_c"] = df["temperature_c"].clip(10, 45).round(1)
    if "population_density" in df.columns:
        df["population_density"] = df["population_density"].clip(lower=1).round(1)
    return df
