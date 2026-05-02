"""Post-generation constraints for skin NTDs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_constraints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "disease_cases" in df.columns:
        df["disease_cases"] = np.maximum(0, df["disease_cases"]).round().astype(int)
    if "humidity_pct" in df.columns:
        df["humidity_pct"] = df["humidity_pct"].clip(0, 100).round(1)
    if "rainfall_mm" in df.columns:
        df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0).round(1)
    if "proximity_to_water_km" in df.columns:
        df["proximity_to_water_km"] = df["proximity_to_water_km"].clip(lower=0).round(2)
    if "household_size" in df.columns:
        df["household_size"] = df["household_size"].clip(1, 30).round().astype(int)
    for col in ("sanitation_index", "healthcare_access_index"):
        if col in df.columns:
            df[col] = df[col].clip(0, 1).round(3)
    return df
