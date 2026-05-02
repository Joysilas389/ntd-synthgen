"""Generate small but realistic seed datasets for each disease module.

Run from project root:

    python -m core.seed_data

Produces CSVs in data/samples/. These are *seed* datasets - they are
plausible, region-aware fixtures used to (a) demonstrate the system
and (b) train the synthesizer on something resembling real-world
structure when no real data is available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .logging_config import get_logger

log = get_logger(__name__)


REGIONS = {
    "ghana": ["Greater Accra", "Ashanti", "Eastern", "Western", "Volta", "Northern"],
    "nigeria": ["Lagos", "Kano", "Kaduna", "Cross River", "Rivers", "Plateau"],
    "kenya": ["Nairobi", "Coast", "Rift Valley", "Western", "Eastern", "Nyanza"],
}
ALL_REGIONS = sum(REGIONS.values(), [])


def _seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def water_borne(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = _seed(seed)
    region = rng.choice(ALL_REGIONS, size=n)
    rainfall = rng.gamma(shape=2.5, scale=40, size=n).clip(0, 500)
    temperature = rng.normal(28, 3, size=n).clip(18, 38)
    pop_density = rng.lognormal(mean=4.5, sigma=1.0, size=n).clip(5, 5000)
    sanitation = rng.beta(2, 4, size=n)  # skewed low (low-resource bias)
    water_source = rng.choice(
        ["piped", "borehole", "well", "river", "lake"],
        size=n, p=[0.20, 0.25, 0.20, 0.25, 0.10],
    )
    open_def = rng.beta(2, 6, size=n)

    # Disease cases: increase with rainfall, density, open defecation;
    # decrease with sanitation. Add Poisson noise.
    base_rate = (
        0.05 * rainfall +
        0.0015 * pop_density +
        80 * open_def -
        60 * sanitation +
        rng.normal(0, 5, size=n)
    )
    base_rate = np.maximum(base_rate, 0)
    cases = rng.poisson(base_rate)

    return pd.DataFrame({
        "region": region,
        "rainfall_mm": rainfall.round(1),
        "temperature_c": temperature.round(1),
        "population_density": pop_density.round(1),
        "sanitation_index": sanitation.round(3),
        "water_source": water_source,
        "open_defecation_rate": open_def.round(3),
        "disease_cases": cases,
    })


def vector_borne(n: int = 300, seed: int = 2) -> pd.DataFrame:
    rng = _seed(seed)
    region = rng.choice(ALL_REGIONS, size=n)
    rainfall = rng.gamma(shape=2.5, scale=40, size=n).clip(0, 500)
    temperature = rng.normal(27, 3, size=n).clip(18, 38)
    humidity = rng.normal(70, 12, size=n).clip(20, 100)
    ndvi = rng.beta(2, 2, size=n)
    vector_density = (
        rng.gamma(shape=2.0, scale=40, size=n)
        * (0.5 + ndvi)
        * (0.5 + humidity / 100)
    ).clip(0, 1000)
    bednet = rng.beta(3, 3, size=n)
    land_use = rng.choice(
        ["urban", "peri_urban", "rural_agriculture", "forest"],
        size=n, p=[0.20, 0.25, 0.40, 0.15],
    )

    base_rate = (
        0.08 * vector_density
        + 0.5 * humidity
        - 60 * bednet
        + rng.normal(0, 5, size=n)
    )
    base_rate = np.maximum(base_rate / 4, 0)
    cases = rng.poisson(base_rate)

    return pd.DataFrame({
        "region": region,
        "rainfall_mm": rainfall.round(1),
        "temperature_c": temperature.round(1),
        "humidity_pct": humidity.round(1),
        "vegetation_index": ndvi.round(3),
        "vector_density": vector_density.round(1),
        "bednet_coverage": bednet.round(3),
        "land_use": land_use,
        "disease_cases": cases,
    })


def skin_ntd(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = _seed(seed)
    region = rng.choice(ALL_REGIONS, size=n)
    humidity = rng.normal(72, 10, size=n).clip(20, 100)
    rainfall = rng.gamma(shape=2.5, scale=40, size=n).clip(0, 500)
    proximity = rng.exponential(scale=4.0, size=n).clip(0, 50)
    household = rng.poisson(lam=5, size=n).clip(1, 20)
    sanitation = rng.beta(2, 4, size=n)
    healthcare = rng.beta(2, 3, size=n)
    occupation = rng.choice(
        ["farming", "fishing", "mining", "trade", "education"],
        size=n, p=[0.40, 0.20, 0.10, 0.20, 0.10],
    )

    base_rate = (
        20 * np.exp(-proximity / 4)        # Buruli-like distance decay
        + 0.3 * humidity
        + 2 * household
        - 40 * healthcare
        - 30 * sanitation
        + rng.normal(0, 3, size=n)
    )
    base_rate = np.maximum(base_rate, 0)
    cases = rng.poisson(base_rate)

    return pd.DataFrame({
        "region": region,
        "humidity_pct": humidity.round(1),
        "rainfall_mm": rainfall.round(1),
        "proximity_to_water_km": proximity.round(2),
        "household_size": household.astype(int),
        "sanitation_index": sanitation.round(3),
        "healthcare_access_index": healthcare.round(3),
        "skin_contact_occupation": occupation,
        "disease_cases": cases,
    })


GENERATORS = {
    "water_borne": water_borne,
    "vector_borne": vector_borne,
    "skin_ntd": skin_ntd,
}


def generate_all(out_dir: Path | str = "data/samples") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in GENERATORS.items():
        df = fn()
        path = out_dir / f"{name}_seed.csv"
        df.to_csv(path, index=False)
        log.info("Wrote %s | shape=%s", path, df.shape)


if __name__ == "__main__":
    generate_all()
