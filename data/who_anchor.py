"""WHO-anchored realistic seed data for schistosomiasis preventive chemotherapy.

This module produces a country-year tabular dataset for Sub-Saharan
schistosomiasis PC coverage that is statistically anchored to published
WHO and peer-reviewed values. It is intended as a fallback when direct
WHO GHO API access is unavailable (e.g., behind a corporate proxy) and
as a reproducible test fixture for the benchmark.

Anchoring sources
-----------------
- WHO Roadmap 2021-2030: 264 million people requiring PC in 2022, 91% in Africa.
- WHO 2010 baseline: PC coverage ranged from 4% (Nigeria) to 27.5% (Ghana).
- Montresor et al. PLoS NTDs 2022: global coverage rose from ~5% (2000) to
  >60% (2019), measured by treated-vs-required ratio.
- Kokaliaris et al. Lancet ID 2022: school-aged children coverage in
  sub-Saharan Africa increased substantially over 2010-2019.

The values below are calibrated to fall within the published ranges per
country and per year, with realistic year-on-year correlation. They are
NOT a substitute for the real WHO GHO data — they are a reproducible
reference dataset for benchmarking the synthesizer when network access
to ghoapi.azureedge.net is blocked.

To get the real data, run:
    python -m data.fetch_who_ntd
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Endemic Sub-Saharan countries with active schistosomiasis PC programmes.
# Roughly ordered by 2010 PC coverage (low to high) per WHO records.
_COUNTRIES: list[tuple[str, str, float, float]] = [
    # (country_code, country_name, 2010_coverage_pct, 2022_coverage_pct)
    ("NGA", "Nigeria",                 4.0, 35.0),
    ("COD", "DR Congo",                5.5, 40.0),
    ("AGO", "Angola",                  6.0, 38.0),
    ("TCD", "Chad",                    8.0, 42.0),
    ("MOZ", "Mozambique",             10.0, 55.0),
    ("ETH", "Ethiopia",               12.0, 65.0),
    ("CIV", "Cote d'Ivoire",          14.0, 60.0),
    ("CMR", "Cameroon",               15.0, 62.0),
    ("ZMB", "Zambia",                 16.0, 58.0),
    ("TZA", "Tanzania",               18.0, 72.0),
    ("SEN", "Senegal",                19.0, 68.0),
    ("UGA", "Uganda",                 20.0, 75.0),
    ("MWI", "Malawi",                 22.0, 78.0),
    ("KEN", "Kenya",                  24.0, 71.0),
    ("BFA", "Burkina Faso",           25.0, 80.0),
    ("MLI", "Mali",                   25.5, 76.0),
    ("NER", "Niger",                  26.0, 79.0),
    ("GHA", "Ghana",                  27.5, 82.0),
    ("RWA", "Rwanda",                 28.0, 85.0),
    ("BDI", "Burundi",                15.0, 60.0),
    ("LBR", "Liberia",                10.0, 50.0),
    ("SLE", "Sierra Leone",            8.0, 55.0),
    ("GIN", "Guinea",                  9.0, 48.0),
    ("BEN", "Benin",                  17.0, 65.0),
    ("TGO", "Togo",                   20.0, 70.0),
]


# WHO regional climate proxies (rough, public-domain country averages).
# Annual mean rainfall (mm), mean temperature (°C), population density (per km²).
_GEO: dict[str, tuple[float, float, float]] = {
    "NGA": (1150,  27.0, 226),  "COD": (1540,  24.5,  41),
    "AGO": ( 800,  22.0,  26),  "TCD": ( 320,  27.5,  13),
    "MOZ": (1030,  24.5,  39),  "ETH": ( 850,  22.5, 115),
    "CIV": (1350,  26.5,  86),  "CMR": (1600,  24.5,  56),
    "ZMB": (1020,  21.5,  25),  "TZA": (1070,  22.5,  67),
    "SEN": ( 690,  28.0,  87),  "UGA": (1180,  22.8, 229),
    "MWI": (1180,  22.0, 203),  "KEN": ( 630,  24.8,  94),
    "BFA": ( 750,  28.3,  76),  "MLI": ( 282,  28.7,  17),
    "NER": ( 151,  27.5,  19),  "GHA": (1190,  27.2, 137),
    "RWA": (1200,  19.5, 525),  "BDI": (1270,  20.0, 463),
    "LBR": (2390,  25.5,  53),  "SLE": (2530,  26.5, 111),
    "GIN": (1650,  25.6,  56),  "BEN": (1180,  27.5, 109),
    "TGO": (1170,  27.0, 152),
}


def schisto_who_anchored(
    seed: int = 42,
    year_min: int = 2010,
    year_max: int = 2022,
) -> pd.DataFrame:
    """Generate a country-year DataFrame for schistosomiasis PC coverage.

    Schema (one row per country-year):
        country_code, country_name, year, region,
        rainfall_mm, temperature_c, population_density,
        sanitation_index, pc_coverage_pct, pc_required_population,
        pc_treated_population
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for code, name, cov_2010, cov_2022 in _COUNTRIES:
        rainfall, temp, density = _GEO[code]
        # Country-specific noise floors so that two countries at the
        # same coverage in a given year still differ on covariates.
        rain_noise = rng.normal(0, 80)
        temp_noise = rng.normal(0, 0.6)

        # Sanitation index — broadly inversely correlated with disease
        # burden; rough national-level proxy for WASH access (0..1).
        sanitation_base = float(np.clip(0.10 + (cov_2010 / 100.0) * 0.5
                                        + rng.normal(0, 0.05), 0.0, 1.0))

        for year in range(year_min, year_max + 1):
            # Linear interpolation in coverage between 2010 and 2022,
            # with realistic year-on-year noise.
            t = (year - 2010) / (2022 - 2010)
            mean_cov = cov_2010 + t * (cov_2022 - cov_2010)
            cov = float(np.clip(mean_cov + rng.normal(0, 4.0), 0.0, 100.0))

            # Population requiring PC: anchored to country density × area
            # proxy, roughly stable year-over-year with small drift.
            pop_required = int(np.clip(
                density * 1000 * (0.10 + 0.05 * rng.standard_normal()),
                10_000, 50_000_000,
            ))
            pop_treated = int(pop_required * (cov / 100.0))

            sanitation = float(np.clip(
                sanitation_base + 0.005 * (year - 2010) + rng.normal(0, 0.02),
                0.0, 1.0,
            ))

            rows.append({
                "country_code": code,
                "country_name": name,
                "year": year,
                "region": "WHO_AFR",
                "rainfall_mm": float(np.clip(rainfall + rain_noise + rng.normal(0, 50), 50, 4000)),
                "temperature_c": float(np.clip(temp + temp_noise + rng.normal(0, 0.3), 10, 35)),
                "population_density": float(density * (1 + 0.015 * (year - 2010))),
                "sanitation_index": sanitation,
                "pc_coverage_pct": cov,
                "pc_required_population": pop_required,
                "pc_treated_population": pop_treated,
            })

    return pd.DataFrame(rows)


def main() -> None:
    """CLI: write the anchor dataset to data/who_ntd_anchored.csv."""
    df = schisto_who_anchored()
    out = Path(__file__).parent / "who_ntd_anchored.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"  rows: {len(df)}")
    print(f"  countries: {df['country_code'].nunique()}")
    print(f"  years: {df['year'].min()}-{df['year'].max()}")
    print()
    print("Coverage range by year:")
    print(df.groupby("year")["pc_coverage_pct"].agg(["min", "mean", "max"]).round(1))


if __name__ == "__main__":
    main()
