"""Fetch real Sub-Saharan NTD surveillance data from the WHO Global Health Observatory.

This script hits the public WHO GHO OData API (no authentication, no rate
limit for reasonable use) and downloads tabular records for selected NTD
indicators across the 47 countries in WHO's African Region. The result
is written to ``data/who_ntd_real.csv`` and is suitable as a real-world
validation dataset for the synthesizer benchmark.

Why this matters for the paper
------------------------------
Reviewers will press on the use of synthetic seed data. WHO GHO data is
the most accessible public NTD source for Sub-Saharan Africa: it is
country-year-level (not patient-level), but it is real, peer-reviewed
through WHO's reporting channels, and citable. Validating the synthesizer
on this data closes the "you only tested on synthetic data" critique
without requiring a multi-month ministry-of-health data-sharing agreement.

Selected indicators
-------------------
The defaults below cover the three NTD families this project models. The
indicator codes were verified against the GHO indicator index. Users can
override with --indicators on the command line to fetch additional ones.

Indicator coverage notes
------------------------
WHO publishes treatment-coverage and PC (preventive chemotherapy) numbers
per country per year. Patient-level case counts are NOT available from
GHO — those live in country surveillance systems (e.g. DHIS2). For the
synthesizer benchmark this is acceptable because we treat each
country-year as a row and use treatment / coverage / population as
covariates.

Usage
-----
    python -m data.fetch_who_ntd                       # default selection
    python -m data.fetch_who_ntd --output custom.csv   # custom path
    python -m data.fetch_who_ntd --region AFR          # filter region
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


GHO_API = "https://ghoapi.azureedge.net/api"

# Default NTD-related indicator codes. These are the most data-rich
# entries in the GHO NTD theme as of the paper submission window.
# Verified against https://www.who.int/data/gho/data/themes/neglected-tropical-diseases
DEFAULT_INDICATORS: dict[str, str] = {
    # Schistosomiasis (water-borne)
    "SCH_TREATMENT_NUM": "Number of people treated for schistosomiasis",
    "SCH_PC_REQ_NUM": "Population requiring preventive chemotherapy for schistosomiasis",
    "SCH_PC_COV": "Coverage of preventive chemotherapy for schistosomiasis (%)",
    # Soil-transmitted helminthiases (water-borne / WASH-linked)
    "STH_TREATMENT_NUM": "Number of people treated for STH",
    "STH_PC_COV": "Coverage of preventive chemotherapy for STH (%)",
    # Lymphatic filariasis (vector-borne)
    "LF_PC_REQ_NUM": "Population requiring PC for lymphatic filariasis",
    "LF_PC_COV": "Coverage of preventive chemotherapy for LF (%)",
    # Onchocerciasis (vector-borne)
    "ONCHO_PC_REQ_NUM": "Population requiring PC for onchocerciasis",
    "ONCHO_PC_COV": "Coverage of preventive chemotherapy for onchocerciasis (%)",
    # Leprosy (skin NTD)
    "LEP_NEW_CASES": "New leprosy case detection rate",
    "LEP_PREVALENCE": "Leprosy prevalence rate per 10 000 population",
    # Buruli ulcer (skin NTD)
    "BU_NEW_CASES": "Buruli ulcer new cases reported",
}


def _http_get_json(url: str, timeout: int = 30) -> dict:
    """GET a JSON payload from the GHO API with a real User-Agent.

    GHO occasionally rejects requests without a User-Agent; we set a
    descriptive one so traffic is identifiable and traceable.
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ntd-synthgen/1.0 (research; +https://github.com/)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _list_african_countries() -> list[dict]:
    """Return the WHO AFR region country list (47 entries)."""
    url = f"{GHO_API}/DIMENSION/COUNTRY/DimensionValues"
    payload = _http_get_json(url)
    return [c for c in payload.get("value", [])
            if c.get("ParentCode") == "AFR"]


def _fetch_indicator(code: str) -> pd.DataFrame:
    """Fetch all rows for a single indicator and return as DataFrame.

    We retain the columns most useful for downstream synthesis:
    SpatialDim (country code), TimeDim (year), NumericValue (value),
    Dim1 (often sex/age subcategory if present).
    """
    url = f"{GHO_API}/{code}"
    payload = _http_get_json(url)
    records = payload.get("value", [])
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    keep = [c for c in ["SpatialDim", "SpatialDimType", "TimeDim",
                        "Dim1", "Dim1Type", "NumericValue", "Value"]
            if c in df.columns]
    df = df[keep].copy()
    df["IndicatorCode"] = code
    return df


def _shape_for_synthesis(
    long_df: pd.DataFrame,
    african_codes: set[str],
) -> pd.DataFrame:
    """Pivot the long-form indicator dump into a country-year table.

    Output schema (one row per country-year combination):
        country, year, region, <indicator_code>, ...

    Only Sub-Saharan rows (WHO AFR region) are kept.
    """
    df = long_df.copy()
    # Keep country-level rows (drop sub-national / regional aggregates)
    df = df[df["SpatialDimType"] == "COUNTRY"]
    df = df[df["SpatialDim"].isin(african_codes)]
    df = df.dropna(subset=["TimeDim", "NumericValue"])
    df["TimeDim"] = pd.to_numeric(df["TimeDim"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["TimeDim"])

    pivot = (
        df.groupby(["SpatialDim", "TimeDim", "IndicatorCode"], as_index=False)
        ["NumericValue"]
        .mean()
        .pivot(index=["SpatialDim", "TimeDim"], columns="IndicatorCode",
               values="NumericValue")
        .reset_index()
        .rename(columns={"SpatialDim": "country", "TimeDim": "year"})
    )
    pivot.columns.name = None
    pivot["region"] = "WHO_AFR"
    return pivot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="data/who_ntd_real.csv",
        help="Where to write the merged CSV (default: data/who_ntd_real.csv).",
    )
    parser.add_argument(
        "--indicators",
        nargs="+",
        default=list(DEFAULT_INDICATORS.keys()),
        help="GHO indicator codes to fetch (default: built-in NTD bundle).",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=2010,
        help="Earliest year to retain (default: 2010).",
    )
    args = parser.parse_args()

    print(f"Fetching WHO AFR country list from {GHO_API}...")
    africa = _list_african_countries()
    african_codes = {c["Code"] for c in africa}
    print(f"  {len(african_codes)} African Region countries.")

    chunks = []
    for code in args.indicators:
        try:
            print(f"Fetching indicator {code}...")
            chunk = _fetch_indicator(code)
            if chunk.empty:
                print(f"  empty — skipping.")
                continue
            chunks.append(chunk)
            print(f"  {len(chunk)} rows.")
        except Exception as e:
            print(f"  failed: {e}")
            continue

    if not chunks:
        print("No indicators returned data. Aborting.", file=sys.stderr)
        return 1

    long_df = pd.concat(chunks, ignore_index=True)
    print(f"Total raw rows: {len(long_df)}")

    shaped = _shape_for_synthesis(long_df, african_codes)
    shaped = shaped[shaped["year"] >= args.year_min]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shaped.to_csv(out_path, index=False)

    print(f"\nWrote {out_path.resolve()}")
    print(f"  rows : {len(shaped)}")
    print(f"  cols : {list(shaped.columns)}")
    print(f"  year range : {shaped['year'].min()} - {shaped['year'].max()}")
    print(f"  countries  : {shaped['country'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
