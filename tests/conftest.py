"""Shared test fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.seed_data import water_borne, vector_borne, skin_ntd  # noqa: E402


@pytest.fixture(scope="session")
def water_df() -> pd.DataFrame:
    return water_borne(n=200, seed=11)


@pytest.fixture(scope="session")
def vector_df() -> pd.DataFrame:
    return vector_borne(n=200, seed=12)


@pytest.fixture(scope="session")
def skin_df() -> pd.DataFrame:
    return skin_ntd(n=200, seed=13)
