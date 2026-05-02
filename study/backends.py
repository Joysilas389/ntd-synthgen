"""
Unified backends for the comparison study.

Each backend exposes the same minimal interface:
    fit(df) -> None
    sample(n) -> pd.DataFrame

This file is study-only. The deployable engine in core/ is unchanged.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Silence SDV's chatty progress bars and deprecation warnings during benchmark runs.
warnings.filterwarnings("ignore")
import logging
logging.getLogger("sdv").setLevel(logging.ERROR)
logging.getLogger("rdt").setLevel(logging.ERROR)
logging.getLogger("copulas").setLevel(logging.ERROR)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.synthesizer import GaussianCopulaSynthesizer as OurCopula


class Backend:
    """Base class for benchmark backends."""

    name: str = "base"

    def __init__(self) -> None:
        self.fit_seconds: float = 0.0
        self.sample_seconds: float = 0.0

    def fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def sample(self, n: int) -> pd.DataFrame:
        raise NotImplementedError

    def disk_bytes(self, tmp_path: Path) -> int:
        """Save the fitted model and return its size on disk."""
        raise NotImplementedError


class OursBackend(Backend):
    """Our hand-rolled Gaussian Copula (the deployable one)."""

    name = "ours_copula"

    def __init__(self) -> None:
        super().__init__()
        self.model = OurCopula()

    def fit(self, df: pd.DataFrame) -> None:
        t = time.perf_counter()
        self.model.fit(df)
        self.fit_seconds = time.perf_counter() - t

    def sample(self, n: int) -> pd.DataFrame:
        t = time.perf_counter()
        out = self.model.sample(n)
        self.sample_seconds = time.perf_counter() - t
        return out

    def disk_bytes(self, tmp_path: Path) -> int:
        path = tmp_path / "ours.npz"
        self.model.save(str(path))
        return path.stat().st_size


class _SDVBackend(Backend):
    """Shared logic for SDV-based backends."""

    sdv_class: Any = None
    sdv_kwargs: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.metadata = None

    def _build_metadata(self, df: pd.DataFrame):
        from sdv.metadata import SingleTableMetadata

        md = SingleTableMetadata()
        md.detect_from_dataframe(df)
        return md

    def fit(self, df: pd.DataFrame) -> None:
        self.metadata = self._build_metadata(df)
        self.model = self.sdv_class(metadata=self.metadata, **self.sdv_kwargs)
        t = time.perf_counter()
        self.model.fit(df)
        self.fit_seconds = time.perf_counter() - t

    def sample(self, n: int) -> pd.DataFrame:
        t = time.perf_counter()
        out = self.model.sample(num_rows=n)
        self.sample_seconds = time.perf_counter() - t
        return out

    def disk_bytes(self, tmp_path: Path) -> int:
        path = tmp_path / f"{self.name}.pkl"
        self.model.save(str(path))
        return path.stat().st_size


class SDVCopulaBackend(_SDVBackend):
    name = "sdv_copula"

    def __init__(self) -> None:
        super().__init__()
        from sdv.single_table import GaussianCopulaSynthesizer as SDVCopula
        self.sdv_class = SDVCopula
        self.sdv_kwargs = {}


class CTGANBackend(_SDVBackend):
    name = "ctgan"

    def __init__(self, epochs: int = 100) -> None:
        super().__init__()
        from sdv.single_table import CTGANSynthesizer
        self.sdv_class = CTGANSynthesizer
        # 100 epochs keeps the benchmark tractable on CPU; reviewers can scale up.
        self.sdv_kwargs = {"epochs": epochs, "verbose": False}


class TVAEBackend(_SDVBackend):
    name = "tvae"

    def __init__(self, epochs: int = 100) -> None:
        super().__init__()
        from sdv.single_table import TVAESynthesizer
        self.sdv_class = TVAESynthesizer
        self.sdv_kwargs = {"epochs": epochs}


def all_backends() -> list[Backend]:
    """Factory returning a fresh instance of each backend."""
    return [
        OursBackend(),
        SDVCopulaBackend(),
        CTGANBackend(epochs=100),
        TVAEBackend(epochs=100),
    ]
