"""Abstract base class defining the synthesizer interface.

Every concrete backend — Gaussian Copula, CTGAN, TVAE, Bayesian Network —
implements this same surface so the API layer, frontend, and benchmark
runner are completely backend-agnostic.

This is the seam where the project's "backend-agnostic" design claim
becomes verifiable rather than rhetorical.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseSynthesizer(ABC):
    """Minimum contract every synthesizer backend must satisfy.

    Concrete subclasses are free to add hyperparameters in their __init__
    but must accept a `seed` kwarg and implement the four methods below
    with these exact signatures.
    """

    name: str = "base"  # subclasses override for logging / display

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str] | None = None,
    ) -> "BaseSynthesizer":
        """Fit the model to the real data.

        Args:
            df: training data (rows = samples, columns = features).
            categorical_cols: columns to treat as categorical. If None,
                inferred from dtypes (object/category).

        Returns:
            self, for chaining.
        """

    @abstractmethod
    def sample(
        self,
        n: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate `n` synthetic rows.

        Args:
            n: number of rows to return.
            conditions: optional dict of column -> value to condition on
                (e.g. {"region": "Lagos"}). Backends that do not support
                native conditional sampling should fall back to rejection.

        Returns:
            A DataFrame with the same columns and dtypes as the training
            data.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseSynthesizer":
        """Restore a fitted model from disk."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
