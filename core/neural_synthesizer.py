"""Neural tabular synthesizers wrapped in the project's BaseSynthesizer interface.

Wraps SDV's CTGAN and TVAE so they are drop-in replacements for the
Gaussian Copula backend in benchmarking, the API, or downstream code.

Why these are kept in a separate file
-------------------------------------
SDV pulls in PyTorch (~800 MB), which exceeds Vercel's 250 MB serverless
function size limit. Importing this module is therefore opt-in: the rest
of the system never imports it, and Vercel deployments don't ship PyTorch.

Install with:
    pip install -r requirements-bench.txt
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .base_synthesizer import BaseSynthesizer
from .logging_config import get_logger

log = get_logger(__name__)


def _build_metadata(df: pd.DataFrame, categorical_cols: list[str]):
    """Build SDV Metadata for a single-table DataFrame.

    Avoids the auto-detection step which mis-types ID-like integer columns
    on small datasets.
    """
    from sdv.metadata import SingleTableMetadata

    md = SingleTableMetadata()
    for col in df.columns:
        if col in categorical_cols:
            md.add_column(col, sdtype="categorical")
        elif pd.api.types.is_numeric_dtype(df[col]):
            md.add_column(col, sdtype="numerical")
        else:
            md.add_column(col, sdtype="categorical")
    return md


class _SDVAdapter(BaseSynthesizer):
    """Common scaffolding shared by CTGAN and TVAE adapters."""

    name = "sdv_base"
    _sdv_class: type | None = None  # subclass overrides

    def __init__(self, seed: int | None = 42, epochs: int = 150, **kwargs) -> None:
        self.seed = seed
        self.epochs = epochs
        self.extra_kwargs = kwargs
        self._model = None
        self._columns: list[str] = []
        self._dtypes: dict[str, str] = {}
        self._categorical_cols: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str] | None = None,
    ) -> "_SDVAdapter":
        if df.empty or len(df) < 10:
            raise ValueError(
                f"{self.name} requires at least 10 rows; got {len(df)}."
            )

        if categorical_cols is None:
            categorical_cols = [
                c for c in df.columns
                if df[c].dtype == object or str(df[c].dtype) == "category"
            ]

        self._columns = list(df.columns)
        self._dtypes = {c: str(df[c].dtype) for c in df.columns}
        self._categorical_cols = list(categorical_cols)

        metadata = _build_metadata(df, categorical_cols)

        # SDV's CTGAN/TVAE accept epochs and a torch device; we keep CPU
        # by default for consistency with the deployment story.
        model = self._sdv_class(  # type: ignore[misc]
            metadata,
            epochs=self.epochs,
            verbose=False,
            **self.extra_kwargs,
        )
        log.info("Fitting %s | rows=%d cols=%d epochs=%d",
                 self.name, len(df), len(df.columns), self.epochs)
        model.fit(df)
        self._model = model
        log.info("Fitted %s", self.name)
        return self

    def sample(
        self,
        n: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError(f"{self.name} not fitted; call fit() first.")

        if conditions:
            # Use rejection sampling for fair comparison across backends.
            # SDV does support conditional sampling but the API differs by
            # version; rejection works uniformly and matches what the
            # Gaussian Copula backend does.
            target = n
            collected: list[pd.DataFrame] = []
            attempts, max_attempts = 0, 20
            oversample = max(target * 5, 200)
            while sum(len(d) for d in collected) < target and attempts < max_attempts:
                draw = self._model.sample(num_rows=oversample)
                mask = pd.Series(True, index=draw.index)
                for col, val in conditions.items():
                    mask &= draw[col] == val
                collected.append(draw[mask])
                attempts += 1
            out = pd.concat(collected, ignore_index=True).head(target)
            log.info("Sampled %d/%d rows from %s with conditions=%s",
                     len(out), target, self.name, conditions)
        else:
            out = self._model.sample(num_rows=n)
            log.info("Sampled %d rows from %s", n, self.name)

        # Restore column order
        return out[self._columns]

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save: model not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # SDV models are picklable
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._model, f)
        meta = {
            "backend": self.name,
            "columns": self._columns,
            "dtypes": self._dtypes,
            "categorical_cols": self._categorical_cols,
            "epochs": self.epochs,
            "seed": self.seed,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "_SDVAdapter":
        path = Path(path)
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        with open(path.with_suffix(".pkl"), "rb") as f:
            model = pickle.load(f)
        obj = cls(seed=meta["seed"], epochs=meta["epochs"])
        obj._model = model
        obj._columns = meta["columns"]
        obj._dtypes = meta["dtypes"]
        obj._categorical_cols = meta["categorical_cols"]
        return obj


class CTGANSynthesizer(_SDVAdapter):
    """SDV CTGAN wrapped in our BaseSynthesizer interface.

    CTGAN models the joint via a conditional GAN with mode-specific
    normalisation for continuous columns and one-hot encoding plus a
    training-by-sampling scheme for categorical columns.

    Reference: Xu et al., "Modeling Tabular Data using Conditional GAN",
    NeurIPS 2019.
    """

    name = "ctgan"

    def __init__(self, seed: int | None = 42, epochs: int = 150, **kwargs) -> None:
        from sdv.single_table import CTGANSynthesizer as _SDVCTGAN
        self._sdv_class = _SDVCTGAN
        super().__init__(seed=seed, epochs=epochs, **kwargs)


class TVAESynthesizer(_SDVAdapter):
    """SDV TVAE wrapped in our BaseSynthesizer interface.

    TVAE is the variational autoencoder counterpart to CTGAN, often more
    stable and faster to train on small datasets but sometimes weaker on
    sharp categorical structure.

    Reference: Xu et al., "Modeling Tabular Data using Conditional GAN",
    NeurIPS 2019 (TVAE is the VAE baseline introduced in the same paper).
    """

    name = "tvae"

    def __init__(self, seed: int | None = 42, epochs: int = 150, **kwargs) -> None:
        from sdv.single_table import TVAESynthesizer as _SDVTVAE
        self._sdv_class = _SDVTVAE
        super().__init__(seed=seed, epochs=epochs, **kwargs)
