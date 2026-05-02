"""Bayesian Network tabular synthesizer.

Discretises continuous columns, learns a DAG via Hill-Climb search with
the BIC score, fits CPDs by maximum likelihood, and samples from the
joint via forward (ancestral) sampling.

This is the graphical-model baseline in the benchmark. It contrasts with
the copula approach (which captures global pairwise structure but no
explicit conditional independencies) and the neural backends (which
trade interpretability for capacity).

Why include it
--------------
1. Different algorithmic family — strengthens the empirical comparison.
2. Lightweight (CPU, < 100 MB), so it could be a Vercel-compatible
   alternative if a deployment ever needed it.
3. Interpretable for public health users — the learned DAG is itself a
   research artefact (suggests which variables drive disease outcomes).
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base_synthesizer import BaseSynthesizer
from .logging_config import get_logger

log = get_logger(__name__)

# Suppress pgmpy's deprecation chatter; not actionable from our side.
warnings.filterwarnings("ignore", category=FutureWarning, module="pgmpy")
warnings.filterwarnings("ignore", category=UserWarning, module="pgmpy")


class BayesianNetworkSynthesizer(BaseSynthesizer):
    """Discrete Bayesian Network synthesizer.

    Pipeline:
      1. Discretise every continuous column into `n_bins` quantile bins.
      2. Learn DAG structure via Hill-Climb search with BIC scoring.
      3. Fit CPDs with maximum likelihood.
      4. Sample categorical assignments from the joint, then map each
         binned-continuous column back to a uniform draw within its bin.
    """

    name = "bayesian_network"

    def __init__(
        self,
        seed: int | None = 42,
        n_bins: int = 8,
        max_iter: int = 50,
    ) -> None:
        self.seed = seed
        self.n_bins = n_bins
        self.max_iter = max_iter
        self._rng = np.random.default_rng(seed)
        self._model = None
        self._columns: list[str] = []
        self._dtypes: dict[str, str] = {}
        self._categorical_cols: list[str] = []
        self._bin_edges: dict[str, np.ndarray] = {}

    # ---- discretisation helpers ------------------------------------------

    def _discretise(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Bin continuous columns into integer bin labels.

        On fit, computes quantile bin edges and stores them. On sample,
        uses the stored edges so train/sample alignment is exact.
        """
        out = df.copy()
        for col in df.columns:
            if col in self._categorical_cols:
                out[col] = df[col].astype(str)
                continue
            if fit:
                # Quantile binning — robust to skew.
                edges = np.quantile(
                    df[col].to_numpy(),
                    np.linspace(0, 1, self.n_bins + 1),
                )
                # Make edges strictly increasing in case of ties.
                edges = np.unique(edges)
                if len(edges) < 2:
                    edges = np.array([df[col].min(), df[col].max() + 1e-9])
                self._bin_edges[col] = edges
            edges = self._bin_edges[col]
            # Use [edges[0], edges[-1]] so all values fall in a bin.
            out[col] = pd.cut(
                df[col],
                bins=edges,
                labels=False,
                include_lowest=True,
            )
            # Any NaNs from out-of-range values get clipped to nearest bin.
            out[col] = out[col].fillna(0).astype(int).astype(str)
        return out

    def _undiscretise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map binned-continuous columns back to uniform draws within bins."""
        out = df.copy()
        for col in df.columns:
            if col in self._categorical_cols:
                continue
            edges = self._bin_edges[col]
            bin_idx = df[col].astype(int).to_numpy()
            bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
            lo = edges[bin_idx]
            hi = edges[bin_idx + 1]
            out[col] = self._rng.uniform(lo, hi)
            # Restore original dtype where possible
            if "int" in self._dtypes[col]:
                out[col] = out[col].round().astype(int)
            else:
                out[col] = out[col].astype(float)
        return out

    # ---- BaseSynthesizer interface ---------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str] | None = None,
    ) -> "BayesianNetworkSynthesizer":
        if df.empty or len(df) < 10:
            raise ValueError(
                f"BayesianNetwork requires at least 10 rows; got {len(df)}."
            )

        # pgmpy imports are lazy because of its noisy deprecation warnings.
        from pgmpy.estimators import HillClimbSearch
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.parameter_estimator import DiscreteMLE

        if categorical_cols is None:
            categorical_cols = [
                c for c in df.columns
                if df[c].dtype == object or str(df[c].dtype) == "category"
            ]

        self._columns = list(df.columns)
        self._dtypes = {c: str(df[c].dtype) for c in df.columns}
        self._categorical_cols = list(categorical_cols)

        log.info("Fitting BayesianNetwork | rows=%d cols=%d bins=%d",
                 len(df), len(df.columns), self.n_bins)

        disc = self._discretise(df, fit=True)

        # Structure learning. 'bic-d' = BIC for purely discrete data,
        # which is what we have after the discretisation step above.
        hc = HillClimbSearch(disc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_dag = hc.estimate(
                scoring_method="bic-d",
                max_iter=self.max_iter,
                show_progress=False,
            )

        model = DiscreteBayesianNetwork(best_dag.edges())
        # Add isolated nodes (HC may drop columns with no detected edges)
        for col in self._columns:
            if col not in model.nodes():
                model.add_node(col)

        # Parameter learning. DiscreteMLE() takes no model in its
        # constructor; the model and data are passed at fit time, which
        # `DiscreteBayesianNetwork.fit` does internally.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(disc, estimator=DiscreteMLE())

        self._model = model
        log.info("Fitted BayesianNetwork | edges=%d nodes=%d",
                 len(model.edges()), len(model.nodes()))
        return self

    def sample(
        self,
        n: int,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("BayesianNetwork not fitted; call fit() first.")

        from pgmpy.sampling import BayesianModelSampling

        sampler = BayesianModelSampling(self._model)

        # We always oversample then post-filter for conditions (uniform
        # behaviour with the other backends).
        target = n
        oversample = max(n * 5, 500) if conditions else n

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            disc_draw = sampler.forward_sample(
                size=oversample,
                seed=self._rng.integers(0, 2**31 - 1),
                show_progress=False,
            )

        # Reorder to original column order
        disc_draw = disc_draw[self._columns]

        # Un-discretise continuous columns
        cont_draw = self._undiscretise(disc_draw)

        if conditions:
            mask = pd.Series(True, index=cont_draw.index)
            for col, val in conditions.items():
                mask &= cont_draw[col] == val
            cont_draw = cont_draw[mask].head(target).reset_index(drop=True)
        else:
            cont_draw = cont_draw.head(target).reset_index(drop=True)

        log.info("Sampled %d rows from BayesianNetwork (conditions=%s)",
                 len(cont_draw), conditions)
        return cont_draw

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save: model not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "bin_edges": self._bin_edges,
                },
                f,
            )
        meta = {
            "backend": self.name,
            "columns": self._columns,
            "dtypes": self._dtypes,
            "categorical_cols": self._categorical_cols,
            "n_bins": self.n_bins,
            "max_iter": self.max_iter,
            "seed": self.seed,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BayesianNetworkSynthesizer":
        path = Path(path)
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        with open(path.with_suffix(".pkl"), "rb") as f:
            blob = pickle.load(f)
        obj = cls(
            seed=meta["seed"],
            n_bins=meta["n_bins"],
            max_iter=meta["max_iter"],
        )
        obj._model = blob["model"]
        obj._bin_edges = blob["bin_edges"]
        obj._columns = meta["columns"]
        obj._dtypes = meta["dtypes"]
        obj._categorical_cols = meta["categorical_cols"]
        return obj
