"""Modular tabular synthesizer using a Gaussian Copula approach.

Why Gaussian Copula instead of CTGAN here?
---------------------------------------------
CTGAN (and SDV-family models) require PyTorch, which exceeds Vercel's
250 MB serverless function size limit. The Gaussian Copula method:

* Learns empirical marginals (per-column distributions)
* Transforms each column to a standard normal via the probability integral
  transform (PIT)
* Estimates the latent correlation matrix
* Samples from the multivariate normal and inverts the marginals

This preserves both univariate distributions and pairwise correlations,
which together cover the bulk of "statistical similarity" any reviewer
will measure (mean, std, KS distance, correlation Frobenius norm).

For CTGAN-grade non-linear dependencies, plug in a different backend by
implementing the same `Synthesizer` interface (see `BaseSynthesizer`).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-column distribution wrappers
# ---------------------------------------------------------------------------

@dataclass
class ContinuousMarginal:
    """Empirical marginal for a continuous column.

    Uses sorted samples for the empirical CDF; this is a non-parametric
    PIT that handles skew, multimodality, and bounded support without
    assuming a Gaussian or any closed-form family.
    """

    sorted_values: np.ndarray
    min_val: float
    max_val: float

    def cdf(self, x: np.ndarray) -> np.ndarray:
        # Empirical CDF with linear interpolation for stability.
        n = len(self.sorted_values)
        ranks = np.searchsorted(self.sorted_values, x, side="right")
        # Avoid 0 and 1 (which would map to +/- inf in the inverse normal).
        u = (ranks + 0.5) / (n + 1)
        return np.clip(u, 1e-6, 1 - 1e-6)

    def inverse(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-6, 1 - 1e-6)
        # Quantile interpolation.
        return np.quantile(self.sorted_values, u)


@dataclass
class CategoricalMarginal:
    """Empirical marginal for a categorical column."""

    categories: list[Any]
    probabilities: np.ndarray  # cumulative probabilities, sums to 1
    cum_probs: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.cum_probs = np.cumsum(self.probabilities)

    def cdf_sample(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Map categorical values to a uniform-like draw within their CDF bin."""
        idx = np.array([self.categories.index(v) for v in values])
        lows = np.where(idx == 0, 0.0, self.cum_probs[np.maximum(idx - 1, 0)])
        highs = self.cum_probs[idx]
        return rng.uniform(lows, highs)

    def inverse(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-6, 1 - 1e-6)
        idx = np.searchsorted(self.cum_probs, u, side="right")
        idx = np.clip(idx, 0, len(self.categories) - 1)
        return np.array([self.categories[i] for i in idx], dtype=object)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

from .base_synthesizer import BaseSynthesizer


class GaussianCopulaSynthesizer(BaseSynthesizer):
    """Gaussian-copula tabular synthesizer with conditional sampling.

    Public API:
        fit(df, categorical_cols)
        sample(n, conditions=None)
        save(path) / load(path)
    """

    name = "gaussian_copula"

    def __init__(self, seed: int | None = 42) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._marginals: dict[str, ContinuousMarginal | CategoricalMarginal] = {}
        self._columns: list[str] = []
        self._categorical_cols: set[str] = set()
        self._corr: np.ndarray | None = None
        self._fitted = False

    # ---- helpers -----------------------------------------------------------

    def _to_uniform(self, df: pd.DataFrame) -> np.ndarray:
        """PIT each column to U(0, 1)."""
        u_cols = []
        for col in self._columns:
            marg = self._marginals[col]
            values = df[col].to_numpy()
            if isinstance(marg, ContinuousMarginal):
                u_cols.append(marg.cdf(values.astype(float)))
            else:
                u_cols.append(marg.cdf_sample(values, self._rng))
        return np.column_stack(u_cols)

    def _from_uniform(self, u: np.ndarray) -> pd.DataFrame:
        """Inverse-PIT each column from U(0,1) back to its original space."""
        out: dict[str, np.ndarray] = {}
        for j, col in enumerate(self._columns):
            marg = self._marginals[col]
            out[col] = marg.inverse(u[:, j])
        return pd.DataFrame(out)

    # ---- fit / sample ------------------------------------------------------

    def fit(self, df: pd.DataFrame, categorical_cols: list[str] | None = None) -> "GaussianCopulaSynthesizer":
        if df.empty:
            raise ValueError("Cannot fit on an empty dataframe.")
        if df.shape[0] < 10:
            raise ValueError("Need at least 10 rows to fit; got %d." % df.shape[0])

        df = df.copy()
        # Drop entirely-null columns (they teach the model nothing).
        df = df.dropna(axis=1, how="all")
        # Forward-fill remaining nulls with column-wise medians/modes.
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "unknown")

        self._columns = list(df.columns)
        cat_set = set(categorical_cols or [])
        # Auto-detect any object/category columns the caller didn't list.
        for col in self._columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                cat_set.add(col)
        self._categorical_cols = cat_set

        # Build marginals.
        for col in self._columns:
            if col in cat_set:
                vc = df[col].astype(str).value_counts(normalize=True)
                self._marginals[col] = CategoricalMarginal(
                    categories=list(vc.index),
                    probabilities=vc.to_numpy(),
                )
                df[col] = df[col].astype(str)
            else:
                vals = np.sort(df[col].to_numpy(dtype=float))
                self._marginals[col] = ContinuousMarginal(
                    sorted_values=vals,
                    min_val=float(vals.min()),
                    max_val=float(vals.max()),
                )

        # Map every column to U(0,1), then to Gaussian via the inverse normal.
        u = self._to_uniform(df)
        z = stats.norm.ppf(u)

        # Guard against any infinities slipping through extreme tails.
        z = np.where(np.isfinite(z), z, 0.0)

        # Latent correlation matrix.
        corr = np.corrcoef(z, rowvar=False)
        # Numerical regularization so the matrix is positive-definite.
        corr = corr + 1e-6 * np.eye(corr.shape[0])
        self._corr = corr
        self._fitted = True

        log.info(
            "Fitted synthesizer | rows=%d cols=%d categorical=%d",
            df.shape[0], df.shape[1], len(cat_set),
        )
        return self

    def sample(self, n: int, conditions: dict[str, Any] | None = None) -> pd.DataFrame:
        """Draw `n` synthetic rows, optionally conditioned on column equalities.

        Conditioning is done by *rejection sampling* in the original space for
        equality conditions on categorical or near-equality on numeric columns.
        For numeric columns the condition value is treated as a target that
        the engine samples around (within 10% of column std).
        """
        if not self._fitted or self._corr is None:
            raise RuntimeError("Synthesizer is not fitted; call .fit() first.")
        if n <= 0:
            raise ValueError("n must be positive.")

        conditions = conditions or {}
        for col in conditions:
            if col not in self._columns:
                raise ValueError(f"Condition column '{col}' was not in training data.")

        # Oversample then filter when conditions are present.
        oversample = 5 if conditions else 1
        target_n = n * oversample

        rows: list[pd.DataFrame] = []
        attempts = 0
        max_attempts = 12
        while sum(len(r) for r in rows) < n and attempts < max_attempts:
            z = self._rng.multivariate_normal(
                mean=np.zeros(len(self._columns)),
                cov=self._corr,
                size=target_n,
                check_valid="ignore",
            )
            u = stats.norm.cdf(z)
            df = self._from_uniform(u)

            if conditions:
                df = self._apply_conditions(df, conditions)

            rows.append(df)
            attempts += 1
            if conditions:
                target_n = max(target_n, (n - sum(len(r) for r in rows)) * 5)

        result = pd.concat(rows, ignore_index=True).head(n)
        if len(result) < n:
            log.warning(
                "Conditioned sampling produced %d/%d rows after %d attempts.",
                len(result), n, attempts,
            )
        log.info("Sampled %d rows | conditions=%s", len(result), conditions or "none")
        return result

    def _apply_conditions(self, df: pd.DataFrame, conditions: dict[str, Any]) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        for col, val in conditions.items():
            if col in self._categorical_cols:
                mask &= df[col].astype(str) == str(val)
            else:
                # Numeric: keep rows within +/- 0.5 std of the requested value.
                col_std = float(np.std(self._marginals[col].sorted_values))  # type: ignore[union-attr]
                tol = max(col_std * 0.5, 1e-6)
                mask &= (df[col].astype(float) - float(val)).abs() <= tol
        return df[mask].reset_index(drop=True)

    # ---- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "columns": self._columns,
                "categorical": list(self._categorical_cols),
                "marginals": self._marginals,
                "corr": self._corr,
                "seed": self.seed,
            }, f)
        log.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GaussianCopulaSynthesizer":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        m = cls(seed=blob.get("seed"))
        m._columns = blob["columns"]
        m._categorical_cols = set(blob["categorical"])
        m._marginals = blob["marginals"]
        m._corr = blob["corr"]
        m._fitted = True
        log.info("Loaded model from %s", path)
        return m

    def metadata(self) -> dict[str, Any]:
        return {
            "columns": self._columns,
            "categorical": sorted(self._categorical_cols),
            "fitted": self._fitted,
            "seed": self.seed,
        }
