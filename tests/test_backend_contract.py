"""Contract tests for the BaseSynthesizer interface.

Confirms that every concrete backend exposes the same surface and behaves
correctly on a small test dataset. CTGAN/TVAE tests are skipped when
their optional dependencies are not installed.
"""
from __future__ import annotations

import importlib
import warnings

import pytest

from core.base_synthesizer import BaseSynthesizer
from core.bayesnet_synthesizer import BayesianNetworkSynthesizer
from core.synthesizer import GaussianCopulaSynthesizer

warnings.filterwarnings("ignore")


# Lazy availability checks for the heavy backends
_HAS_SDV = importlib.util.find_spec("sdv") is not None


# ---- Common contract -----------------------------------------------------

@pytest.mark.parametrize(
    "backend_factory",
    [
        pytest.param(
            lambda: GaussianCopulaSynthesizer(seed=42),
            id="gaussian_copula",
        ),
        pytest.param(
            lambda: BayesianNetworkSynthesizer(seed=42, n_bins=5, max_iter=10),
            id="bayesian_network",
        ),
        pytest.param(
            lambda: _make_ctgan() if _HAS_SDV else None,
            id="ctgan",
            marks=pytest.mark.skipif(not _HAS_SDV, reason="SDV not installed"),
        ),
        pytest.param(
            lambda: _make_tvae() if _HAS_SDV else None,
            id="tvae",
            marks=pytest.mark.skipif(not _HAS_SDV, reason="SDV not installed"),
        ),
    ],
)
class TestSynthesizerContract:
    """Every backend must satisfy this minimal contract."""

    def test_is_base_synthesizer(self, backend_factory, water_df):
        s = backend_factory()
        assert isinstance(s, BaseSynthesizer)
        assert hasattr(s, "name") and isinstance(s.name, str)

    def test_fit_returns_self(self, backend_factory, water_df):
        s = backend_factory()
        out = s.fit(water_df)
        assert out is s

    def test_sample_shape_and_columns(self, backend_factory, water_df):
        s = backend_factory()
        s.fit(water_df)
        synth = s.sample(50)
        assert len(synth) == 50
        assert list(synth.columns) == list(water_df.columns)

    def test_sample_preserves_categorical_values(
        self, backend_factory, water_df,
    ):
        s = backend_factory()
        s.fit(water_df)
        synth = s.sample(200)
        real_regions = set(water_df["region"].unique())
        synth_regions = set(synth["region"].unique())
        # Synthetic regions must be a subset of real regions; backends
        # are not allowed to invent new categories.
        assert synth_regions.issubset(real_regions)


# ---- Helpers for optional neural backends --------------------------------

def _make_ctgan():
    from core.neural_synthesizer import CTGANSynthesizer
    return CTGANSynthesizer(seed=42, epochs=10)


def _make_tvae():
    from core.neural_synthesizer import TVAESynthesizer
    return TVAESynthesizer(seed=42, epochs=10)
