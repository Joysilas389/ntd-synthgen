"""Tests for the Gaussian Copula synthesizer."""
from __future__ import annotations

import pandas as pd
import pytest

from core.synthesizer import GaussianCopulaSynthesizer


def test_fit_and_sample_shape(water_df: pd.DataFrame):
    m = GaussianCopulaSynthesizer(seed=0)
    m.fit(water_df, categorical_cols=["region", "water_source"])
    out = m.sample(150)
    assert out.shape[0] == 150
    assert list(out.columns) == list(water_df.columns)


def test_categorical_marginal_preserved(water_df: pd.DataFrame):
    m = GaussianCopulaSynthesizer(seed=1)
    m.fit(water_df, categorical_cols=["region", "water_source"])
    out = m.sample(2000)
    real_p = water_df["water_source"].value_counts(normalize=True).sort_index()
    synth_p = out["water_source"].astype(str).value_counts(normalize=True).sort_index()
    # Allow up to 0.10 TVD on a small sample.
    tvd = 0.5 * (real_p - synth_p).abs().sum()
    assert tvd < 0.15


def test_conditional_sampling_filters_categorical(water_df: pd.DataFrame):
    m = GaussianCopulaSynthesizer(seed=2)
    m.fit(water_df, categorical_cols=["region", "water_source"])
    out = m.sample(50, conditions={"region": "Lagos"})
    assert len(out) > 0
    assert (out["region"] == "Lagos").all()


def test_save_and_load_round_trip(tmp_path, water_df: pd.DataFrame):
    m = GaussianCopulaSynthesizer(seed=3)
    m.fit(water_df, categorical_cols=["region", "water_source"])
    p = tmp_path / "model.pkl"
    m.save(p)
    m2 = GaussianCopulaSynthesizer.load(p)
    out = m2.sample(20)
    assert out.shape == (20, water_df.shape[1])


def test_fit_rejects_tiny_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    m = GaussianCopulaSynthesizer()
    with pytest.raises(ValueError):
        m.fit(df)


def test_fit_rejects_empty_df():
    m = GaussianCopulaSynthesizer()
    with pytest.raises(ValueError):
        m.fit(pd.DataFrame())


def test_sample_before_fit_raises():
    m = GaussianCopulaSynthesizer()
    with pytest.raises(RuntimeError):
        m.sample(5)
