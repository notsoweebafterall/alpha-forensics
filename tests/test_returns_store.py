"""
Unit tests for returns_store module (Phase 13).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from statistics.returns_store import (
    save_returns,
    load_returns,
    load_returns_matrix,
)


def test_save_load_returns_roundtrip(tmp_path):
    store_file = tmp_path / "trial_returns.parquet"
    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    np.random.seed(42)
    series = pd.Series(np.random.randn(100), index=dates, name="test_cand_1")

    # Save returns
    save_returns("test_cand_1", series, store_path=store_file)

    assert store_file.exists()

    # Load returns
    loaded = load_returns("test_cand_1", store_path=store_file)
    assert loaded is not None
    assert len(loaded) == 100
    assert loaded.name == "test_cand_1"
    np.testing.assert_allclose(loaded.values, series.values, rtol=1e-5)


def test_immutable_duplicate_rejection(tmp_path):
    store_file = tmp_path / "trial_returns.parquet"
    dates = pd.date_range("2022-01-01", periods=50, freq="B")
    s1 = pd.Series(np.random.randn(50), index=dates)

    # First write succeeds
    save_returns("cand_dup", s1, store_path=store_file)

    # Second write for same candidate_id MUST raise ValueError (immutability discipline)
    with pytest.raises(ValueError, match="already exists in returns store"):
        save_returns("cand_dup", s1, store_path=store_file)


def test_multiple_candidates_matrix(tmp_path):
    store_file = tmp_path / "trial_returns.parquet"
    dates = pd.date_range("2022-01-01", periods=60, freq="B")

    s1 = pd.Series(np.random.randn(60), index=dates)
    s2 = pd.Series(np.random.randn(60), index=dates)

    save_returns("cand_A", s1, store_path=store_file)
    save_returns("cand_B", s2, store_path=store_file)

    matrix = load_returns_matrix(["cand_A", "cand_B"], store_path=store_file)
    assert not matrix.empty
    assert list(matrix.columns) == ["cand_A", "cand_B"]
    assert len(matrix) == 60


def test_missing_candidate_handling(tmp_path):
    store_file = tmp_path / "trial_returns.parquet"

    # Non-existent file
    assert load_returns("non_existent", store_path=store_file) is None
    empty_matrix = load_returns_matrix(["non_existent"], store_path=store_file)
    assert empty_matrix.empty

    # File exists with cand_1, search for missing cand_2
    dates = pd.date_range("2022-01-01", periods=10, freq="B")
    save_returns("cand_1", pd.Series(np.ones(10), index=dates), store_path=store_file)

    assert load_returns("cand_2", store_path=store_file) is None
    partial_matrix = load_returns_matrix(["cand_1", "cand_2"], store_path=store_file)
    assert list(partial_matrix.columns) == ["cand_1"]
