"""
Unit tests for registry query API (Phase 14).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from statistics import (
    TrialRecord,
    log_trial,
    append_status_change,
    DSRResult,
    log_dsr_result,
)
from statistics.returns_store import save_returns
from registry import (
    build_registry,
    get_all_candidates,
    get_valid_non_redundant_candidates,
    get_candidates_by_verdict,
    get_candidate_by_id,
    get_status_history,
)


@pytest.fixture
def sample_registry_db(tmp_path):
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    store_file = tmp_path / "trial_returns.parquet"
    dsr_file = tmp_path / "dsr_results.jsonl"
    db_file = tmp_path / "alpha_registry.db"

    dates = pd.date_range("2022-01-01", periods=50, freq="B")
    np.random.seed(42)

    # Candidate 1: Valid, survives DSR, high Sharpe
    cand1 = "alpha_momentum"
    log_trial(TrialRecord(candidate_id=cand1, phase="P7", oos_sharpe=1.8, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand1, pd.Series(np.random.randn(50), index=dates), store_path=store_file)
    log_dsr_result(DSRResult(candidate_id=cand1, observed_sharpe=1.8, n_trials=10, expected_max_sharpe=1.2, dsr=0.97, verdict="survives_dsr"), log_path=dsr_file)

    # Candidate 2: Valid, fails DSR, medium Sharpe
    cand2 = "alpha_reversal"
    log_trial(TrialRecord(candidate_id=cand2, phase="P7", oos_sharpe=0.9, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand2, pd.Series(np.random.randn(50), index=dates), store_path=store_file)
    log_dsr_result(DSRResult(candidate_id=cand2, observed_sharpe=0.9, n_trials=10, expected_max_sharpe=1.2, dsr=0.40, verdict="fails_dsr"), log_path=dsr_file)

    # Candidate 3: Invalidated (redundant with cand1)
    cand3 = "alpha_momentum_clone"
    log_trial(TrialRecord(candidate_id=cand3, phase="P9", oos_sharpe=1.6, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    append_status_change(cand3, "INVALIDATED", reason="redundant with alpha_momentum", status_log_path=status_file, metadata={"redundant_with": cand1, "correlation": 0.96})

    # Candidate 4: Unevaluated for DSR (no DSR log entry)
    cand4 = "alpha_new"
    log_trial(TrialRecord(candidate_id=cand4, phase="P7", oos_sharpe=1.1, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    # Candidate 5: Valid sub-slice candidate
    cand5 = "alpha_momentum__sector_Tech"
    log_trial(TrialRecord(candidate_id=cand5, phase="P11", oos_sharpe=2.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)

    build_registry(
        db_path=db_file,
        log_path=log_file,
        status_log_path=status_file,
        store_path=store_file,
        dsr_log_path=dsr_file,
    )
    return db_file


def test_get_all_candidates(sample_registry_db):
    df = get_all_candidates(sample_registry_db)
    assert len(df) == 5
    # Check ranking by oos_sharpe descending
    assert list(df["candidate_id"]) == ["alpha_momentum__sector_Tech", "alpha_momentum", "alpha_momentum_clone", "alpha_new", "alpha_reversal"]


def test_get_valid_non_redundant_candidates(sample_registry_db):
    # Default (top_level_only=True): excludes sub-slice candidate cand5
    df_top = get_valid_non_redundant_candidates(sample_registry_db, top_level_only=True)
    assert len(df_top) == 3
    assert set(df_top["candidate_id"]) == {"alpha_momentum", "alpha_reversal", "alpha_new"}

    # All valid non-redundant (top_level_only=False): includes sub-slice candidate cand5
    df_all_valid = get_valid_non_redundant_candidates(sample_registry_db, top_level_only=False)
    assert len(df_all_valid) == 4
    assert set(df_all_valid["candidate_id"]) == {"alpha_momentum__sector_Tech", "alpha_momentum", "alpha_reversal", "alpha_new"}


def test_get_candidates_by_verdict(sample_registry_db):
    survives_df = get_candidates_by_verdict("survives_dsr", db_path=sample_registry_db)
    assert len(survives_df) == 1
    assert survives_df.iloc[0]["candidate_id"] == "alpha_momentum"

    fails_df = get_candidates_by_verdict("fails_dsr", db_path=sample_registry_db)
    assert len(fails_df) == 1
    assert fails_df.iloc[0]["candidate_id"] == "alpha_reversal"


def test_get_candidate_by_id(sample_registry_db):
    c1 = get_candidate_by_id("alpha_momentum", db_path=sample_registry_db)
    assert c1 is not None
    assert c1["candidate_id"] == "alpha_momentum"
    assert c1["dsr_verdict"] == "survives_dsr"

    c_missing = get_candidate_by_id("non_existent", db_path=sample_registry_db)
    assert c_missing is None


def test_get_status_history(sample_registry_db):
    history_df = get_status_history("alpha_momentum_clone", db_path=sample_registry_db)
    assert len(history_df) == 1
    assert history_df.iloc[0]["new_status"] == "INVALIDATED"
    assert "alpha_momentum" in history_df.iloc[0]["metadata_json"]
