"""
Unit tests for registry build module (Phase 14).
"""

import sqlite3
import pytest
import pandas as pd
from pathlib import Path

from statistics import (
    TrialRecord,
    log_trial,
    append_status_change,
    DSRResult,
    log_dsr_result,
)
from statistics.returns_store import save_returns
from registry import build_registry, get_candidate_by_id, get_all_candidates


def test_build_registry_joins_and_idempotency(tmp_path):
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    store_file = tmp_path / "trial_returns.parquet"
    dsr_file = tmp_path / "dsr_results.jsonl"
    db_file = tmp_path / "alpha_registry.db"

    # Candidate 1: Valid candidate with returns & DSR verdict
    cand1 = "strat_alpha"
    log_trial(TrialRecord(candidate_id=cand1, phase="P7", oos_sharpe=1.5, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand1, pd.Series(np_random_returns(50), index=pd.date_range("2022-01-01", periods=50, freq="B")), store_path=store_file)
    dsr_res1 = DSRResult(candidate_id=cand1, observed_sharpe=1.5, n_trials=10, expected_max_sharpe=1.2, dsr=0.96, verdict="survives_dsr")
    log_dsr_result(dsr_res1, log_path=dsr_file)

    # Candidate 2: Redundant candidate (INVALIDATED with metadata)
    cand2 = "strat_alpha_variant"
    log_trial(TrialRecord(candidate_id=cand2, phase="P9", oos_sharpe=1.2, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    append_status_change(cand2, "INVALIDATED", reason="redundant with strat_alpha", status_log_path=status_file, metadata={"redundant_with": cand1, "correlation": 0.95})

    # Build registry
    build_registry(db_path=db_file, log_path=log_file, status_log_path=status_file, store_path=store_file, dsr_log_path=dsr_file)

    assert db_file.exists()

    # Query cand1
    c1 = get_candidate_by_id(cand1, db_path=db_file)
    assert c1 is not None
    assert c1["effective_status"] == "VALID"
    assert c1["has_returns_series"] == 1
    assert c1["dsr_verdict"] == "survives_dsr"
    assert c1["dsr_score"] == pytest.approx(0.96)
    assert c1["redundant_with"] is None

    # Query cand2
    c2 = get_candidate_by_id(cand2, db_path=db_file)
    assert c2 is not None
    assert c2["effective_status"] == "INVALIDATED"
    assert c2["redundant_with"] == cand1
    assert c2["correlation_with_redundant"] == pytest.approx(0.95)

    # Rebuild idempotency check: build a second time
    build_registry(db_path=db_file, log_path=log_file, status_log_path=status_file, store_path=store_file, dsr_log_path=dsr_file)
    df_all = get_all_candidates(db_path=db_file)
    assert len(df_all) == 2  # Exactly 2 candidates, no duplicates


def test_toggle_sequence_redundancy_metadata_null(tmp_path):
    """
    Tests the toggle sequence:
    VALID -> INVALIDATED (redundant) -> VALID -> INVALIDATED (other reason without redundancy metadata)
    Asserts that redundant_with and correlation_with_redundant are NULL.
    """
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    db_file = tmp_path / "alpha_registry.db"

    cand = "cand_toggle"
    log_trial(TrialRecord(candidate_id=cand, phase="P7", oos_sharpe=1.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)

    # Event 1: INVALIDATED (redundant)
    append_status_change(cand, "INVALIDATED", reason="redundant", status_log_path=status_file, metadata={"redundant_with": "other_cand", "correlation": 0.92})

    # Event 2: Re-validated (VALID)
    append_status_change(cand, "VALID", reason="re-evaluated standalone", status_log_path=status_file)

    # Event 3: INVALIDATED (unrelated reason, no redundancy metadata)
    append_status_change(cand, "INVALIDATED", reason="failed risk limit", status_log_path=status_file)

    build_registry(db_path=db_file, log_path=log_file, status_log_path=status_file)

    c = get_candidate_by_id(cand, db_path=db_file)
    assert c is not None
    assert c["effective_status"] == "INVALIDATED"
    # Because latest status change event (Event 3) has no redundancy metadata, redundant_with MUST be NULL
    assert c["redundant_with"] is None
    assert c["correlation_with_redundant"] is None


def np_random_returns(n):
    import numpy as np
    np.random.seed(42)
    return np.random.randn(n) * 0.01
