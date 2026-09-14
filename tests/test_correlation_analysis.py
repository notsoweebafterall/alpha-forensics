"""
Unit tests for redundancy detection & correlation analysis (Phase 13).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from statistics import (
    TrialRecord,
    log_trial,
    effective_trial_records,
    load_returns,
)
from statistics.trial_log import StatusChangeRecord
from statistics.returns_store import save_returns
from redundancy import (
    is_top_level_candidate_id,
    run_redundancy_analysis,
)


def test_scope_restriction_filter():
    assert is_top_level_candidate_id("cross_sectional_momentum") is True
    assert is_top_level_candidate_id("cross_sectional_momentum__param_lookback_20") is True
    assert is_top_level_candidate_id("alpha_1") is True

    assert is_top_level_candidate_id("alpha_1__sector_tech") is False
    assert is_top_level_candidate_id("alpha_1__period_h1") is False
    assert is_top_level_candidate_id("alpha_1__regime_bull_low_vol") is False
    assert is_top_level_candidate_id("alpha_1__cost_sweep_base") is False


def test_synthetic_redundancy_clustering_and_invalidation(tmp_path):
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    store_file = tmp_path / "trial_returns.parquet"

    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    np.random.seed(42)
    base_returns = pd.Series(np.random.randn(100) * 0.01 + 0.001, index=dates)

    # Candidate 1: High Sharpe representative
    cand1 = "strat_alpha"
    log_trial(TrialRecord(candidate_id=cand1, phase="P7", oos_sharpe=1.5, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand1, base_returns, store_path=store_file)

    # Candidate 2: Highly correlated with cand1 (r ~ 0.99), lower Sharpe (1.2) -> Should be INVALIDATED
    cand2 = "strat_alpha_variant"
    correlated_returns = base_returns + np.random.randn(100) * 0.0001
    log_trial(TrialRecord(candidate_id=cand2, phase="P9", oos_sharpe=1.2, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand2, correlated_returns, store_path=store_file)

    # Candidate 3: Independent candidate (r ~ 0.0) -> Should NOT cluster
    cand3 = "strat_beta"
    independent_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
    log_trial(TrialRecord(candidate_id=cand3, phase="P7", oos_sharpe=0.8, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand3, independent_returns, store_path=store_file)

    # Run redundancy analysis
    res = run_redundancy_analysis(
        log_path=log_file,
        status_log_path=status_file,
        store_path=store_file,
        correlation_threshold=0.90,
        min_overlap_days=60,
        apply_status_changes=True,
    )

    assert set(res.evaluated_candidates) == {cand1, cand2, cand3}
    assert len(res.clusters) == 1

    cluster = res.clusters[0]
    assert cluster.representative_id == cand1
    assert cluster.redundant_ids == [cand2]

    assert res.invalidated_candidates == [cand2]

    # Verify status log lifecycle integration
    eff_list = effective_trial_records(log_path=log_file, status_log_path=status_file)
    eff_records = {r.candidate_id: r for r in eff_list}
    assert eff_records[cand1].status == "VALID"
    assert eff_records[cand2].status == "INVALIDATED"
    assert eff_records[cand3].status == "VALID"


def test_insufficient_overlap_skipping(tmp_path):
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    store_file = tmp_path / "trial_returns.parquet"

    # Candidate 1: Jan-Feb dates (30 days)
    dates1 = pd.date_range("2022-01-01", periods=30, freq="B")
    cand1 = "cand_short_1"
    log_trial(TrialRecord(candidate_id=cand1, phase="P7", oos_sharpe=1.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand1, pd.Series(np.random.randn(30), index=dates1), store_path=store_file)

    # Candidate 2: Mar-Apr dates (30 non-overlapping days)
    dates2 = pd.date_range("2022-03-01", periods=30, freq="B")
    cand2 = "cand_short_2"
    log_trial(TrialRecord(candidate_id=cand2, phase="P7", oos_sharpe=1.1, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand2, pd.Series(np.random.randn(30), index=dates2), store_path=store_file)

    res = run_redundancy_analysis(
        log_path=log_file,
        status_log_path=status_file,
        store_path=store_file,
        correlation_threshold=0.90,
        min_overlap_days=60,
        apply_status_changes=False,
    )

    pair_key = f"pair ({cand1}, {cand2})"
    assert pair_key in res.skipped_candidates or f"pair ({cand2}, {cand1})" in res.skipped_candidates
    assert len(res.clusters) == 0


def test_transitive_redundancy_reason_string(tmp_path):
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    store_file = tmp_path / "trial_returns.parquet"

    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    np.random.seed(42)
    
    # Create 3 series: A, B (corr(A,B) >= 0.90), C (corr(B,C) >= 0.90, but corr(A,C) < 0.90)
    ret_a = pd.Series(np.random.randn(100) * 0.01 + 0.001, index=dates)
    # B is A + noise -> corr(A, B) ~ 0.92
    ret_b = ret_a + np.random.randn(100) * 0.0042
    # C is B + noise -> corr(B, C) ~ 0.92, corr(A, C) ~ 0.85
    ret_c = ret_b + np.random.randn(100) * 0.0042

    corr_ab = float(np.corrcoef(ret_a, ret_b)[0, 1])
    corr_bc = float(np.corrcoef(ret_b, ret_c)[0, 1])
    corr_ac = float(np.corrcoef(ret_a, ret_c)[0, 1])

    # Ensure our synthetic setup matches: corr_ab >= 0.90, corr_bc >= 0.90, corr_ac < 0.90
    assert corr_ab >= 0.90
    assert corr_bc >= 0.90
    assert corr_ac < 0.90

    cand_a = "cand_a" # Sharpe 2.0 (Representative)
    cand_b = "cand_b" # Sharpe 1.5 (Directly redundant with A)
    cand_c = "cand_c" # Sharpe 1.0 (Transitively redundant with A via B)

    log_trial(TrialRecord(candidate_id=cand_a, phase="P7", oos_sharpe=2.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_a, ret_a, store_path=store_file)

    log_trial(TrialRecord(candidate_id=cand_b, phase="P7", oos_sharpe=1.5, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_b, ret_b, store_path=store_file)

    log_trial(TrialRecord(candidate_id=cand_c, phase="P7", oos_sharpe=1.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_c, ret_c, store_path=store_file)

    res = run_redundancy_analysis(
        log_path=log_file,
        status_log_path=status_file,
        store_path=store_file,
        correlation_threshold=0.90,
        min_overlap_days=60,
        apply_status_changes=True,
    )

    assert len(res.clusters) == 1
    cluster = res.clusters[0]
    assert cluster.representative_id == cand_a
    assert set(cluster.redundant_ids) == {cand_b, cand_c}

    # Verify status log entries for cand_b and cand_c directly from status log
    import json
    status_records = {}
    with status_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = StatusChangeRecord.from_dict(json.loads(line.strip()))
                status_records[rec.candidate_id] = rec

    # B has direct corr >= 0.90 with representative A
    assert "redundant with cand_a, corr=" in status_records[cand_b].reason
    assert ">= 0.90" in status_records[cand_b].reason

    # C is transitively redundant (direct corr < 0.90 with representative A)
    assert "transitively redundant via cluster containing cand_a" in status_records[cand_c].reason
    assert "below threshold 0.90" in status_records[cand_c].reason

