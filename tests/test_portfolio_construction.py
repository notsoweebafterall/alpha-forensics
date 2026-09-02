"""
Unit tests for portfolio construction module (Phase 15).
"""

import math
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
from registry import build_registry
from portfolio import build_portfolio, PortfolioResult


@pytest.fixture
def portfolio_fixture(tmp_path):
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    store_file = tmp_path / "trial_returns.parquet"
    dsr_file = tmp_path / "dsr_results.jsonl"
    db_file = tmp_path / "alpha_registry.db"

    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    np.random.seed(42)

    # Candidate A: Valid, top-level, stored returns, survives DSR, vol ~ 0.01
    cand_A = "alpha_momentum"
    log_trial(TrialRecord(candidate_id=cand_A, phase="P7", oos_sharpe=1.8, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_A, pd.Series(np.random.randn(100) * 0.01 + 0.001, index=dates), store_path=store_file)
    log_dsr_result(DSRResult(candidate_id=cand_A, observed_sharpe=1.8, n_trials=10, expected_max_sharpe=1.2, dsr=0.96, verdict="survives_dsr"), log_path=dsr_file)

    # Candidate B: Valid, top-level, stored returns, survives DSR, higher vol ~ 0.02
    cand_B = "alpha_reversal"
    log_trial(TrialRecord(candidate_id=cand_B, phase="P7", oos_sharpe=1.2, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_B, pd.Series(np.random.randn(100) * 0.02 + 0.0005, index=dates), store_path=store_file)
    log_dsr_result(DSRResult(candidate_id=cand_B, observed_sharpe=1.2, n_trials=10, expected_max_sharpe=1.2, dsr=0.95, verdict="survives_dsr"), log_path=dsr_file)

    # Candidate C: Valid, top-level, stored returns, FAILS DSR
    cand_C = "alpha_trend"
    log_trial(TrialRecord(candidate_id=cand_C, phase="P7", oos_sharpe=0.8, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_C, pd.Series(np.random.randn(100) * 0.01, index=dates), store_path=store_file)
    log_dsr_result(DSRResult(candidate_id=cand_C, observed_sharpe=0.8, n_trials=10, expected_max_sharpe=1.2, dsr=0.40, verdict="fails_dsr"), log_path=dsr_file)

    # Candidate D: INVALIDATED (redundant)
    cand_D = "alpha_momentum_clone"
    log_trial(TrialRecord(candidate_id=cand_D, phase="P9", oos_sharpe=1.7, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    append_status_change(cand_D, "INVALIDATED", reason="redundant", status_log_path=status_file, metadata={"redundant_with": cand_A, "correlation": 0.95})
    save_returns(cand_D, pd.Series(np.random.randn(100) * 0.01, index=dates), store_path=store_file)

    # Candidate E: Sub-slice candidate ID (contains __sector_)
    cand_E = "alpha_momentum__sector_Tech"
    log_trial(TrialRecord(candidate_id=cand_E, phase="P11", oos_sharpe=2.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_E, pd.Series(np.random.randn(100) * 0.01, index=dates), store_path=store_file)

    build_registry(db_path=db_file, log_path=log_file, status_log_path=status_file, store_path=store_file, dsr_log_path=dsr_file)

    return {
        "db_path": db_file,
        "store_path": store_file,
        "cand_A": cand_A,
        "cand_B": cand_B,
        "cand_C": cand_C,
        "cand_D": cand_D,
        "cand_E": cand_E,
    }


def test_weight_sums_and_schemes(portfolio_fixture):
    db_path = portfolio_fixture["db_path"]
    store_path = portfolio_fixture["store_path"]

    # Equal Weight (require_dsr_survival=True) -> Constituents: cand_A and cand_B
    res_eq = build_portfolio(db_path=db_path, returns_store_path=store_path, weighting="equal_weight", require_dsr_survival=True)
    assert res_eq.eligible_candidates == [portfolio_fixture["cand_A"], portfolio_fixture["cand_B"]]
    assert len(res_eq.weights) == 2
    assert sum(res_eq.weights.values()) == pytest.approx(1.0)
    assert res_eq.weights[portfolio_fixture["cand_A"]] == pytest.approx(0.5)
    assert res_eq.weights[portfolio_fixture["cand_B"]] == pytest.approx(0.5)
    assert res_eq.portfolio_returns is not None
    assert res_eq.portfolio_sharpe is not None

    # Inverse Volatility (require_dsr_survival=True) -> cand_A (lower vol) gets higher weight than cand_B (higher vol)
    res_iv = build_portfolio(db_path=db_path, returns_store_path=store_path, weighting="inverse_volatility", require_dsr_survival=True)
    assert sum(res_iv.weights.values()) == pytest.approx(1.0)
    assert res_iv.weights[portfolio_fixture["cand_A"]] > res_iv.weights[portfolio_fixture["cand_B"]]


def test_stage_filtering_funnel(portfolio_fixture):
    db_path = portfolio_fixture["db_path"]
    store_path = portfolio_fixture["store_path"]

    # require_dsr_survival=False -> Constituents: cand_A, cand_B, cand_C
    res_no_dsr = build_portfolio(db_path=db_path, returns_store_path=store_path, weighting="equal_weight", require_dsr_survival=False)
    assert len(res_no_dsr.eligible_candidates) == 3
    assert portfolio_fixture["cand_C"] in res_no_dsr.eligible_candidates
    # cand_D (INVALIDATED) and cand_E (sub-slice) must be excluded from Stage 1
    assert portfolio_fixture["cand_D"] not in res_no_dsr.stage1_candidates
    assert portfolio_fixture["cand_E"] not in res_no_dsr.stage1_candidates


def test_zero_volatility_guard(tmp_path):
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    store_file = tmp_path / "trial_returns.parquet"
    dsr_file = tmp_path / "dsr_results.jsonl"
    db_file = tmp_path / "alpha_registry.db"

    dates = pd.date_range("2022-01-01", periods=80, freq="B")

    # Cand 1: Normal returns
    cand1 = "cand_normal"
    log_trial(TrialRecord(candidate_id=cand1, phase="P7", oos_sharpe=1.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand1, pd.Series(np.random.randn(80) * 0.01, index=dates), store_path=store_file)

    # Cand 2: Zero volatility returns (constant 0.0)
    cand2 = "cand_flat"
    log_trial(TrialRecord(candidate_id=cand2, phase="P7", oos_sharpe=1.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand2, pd.Series(np.zeros(80), index=dates), store_path=store_file)

    build_registry(db_path=db_file, log_path=log_file, status_log_path=status_file, store_path=store_file, dsr_log_path=dsr_file)

    # Inverse volatility MUST NOT raise ZeroDivisionError or produce NaN/Inf weights
    res = build_portfolio(db_path=db_file, returns_store_path=store_file, weighting="inverse_volatility", require_dsr_survival=False)
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert not math.isnan(res.weights[cand2])
    assert not math.isinf(res.weights[cand2])


def test_aligned_date_window(tmp_path):
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    store_file = tmp_path / "trial_returns.parquet"
    db_file = tmp_path / "alpha_registry.db"

    # Cand 1: Dates 1 to 100
    dates1 = pd.date_range("2022-01-01", periods=100, freq="B")
    cand1 = "cand_long_1"
    log_trial(TrialRecord(candidate_id=cand1, phase="P7", oos_sharpe=1.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand1, pd.Series(np.random.randn(100) * 0.01, index=dates1), store_path=store_file)

    # Cand 2: Dates 21 to 100 (80 overlapping dates)
    dates2 = dates1[20:]
    cand2 = "cand_short_2"
    log_trial(TrialRecord(candidate_id=cand2, phase="P7", oos_sharpe=1.1, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand2, pd.Series(np.random.randn(80) * 0.02, index=dates2), store_path=store_file)

    build_registry(db_path=db_file, log_path=log_file, status_log_path=status_file, store_path=store_file)

    res = build_portfolio(db_path=db_file, returns_store_path=store_file, weighting="inverse_volatility", require_dsr_survival=False)

    # Aligned return series MUST have exactly 80 dates (inner-join overlap)
    assert len(res.portfolio_returns) == 80
    assert (res.portfolio_returns.index == dates2).all()


def test_empty_eligible_set_handling(tmp_path):
    db_file = tmp_path / "empty_registry.db"
    store_file = tmp_path / "empty_returns.parquet"

    # Non-existent DB / empty registry
    res = build_portfolio(db_path=db_file, returns_store_path=store_file, weighting="equal_weight")

    assert res.eligible_candidates == []
    assert res.weights == {}
    assert res.portfolio_returns is None
    assert res.portfolio_sharpe is None
    assert "Empty portfolio" in res.message
