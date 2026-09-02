"""
Unit tests for the Phase 16 report builder.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from statistics import TrialRecord, log_trial, append_status_change
from statistics.dsr import DSRResult, log_dsr_result
from statistics.returns_store import save_returns
from registry import build_registry
from reporting import build_report


@pytest.fixture
def report_fixture(tmp_path):
    """
    Synthetic fixture: 4 candidates covering all filter-stage cases.
      cand_A: Valid, top-level, has returns, survives DSR
      cand_B: Valid, top-level, has returns, fails DSR
      cand_C: Valid, top-level, has returns, NOT dsr-evaluated (NULL)
      cand_D: INVALIDATED (redundant with A), top-level, has returns
      cand_E: Valid, sub-slice (__sector_), has returns
    """
    log_file = tmp_path / "trial_log.jsonl"
    status_file = tmp_path / "trial_status_log.jsonl"
    store_file = tmp_path / "trial_returns.parquet"
    dsr_file = tmp_path / "dsr_results.jsonl"
    db_file = tmp_path / "alpha_registry.db"
    output_dir = tmp_path / "reports"

    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    rng = np.random.default_rng(99)

    cand_A = "alpha_momentum"
    log_trial(TrialRecord(candidate_id=cand_A, phase="P7", oos_sharpe=1.8, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_A, pd.Series(rng.standard_normal(100) * 0.01 + 0.001, index=dates), store_path=store_file)
    log_dsr_result(DSRResult(candidate_id=cand_A, observed_sharpe=1.8, n_trials=10, expected_max_sharpe=1.2, dsr=0.96, verdict="survives_dsr"), log_path=dsr_file)

    cand_B = "alpha_reversal"
    log_trial(TrialRecord(candidate_id=cand_B, phase="P7", oos_sharpe=0.5, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_B, pd.Series(rng.standard_normal(100) * 0.02, index=dates), store_path=store_file)
    log_dsr_result(DSRResult(candidate_id=cand_B, observed_sharpe=0.5, n_trials=10, expected_max_sharpe=1.2, dsr=0.30, verdict="fails_dsr"), log_path=dsr_file)

    cand_C = "alpha_trend"
    log_trial(TrialRecord(candidate_id=cand_C, phase="P7", oos_sharpe=0.8, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_C, pd.Series(rng.standard_normal(100) * 0.015, index=dates), store_path=store_file)
    # No DSR entry for cand_C — dsr_verdict stays NULL

    cand_D = "alpha_momentum_clone"
    log_trial(TrialRecord(candidate_id=cand_D, phase="P9", oos_sharpe=1.75, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    append_status_change(cand_D, "INVALIDATED", reason="redundant", status_log_path=status_file, metadata={"redundant_with": cand_A, "correlation": 0.95})
    save_returns(cand_D, pd.Series(rng.standard_normal(100) * 0.01, index=dates), store_path=store_file)

    cand_E = "alpha_momentum__sector_Tech"
    log_trial(TrialRecord(candidate_id=cand_E, phase="P11", oos_sharpe=2.0, timestamp="2026-01-01T00:00:00"), log_path=log_file)
    save_returns(cand_E, pd.Series(rng.standard_normal(100) * 0.01, index=dates), store_path=store_file)

    build_registry(db_path=db_file, log_path=log_file, status_log_path=status_file, store_path=store_file, dsr_log_path=dsr_file)

    return dict(
        db_file=db_file, store_file=store_file, dsr_file=dsr_file,
        log_file=log_file, status_file=status_file,
        output_dir=output_dir,
        cand_A=cand_A, cand_B=cand_B, cand_C=cand_C, cand_D=cand_D, cand_E=cand_E,
    )


def _build(fixture, **kwargs) -> "ReportResult":
    from reporting import build_report
    return build_report(
        db_path=fixture["db_file"],
        returns_store_path=fixture["store_file"],
        dsr_log_path=fixture["dsr_file"],
        log_path=fixture["log_file"],
        status_log_path=fixture["status_file"],
        output_dir=fixture["output_dir"],
        skip_rebuild=True,   # already built by fixture
        **kwargs,
    )


class TestReportResult:
    def test_funnel_counts_match_fixture(self, report_fixture):
        result = _build(report_fixture)
        # 5 total (A, B, C, D-invalidated, E-subslice)
        assert result.total_candidates == 5
        # Valid non-redundant top-level: A, B, C (D is INVALIDATED, E is sub-slice)
        assert result.valid_top_level_candidates == 3
        # All 3 have returns
        assert result.candidates_with_returns == 3
        # DSR breakdown
        assert result.candidates_surviving_dsr == 1   # cand_A
        assert result.candidates_failing_dsr == 1     # cand_B
        assert result.candidates_not_dsr_evaluated == 1  # cand_C

    def test_rigorous_portfolio_eligible(self, report_fixture):
        result = _build(report_fixture)
        # Only cand_A survives DSR -> 1 eligible
        assert result.portfolio_rigorous_eligible == 1

    def test_relaxed_portfolio_eligible(self, report_fixture):
        result = _build(report_fixture)
        # A, B, C all have returns -> 3 eligible
        assert result.portfolio_relaxed_eligible == 3

    def test_report_file_written_and_non_empty(self, report_fixture):
        result = _build(report_fixture)
        assert result.markdown_path.exists()
        content = result.markdown_path.read_text(encoding="utf-8")
        assert len(content) > 200

    def test_json_companion_written(self, report_fixture):
        result = _build(report_fixture, write_json=True)
        assert result.json_path is not None
        assert result.json_path.exists()
        data = json.loads(result.json_path.read_text(encoding="utf-8"))
        assert data["funnel"]["total_logged"] == 5
        assert data["funnel"]["survives_dsr"] == 1

    def test_json_skipped_when_write_json_false(self, report_fixture):
        result = _build(report_fixture, write_json=False)
        assert result.json_path is None

    def test_empty_portfolio_section_clear_message(self, report_fixture):
        """With rigorous=True, only cand_A eligible. Test the empty case by
        checking a fixture where no one survives DSR."""
        result = _build(report_fixture)
        content = result.markdown_path.read_text(encoding="utf-8")
        # Rigorous portfolio section: 1 eligible (cand_A) — not empty in this fixture.
        # But test that when eligible=0 the word "empty" appears (check relaxed message or
        # verify the rigorous section shows constituent table vs empty message).
        # Either "eligible_constituents: 1" or the word "alpha_momentum" must appear:
        assert "alpha_momentum" in content

    def test_fails_dsr_verdict_visible_in_table(self, report_fixture):
        result = _build(report_fixture)
        content = result.markdown_path.read_text(encoding="utf-8")
        assert "fails_dsr" in content

    def test_null_dsr_candidate_shows_not_evaluated(self, report_fixture):
        result = _build(report_fixture)
        content = result.markdown_path.read_text(encoding="utf-8")
        # NULL verdict should appear as "NULL (not evaluated)" in the table
        assert "NULL (not evaluated)" in content

    def test_executive_summary_in_result(self, report_fixture):
        result = _build(report_fixture)
        assert len(result.executive_summary) > 50
        # Should mention candidate counts plainly
        assert "5" in result.executive_summary or "3" in result.executive_summary

    def test_caveats_section_present(self, report_fixture):
        result = _build(report_fixture)
        content = result.markdown_path.read_text(encoding="utf-8")
        assert "DSR Limitations" in content
        assert "Regime Coverage Limitation" in content
        assert "Redundancy Analysis Scope Restriction" in content


class TestEmptyPortfolioCase:
    """Verify the empty-portfolio case renders clearly — no crash, no blank section."""

    def test_rigorous_empty_renders_clearly(self, tmp_path):
        """Create a fixture where NO candidate survives DSR."""
        log_file = tmp_path / "trial_log.jsonl"
        status_file = tmp_path / "trial_status_log.jsonl"
        store_file = tmp_path / "trial_returns.parquet"
        dsr_file = tmp_path / "dsr_results.jsonl"
        db_file = tmp_path / "alpha_registry.db"
        output_dir = tmp_path / "reports"

        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        rng = np.random.default_rng(7)

        cand = "alpha_weak"
        log_trial(TrialRecord(candidate_id=cand, phase="P7", oos_sharpe=0.3, timestamp="2026-01-01T00:00:00"), log_path=log_file)
        save_returns(cand, pd.Series(rng.standard_normal(100) * 0.01, index=dates), store_path=store_file)
        log_dsr_result(DSRResult(candidate_id=cand, observed_sharpe=0.3, n_trials=5, expected_max_sharpe=1.5, dsr=0.1, verdict="fails_dsr"), log_path=dsr_file)

        build_registry(db_path=db_file, log_path=log_file, status_log_path=status_file, store_path=store_file, dsr_log_path=dsr_file)

        result = build_report(
            db_path=db_file, returns_store_path=store_file, dsr_log_path=dsr_file,
            log_path=log_file, status_log_path=status_file,
            output_dir=output_dir, skip_rebuild=True,
        )
        assert result.portfolio_rigorous_eligible == 0
        assert result.markdown_path.exists()
        content = result.markdown_path.read_text(encoding="utf-8")
        # Must not be blank, must contain "empty" explanation
        assert len(content) > 100
        assert "empty" in content.lower() or "0 candidate" in content.lower()
        # Executive summary must mention empty portfolio
        assert "empty" in result.executive_summary.lower()
