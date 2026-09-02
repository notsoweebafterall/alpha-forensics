"""
Phase 17 Flagship Experiment Orchestrator.

Executes the full Alpha Forensics evaluation pipeline (Phases 7–16) on scale:
- Universe: UNIVERSE_150 (169 liquid clean tickers)
- Date Range: 2019-01-01 to 2023-12-31 (5 full calendar years)
- Isolated Path Architecture: data/flagship/* and reports/flagship/

Path Isolation Rationale:
    build_candidate_id() does NOT encode universe size or date ranges. Re-running
    on default paths would match existing trial_log.jsonl entries and silently reuse
    small-universe results. Isolated data/flagship/* paths prevent candidate collision,
    maintain single-look data discipline, and preserve statistical validity for DSR N count.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_150
from walk_forward_validation_demo import run_demo as run_wf_demo
from parameter_robustness_demo import run_demo as run_param_demo
from cost_sensitivity_demo import run_demo as run_cost_demo
from statistical_forensics_demo import run_demo as run_stats_demo
from factor_exposure_demo import run_demo as run_factor_demo
from regime_analysis_demo import run_demo as run_regime_demo
from redundancy_analysis_demo import run_demo as run_redundancy_demo
from registry_build_demo import run_demo as run_registry_demo
from portfolio_construction_demo import run_demo as run_portfolio_demo
from reporting_demo import run_demo as run_reporting_demo
from reporting import build_report


@dataclass(frozen=True)
class FlagshipResult:
    base_dir: Path
    reports_dir: Path
    log_path: Path
    status_log_path: Path
    store_path: Path
    dsr_log_path: Path
    db_path: Path
    report_markdown_path: Path
    report_json_path: Optional[Path]
    total_logged_candidates: int
    valid_top_level_candidates: int
    surviving_dsr_candidates: int


def run_flagship_experiment(
    tickers: Optional[list[str]] = None,
    start_date: str = "2019-01-01",
    end_date: str = "2023-12-31",
    base_dir: Union[str, Path] = "data/flagship",
    reports_dir: Union[str, Path] = "reports/flagship",
    cache_dir: Union[str, Path] = "data/cache",
) -> FlagshipResult:
    """
    Executes the flagship end-to-end experiment pipeline over isolated ledger paths.
    """
    target_tickers = UNIVERSE_150 if tickers is None else tickers
    base_dir = Path(base_dir)
    reports_dir = Path(reports_dir)
    cache_dir = Path(cache_dir)

    base_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    log_path = base_dir / "trial_log.jsonl"
    status_log_path = base_dir / "trial_status_log.jsonl"
    store_path = base_dir / "trial_returns.parquet"
    dsr_log_path = base_dir / "dsr_results.jsonl"
    db_path = base_dir / "alpha_registry.db"

    print("=" * 100, flush=True)
    print("ALPHA FORENSICS — PHASE 17 FLAGSHIP EXPERIMENT ORCHESTRATOR", flush=True)
    print("=" * 100, flush=True)
    print(f"Universe Tickers Count : {len(target_tickers)}", flush=True)
    print(f"Evaluation Window      : [{start_date} to {end_date}]", flush=True)
    print(f"Ledger Base Directory  : {base_dir}", flush=True)
    print(f"Reports Output Dir     : {reports_dir}", flush=True)
    print("-" * 100, flush=True)

    # 1. Base Strategy Walk-Forward Validation (Phase 7)
    print("\n>>> Step 1/10: Base Strategy Walk-Forward Validation (Phase 7)...", flush=True)
    run_wf_demo(
        tickers=target_tickers,
        start_date=start_date,
        end_date=end_date,
        log_path=log_path,
        store_path=store_path,
        cache_dir=cache_dir,
    )

    # 2. Cost Sensitivity Analysis (Phase 8)
    print("\n>>> Step 2/10: Cost Sensitivity Analysis (Phase 8)...", flush=True)
    run_cost_demo(
        tickers=target_tickers,
        start_date=start_date,
        end_date=end_date,
        log_path=log_path,
        store_path=store_path,
        cache_dir=cache_dir,
    )

    # 3. Parameter Robustness & Landscape Search (Phase 9)
    print("\n>>> Step 3/10: Parameter Robustness & Landscape Search (Phase 9)...", flush=True)
    run_param_demo(
        tickers=target_tickers,
        start_date=start_date,
        end_date=end_date,
        log_path=log_path,
        store_path=store_path,
        cache_dir=cache_dir,
    )

    # 4. Statistical Forensics & DSR Evaluation (Phase 10)
    print("\n>>> Step 4/10: Statistical Forensics & Deflated Sharpe Ratio (DSR) (Phase 10)...", flush=True)
    run_stats_demo(
        tickers=target_tickers,
        start_date=start_date,
        end_date=end_date,
        log_path=log_path,
        status_log_path=status_log_path,
        store_path=store_path,
        dsr_log_path=dsr_log_path,
        cache_dir=cache_dir,
    )

    # 5. Factor Exposure & Sub-Period Generalization (Phase 11)
    print("\n>>> Step 5/10: Factor Exposure & Sub-Period Generalization (Phase 11)...", flush=True)
    run_factor_demo(
        tickers=target_tickers,
        start_date=start_date,
        end_date=end_date,
        log_path=log_path,
        store_path=store_path,
        cache_dir=cache_dir,
    )

    # 6. Regime Analysis (Phase 12)
    print("\n>>> Step 6/10: Market Trend & Volatility Regime Analysis (Phase 12)...", flush=True)
    run_regime_demo(
        tickers=target_tickers,
        start_date=start_date,
        end_date=end_date,
        log_path=log_path,
        store_path=store_path,
        cache_dir=cache_dir,
    )

    # 7. Redundancy Analysis (Phase 13)
    print("\n>>> Step 7/10: Redundancy Detection & Invalidation Logging (Phase 13)...", flush=True)
    run_redundancy_demo(
        tickers=target_tickers,
        start_date=start_date,
        end_date=end_date,
        log_path=log_path,
        status_log_path=status_log_path,
        store_path=store_path,
        cache_dir=cache_dir,
    )

    # 8. Build Alpha Registry (Phase 14)
    print("\n>>> Step 8/10: Building SQLite Alpha Registry Catalog (Phase 14)...", flush=True)
    run_registry_demo(
        db_path=db_path,
        log_path=log_path,
        status_log_path=status_log_path,
        store_path=store_path,
        dsr_log_path=dsr_log_path,
    )

    # 9. Portfolio Construction (Phase 15)
    print("\n>>> Step 9/10: Meta-Portfolio Capital Allocation (Phase 15)...", flush=True)
    run_portfolio_demo(
        db_path=db_path,
        store_path=store_path,
    )

    # 10. Report Generation (Phase 16)
    print("\n>>> Step 10/10: Generating Final Flagship Report Artifacts (Phase 16)...", flush=True)
    run_reporting_demo(
        db_path=db_path,
        returns_store_path=store_path,
        dsr_log_path=dsr_log_path,
        log_path=log_path,
        status_log_path=status_log_path,
        output_dir=reports_dir,
        skip_rebuild=False,
        write_json=True,
    )

    # Build report directly to capture precise result metrics
    rep_res = build_report(
        db_path=db_path,
        returns_store_path=store_path,
        dsr_log_path=dsr_log_path,
        log_path=log_path,
        status_log_path=status_log_path,
        output_dir=reports_dir,
        skip_rebuild=True,
        write_json=True,
    )

    print("\n" + "=" * 100, flush=True)
    print("FLAGSHIP EXPERIMENT EXECUTION COMPLETE!", flush=True)
    print("=" * 100, flush=True)
    print(f"Report Markdown : {rep_res.markdown_path}", flush=True)
    print(f"Report JSON     : {rep_res.json_path}", flush=True)

    return FlagshipResult(
        base_dir=base_dir,
        reports_dir=reports_dir,
        log_path=log_path,
        status_log_path=status_log_path,
        store_path=store_path,
        dsr_log_path=dsr_log_path,
        db_path=db_path,
        report_markdown_path=rep_res.markdown_path,
        report_json_path=rep_res.json_path,
        total_logged_candidates=rep_res.total_candidates,
        valid_top_level_candidates=rep_res.valid_top_level_candidates,
        surviving_dsr_candidates=rep_res.candidates_surviving_dsr,
    )


if __name__ == "__main__":
    run_flagship_experiment()
