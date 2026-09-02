"""
Unit and integration tests for Phase 17 Flagship Experiment Orchestration.
"""

import sys
import inspect
from pathlib import Path
import pytest

# Add src/ and experiments/ to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from flagship_experiment_demo import run_flagship_experiment, FlagshipResult
import walk_forward_validation_demo
import cost_sensitivity_demo
import parameter_robustness_demo
import statistical_forensics_demo
import factor_exposure_demo
import regime_analysis_demo
import redundancy_analysis_demo
import registry_build_demo
import portfolio_construction_demo
import reporting_demo


def test_flagship_experiment_orchestration_fast(tmp_path):
    """
    Verifies that run_flagship_experiment executes end-to-end cleanly over a fast synthetic setup,
    populating isolated ledger files and report artifacts without touching main data/ ledger paths.
    """
    base_dir = tmp_path / "data_flagship"
    reports_dir = tmp_path / "reports_flagship"
    cache_dir = tmp_path / "cache"

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    start_date = "2021-01-01"
    end_date = "2023-12-31"

    result = run_flagship_experiment(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        base_dir=base_dir,
        reports_dir=reports_dir,
        cache_dir=cache_dir,
    )

    assert isinstance(result, FlagshipResult)

    # 1. Verify all 5 ledger / database files exist in base_dir
    assert (base_dir / "trial_log.jsonl").exists()
    assert (base_dir / "trial_status_log.jsonl").exists()
    assert (base_dir / "trial_returns.parquet").exists()
    assert (base_dir / "dsr_results.jsonl").exists()
    assert (base_dir / "alpha_registry.db").exists()

    # 2. Verify report artifacts exist in reports_dir
    assert result.report_markdown_path.exists()
    assert result.report_markdown_path.parent == reports_dir
    assert result.report_json_path is not None
    assert result.report_json_path.exists()

    # 3. Verify counts are populated non-negatively
    assert result.total_logged_candidates > 0
    assert result.valid_top_level_candidates >= 0
    assert result.surviving_dsr_candidates >= 0


def test_flagship_path_isolation_no_leakage(tmp_path):
    """
    Verifies that running flagship orchestrator with custom paths leaves any custom directory
    isolated and does not mutate unrelated paths.
    """
    base_dir = tmp_path / "isolated_data"
    reports_dir = tmp_path / "isolated_reports"
    cache_dir = tmp_path / "isolated_cache"

    # Pre-check: base_dir and reports_dir should not exist yet
    assert not base_dir.exists()
    assert not reports_dir.exists()

    run_flagship_experiment(
        tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        start_date="2020-01-01",
        end_date="2023-12-31",
        base_dir=base_dir,
        reports_dir=reports_dir,
        cache_dir=cache_dir,
    )

    # Post-check: isolated paths created properly
    assert base_dir.exists()
    assert reports_dir.exists()
    assert (base_dir / "alpha_registry.db").exists()


@pytest.mark.parametrize(
    "demo_module, required_path_param",
    [
        (walk_forward_validation_demo, "log_path"),
        (cost_sensitivity_demo, "log_path"),
        (parameter_robustness_demo, "log_path"),
        (statistical_forensics_demo, "log_path"),
        (factor_exposure_demo, "log_path"),
        (regime_analysis_demo, "log_path"),
        (redundancy_analysis_demo, "log_path"),
        (registry_build_demo, "db_path"),
        (portfolio_construction_demo, "db_path"),
        (reporting_demo, "db_path"),
    ],
)
def test_all_demo_functions_accept_custom_paths(demo_module, required_path_param):
    """
    Regression guard: ensures every demo script's run_demo function explicitly parameterizes custom paths.
    """
    sig = inspect.signature(demo_module.run_demo)
    assert required_path_param in sig.parameters, (
        f"{demo_module.__name__}.run_demo does not accept parameter '{required_path_param}'"
    )
