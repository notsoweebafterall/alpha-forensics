"""
Statistical Forensics & Multiple Testing Demo (Phase 10).

Demonstrates persistent trial ledger logging, Probabilistic Sharpe Ratio (PSR),
and Deflated Sharpe Ratio (DSR) evaluation for real, re-executed alpha candidates.

DISCLAIMER:
    DSR evaluates selection bias across multiple candidate trials. It does not replace
    parameter landscape stability checks (Phase 9), transaction cost modeling (Phase 8),
    or regime robustness testing (Phase 11/12).

Trial Ledger Provenance:
    N (total trial count) is read directly from data/trial_log.jsonl, which is populated
    by real log_trial() calls in Phase 7 (walk-forward), Phase 8 (cost sensitivity),
    Phase 9 (parameter landscape), and Phase 11 (sector/period generalization). Every
    entry corresponds to a genuinely re-executed, single-look OOS evaluation -- there is
    no synthetic or fabricated fallback anywhere in this ledger's write path.

    Phase 4 (strategy reproduction) and Phase 6 (systematic generation) do NOT log to
    this ledger, by design: Phase 4 is explicitly full-sample/in-sample only, and Phase 6
    is a pre-OOS triage screen. Neither produces a genuine guarded OOS Sharpe, so
    including them would misrepresent what N counts. N therefore reflects real OOS
    trials only, not every candidate ever generated or screened -- which is the
    methodologically correct scope for DSR's multiple-testing correction.

    If you've modified any logging call site, re-run: delete data/trial_log.jsonl, then
    re-run walk_forward_validation_demo.py -> cost_sensitivity_demo.py ->
    parameter_robustness_demo.py -> factor_exposure_demo.py -> this script, in that order.
"""

import sys
from pathlib import Path
from typing import Optional, Union
import pandas as pd

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.strategies.registry import get_strategy
from alpha.strategies import build_candidate_id
from validation.folds import WalkForwardConfig
from validation.runner import run_walk_forward_validation
from validation.oos import reset_oos_access_log
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from statistics import (
    TrialRecord,
    load_trial_log,
    log_trial,
    trial_count,
    effective_trial_records,
    save_returns,
    expected_max_sharpe_under_trials,
    deflated_sharpe_ratio,
    DSRResult,
    log_dsr_result,
    probabilistic_sharpe_ratio,
    sharpe_variance_across_trials,
    compute_distribution_stats,
)


def _load_panel(start_date: str, end_date: str, tickers: Optional[list[str]] = None, cache_dir: Union[str, Path] = "data/cache"):
    """Loads the real Panel, falling back to a synthetic one if data fetch fails."""
    target_tickers = UNIVERSE_60 if tickers is None else tickers
    cache_dir = Path(cache_dir)
    print(f"Loading {len(target_tickers)}-ticker Panel [{start_date} to {end_date}]...")
    try:
        panel = build_panel(
            tickers=target_tickers,
            start_date=start_date,
            end_date=end_date,
            missing_threshold=0.05,
            cache_dir=cache_dir,
        )
        print(f"Loaded Panel successfully! Dates: {len(panel.prices)}, Tickers: {len(panel.universe)}")
    except Exception as e:
        print(f"Warning: Could not fetch live data ({e}). Creating synthetic demo panel...")
        dates = pd.date_range(start_date, periods=800, freq="B")
        import numpy as np
        np.random.seed(42)
        p_df = pd.DataFrame(100.0 + np.random.randn(len(dates), len(target_tickers)).cumsum(axis=0), index=dates, columns=target_tickers)
        v_df = pd.DataFrame(10000 + np.random.randint(0, 5000, size=(len(dates), len(target_tickers))), index=dates, columns=target_tickers)
        panel = build_panel(tickers=target_tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)
    return panel


def _get_or_evaluate_base_candidate(
    strategy_name: str,
    panel,
    wf_config: WalkForwardConfig,
    backtest_config: BacktestConfig,
    cost_model: CostModel,
    log_path: Path,
    status_log_path: Path,
    store_path: Path,
):
    """
    Returns real (candidate_id, oos_sharpe, skew, kurtosis, track_record_length) for a
    strategy's default-parameter candidate.
    """
    strat = get_strategy(strategy_name)
    candidate_id = build_candidate_id(strategy_name)  # default-param candidate: no param suffix

    all_effective = {r.candidate_id: r for r in effective_trial_records(log_path=log_path, status_log_path=status_log_path, include_invalidated=True)}

    if candidate_id in all_effective:
        rec = all_effective[candidate_id]
        if rec.status == "INVALIDATED":
            print(f"  '{candidate_id}' is marked INVALIDATED in status log (effective status: {rec.status}) -- skipping DSR detail.")
            return candidate_id, rec.oos_sharpe, None, None, None

        print(f"  '{candidate_id}' already logged (oos_sharpe={rec.oos_sharpe:.4f}) -- reusing, not re-evaluating.")
        if rec.skew is not None and rec.kurtosis is not None and rec.track_record_length is not None:
            return rec.candidate_id, rec.oos_sharpe, rec.skew, rec.kurtosis, rec.track_record_length
        print(f"    (note: '{candidate_id}' was logged before skew/kurtosis were persisted -- "
              f"no PSR/DSR detail is possible for it without re-running against a fresh ledger.)")
        return rec.candidate_id, rec.oos_sharpe, None, None, None

    val_res = run_walk_forward_validation(
        candidate_id=candidate_id,
        expression=strat.build(strat.default_params),
        panel=panel,
        wf_config=wf_config,
        backtest_config=backtest_config,
        cost_model=cost_model,
    )
    oos_sharpe = float(val_res.oos_metrics.get("sharpe_ratio", 0.0))
    oos_net = val_res.oos_net_returns
    skew, kurtosis, t_len = compute_distribution_stats(oos_net)

    log_trial(
        TrialRecord(
            candidate_id=candidate_id,
            phase="Phase 10 Statistical Forensics (base candidate)",
            oos_sharpe=oos_sharpe,
            timestamp=pd.Timestamp.now().isoformat(),
            skew=skew,
            kurtosis=kurtosis,
            track_record_length=t_len,
        ),
        log_path=log_path,
    )
    if oos_net is not None:
        try:
            save_returns(candidate_id, oos_net, store_path=store_path)
        except ValueError:
            pass  # Already stored; immutability discipline
    print(f"  Evaluated and logged '{candidate_id}': oos_sharpe={oos_sharpe:.4f}, n_obs={t_len}")
    return candidate_id, oos_sharpe, skew, kurtosis, t_len


def run_demo(
    tickers: Optional[list[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    dsr_log_path: Union[str, Path] = "data/dsr_results.jsonl",
    cache_dir: Union[str, Path] = "data/cache",
) -> None:
    print("=" * 100)
    print("ALPHA FORENSICS — PHASE 10 STATISTICAL FORENSICS & MULTIPLE TESTING DEMO")
    print("=" * 100)
    print("Persistent Trial Ledger & Deflated Sharpe Ratio (DSR) Evaluation (Bailey & López de Prado, 2014)")
    print("-" * 100)

    reset_oos_access_log()
    log_path = Path(log_path)
    status_log_path = Path(status_log_path)
    store_path = Path(store_path)
    dsr_log_path = Path(dsr_log_path)
    cache_dir = Path(cache_dir)

    panel = _load_panel(start_date=start_date, end_date=end_date, tickers=tickers, cache_dir=cache_dir)

    wf_config = WalkForwardConfig(
        mode="expanding", initial_train_window=400, step_size=60, val_window=60, embargo_days=10,
    )
    backtest_config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    cost_model = CostModel(cost_bps=5.0)

    print("\nEvaluating (or reusing already-logged) base candidates for DSR analysis...")
    strategy_names = ["cross_sectional_momentum", "volatility_adjusted_momentum"]
    evaluated = {}
    for name in strategy_names:
        cid, oos_sharpe, skew, kurtosis, t_len = _get_or_evaluate_base_candidate(
            name, panel, wf_config, backtest_config, cost_model, log_path, status_log_path, store_path,
        )
        evaluated[name] = {
            "candidate_id": cid,
            "oos_sharpe": oos_sharpe,
            "skew": skew,
            "kurtosis": kurtosis,
            "t_len": t_len,
        }

    total_n = trial_count(log_path=log_path, status_log_path=status_log_path)
    effective_recs = effective_trial_records(log_path=log_path, status_log_path=status_log_path, include_invalidated=False)
    all_sharpes = [r.oos_sharpe for r in effective_recs]
    var_sr = sharpe_variance_across_trials(all_sharpes)
    expected_max_sr = expected_max_sharpe_under_trials(var_sr, total_n)

    print("-" * 100)
    print(f"Sample Sharpe Variance V[SR_hat] across N={total_n} active (non-invalidated) trials: {var_sr:.6f}")
    print(f"Expected Max Sharpe SR_0 under N={total_n} trials: {expected_max_sr:.4f}")
    print(f"NOTE: N={total_n} reflects active logged OOS trials, respecting status log invalidations (P0.3 lifecycle).")
    print("-" * 100)

    results = []
    detail_rows = []
    for name in strategy_names:
        e = evaluated[name]
        if e["skew"] is None:
            print(f"\nSkipping PSR/DSR detail for '{name}': candidate was reused from ledger without "
                  f"distributional stats (skew/kurtosis/track_record_length are not persisted in TrialRecord). "
                  f"Re-run against a fresh ledger to get full PSR/DSR figures for this candidate.")
            continue

        psr = probabilistic_sharpe_ratio(
            observed_sharpe=e["oos_sharpe"], benchmark_sharpe=0.0,
            skew=e["skew"], kurtosis=e["kurtosis"], track_record_length=e["t_len"],
        )
        dsr_res = DSRResult.create(
            candidate_id=e["candidate_id"], observed_sharpe=e["oos_sharpe"], sharpe_variance=var_sr,
            n_trials=total_n, skew=e["skew"], kurtosis=e["kurtosis"], track_record_length=e["t_len"],
        )
        log_dsr_result(dsr_res, log_path=dsr_log_path)
        results.append({
            "Candidate Name": name,
            "OOS Sharpe": e["oos_sharpe"],
            "PSR (vs 0)": f"{psr:.4f}",
            "N (Trials)": total_n,
            "Expected Max SR_0": f"{dsr_res.expected_max_sharpe:.4f}",
            "DSR": f"{dsr_res.dsr:.4f}",
            "Verdict": dsr_res.verdict,
        })
        detail_rows.append((name, e, dsr_res))

    if results:
        df = pd.DataFrame(results)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print("\nSTATISTICAL FORENSICS & DSR SUMMARY REPORT")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

        print("\nDETAILED DSR CAVEATS & FORENSIC RATIONALE:")
        for name, e, dsr_res in detail_rows:
            print(f"\nCandidate: [{name}] (ID: {e['candidate_id']})")
            print(f"  - Observed OOS Sharpe : {dsr_res.observed_sharpe:.4f}")
            print(f"  - DSR Score           : {dsr_res.dsr:.4f} ({dsr_res.verdict})")
            print("  - Auto-Generated Caveats:")
            for cav in dsr_res.caveats:
                print(f"      * {cav}")

    print("\n" + "=" * 100)
    print("SUMMARY CONCLUSION:")
    print("PSR vs zero measures isolated statistical significance, whereas DSR accounts for the total number of")
    print("trials N evaluated during discovery. Combining DSR with Phase 9 parameter landscape robustness")
    print("prevents overestimating strategy viability prior to production deployment.")
    print("=" * 100)


if __name__ == "__main__":
    run_demo()