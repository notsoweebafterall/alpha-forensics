"""
Redundancy Analysis Demo (Phase 13).

Evaluates pairwise return correlations across active top-level alpha strategy candidates
in the trial ledger to detect redundant performance. Candidates with return correlation >= 0.90
are grouped into redundancy clusters.

Within each cluster:
  - The candidate with the highest logged OOS Sharpe is preserved as VALID (representative).
  - Redundant candidates are marked INVALIDATED via append_status_change() in trial_status_log.jsonl.

Honest Limitation:
  Candidates logged to trial_log.jsonl before Phase 13 was deployed do not have a stored returns
  series in trial_returns.parquet. They are explicitly reported in skipped candidates.
"""

import sys
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.strategies.registry import get_strategy
from alpha.strategies import build_candidate_id
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from validation.folds import WalkForwardConfig
from validation.runner import run_walk_forward_validation
from statistics import (
    load_trial_log,
    effective_trial_records,
    log_trial,
    save_returns,
    compute_distribution_stats,
    TrialRecord,
)
from redundancy import run_redundancy_analysis, is_top_level_candidate_id


def _ensure_demo_returns_exist(
    log_path: Path,
    store_path: Path,
    tickers: Optional[list[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    cache_dir: Union[str, Path] = "data/cache",
):
    """
    Populates Phase 13 returns store for key base strategy candidates if they
    were logged prior to Phase 13 deployment.
    """
    target_tickers = UNIVERSE_60 if tickers is None else tickers
    cache_dir = Path(cache_dir)
    try:
        panel = build_panel(
            tickers=target_tickers,
            start_date=start_date,
            end_date=end_date,
            missing_threshold=0.05,
            cache_dir=cache_dir,
        )
    except Exception:
        return

    wf_config = WalkForwardConfig(
        mode="expanding", initial_train_window=400, step_size=60, val_window=60, embargo_days=10
    )
    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    cost_model = CostModel(cost_bps=5.0)

    demo_candidates = [
        {"strat_name": "cross_sectional_momentum", "params": {"lookback": 20}},
        {"strat_name": "cross_sectional_momentum", "params": {"lookback": 60}},
        {"strat_name": "volatility_adjusted_momentum", "params": {"lookback": 60, "window": 20}},
    ]

    print("\nEnsuring OOS returns series are stored in returns_store for base candidates...", flush=True)

    for item in demo_candidates:
        strat = get_strategy(item["strat_name"])
        cid = build_candidate_id(item["strat_name"], params=item["params"] if item["params"] != strat.default_params else None)
        expr = strat.build(item["params"])

        val_res = run_walk_forward_validation(
            candidate_id=cid,
            expression=expr,
            panel=panel,
            wf_config=wf_config,
            backtest_config=config,
            cost_model=cost_model,
            oos_fraction=0.15,
        )

        oos_net = val_res.oos_net_returns
        oos_sh = float(val_res.oos_metrics.get("sharpe_ratio", 0.0))
        skew, kurtosis, t_len = compute_distribution_stats(oos_net)

        try:
            log_trial(
                TrialRecord(
                    candidate_id=cid,
                    phase="Phase 13 Redundancy Demo Bootstrap",
                    oos_sharpe=oos_sh,
                    timestamp=pd.Timestamp.now().isoformat(),
                    skew=skew,
                    kurtosis=kurtosis,
                    track_record_length=t_len,
                ),
                log_path=log_path,
            )
        except ValueError:
            pass  # Already in ledger

        if oos_net is not None:
            try:
                save_returns(cid, oos_net, store_path=store_path)
                print(f"  * Saved returns for [{cid}] -> {store_path}", flush=True)
            except ValueError:
                pass  # Already stored


def run_demo(
    tickers: Optional[list[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    cache_dir: Union[str, Path] = "data/cache",
) -> None:
    log_path = Path(log_path)
    status_log_path = Path(status_log_path)
    store_path = Path(store_path)
    cache_dir = Path(cache_dir)

    print("=" * 100, flush=True)
    print("ALPHA FORENSICS — PHASE 13 REDUNDANCY DETECTION & CORRELATION DEMO", flush=True)
    print("=" * 100, flush=True)
    print("Detecting structurally different expressions with redundant OOS daily returns", flush=True)
    print(f"Correlation Threshold: 0.90 | Min Overlap Days: 60 | Store: {store_path}", flush=True)
    print("-" * 100, flush=True)

    # Populate returns store if empty
    _ensure_demo_returns_exist(
        log_path=log_path,
        store_path=store_path,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )

    # 1. Inspect existing trial ledger
    records_list = effective_trial_records(log_path=log_path, status_log_path=status_log_path)
    records_dict = {r.candidate_id: r for r in records_list}
    total_logged = len(records_dict)
    top_level_ids = [cid for cid in records_dict if is_top_level_candidate_id(cid)]

    print(f"\nTrial Ledger Audit:", flush=True)
    print(f"  * Total Active Ledger Candidates : {total_logged}", flush=True)
    print(f"  * Top-Level Full-Window Candidates: {len(top_level_ids)}", flush=True)
    print(f"  * Excluded Sub-Slice Candidates   : {total_logged - len(top_level_ids)}", flush=True)

    # 2. Run Redundancy Analysis
    res = run_redundancy_analysis(
        log_path=log_path,
        status_log_path=status_log_path,
        store_path=store_path,
        correlation_threshold=0.90,
        min_overlap_days=60,
        apply_status_changes=True,
    )

    print("\n" + "=" * 100, flush=True)
    print("EVALUATION & SKIPPED CANDIDATES SUMMARY", flush=True)
    print("=" * 100, flush=True)
    print(f"Evaluated Candidates for Redundancy ({len(res.evaluated_candidates)}):", flush=True)
    for cid in res.evaluated_candidates:
        sh = records_dict[cid].oos_sharpe
        print(f"  - [{cid:<45}] OOS Sharpe = {sh:+.4f}", flush=True)

    if res.skipped_candidates:
        top_level_missing = {k: v for k, v in res.skipped_candidates.items() if not k.startswith("pair (") and "sub-slice" not in v}
        pairs_skipped = {k: v for k, v in res.skipped_candidates.items() if k.startswith("pair (")}
        sub_slices = {k: v for k, v in res.skipped_candidates.items() if "sub-slice" in v}

        if top_level_missing:
            print(f"\nSkipped Top-Level Candidates ({len(top_level_missing)}) — Honest Limitation (Missing Stored Returns):", flush=True)
            for item_id, reason in top_level_missing.items():
                print(f"  * Skipped [{item_id:<45}]: {reason}", flush=True)

        if pairs_skipped:
            print(f"\nSkipped Candidate Pairs ({len(pairs_skipped)}) — Insufficient Date Overlap:", flush=True)
            for pair_id, reason in pairs_skipped.items():
                print(f"  * Skipped [{pair_id:<45}]: {reason}", flush=True)

        if sub_slices:
            print(f"\nExcluded Sub-Slice Candidates ({len(sub_slices)}) — Scope Restriction:", flush=True)
            for sub_id, reason in list(sub_slices.items())[:5]:  # Show first 5 examples
                print(f"  * Excluded [{sub_id:<45}]: {reason}", flush=True)
            if len(sub_slices) > 5:
                print(f"  * ... and {len(sub_slices) - 5} additional sub-slice candidates excluded.", flush=True)

    # 3. Print Pairwise Correlation Matrix
    if not res.correlation_matrix.empty and len(res.evaluated_candidates) >= 2:
        print("\n" + "=" * 100, flush=True)
        print("PAIRWISE DAILY RETURN CORRELATION MATRIX", flush=True)
        print("=" * 100, flush=True)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.float_format", lambda x: f"{x:+.4f}" if pd.notna(x) else " N/A ")
        print(res.correlation_matrix.to_string(), flush=True)

    # 4. Print Redundancy Clusters & Status Log Actions
    print("\n" + "=" * 100, flush=True)
    print("REDUNDANCY CLUSTERING & INVALIDATION RESULTS", flush=True)
    print("=" * 100, flush=True)

    if not res.clusters:
        print("No redundant candidate clusters found (all evaluated candidates are return-orthogonal < 0.90).", flush=True)
    else:
        for cl in res.clusters:
            print(f"\nCluster #{cl.cluster_id}: Representative -> [{cl.representative_id}]", flush=True)
            print(f"  * Preserved Representative (VALID) : [{cl.representative_id}] (OOS Sharpe = {records_dict[cl.representative_id].oos_sharpe:+.4f})", flush=True)
            for red_id in cl.redundant_ids:
                sh = records_dict[red_id].oos_sharpe
                corr_val = cl.pairwise_correlations.get((cl.representative_id, red_id), 1.0)
                print(f"  * Marked INVALIDATED (Redundant)   : [{red_id}] (OOS Sharpe = {sh:+.4f}, corr={corr_val:.4f})", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("SUMMARY CONCLUSION:", flush=True)
    print("Phase 13 Redundancy Detection successfully audits candidate OOS daily return correlations.", flush=True)
    print("Structurally distinct expressions producing redundant returns are grouped into clusters,", flush=True)
    print("keeping only the highest-Sharpe candidate valid while invalidating redundant variants.", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    run_demo()
