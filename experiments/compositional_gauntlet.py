"""
Compositional Candidates Full Gauntlet Runner (Task 2).

Runs the 25 compositional candidates (and all logged candidates) through:
  - Phase 8 Cost Realism (10bps survival check)
  - Phase 10 DSR Re-run (N total trials updated, DSR deflator recomputed across active trials)
  - Phase 11 Factor Exposure (OLS regression with Newey-West HAC SEs against MKT/MOM/VOL/SECTOR)
  - Phase 13 Redundancy Detection (pairwise OOS net return correlation threshold = 0.90)
  - Phase 14 Alpha Registry Rebuild & Phase 16 Report Generation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.generator.compositional_generator import generate_random_candidates
from alpha.generator.screening import screen_candidates
from alpha.strategies import build_candidate_id
from validation.runner import run_walk_forward_validation
from validation.oos import reserve_oos_holdout, evaluate_oos, reset_oos_access_log
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from backtesting.engine import run_backtest
from backtesting.metrics import economic_metrics
from statistics import (
    TrialRecord,
    load_trial_log,
    log_trial,
    effective_trial_records,
    load_returns_matrix,
    load_returns,
    save_returns,
    sharpe_variance_across_trials,
    expected_max_sharpe_under_trials,
    deflated_sharpe_ratio,
    DSRResult,
    log_dsr_result,
    load_latest_dsr_verdicts,
    compute_distribution_stats,
)
from factors.factor_construction import build_factor_panel
from factors.exposure import run_factor_regression
from redundancy.correlation_analysis import run_redundancy_analysis
from registry import build_registry
from reporting import build_report


def _candidate_id(cand) -> str:
    if cand.generation_method == "variant_expansion" and cand.parent_family:
        return build_candidate_id(cand.parent_family, params=cand.param_values or None)
    else:
        return f"generated_{cand.canonical_hash()[:12]}"


def run_gauntlet(
    log_path: str = "data/trial_log.jsonl",
    status_log_path: str = "data/trial_status_log.jsonl",
    store_path: str = "data/trial_returns.parquet",
    dsr_log_path: str = "data/dsr_results.jsonl",
    db_path: str = "data/alpha_registry.db",
    reports_dir: str = "reports",
    cache_dir: str = "data/cache",
):
    print("=" * 90)
    print("ALPHA FORENSICS — COMPOSITIONAL CANDIDATES FULL GAUNTLET")
    print("=" * 90)

    log_p = Path(log_path)
    status_p = Path(status_log_path)
    store_p = Path(store_path)
    dsr_p = Path(dsr_log_path)
    db_p = Path(db_path)
    cache_p = Path(cache_dir)

    # ── Step 1: Load Panel & OOS holdout ───────────────────────────────────────────────
    print("\n[1/5] Loading Panel...")
    panel = build_panel(
        tickers=UNIVERSE_60,
        start_date="2020-01-01",
        end_date="2023-12-31",
        missing_threshold=0.05,
        cache_dir=cache_p,
    )
    print(f"Panel loaded: {len(panel.prices)} dates x {len(panel.universe)} tickers.")
    dev_panel, oos_panel = reserve_oos_holdout(panel, oos_fraction=0.15)

    # Re-generate Phase 6 candidates to map candidate_id -> expression
    raw_candidates = generate_random_candidates(budget=200, max_depth=4, seed=42)
    passed, _ = screen_candidates(raw_candidates, panel, ic_threshold=0.005, min_coverage=0.8)
    passed_25 = passed[:25]
    id_to_candidate = {_candidate_id(c): c for c in passed_25}

    # ── Step 2: Phase 8 — Cost Sensitivity Sweeps for Compositional Candidates ────────
    print("\n[2/5] Phase 8 — Running Cost Sensitivity Sweeps (10bps) for Compositional Candidates...")
    all_trials = load_trial_log(log_p)
    comp_records = [
        r for r in all_trials
        if r.candidate_id.startswith("generated_") and r.phase == "Phase 6->7 Compositional Validation"
    ]
    print(f"Found {len(comp_records)} compositional candidates in trial_log.jsonl.")

    backtest_config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    cost_model_10bps = CostModel(cost_bps=10.0)

    cost_table_rows = []

    for rec in comp_records:
        cid = rec.candidate_id
        cost_cid = f"{cid}__cost_10bps"
        
        existing_ledger = {r.candidate_id: r for r in load_trial_log(log_p)}

        if cost_cid in existing_ledger:
            c_rec = existing_ledger[cost_cid]
            cost_table_rows.append({
                "Candidate ID": cid,
                "5bps OOS Sharpe": rec.oos_sharpe,
                "10bps OOS Sharpe": c_rec.oos_sharpe,
                "10bps Survival": "SURVIVES" if c_rec.oos_sharpe > 0.0 else "FAILS",
            })
            continue

        # Evaluate at 10bps cost model on OOS panel
        cand = id_to_candidate.get(cid)
        if cand is not None:
            reset_oos_access_log()
            oos_res_10bps = evaluate_oos(
                candidate_id=cost_cid,
                expression=cand.expression,
                oos_panel=oos_panel,
                config=backtest_config,
                cost_model=cost_model_10bps,
            )
            sharpe_10bps = float(oos_res_10bps.get("sharpe_ratio", 0.0))
            net_rets_10bps = oos_res_10bps.get("oos_net_returns")

            skew_10, kurt_10, t_len_10 = compute_distribution_stats(net_rets_10bps)

            log_trial(
                TrialRecord(
                    candidate_id=cost_cid,
                    phase="Phase 8 Cost Sensitivity",
                    oos_sharpe=sharpe_10bps,
                    timestamp=pd.Timestamp.now().isoformat(),
                    skew=skew_10,
                    kurtosis=kurt_10,
                    track_record_length=t_len_10,
                ),
                log_path=log_p,
            )
            if net_rets_10bps is not None:
                try:
                    save_returns(cost_cid, net_rets_10bps, store_path=store_p)
                except ValueError:
                    pass

            cost_table_rows.append({
                "Candidate ID": cid,
                "5bps OOS Sharpe": rec.oos_sharpe,
                "10bps OOS Sharpe": sharpe_10bps,
                "10bps Survival": "SURVIVES" if sharpe_10bps > 0.0 else "FAILS",
            })

    print("\nPhase 8 Cost Sensitivity (10bps) Results:")
    if cost_table_rows:
        df_cost = pd.DataFrame(cost_table_rows)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print(df_cost.to_string(index=False))

    # ── Step 3: Phase 10 — DSR Re-Run Across All Active Trials ────────────────────────
    print("\n[3/5] Phase 10 — Recomputing DSR Across Updated Ledger (N trials)...")
    active_records = effective_trial_records(log_path=log_p, status_log_path=status_p)
    active_sharpes = [r.oos_sharpe for r in active_records]
    N_trials = len(active_records)
    v_sr = sharpe_variance_across_trials(active_sharpes)
    sr_0 = expected_max_sharpe_under_trials(n_trials=N_trials, sharpe_variance=v_sr)

    print(f"Active (non-invalidated) Trials N : {N_trials}")
    print(f"Sample Sharpe Variance V[SR_hat] : {v_sr:.6f}")
    print(f"Expected Max Sharpe SR_0          : {sr_0:.4f}")

    dsr_results = []
    for rec in active_records:
        dsr_res = DSRResult.create(
            candidate_id=rec.candidate_id,
            observed_sharpe=rec.oos_sharpe,
            sharpe_variance=v_sr,
            n_trials=N_trials,
            skew=rec.skew,
            kurtosis=rec.kurtosis,
            track_record_length=rec.track_record_length or 150,
        )
        log_dsr_result(dsr_res, log_path=dsr_p)
        dsr_results.append(dsr_res)

    print(f"DSR verdicts updated and appended to {dsr_p}.")

    # Print top DSR results
    top_dsr = sorted(dsr_results, key=lambda x: x.observed_sharpe, reverse=True)
    print("\nTop Candidates by OOS Sharpe (with updated DSR):")
    print(f"{'Candidate ID':<45} | {'OOS Sharpe':<10} | {'DSR Score':<10} | {'Verdict':<12}")
    print("-" * 85)
    for r in top_dsr[:20]:
        print(f"{r.candidate_id:<45} | {r.observed_sharpe:<10.4f} | {r.dsr:<10.4f} | {r.verdict:<12}")

    # ── Step 4: Phase 11 — Factor Exposure OLS for Top Positive Candidates ────────────
    print("\n[4/5] Phase 11 — Running Factor Exposure Regression on Positive Sharpe Candidates...")
    factor_panel = build_factor_panel(oos_panel)
    all_sectors = list(factor_panel.sector_returns.columns)

    pos_cands = [r for r in comp_records if r.oos_sharpe > 0.0]
    factor_results = []

    for rec in pos_cands:
        cid = rec.candidate_id
        ret_series = load_returns(cid, store_path=store_p)
        if ret_series is None or ret_series.empty:
            continue

        f_res = run_factor_regression(
            candidate_returns=ret_series,
            factor_panel=factor_panel,
            candidate_sectors=all_sectors,
            candidate_id=cid,
            nw_lags=5,
        )
        factor_results.append(f_res)

        print(f"\n" + "-" * 80)
        print(f"Candidate: [{cid}] ({id_to_candidate[cid].expression_string})")
        print("-" * 80)
        print(f"  Raw Sharpe        : {f_res.raw_sharpe:.4f}")
        print(f"  R-Squared (R^2)   : {f_res.r_squared:.4f}")
        print(f"  Residual Sharpe   : {f_res.residual_sharpe:.4f}")
        print(f"  Alpha Verdict     : {f_res.verdict}")
        print("  Factor Betas & t-stats:")
        for fname, beta in f_res.betas.items():
            tstat = f_res.t_stats.get(fname, 0.0)
            corr = f_res.factor_correlations.get(fname, 0.0)
            print(f"    * {fname:<25}: Beta = {beta:>8.4f} | t-stat = {tstat:>8.4f} | Corr = {corr:>8.4f}")

    # ── Step 5: Phase 13 & 14 & 16 — Redundancy Analysis & Registry Rebuild & Report ─
    print("\n[5/5] Phase 13 & 14 & 16 — Redundancy Analysis, Registry Rebuild & Report Generation...")
    red_res = run_redundancy_analysis(
        log_path=log_p,
        status_log_path=status_p,
        store_path=store_p,
        correlation_threshold=0.90,
        min_overlap_days=60,
        apply_status_changes=True,
    )
    print(f"Evaluated candidates in redundancy analysis : {len(red_res.evaluated_candidates)}")
    print(f"Redundancy clusters found                 : {len(red_res.clusters)}")
    print(f"Invalidated candidates (redundant)         : {len(red_res.invalidated_candidates)}")
    for cluster in red_res.clusters:
        print(f"  Cluster {cluster.cluster_id}: Representative={cluster.representative_id}, Redundant={cluster.redundant_ids}")

    # Rebuild SQLite Alpha Registry
    print("\nRebuilding Alpha Registry SQLite catalog (data/alpha_registry.db)...")
    db_path_out = build_registry(
        log_path=log_p,
        status_log_path=status_p,
        dsr_log_path=dsr_p,
        store_path=store_p,
        db_path=db_p,
    )
    print(f"Registry rebuilt at: {db_path_out}")

    # Generate Phase 16 Report
    print("\nGenerating Phase 16 Report...")
    rep_res = build_report(
        db_path=db_p,
        returns_store_path=store_p,
        output_dir=reports_dir,
    )
    print(f"Report Markdown : {rep_res.markdown_path}")
    print(f"Report JSON     : {rep_res.json_path}")


if __name__ == "__main__":
    run_gauntlet()
