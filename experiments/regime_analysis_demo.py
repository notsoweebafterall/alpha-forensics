"""
Regime Analysis Demo (Phase 12).

Classifies market trend and volatility regimes directly from yfinance Panel data,
and evaluates out-of-sample (OOS) strategy robustness across regime subsets for key
alpha strategy candidates.

Key Design Discipline:
    1. Pure classification logic (Panel -> RegimeLabels) separated from backtesting.
    2. Single guarded OOS walk-forward evaluation per candidate variant, with the resulting
       OOS net returns series sliced by per-date regime labels (preventing gapped walk-forward
       validation errors).
    3. Every sub-evaluation is logged to data/trial_log.jsonl with full distributional stats
       (skew, kurtosis, track_record_length) via compute_distribution_stats().
    4. Single-look guard: existing ledger entries are reused without re-logging.
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
from validation.oos import reserve_oos_holdout, reset_oos_access_log
from validation.runner import run_walk_forward_validation
from statistics import load_trial_log, load_returns, save_returns
from regimes import classify_regimes, evaluate_regime_robustness


def _create_synthetic_panel(start_date: str, end_date: str, tickers: list[str]):
    dates = pd.date_range(start_date, periods=800, freq="B")
    np.random.seed(42)
    p_df = pd.DataFrame(100.0 + np.random.randn(len(dates), len(tickers)).cumsum(axis=0), index=dates, columns=tickers)
    v_df = pd.DataFrame(10000 + np.random.randint(0, 5000, size=(len(dates), len(tickers))), index=dates, columns=tickers)
    return build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)


def run_demo(
    tickers: Optional[list[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    cache_dir: Union[str, Path] = "data/cache",
) -> None:
    print("=" * 100, flush=True)
    print("ALPHA FORENSICS — PHASE 12 REGIME ANALYSIS DEMO", flush=True)
    print("=" * 100, flush=True)
    print("Market Trend & Volatility Regime Classification & Sub-Period Robustness Auditing", flush=True)
    print("-" * 100, flush=True)

    reset_oos_access_log()
    log_path = Path(log_path)
    store_path = Path(store_path)
    cache_dir = Path(cache_dir)
    target_tickers = UNIVERSE_60 if tickers is None else tickers

    trial_records = load_trial_log(log_path)
    trial_dict = {r.candidate_id: r for r in trial_records}
    print(f"Loaded existing trial ledger ({log_path}): {len(trial_records)} total records logged.", flush=True)

    print(f"\nLoading {len(target_tickers)}-ticker Panel [{start_date} to {end_date}]...", flush=True)
    try:
        panel = build_panel(
            tickers=target_tickers,
            start_date=start_date,
            end_date=end_date,
            missing_threshold=0.05,
            cache_dir=cache_dir,
        )
        print(f"Loaded Panel successfully! Dates: {len(panel.prices)}, Tickers: {len(panel.universe)}", flush=True)
    except Exception as e:
        print(f"Warning: Could not fetch live data ({e}). Creating synthetic demo panel...", flush=True)
        panel = _create_synthetic_panel(start_date, end_date, target_tickers)

    # 3. Classify regimes over full Panel to showcase classification statistics
    print("\nClassifying Market Trend & Volatility Regimes (60-day trailing window)...", flush=True)
    reg_labels = classify_regimes(panel, lookback_days=60)

    label_counts = reg_labels.combined.value_counts(dropna=True)
    print("Market Regime Distribution across Panel dates:", flush=True)
    for lbl, cnt in label_counts.items():
        pct = cnt / len(panel.prices)
        print(f"  * Regime [{lbl:<16}]: {cnt:>4} dates ({pct:>6.1%})", flush=True)

    # 4. Candidates to evaluate
    candidates_to_test = [
        {
            "strat_name": "cross_sectional_momentum",
            "params": {"lookback": 20},
        },
        {
            "strat_name": "volatility_adjusted_momentum",
            "params": {"lookback": 60, "window": 20},
        },
    ]
    for item in candidates_to_test:
        item["strat"] = get_strategy(item["strat_name"])
        item["candidate_id"] = (
            build_candidate_id(item["strat_name"])
            if item["params"] == item["strat"].default_params
            else build_candidate_id(item["strat_name"], params=item["params"])
        )

    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    cost_model = CostModel(cost_bps=5.0)
    wf_config = WalkForwardConfig(
        mode="expanding",
        initial_train_window=400,
        step_size=60,
        val_window=60,
        embargo_days=10,
    )

    print("\n" + "=" * 100, flush=True)
    print("REGIME ROBUSTNESS EVALUATION", flush=True)
    print("=" * 100, flush=True)

    for item in candidates_to_test:
        cid = item["candidate_id"]
        strat = item["strat"]
        expr = strat.build(item["params"])

        print(f"\nEvaluating OOS performance for [{cid}]...", flush=True)

        store_path = log_path.parent / "trial_returns.parquet"

        # Phase 13: use already-stored returns if available — avoid a second OOS look
        oos_net_returns = load_returns(cid, store_path=store_path)
        if oos_net_returns is None:
            print(f"  No stored returns found for [{cid}]; running single walk-forward evaluation...", flush=True)
            # Use canonical cid (no demo_ alias) so the single-look guard is respected
            val_res = run_walk_forward_validation(
                candidate_id=cid,
                expression=expr,
                panel=panel,
                wf_config=wf_config,
                backtest_config=config,
                cost_model=cost_model,
                oos_fraction=0.15,
            )
            oos_net_returns = val_res.oos_net_returns
            base_oos_sharpe = float(val_res.oos_metrics.get("sharpe_ratio", 0.0))
            if oos_net_returns is not None:
                try:
                    save_returns(cid, oos_net_returns, store_path=store_path)
                except ValueError:
                    pass  # Already stored; immutability discipline
        else:
            print(f"  Loaded stored OOS returns for [{cid}] from returns store.", flush=True)
            # Compute Sharpe from stored net returns (consistent with saved series)
            mean_r = float(oos_net_returns.mean())
            std_r = float(oos_net_returns.std())
            base_oos_sharpe = (mean_r / std_r * (252.0 ** 0.5)) if std_r > 1e-8 else 0.0

        # Evaluate regime robustness using the pre-computed oos_net_returns
        reg_res = evaluate_regime_robustness(
            candidate_id=cid,
            build_fn=strat.build,
            default_params=item["params"],
            panel=panel,
            wf_config=wf_config,
            backtest_config=config,
            cost_model=cost_model,
            log_path=log_path,
            store_path=store_path,
            min_regime_days=20,
            regime_lookback=60,
            oos_returns=oos_net_returns,
        )

        print(f"\n--- Strategy Candidate ID: [{cid}] ---", flush=True)
        print(f"Base Strategy Family   : {strat.name}", flush=True)
        print(f"Evaluated Parameters   : {item['params']}", flush=True)
        print(f"Full OOS Net Sharpe    : {base_oos_sharpe:.4f}", flush=True)
        print(f"Regime Robustness Class: {reg_res.classification.upper()}", flush=True)
        print(f"Classification Rationale: {reg_res.rationale}", flush=True)

        print("\n  Per-Regime OOS Sharpe Ratios & Observations:", flush=True)
        for r_name, r_sh in reg_res.regime_sharpes.items():
            r_cnt = reg_res.regime_counts.get(r_name, 0)
            # Fetch ledger record details for verification
            sub_cid = build_candidate_id(cid, suffix=f"regime_{r_name}")
            rec = trial_dict.get(sub_cid)
            logged_info = f"(Logged skew={rec.skew:.2f}, kurt={rec.kurtosis:.2f})" if rec and rec.skew is not None else ""
            print(f"    * Evaluated Regime [{r_name:<16}]: OOS Sharpe = {r_sh:>7.4f} | Days = {r_cnt:>4} {logged_info}", flush=True)

        if reg_res.skipped_regimes:
            print("\n  Skipped Regime Subsets (Coverage Limitation):", flush=True)
            for r_name, reason in reg_res.skipped_regimes.items():
                print(f"    * Skipped Regime   [{r_name:<16}]: {reason}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("SUMMARY CONCLUSION:", flush=True)
    print("Phase 12 Regime Analysis successfully classifies market trend and volatility regimes directly", flush=True)
    print("from Panel data and evaluates candidate robustness across regime subsets using single-look OOS returns.", flush=True)
    print("All sub-evaluations are recorded to the persistent trial ledger with full distribution stats.", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    run_demo()
