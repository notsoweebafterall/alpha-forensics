"""
Transaction Cost Sensitivity Analysis Demo (Phase 8).

Demonstrates single-look OOS cost sensitivity sweeps across transaction cost grids [0, 5, 10, 25, 50] bps
and evaluates realistic cost survival (10 bps threshold).

DISCLAIMER:
    This experiment performs single-look OOS cost sensitivity sweeps. No multiple-testing corrections
    have been applied yet (Phase 10).
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
from alpha.expressions.tree import Leaf, UnaryNode, BinaryNode
import alpha.primitives.inputs as in_prim
import alpha.primitives.operators as op_prim
from validation.folds import WalkForwardConfig
from validation.oos import reset_oos_access_log
from backtesting.config import BacktestConfig
from backtesting.cost_sensitivity import run_cost_sensitivity, STANDARD_COST_GRID, REALISTIC_COST_BPS
from statistics import TrialRecord, load_trial_log, log_trial, save_returns


def run_demo(
    tickers: Optional[list[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    cache_dir: Union[str, Path] = "data/cache",
) -> None:
    print("=" * 100)
    print("ALPHA FORENSICS — PHASE 8 COST SENSITIVITY ANALYSIS DEMO")
    print("=" * 100)
    print(f"Executing single-look OOS cost sweeps across cost grid: {STANDARD_COST_GRID} bps")
    print(f"Realistic Cost Threshold: {REALISTIC_COST_BPS} bps")
    print("-" * 100)

    reset_oos_access_log()
    log_path = Path(log_path)
    store_path = Path(store_path)
    cache_dir = Path(cache_dir)
    target_tickers = UNIVERSE_60 if tickers is None else tickers

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

    # 3 candidates. Base IDs use the same canonical build_candidate_id() convention
    # as Phase 7 (bare name when params == strategy default_params), so this script
    # refers to the exact same underlying expression Phase 7 evaluates. The ledger
    # entry THIS script writes is still a separate trial from Phase 7's, though --
    # see the cost-suffix note below -- because it represents a genuinely different
    # cost assumption (10bps realistic vs Phase 7's 5bps), not a re-look at the
    # same trial.
    strat_mom = get_strategy("cross_sectional_momentum")
    strat_vol = get_strategy("low_volatility")

    cand1_params = {"lookback": 60}
    cand1_expr = strat_mom.build(cand1_params)
    cand1_base_id = (
        build_candidate_id(strat_mom.name)
        if cand1_params == strat_mom.default_params
        else build_candidate_id(strat_mom.name, params=cand1_params)
    )

    cand2_params = {"window": 60}
    cand2_expr = strat_vol.build(cand2_params)
    cand2_base_id = (
        build_candidate_id(strat_vol.name)
        if cand2_params == strat_vol.default_params
        else build_candidate_id(strat_vol.name, params=cand2_params)
    )

    cand3_expr = UnaryNode(
        op_prim.Rank(),
        BinaryNode(
            op_prim.Multiply(),
            Leaf(in_prim.Momentum(20)),
            Leaf(in_prim.VolumeChange(5)),
        ),
    )
    cand3_base_id = build_candidate_id("volume_momentum_combo")

    # The ledger candidate_id for THIS phase's trial gets an explicit cost-bps
    # suffix. Without it, this would collide with Phase 7's ID for the same
    # expression and silently reuse Phase 7's 5bps Sharpe as if it were this
    # phase's 10bps realistic-cost figure -- wrong number under the right label.
    candidates = [
        (cand1_base_id, build_candidate_id(cand1_base_id, suffix=f"cost_{int(REALISTIC_COST_BPS)}bps"), cand1_expr),
        (cand2_base_id, build_candidate_id(cand2_base_id, suffix=f"cost_{int(REALISTIC_COST_BPS)}bps"), cand2_expr),
        (cand3_base_id, build_candidate_id(cand3_base_id, suffix=f"cost_{int(REALISTIC_COST_BPS)}bps"), cand3_expr),
    ]

    wf_config = WalkForwardConfig(
        mode="expanding",
        initial_train_window=400,
        step_size=60,
        val_window=60,
        embargo_days=10,
    )
    backtest_config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)

    results = []

    # 2. Run Cost Sensitivity Sweeps
    existing_ledger = {r.candidate_id: r for r in load_trial_log(log_path)}

    for base_id, ledger_id, expr in candidates:
        print(f"\nRunning Cost Sensitivity Sweep for Candidate: '{base_id}' (ledger id: '{ledger_id}')...")

        if ledger_id in existing_ledger:
            rec = existing_ledger[ledger_id]
            print(f"  '{ledger_id}' already logged (oos_sharpe @ {int(REALISTIC_COST_BPS)}bps = {rec.oos_sharpe:.4f}) "
                  f"-- reusing per single-look discipline, not re-running the sweep.")
            results.append({
                "Candidate ID": base_id,
                "Sharpe @ 0bps": None,
                "Sharpe @ 5bps": None,
                "Sharpe @ 10bps": rec.oos_sharpe,
                "Sharpe @ 25bps": None,
                "Sharpe @ 50bps": None,
                "Breakeven (bps)": None,
                "Survives 10bps": rec.oos_sharpe > 0.0,
            })
            continue

        res = run_cost_sensitivity(
            candidate_id=base_id,
            expression=expr,
            panel=panel,
            wf_config=wf_config,
            backtest_config=backtest_config,
            cost_grid=STANDARD_COST_GRID,
            oos_fraction=0.15,
        )

        log_trial(
            TrialRecord(
                candidate_id=ledger_id,
                phase="Phase 8 Cost Sensitivity",
                oos_sharpe=float(res.oos_sharpe_by_cost[REALISTIC_COST_BPS]),
                timestamp=pd.Timestamp.now().isoformat(),
                skew=res.realistic_cost_skew,
                kurtosis=res.realistic_cost_kurtosis,
                track_record_length=res.realistic_cost_track_record_length,
            ),
            log_path=log_path,
        )
        if res.realistic_cost_returns is not None:
            try:
                save_returns(ledger_id, res.realistic_cost_returns, store_path=store_path)
            except ValueError:
                pass  # Already stored; immutability discipline

        row = {
            "Candidate ID": base_id,
            "Sharpe @ 0bps": res.oos_sharpe_by_cost[0.0],
            "Sharpe @ 5bps": res.oos_sharpe_by_cost[5.0],
            "Sharpe @ 10bps": res.oos_sharpe_by_cost[10.0],
            "Sharpe @ 25bps": res.oos_sharpe_by_cost[25.0],
            "Sharpe @ 50bps": res.oos_sharpe_by_cost[50.0],
            "Breakeven (bps)": res.breakeven_cost_bps if res.breakeven_cost_bps is not None else "N/A",
            "Survives 10bps": res.survives_realistic_costs,
        }
        results.append(row)

    # 3. Format & Display Summary Table
    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 115)
    print("TRANSACTION COST SENSITIVITY SUMMARY TABLE")
    print("=" * 115)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary_df.to_string(index=False))
    print("=" * 115)
    print("NOTE: Alphas that decay under modest transaction costs are transparently flagged above.")


if __name__ == "__main__":
    run_demo()