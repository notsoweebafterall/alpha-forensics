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
import pandas as pd

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.strategies.registry import get_strategy
from alpha.expressions.tree import Leaf, UnaryNode, BinaryNode
import alpha.primitives.inputs as in_prim
import alpha.primitives.operators as op_prim
from validation.folds import WalkForwardConfig
from validation.oos import reset_oos_access_log
from backtesting.config import BacktestConfig
from backtesting.cost_sensitivity import run_cost_sensitivity, STANDARD_COST_GRID, REALISTIC_COST_BPS


def run_demo():
    print("=" * 100)
    print("ALPHA FORENSICS — PHASE 8 COST SENSITIVITY ANALYSIS DEMO")
    print("=" * 100)
    print(f"Executing single-look OOS cost sweeps across cost grid: {STANDARD_COST_GRID} bps")
    print(f"Realistic Cost Threshold: {REALISTIC_COST_BPS} bps")
    print("-" * 100)

    reset_oos_access_log()

    # 1. Load real 60-ticker Panel
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    print(f"Loading 60-ticker Panel [{start_date} to {end_date}]...")

    try:
        panel = build_panel(
            tickers=UNIVERSE_60,
            start_date=start_date,
            end_date=end_date,
            missing_threshold=0.05,
            cache_dir="data/cache",
        )
        print(f"Loaded Panel successfully! Dates: {len(panel.prices)}, Tickers: {len(panel.universe)}")
    except Exception as e:
        print(f"Warning: Could not fetch live data ({e}). Creating synthetic demo panel...")
        dates = pd.date_range(start_date, periods=800, freq="B")
        import numpy as np
        np.random.seed(42)
        p_df = pd.DataFrame(100.0 + np.random.randn(len(dates), 60).cumsum(axis=0), index=dates, columns=UNIVERSE_60)
        v_df = pd.DataFrame(10000 + np.random.randint(0, 5000, size=(len(dates), 60)), index=dates, columns=UNIVERSE_60)
        panel = build_panel(tickers=UNIVERSE_60, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

    # 3 candidates
    cand1_expr = get_strategy("cross_sectional_momentum").build({"lookback": 60})
    cand2_expr = get_strategy("low_volatility").build({"window": 60})
    cand3_expr = UnaryNode(
        op_prim.Rank(),
        BinaryNode(
            op_prim.Multiply(),
            Leaf(in_prim.Momentum(20)),
            Leaf(in_prim.VolumeChange(5)),
        ),
    )

    candidates = [
        ("cross_sectional_momentum_60d", cand1_expr),
        ("low_volatility_60d", cand2_expr),
        ("volume_momentum_combo", cand3_expr),
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
    for cid, expr in candidates:
        print(f"\nRunning Cost Sensitivity Sweep for Candidate: '{cid}'...")
        res = run_cost_sensitivity(
            candidate_id=cid,
            expression=expr,
            panel=panel,
            wf_config=wf_config,
            backtest_config=backtest_config,
            cost_grid=STANDARD_COST_GRID,
            oos_fraction=0.15,
        )

        row = {
            "Candidate ID": cid,
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
