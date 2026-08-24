"""
Strategy Reproduction Demo (Phase 4).

Loads the 60-ticker Panel, evaluates all 5 canonical strategy families (and their parameter variant grids),
runs each through the backtesting engine, and displays in-sample economic & predictive metrics.

DISCLAIMER:
    These results are full-sample, in-sample illustrative metrics only. No out-of-sample (OOS) splits
    (Phase 7) or cost sensitivity sweeps (Phase 8) have been applied yet. Do NOT interpret these
    numbers as validated trading signals.
"""

import sys
import itertools
from pathlib import Path
import pandas as pd

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.strategies.registry import get_all_strategies
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from backtesting.engine import run_backtest
from backtesting.metrics import economic_metrics, predictive_metrics


def run_demo():
    print("=" * 90)
    print("ALPHA FORENSICS — PHASE 4 STRATEGY REPRODUCTION DEMO")
    print("=" * 90)
    print("DISCLAIMER: Full-sample, in-sample illustrative metrics only.")
    print("No out-of-sample split (Phase 7) or cost sensitivity sweep (Phase 8) applied.")
    print("-" * 90)

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
        dates = pd.date_range(start_date, periods=250, freq="B")
        import numpy as np
        np.random.seed(42)
        p_df = pd.DataFrame(100.0 + np.random.randn(len(dates), 60).cumsum(axis=0), index=dates, columns=UNIVERSE_60)
        v_df = pd.DataFrame(10000 + np.random.randint(0, 5000, size=(len(dates), 60)), index=dates, columns=UNIVERSE_60)
        panel = build_panel(tickers=UNIVERSE_60, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True, gross_exposure=1.0)
    cost_model = CostModel(cost_bps=5.0)

    results = []

    # 2. Iterate over all strategy definitions
    strategies = get_all_strategies()
    for strat in strategies:
        print(f"\nEvaluating strategy family: '{strat.name}' ({strat.family})")
        print(f"Rationale: \"{strat.hypothesis_text}\"")

        # Build variant grid parameter combinations
        keys = list(strat.variant_params.keys())
        param_values = list(strat.variant_params.values())
        grid_combos = [dict(zip(keys, v)) for v in itertools.product(*param_values)]

        for params in grid_combos:
            is_default = (params == strat.default_params)
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            if is_default:
                param_str += " (default)"

            # Build Expression tree
            expr = strat.build(params)

            # Evaluate Expression -> Signal
            signal = expr.evaluate(panel)

            # Run Backtest
            backtest_res = run_backtest(signal, panel, config, cost_model)

            # Compute Metrics
            econ = economic_metrics(backtest_res)
            pred = predictive_metrics(signal, panel, execution_lag_days=config.execution_lag_days)

            results.append({
                "Strategy": strat.name,
                "Parameters": param_str,
                "Sharpe": econ["sharpe_ratio"],
                "Ann Return": econ["annualized_return"],
                "Ann Vol": econ["annualized_volatility"],
                "Max DD": econ["max_drawdown"],
                "Turnover": econ["average_turnover"],
                "IC Mean": pred["ic_mean"],
                "Rank IC": pred["rank_ic_mean"],
            })

    # 3. Format and display summary results table
    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 110)
    print("SUMMARY PERFORMANCE TABLE (FULL-SAMPLE / IN-SAMPLE ONLY)")
    print("=" * 110)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary_df.to_string(index=False))
    print("=" * 110)


if __name__ == "__main__":
    run_demo()
