"""
Parameter Robustness & Stability Analysis Demo (Phase 9).

Evaluates parameter landscapes across strategy parameter grids to determine whether alpha performance
is broadly robust across parameter neighborhoods or represents an isolated optimum.
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
from validation.folds import WalkForwardConfig
from validation.oos import reset_oos_access_log
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from robustness.parameter_landscape import evaluate_parameter_landscape


def run_demo(
    tickers: Optional[list[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    cache_dir: Union[str, Path] = "data/cache",
) -> None:
    print("=" * 100)
    print("ALPHA FORENSICS — PHASE 9 PARAMETER ROBUSTNESS ANALYSIS DEMO")
    print("=" * 100)
    print("Evaluates candidate performance across pre-specified parameter neighborhoods.")
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

    wf_config = WalkForwardConfig(
        mode="expanding",
        initial_train_window=400,
        step_size=60,
        val_window=60,
        embargo_days=10,
    )
    backtest_config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    cost_model = CostModel(cost_bps=5.0)

    # Strategy 1: cross_sectional_momentum (1D grid)
    strat1 = get_strategy("cross_sectional_momentum")
    print(f"\nEvaluating 1D Parameter Landscape for '{strat1.name}'...")
    res1 = evaluate_parameter_landscape(
        base_candidate_id=strat1.name,
        build_fn=strat1.build,
        param_grid=strat1.variant_params,
        panel=panel,
        wf_config=wf_config,
        backtest_config=backtest_config,
        cost_model=cost_model,
        max_variants=25,
        log_path=log_path,
        store_path=store_path,
        default_params=strat1.default_params,
    )

    rows1 = [
        {
            "Variant ID": vr.candidate_id,
            "Params": str(vr.params),
            "Train Sharpe (Mean)": vr.fold_sharpe_mean,
            "OOS Sharpe": vr.oos_sharpe,
            "OOS IC (Mean)": vr.oos_ic_mean,
            "Reused From Ledger": vr.reused_from_ledger,
        }
        for vr in res1.variant_results
    ]
    df1 = pd.DataFrame(rows1)

    print("\n--- Strategy 1: Cross-Sectional Momentum Landscape ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df1.to_string(index=False))
    print(f"Stability Classification : {res1.stability_label}")
    print(f"Stability Rationale      : {res1.stability_rationale}")

    # Strategy 2: volatility_adjusted_momentum (2D grid)
    strat2 = get_strategy("volatility_adjusted_momentum")
    print(f"\nEvaluating 2D Parameter Landscape for '{strat2.name}'...")
    res2 = evaluate_parameter_landscape(
        base_candidate_id=strat2.name,
        build_fn=strat2.build,
        param_grid=strat2.variant_params,
        panel=panel,
        wf_config=wf_config,
        backtest_config=backtest_config,
        cost_model=cost_model,
        max_variants=25,
        log_path=log_path,
        store_path=store_path,
        default_params=strat2.default_params,
    )

    rows2 = [
        {
            "Variant ID": vr.candidate_id,
            "Params": str(vr.params),
            "Train Sharpe (Mean)": vr.fold_sharpe_mean,
            "OOS Sharpe": vr.oos_sharpe,
            "OOS IC (Mean)": vr.oos_ic_mean,
            "Reused From Ledger": vr.reused_from_ledger,
        }
        for vr in res2.variant_results
    ]
    df2 = pd.DataFrame(rows2)

    print("\n--- Strategy 2: Volatility-Adjusted Momentum Landscape ---")
    print(df2.to_string(index=False))
    print(f"Stability Classification : {res2.stability_label}")
    print(f"Stability Rationale      : {res2.stability_rationale}")

    print("\n" + "=" * 100)
    print("NOTE: Parameter stability classification distinguishes broad robust effects from isolated peaks.")
    print("=" * 100)


if __name__ == "__main__":
    run_demo()