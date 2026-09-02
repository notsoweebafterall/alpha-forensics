"""
Walk-Forward / Out-of-Sample Validation Demo (Phase 7).

Demonstrates chronological expanding-window walk-forward fold validation and true out-of-sample
holdout evaluation for alpha candidates.

DISCLAIMER:
    This experiment performs real out-of-sample (OOS) validation. However, transaction costs are
    still modeled via a simple flat bps assumption (Phase 8 cost sensitivity sweep pending)
    and multiple-testing significance corrections have not yet been applied (Phase 10).
"""

import sys
from pathlib import Path
from typing import Optional, Union
import numpy as np
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
from validation.runner import run_walk_forward_validation
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from statistics import TrialRecord, load_trial_log, log_trial, compute_distribution_stats, save_returns


def run_demo(
    tickers: Optional[list[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    cache_dir: Union[str, Path] = "data/cache",
) -> None:
    print("=" * 90)
    print("ALPHA FORENSICS — PHASE 7 WALK-FORWARD / OOS VALIDATION DEMO")
    print("=" * 90)
    print("DISCLAIMER: Real Out-of-Sample (OOS) validation enabled.")
    print("Transaction costs are flat bps (Phase 8 sweep pending); multiple testing applies in Phase 10.")
    print("-" * 90)

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

    # Prepare 3 candidates. Candidate IDs are built via the canonical
    # build_candidate_id() -- and for the two that use a registered strategy at
    # exactly its default_params, the ID is the bare strategy name (no "__param_"
    # suffix), matching the convention Phase 9/10 use for the same default-param
    # expression. Without this, the identical expression would get logged under
    # two different candidate_ids from two different phases, double-counting it
    # in N for DSR purposes.
    strat_mom = get_strategy("cross_sectional_momentum")
    strat_vol = get_strategy("low_volatility")

    cand1_params = {"lookback": 60}
    cand1_expr = strat_mom.build(cand1_params)
    cand1_id = (
        build_candidate_id(strat_mom.name)
        if cand1_params == strat_mom.default_params
        else build_candidate_id(strat_mom.name, params=cand1_params)
    )

    cand2_params = {"window": 60}
    cand2_expr = strat_vol.build(cand2_params)
    cand2_id = (
        build_candidate_id(strat_vol.name)
        if cand2_params == strat_vol.default_params
        else build_candidate_id(strat_vol.name, params=cand2_params)
    )

    # Ad hoc one-off expression, not part of a registered strategy family --
    # no default_params concept applies, so it just gets its own bare base id.
    cand3_expr = UnaryNode(
        op_prim.Rank(),
        BinaryNode(
            op_prim.Multiply(),
            Leaf(in_prim.Momentum(20)),
            Leaf(in_prim.VolumeChange(5)),
        ),
    )
    cand3_id = build_candidate_id("volume_momentum_combo")

    candidates = [
        (cand1_id, cand1_expr),
        (cand2_id, cand2_expr),
        (cand3_id, cand3_expr),
    ]

    wf_config = WalkForwardConfig(
        mode="expanding",
        initial_train_window=400,
        step_size=60,
        val_window=60,
        embargo_days=10,
    )

    backtest_config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    cost_model = CostModel(cost_bps=5.0)

    summary_rows = []

    # 2. Execute Walk-Forward Validation
    existing_ledger = {r.candidate_id: r for r in load_trial_log(log_path)}

    for cid, expr in candidates:
        print(f"\n" + "-" * 80)
        print(f"Running Walk-Forward Validation for Candidate: '{cid}'")
        print(f"Expression: {expr.to_string()}")
        print("-" * 80)

        if cid in existing_ledger:
            rec = existing_ledger[cid]
            print(f"'{cid}' already logged in trial_log.jsonl (oos_sharpe={rec.oos_sharpe:.4f}) "
                  f"-- reusing per single-look discipline, not re-evaluating fold-by-fold detail.")
            summary_rows.append({
                "Candidate ID": cid,
                "Folds Count": None,
                "Mean Train Sharpe": None,
                "OOS Sharpe": rec.oos_sharpe,
                "Degradation Ratio": None,
                "Mean Train IC": None,
                "OOS IC": None,
            })
            continue

        val_res = run_walk_forward_validation(
            candidate_id=cid,
            expression=expr,
            panel=panel,
            wf_config=wf_config,
            backtest_config=backtest_config,
            cost_model=cost_model,
            oos_fraction=0.15,
        )

        print(f"Generated {len(val_res.folds)} expanding walk-forward folds.")
        for fold, m in zip(val_res.folds, val_res.fold_metrics):
            print(
                f"  Fold {fold.fold_id}: Train [{fold.train_range[0].strftime('%Y-%m-%d')} .. {fold.train_range[1].strftime('%Y-%m-%d')}] | "
                f"Val [{fold.val_range[0].strftime('%Y-%m-%d')} .. {fold.val_range[1].strftime('%Y-%m-%d')}] -> "
                f"Train Sharpe: {m['sharpe_ratio']:.4f}, IC: {m['ic_mean']:.4f}"
            )

        print("\nTrue OOS Holdout Evaluation:")
        oos_m = val_res.oos_metrics
        print(f"  OOS Sharpe: {oos_m['sharpe_ratio']:.4f}, OOS IC: {oos_m['ic_mean']:.4f}, OOS Max DD: {oos_m['max_drawdown']:.4f}")

        deg = val_res.degradation
        print(f"Degradation Analysis:")
        print(f"  Mean Train Sharpe : {deg['mean_train_sharpe']:.4f}")
        print(f"  OOS Sharpe        : {deg['oos_sharpe']:.4f}")
        print(f"  Degradation Ratio : {deg['degradation_ratio']:.4f}")

        # Use net returns (same series as oos_sharpe) for distribution stats and storage
        oos_net_returns = val_res.oos_net_returns
        skew, kurtosis, t_len = compute_distribution_stats(oos_net_returns)
        log_trial(
            TrialRecord(
                candidate_id=cid,
                phase="Phase 7 Walk-Forward Validation",
                oos_sharpe=float(deg["oos_sharpe"]),
                timestamp=pd.Timestamp.now().isoformat(),
                skew=skew,
                kurtosis=kurtosis,
                track_record_length=t_len,
            ),
            log_path=log_path,
        )
        if oos_net_returns is not None:
            try:
                save_returns(cid, oos_net_returns, store_path=store_path)
            except ValueError:
                pass  # Already stored; immutability discipline

        summary_rows.append({
            "Candidate ID": cid,
            "Folds Count": len(val_res.folds),
            "Mean Train Sharpe": deg["mean_train_sharpe"],
            "OOS Sharpe": deg["oos_sharpe"],
            "Degradation Ratio": deg["degradation_ratio"],
            "Mean Train IC": deg["mean_train_ic"],
            "OOS IC": deg["oos_ic"],
        })

    # 3. Format & Display Summary Results
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 100)
    print("WALK-FORWARD & OOS VALIDATION SUMMARY TABLE")
    print("=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary_df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    run_demo()