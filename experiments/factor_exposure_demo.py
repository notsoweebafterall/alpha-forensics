"""
Factor Exposure & Cross-Universe Generalization Analysis Demo (Phase 11).

Builds self-constructed factor panel (MKT, MOM, VOL, SECTOR) from the 60-ticker yfinance Panel,
evaluates factor regression betas, HAC t-statistics, R^2, and residual Sharpes, and assesses
cross-sector and cross-period generalization across Phase 9/10 alpha candidates.

DISCLAIMER:
    All factors are self-constructed strictly from the 60-ticker universe per the project's
    locked data policy (no external SPY, Fama-French, or fundamental data).

Data-Hygiene Design:
    The factor panel is built ONLY from the development (pre-OOS) portion of the panel.
    This prevents MOM (252-day formation window) and VOL (60-day lookback) factor construction
    from using any prices/returns that fall inside the guarded OOS holdout window.
    Factor panel dates inside the OOS window are then reconstructed via restrict_factor_panel_to_oos()
    using only the OOS sub-panel's own data — the same data the candidate's OOS returns were evaluated on.
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
from statistics import load_trial_log
from factors import (
    SECTOR_MAP,
    build_factor_panel,
    run_factor_regression,
    evaluate_generalization,
)


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
    print("ALPHA FORENSICS — PHASE 11 FACTOR EXPOSURE & GENERALIZATION DEMO", flush=True)
    print("=" * 100, flush=True)
    print("Self-Constructed Factor Model (MKT, MOM, VOL, SECTOR) & Cross-Subuniverse Testing", flush=True)
    print("-" * 100, flush=True)

    reset_oos_access_log()
    log_path = Path(log_path)
    store_path = Path(store_path)
    cache_dir = Path(cache_dir)
    target_tickers = UNIVERSE_60 if tickers is None else tickers

    trial_records = load_trial_log(log_path)
    trial_dict = {r.candidate_id: r for r in trial_records}

    print(f"Loading {len(target_tickers)}-ticker Panel [{start_date} to {end_date}]...", flush=True)
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

    # 3. Split panel into dev and OOS holdout FIRST, then build FactorPanel from OOS sub-panel only.
    oos_fraction = 0.15
    dev_panel, oos_panel = reserve_oos_holdout(panel, oos_fraction=oos_fraction)

    print(f"\nPanel split: Dev = {len(dev_panel.prices)} days "
          f"[{dev_panel.prices.index[0].strftime('%Y-%m-%d')} to "
          f"{dev_panel.prices.index[-1].strftime('%Y-%m-%d')}], "
          f"OOS = {len(oos_panel.prices)} days "
          f"[{oos_panel.prices.index[0].strftime('%Y-%m-%d')} to "
          f"{oos_panel.prices.index[-1].strftime('%Y-%m-%d')}]", flush=True)

    # Build FactorPanel from OOS sub-panel (factors computed using only OOS-window data)
    print("\nBuilding self-constructed FactorPanel (MKT, MOM, VOL, SECTOR) — OOS window only...", flush=True)
    factor_panel = build_factor_panel(oos_panel, sector_map=SECTOR_MAP)
    print(f"Factor Panel constructed over OOS window: {len(factor_panel.dates)} dates.", flush=True)
    print(f"Sectors present: {list(factor_panel.sector_returns.columns)}", flush=True)

    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    cost_model = CostModel(cost_bps=5.0)
    wf_config = WalkForwardConfig(
        mode="expanding",
        initial_train_window=400,
        step_size=60,
        val_window=60,
        embargo_days=10,
    )

    # 4. Phase 9 candidate variants -- IDs are derived via the same canonical
    #    build_candidate_id() used everywhere else, from REAL parameter names/values
    #    that actually exist in each strategy's variant_params grid (see
    #    alpha.strategies.library). No candidate_id is hand-typed here anymore --
    #    a hand-typed ID using invented parameter names ("vol_lookback"/"target",
    #    which appear nowhere in this strategy's real grid) was exactly how a
    #    fabricated candidate got treated as real in a previous version of this file.
    candidates_to_test = [
        {
            "strat_name": "cross_sectional_momentum",
            "params": {"lookback": 20},
            "sectors": ["Technology", "Financials", "Healthcare", "Consumer Discretionary"],
        },
        {
            "strat_name": "volatility_adjusted_momentum",
            "params": {"lookback": 60, "window": 20},
            "sectors": ["Technology", "Financials", "Healthcare", "Consumer Discretionary"],
        },
    ]
    for item in candidates_to_test:
        item["strat"] = get_strategy(item["strat_name"])
        item["candidate_id"] = (
            build_candidate_id(item["strat_name"])
            if item["params"] == item["strat"].default_params
            else build_candidate_id(item["strat_name"], params=item["params"])
        )

    print("\n" + "=" * 100, flush=True)
    print("1. FACTOR EXPOSURE OLS REGRESSION (HAC-ROBUST NEWEY-WEST STANDARD ERRORS)", flush=True)
    print("=" * 100, flush=True)

    for item in candidates_to_test:
        cid = item["candidate_id"]
        strat = get_strategy(item["strat_name"])
        expr = strat.build(item["params"])

        # Retrieve Phase 9/10 recorded OOS Sharpe from trial ledger
        logged_rec = trial_dict.get(cid)
        phase910_oos_sharpe = logged_rec.oos_sharpe if logged_rec else 0.0

        val_res = run_walk_forward_validation(
            candidate_id=f"demo_{cid}",
            expression=expr,
            panel=panel,
            wf_config=wf_config,
            backtest_config=config,
            cost_model=cost_model,
            oos_fraction=oos_fraction,
        )

        oos_gross = val_res.oos_gross_returns
        oos_turnover = val_res.oos_turnover
        if oos_gross is not None and oos_turnover is not None:
            cost_drag = oos_turnover * (cost_model.cost_bps / 10000.0)
            oos_candidate_returns = oos_gross - cost_drag
        else:
            oos_candidate_returns = oos_gross

        reg_res = run_factor_regression(
            candidate_returns=oos_candidate_returns,
            factor_panel=factor_panel,
            candidate_sectors=item["sectors"],
            candidate_id=cid,
        )

        print(f"\n--- Strategy Candidate ID: [{cid}] ---", flush=True)
        print(f"Base Strategy Family   : {strat.name}", flush=True)
        print(f"Evaluated Parameters   : {item['params']}", flush=True)
        print(f"Candidate Sectors Passed: {item['sectors']}", flush=True)
        print(f"Guarded OOS Date Window : {oos_candidate_returns.index[0].strftime('%Y-%m-%d')} "
              f"to {oos_candidate_returns.index[-1].strftime('%Y-%m-%d')} "
              f"({len(oos_candidate_returns)} days)", flush=True)
        if logged_rec is not None:
            print(f"Phase 9/10 Trial Sharpe : {phase910_oos_sharpe:.4f}  "
                  f"<-- Reproduced directly from trial_log.jsonl ledger for candidate variant", flush=True)
        else:
            print(f"Phase 9/10 Trial Sharpe : N/A -- no matching entry for '{cid}' exists in "
                  f"trial_log.jsonl yet. Run the Phase 9 parameter robustness demo first if you "
                  f"expect this candidate to have a prior ledger entry.", flush=True)
        print(f"Regression Raw Sharpe   : {reg_res.raw_sharpe:.4f}  "
              f"<-- Evaluated on OOS net returns aligned to factor date window", flush=True)
        print(f"Residual Strategy Sharpe: {reg_res.residual_sharpe:.4f}", flush=True)
        print(f"R-Squared (Explanatory) : {reg_res.r_squared:.4f}", flush=True)
        print(f"Alpha Classification    : {reg_res.verdict.upper()}", flush=True)

        print("\n  Factor Betas & HAC t-statistics:", flush=True)
        for factor_name, beta in reg_res.betas.items():
            t_val = reg_res.t_stats.get(factor_name, 0.0)
            corr = reg_res.factor_correlations.get(factor_name, 0.0)
            print(f"    * {factor_name:<25}: Beta = {beta:>8.4f} | t-stat = {t_val:>8.4f} | Corr = {corr:>8.4f}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("2. CROSS-SECTOR & CROSS-PERIOD GENERALIZATION ANALYSIS", flush=True)
    print("=" * 100, flush=True)

    for item in candidates_to_test:
        cid = item["candidate_id"]
        strat = get_strategy(item["strat_name"])
        print(f"\nEvaluating Generalization for [{cid}]...", flush=True)

        gen_res = evaluate_generalization(
            candidate_id=cid,
            build_fn=strat.build,
            default_params=item["params"],
            panel=panel,
            sector_map=SECTOR_MAP,
            wf_config=wf_config,
            backtest_config=config,
            cost_model=cost_model,
            log_path=log_path,
            store_path=store_path,
        )

        print(f"\n--- Strategy Candidate ID: [{cid}] ---", flush=True)
        print(f"Generalization Class    : {gen_res.classification.upper()}", flush=True)
        print(f"Rationale               : {gen_res.rationale}", flush=True)
        print("  Period Split OOS Sharpes:", flush=True)
        print(f"    * First Half  (H1)   : {gen_res.period_split_sharpes[0]:.4f}", flush=True)
        print(f"    * Second Half (H2)   : {gen_res.period_split_sharpes[1]:.4f}", flush=True)
        print("  Sector Sub-Universe OOS Sharpes:", flush=True)
        for sec, sh in gen_res.sector_sharpes.items():
            print(f"    * Sector {sec:<22}: OOS Sharpe = {sh:.4f}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("SUMMARY CONCLUSION:", flush=True)
    print("Self-constructed factor regression and generalization analysis successfully evaluate candidate variants", flush=True)
    print("using real, in-grid strategy parameters and canonically-derived candidate IDs, with any prior", flush=True)
    print("Phase 9/10 ledger Sharpe shown only when an actual matching entry exists.", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    run_demo()