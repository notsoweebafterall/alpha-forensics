"""
Cross-sector and cross-period generalization testing for alpha candidates.

Scope & Design Constraints:
    Cross-market testing is explicitly out of scope because this project operates on a single yfinance market universe.
    Generalization is evaluated via:
    1. Cross-sector sub-universes: Evaluating candidate performance restricted to specific sector ticker subsets.
    2. Cross-period splits: Evaluating candidate performance across chronological first-half vs second-half date windows.

Single-Look Guard & Persistent Trial Discipline:
    Every sub-evaluation represents a distinct trial. Unique candidate IDs (e.g. candidate__sector_Technology,
    candidate__period_h1) are generated and logged to Phase 10's persistent trial ledger (data/trial_log.jsonl).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd

from core.panel import Panel
from data.panel import build_panel
from alpha.expressions.tree import Expression
from validation.folds import WalkForwardConfig
from validation.runner import run_walk_forward_validation
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from statistics import TrialRecord, log_trial, load_trial_log
from .sector_map import SECTOR_MAP


@dataclass
class GeneralizationResult:
    candidate_id: str
    sector_sharpes: Dict[str, float]
    period_split_sharpes: Tuple[float, float]  # (h1_sharpe, h2_sharpe)
    classification: str  # "universal", "sector_specific", "period_specific", "fragile"
    rationale: str


def evaluate_generalization(
    candidate_id: str,
    build_fn: Callable[[Dict[str, Any]], Expression],
    default_params: Dict[str, Any],
    panel: Panel,
    sector_map: Dict[str, str] = SECTOR_MAP,
    wf_config: WalkForwardConfig = WalkForwardConfig(),
    backtest_config: BacktestConfig = BacktestConfig(),
    cost_model: CostModel = CostModel(),
    log_path: Union[str, Path] = "data/trial_log.jsonl",
) -> GeneralizationResult:
    """
    Evaluates alpha candidate generalization across sector sub-universes and period half-splits.

    Args:
        candidate_id: Parent candidate identifier string.
        build_fn: Strategy expression builder function.
        default_params: Default parameter dictionary for build_fn.
        panel: Market data Panel.
        sector_map: Ticker-to-sector mapping.
        wf_config: WalkForwardConfig.
        backtest_config: BacktestConfig.
        cost_model: CostModel.
        log_path: Path to trial log ledger.

    Returns:
        GeneralizationResult: Sub-universe Sharpes, period split Sharpes, and classification.
    """
    expr = build_fn(default_params)
    existing_records = load_trial_log(log_path)
    existing_ids = {r.candidate_id for r in existing_records}
    now_str = datetime.now().isoformat()

    # 1. Evaluate Sector Sub-Universes
    sector_sharpes: Dict[str, float] = {}
    unique_sectors = sorted(list(set(sector_map.values())))

    for sec in unique_sectors:
        sec_tickers = [t for t in panel.universe if sector_map.get(t) == sec]
        # Require at least 3 tickers to form a meaningful sub-universe
        if len(sec_tickers) < 3:
            continue

        var_id = f"{candidate_id}__sector_{sec}"
        sec_prices = panel.prices[sec_tickers]
        sec_volume = panel.volume[sec_tickers]

        sec_panel = build_panel(
            tickers=sec_tickers,
            start_date=sec_prices.index[0],
            end_date=sec_prices.index[-1],
            missing_threshold=0.05,
            prices_df=sec_prices,
            volume_df=sec_volume,
        )

        try:
            val_res = run_walk_forward_validation(
                candidate_id=var_id,
                expression=expr,
                panel=sec_panel,
                wf_config=wf_config,
                backtest_config=backtest_config,
                cost_model=cost_model,
            )
            oos_sh = float(val_res.oos_metrics.get("sharpe_ratio", 0.0))
        except Exception:
            oos_sh = 0.0

        sector_sharpes[sec] = oos_sh

        # Log trial to persistent trial ledger
        if var_id not in existing_ids:
            try:
                log_trial(
                    TrialRecord(
                        candidate_id=var_id,
                        phase="Phase 11 Generalization Sector",
                        oos_sharpe=oos_sh,
                        timestamp=now_str,
                        backfilled=False,
                    ),
                    log_path=log_path,
                )
                existing_ids.add(var_id)
            except ValueError:
                pass

    # 2. Evaluate Period Half-Splits (First Half vs Second Half)
    total_dates = len(panel.prices.index)
    mid_idx = total_dates // 2

    # First Half Panel
    h1_prices = panel.prices.iloc[:mid_idx]
    h1_volume = panel.volume.iloc[:mid_idx]
    h1_panel = build_panel(
        tickers=panel.universe,
        start_date=h1_prices.index[0],
        end_date=h1_prices.index[-1],
        missing_threshold=0.05,
        prices_df=h1_prices,
        volume_df=h1_volume,
    )

    # Second Half Panel
    h2_prices = panel.prices.iloc[mid_idx:]
    h2_volume = panel.volume.iloc[mid_idx:]
    h2_panel = build_panel(
        tickers=panel.universe,
        start_date=h2_prices.index[0],
        end_date=h2_prices.index[-1],
        missing_threshold=0.05,
        prices_df=h2_prices,
        volume_df=h2_volume,
    )

    # Evaluate H1
    h1_var_id = f"{candidate_id}__period_h1"
    try:
        wf_h1 = WalkForwardConfig(
            mode="expanding",
            initial_train_window=max(60, len(h1_prices) // 2),
            step_size=30,
            val_window=30,
            embargo_days=5,
        )
        val_h1 = run_walk_forward_validation(
            candidate_id=h1_var_id,
            expression=expr,
            panel=h1_panel,
            wf_config=wf_h1,
            backtest_config=backtest_config,
            cost_model=cost_model,
        )
        h1_sharpe = float(val_h1.oos_metrics.get("sharpe_ratio", 0.0))
    except Exception:
        h1_sharpe = 0.0

    if h1_var_id not in existing_ids:
        try:
            log_trial(
                TrialRecord(
                    candidate_id=h1_var_id,
                    phase="Phase 11 Generalization Period H1",
                    oos_sharpe=h1_sharpe,
                    timestamp=now_str,
                    backfilled=False,
                ),
                log_path=log_path,
            )
            existing_ids.add(h1_var_id)
        except ValueError:
            pass

    # Evaluate H2
    h2_var_id = f"{candidate_id}__period_h2"
    try:
        wf_h2 = WalkForwardConfig(
            mode="expanding",
            initial_train_window=max(60, len(h2_prices) // 2),
            step_size=30,
            val_window=30,
            embargo_days=5,
        )
        val_h2 = run_walk_forward_validation(
            candidate_id=h2_var_id,
            expression=expr,
            panel=h2_panel,
            wf_config=wf_h2,
            backtest_config=backtest_config,
            cost_model=cost_model,
        )
        h2_sharpe = float(val_h2.oos_metrics.get("sharpe_ratio", 0.0))
    except Exception:
        h2_sharpe = 0.0

    if h2_var_id not in existing_ids:
        try:
            log_trial(
                TrialRecord(
                    candidate_id=h2_var_id,
                    phase="Phase 11 Generalization Period H2",
                    oos_sharpe=h2_sharpe,
                    timestamp=now_str,
                    backfilled=False,
                ),
                log_path=log_path,
            )
            existing_ids.add(h2_var_id)
        except ValueError:
            pass

    period_sharpes = (h1_sharpe, h2_sharpe)

    # 3. Classify Generalization
    pos_sectors = [s for s, sh in sector_sharpes.items() if sh > 0.0]
    total_sectors = len(sector_sharpes)
    sector_pos_ratio = (len(pos_sectors) / total_sectors) if total_sectors > 0 else 0.0

    h1_pos = h1_sharpe > 0.0
    h2_pos = h2_sharpe > 0.0

    if sector_pos_ratio >= 0.70 and h1_pos and h2_pos:
        classification = "universal"
        rationale = f"Positive OOS Sharpe across {len(pos_sectors)}/{total_sectors} sectors ({sector_pos_ratio:.0%}) and positive across both time period halves."
    elif sector_pos_ratio >= 0.50 and (h1_pos or h2_pos):
        classification = "sector_specific"
        rationale = f"Positive OOS Sharpe in moderate sector subset ({len(pos_sectors)}/{total_sectors}), with performance variation across time windows."
    elif not (h1_pos and h2_pos) and sector_pos_ratio >= 0.30:
        classification = "period_specific"
        rationale = f"Performance is split across time windows (H1: {h1_sharpe:.2f}, H2: {h2_sharpe:.2f}) and sector sub-universes."
    else:
        classification = "fragile"
        rationale = f"Negative or inconsistent OOS Sharpes across sectors ({len(pos_sectors)}/{total_sectors} positive) and period halves."

    return GeneralizationResult(
        candidate_id=candidate_id,
        sector_sharpes=sector_sharpes,
        period_split_sharpes=period_sharpes,
        classification=classification,
        rationale=rationale,
    )
