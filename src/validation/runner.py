"""
Chronological walk-forward validation runner combining development folds and OOS evaluation.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from core.panel import Panel
from data.panel import build_panel
from alpha.expressions.tree import Expression
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from backtesting.engine import run_backtest
from backtesting.metrics import economic_metrics, predictive_metrics

from .folds import WalkForwardConfig, Fold, generate_folds
from .oos import reserve_oos_holdout, evaluate_oos


@dataclass
class ValidationResult:
    candidate_id: str
    folds: List[Fold]
    fold_metrics: List[Dict[str, Any]]
    oos_metrics: Dict[str, Any]
    degradation: Dict[str, Any]
    oos_gross_returns: Optional[pd.Series] = None
    oos_turnover: Optional[pd.Series] = None
    oos_net_returns: Optional[pd.Series] = None


def run_walk_forward_validation(
    candidate_id: str,
    expression: Expression,
    panel: Panel,
    wf_config: WalkForwardConfig = WalkForwardConfig(),
    backtest_config: BacktestConfig = BacktestConfig(),
    cost_model: CostModel = CostModel(),
    oos_fraction: float = 0.15,
) -> ValidationResult:
    """
    Executes complete walk-forward validation pipeline for an alpha expression.

    Pipeline Steps:
        1. Reserves chronologically last oos_fraction of dates as a true OOS holdout.
        2. Generates expanding/rolling walk-forward folds over the development panel with embargo enforcement.
        3. Evaluates expression and backtests across all development folds.
        4. Evaluates expression exactly once on the true OOS holdout panel.
        5. Computes performance degradation (mean development fold Sharpe vs OOS Sharpe).

    Args:
        candidate_id: Unique candidate identifier string.
        expression: Expression tree.
        panel: Full Panel object.
        wf_config: WalkForwardConfig.
        backtest_config: BacktestConfig.
        cost_model: CostModel.
        oos_fraction: OOS holdout fraction (default 0.15).

    Returns:
        ValidationResult: Complete validation metrics and degradation analysis.
    """
    # 1. Split Panel into development and OOS holdouts
    dev_panel, oos_panel = reserve_oos_holdout(panel, oos_fraction=oos_fraction)

    # 2. Generate walk-forward folds over development panel
    folds = generate_folds(dev_panel, wf_config)

    fold_metrics: List[Dict[str, Any]] = []

    # 3. Evaluate each development fold
    #
    # IMPLEMENTATION NOTE — Option A (required by primitive evaluation semantics):
    #
    # All rolling-window primitives (Momentum, RollingMean, RollingStd, RollingZScore,
    # RollingRank, RealizedVolatility, Drawdown) are implemented via pandas .shift() or
    # .rolling(), which produce NaN for the first (window-1) rows of whatever DataFrame
    # they receive.  They do NOT look outside the supplied DataFrame's index.
    #
    # Consequence: if we built the panel only from val_range, Momentum(252) would produce
    # NaN for the first 251 dates of val_range — no valid signal to score.
    #
    # Correct approach:
    #   1. Evaluate the expression on a panel spanning [train_range[0], val_range[1]],
    #      so that rolling primitives have their full lookback history available at the
    #      start of val_range and produce valid, non-NaN signal values there.
    #   2. Slice the resulting signal AND the panel's returns to [val_range[0], val_range[1]]
    #      BEFORE computing economic_metrics / predictive_metrics.
    #
    # The reported fold Sharpe and IC therefore measure genuine out-of-fold performance:
    # the expression was never fitted to val_range, and the metrics are computed only on
    # dates within that unseen validation window.
    for fold in folds:
        # Build a wide panel: history from train_range[0] through val_range[1]
        # so that rolling primitives compute correctly at the start of val_range.
        wide_p = dev_panel.prices.loc[fold.train_range[0] : fold.val_range[1]]
        wide_v = dev_panel.volume.loc[fold.train_range[0] : fold.val_range[1]]

        wide_panel = build_panel(
            tickers=dev_panel.universe,
            start_date=fold.train_range[0],
            end_date=fold.val_range[1],
            missing_threshold=0.05,
            prices_df=wide_p,
            volume_df=wide_v,
        )

        # Evaluate expression on the full wide panel (train + embargo + val)
        full_signal = expression.evaluate(wide_panel)

        # Restrict both the signal and the panel slice to val_range only
        # so that metrics are computed on the genuinely held-out validation window.
        val_start, val_end = fold.val_range
        val_signal = full_signal.loc[val_start:val_end]

        val_p = dev_panel.prices.loc[val_start:val_end]
        val_v = dev_panel.volume.loc[val_start:val_end]
        val_panel = build_panel(
            tickers=dev_panel.universe,
            start_date=val_start,
            end_date=val_end,
            missing_threshold=0.05,
            prices_df=val_p,
            volume_df=val_v,
        )

        res = run_backtest(val_signal, val_panel, backtest_config, cost_model)

        econ = economic_metrics(res)
        pred = predictive_metrics(
            val_signal, val_panel, execution_lag_days=backtest_config.execution_lag_days
        )

        m = {**econ, **pred, "fold_id": fold.fold_id}
        fold_metrics.append(m)

    # 4. Evaluate OOS panel using runtime guard
    oos_metrics = evaluate_oos(
        candidate_id, expression, oos_panel, backtest_config, cost_model
    )

    # 5. Compute degradation ratio
    train_sharpes = [m["sharpe_ratio"] for m in fold_metrics]
    mean_train_sharpe = float(np.mean(train_sharpes)) if train_sharpes else 0.0
    oos_sharpe = float(oos_metrics.get("sharpe_ratio", 0.0))

    train_ics = [m["ic_mean"] for m in fold_metrics]
    mean_train_ic = float(np.mean(train_ics)) if train_ics else 0.0
    oos_ic = float(oos_metrics.get("ic_mean", 0.0))

    # Handle division by zero or sign flips explicitly
    if abs(mean_train_sharpe) < 1e-8:
        degradation_ratio = 1.0 if abs(oos_sharpe) < 1e-8 else 0.0
    else:
        degradation_ratio = oos_sharpe / mean_train_sharpe

    degradation = {
        "mean_train_sharpe": mean_train_sharpe,
        "oos_sharpe": oos_sharpe,
        "mean_train_ic": mean_train_ic,
        "oos_ic": oos_ic,
        "degradation_ratio": degradation_ratio,
    }

    return ValidationResult(
        candidate_id=candidate_id,
        folds=folds,
        fold_metrics=fold_metrics,
        oos_metrics=oos_metrics,
        degradation=degradation,
        oos_gross_returns=oos_metrics.get("oos_gross_returns"),
        oos_turnover=oos_metrics.get("oos_turnover"),
        oos_net_returns=oos_metrics.get("oos_net_returns"),
    )
