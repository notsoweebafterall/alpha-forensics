"""
Out-of-sample (OOS) holdout reservation and process-local runtime guard.

Disclaimers & Safeguards:
    `reserve_oos_holdout` reserves the final `oos_fraction` portion of the date range
    strictly for out-of-sample testing.

    `evaluate_oos` implements an in-memory runtime access guard (_oos_access_log) that
    prevents accidental multiple evaluations of the same candidate_id within a single execution run.
"""

from typing import Set, Tuple
import pandas as pd

from core.panel import Panel
from data.panel import build_panel
from alpha.expressions.tree import Expression
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from backtesting.engine import run_backtest
from backtesting.metrics import economic_metrics, predictive_metrics

# In-memory access log for process-local OOS evaluation guard
_oos_access_log: Set[str] = set()


def reset_oos_access_log() -> None:
    """Resets the process-local OOS access log (primarily for testing)."""
    _oos_access_log.clear()


def reserve_oos_holdout(
    panel: Panel, oos_fraction: float = 0.15
) -> Tuple[Panel, Panel]:
    """
    Splits a Panel chronologically into (dev_panel, oos_panel).

    The oos_panel is ALWAYS the final oos_fraction portion of the date range.

    Args:
        panel: Full Panel object.
        oos_fraction: Fraction of dates to reserve for OOS holdout (0.0 < oos_fraction < 1.0).

    Returns:
        Tuple[Panel, Panel]: (development_panel, oos_panel)
    """
    if not (0.0 < oos_fraction < 1.0):
        raise ValueError(f"oos_fraction must be between 0.0 and 1.0, got {oos_fraction}")

    total_dates = len(panel.prices.index)
    oos_count = max(1, int(total_dates * oos_fraction))
    dev_count = total_dates - oos_count

    if dev_count <= 0:
        raise ValueError("Panel has insufficient dates to reserve an OOS holdout.")

    dev_prices = panel.prices.iloc[:dev_count]
    dev_volume = panel.volume.iloc[:dev_count]

    oos_prices = panel.prices.iloc[dev_count:]
    oos_volume = panel.volume.iloc[dev_count:]

    dev_panel = build_panel(
        tickers=panel.universe,
        start_date=dev_prices.index[0],
        end_date=dev_prices.index[-1],
        missing_threshold=0.05,
        prices_df=dev_prices,
        volume_df=dev_volume,
    )

    oos_panel = build_panel(
        tickers=panel.universe,
        start_date=oos_prices.index[0],
        end_date=oos_prices.index[-1],
        missing_threshold=0.05,
        prices_df=oos_prices,
        volume_df=oos_volume,
    )

    return dev_panel, oos_panel


def evaluate_oos(
    candidate_id: str,
    expression: Expression,
    oos_panel: Panel,
    config: BacktestConfig,
    cost_model: CostModel,
) -> dict:
    """
    Evaluates candidate expression on the true OOS holdout panel.

    Runtime Guard:
        Enforces a process-local single-evaluation policy per candidate_id to prevent
        accidental double-dipping or data snooping within a single execution run.

    Args:
        candidate_id: Unique candidate identifier string.
        expression: Expression tree.
        oos_panel: OOS Panel holdout.
        config: BacktestConfig.
        cost_model: CostModel.

    Returns:
        dict: Performance and predictive metrics on the OOS panel.

    Raises:
        RuntimeError: If candidate_id has already been evaluated on OOS in this process.
    """
    if candidate_id in _oos_access_log:
        raise RuntimeError(
            f"OOS already evaluated for '{candidate_id}' — re-running OOS on the same candidate "
            f"defeats the purpose of a holdout."
        )

    _oos_access_log.add(candidate_id)

    signal = expression.evaluate(oos_panel)
    res = run_backtest(signal, oos_panel, config, cost_model)

    econ = economic_metrics(res)
    pred = predictive_metrics(signal, oos_panel, execution_lag_days=config.execution_lag_days)

    return {
        **econ,
        **pred,
        "oos_gross_returns": res.gross_returns,
        "oos_net_returns": res.net_returns,
        "oos_turnover": res.turnover,
    }
