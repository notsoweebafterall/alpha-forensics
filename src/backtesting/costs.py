"""
Transaction cost modeling for portfolio backtesting.

Explicit Cost Model Disclaimer:
    This module implements a flat-cost model proportional to traded notional
    (cost_bps / 10000 * sum(|weight_changes|)). It does NOT model institutional
    market impact, order book depth, bid-ask spread dynamics, or non-linear slippage.
"""

from dataclasses import dataclass
import pandas as pd


@dataclass
class CostModel:
    cost_bps: float = 5.0
    apply_on: str = "traded_notional"


def compute_costs(weight_changes: pd.DataFrame, cost_model: CostModel) -> pd.Series:
    """
    Computes daily portfolio cost drag as a fraction of portfolio value.

    Formula:
        traded_notional_t = sum_i(|w_{t, i} - w_{t-1, i}|)
        cost_t = traded_notional_t * (cost_bps / 10000.0)

    Args:
        weight_changes: DataFrame of absolute daily weight changes (date x ticker).
        cost_model: CostModel configuration.

    Returns:
        pd.Series: Daily portfolio cost drag (fraction of portfolio value).
    """
    if cost_model.apply_on != "traded_notional":
        raise NotImplementedError(
            f"Unsupported cost application basis: {cost_model.apply_on}"
        )

    # Sum traded notional across tickers per date
    traded_notional = weight_changes.abs().sum(axis=1)

    # Cost drag in portfolio return units
    bps_factor = cost_model.cost_bps / 10000.0
    costs = traded_notional * bps_factor
    costs.name = "costs"
    return costs
