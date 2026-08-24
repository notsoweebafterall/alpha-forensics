"""
Core backtesting engine enforcing point-in-time execution semantics.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from core.panel import Panel
from core.signal import Signal, validate_signal
from .config import BacktestConfig
from .costs import CostModel, compute_costs


@dataclass
class BacktestResult:
    weights: pd.DataFrame        # date x ticker, actual positions held (post-lag)
    turnover: pd.Series          # daily, sum of abs weight changes
    gross_returns: pd.Series     # daily portfolio return before costs
    net_returns: pd.Series       # daily portfolio return after costs
    costs: pd.Series             # daily cost drag


def signal_to_weights(signal: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """
    Converts cross-sectional signal scores to target portfolio weights.

    NaN Handling Discipline:
        - On each rebalance date t, NaN tickers are strictly excluded before ranking and demeaning.
        - Demeaning and gross exposure scaling are computed ONLY across valid (non-NaN) tickers V_t.
        - Tickers with NaN signal receive target weight 0.0.

    Args:
        signal: DataFrame of signal scores (date x ticker).
        config: BacktestConfig with position sizing rules.

    Returns:
        pd.DataFrame: Target portfolio weights (date x ticker).
    """
    if config.position_sizing != "rank_weighted":
        raise NotImplementedError(
            f"Unsupported position sizing method: {config.position_sizing}"
        )

    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for idx in signal.index:
        row = signal.loc[idx]
        valid_mask = row.notna()
        valid_tickers = row.index[valid_mask]
        n_valid = len(valid_tickers)

        if n_valid <= 1:
            continue

        valid_vals = row.loc[valid_tickers]
        # Cross-sectional ranking across valid tickers (1 to n_valid)
        ranks = valid_vals.rank(method="average")

        if config.dollar_neutral:
            # Demean ranks across valid tickers so sum(weights_valid) == 0
            demeaned_ranks = ranks - ranks.mean()
        else:
            demeaned_ranks = ranks

        gross_sum = demeaned_ranks.abs().sum()
        if gross_sum > 0:
            target_w = demeaned_ranks * (config.gross_exposure / gross_sum)
            weights.loc[idx, valid_tickers] = target_w

    return weights


def align_forward_returns(
    returns: pd.DataFrame, execution_lag_days: int = 1
) -> pd.DataFrame:
    """
    Aligns daily returns matrix to match signal dates under the engine's execution lag.

    For signal at date t executed at t + execution_lag_days, the realized forward return
    for signal at t is returns.shift(-execution_lag_days).

    Args:
        returns: Daily asset returns DataFrame (date x ticker).
        execution_lag_days: Execution lag in days (default 1).

    Returns:
        pd.DataFrame: Aligned forward returns DataFrame.
    """
    return returns.shift(-execution_lag_days)


def run_backtest(
    signal: Signal,
    panel: Panel,
    config: BacktestConfig = BacktestConfig(),
    cost_model: CostModel = CostModel(),
) -> BacktestResult:
    """
    Executes backtest on a Signal and Panel under point-in-time execution semantics.

    Args:
        signal: Signal DataFrame (date x ticker).
        panel: Panel containing asset prices, volume, and daily returns.
        config: BacktestConfig.
        cost_model: CostModel.

    Returns:
        BacktestResult: Output containing weights, turnover, gross/net returns, and costs.
    """
    # 1. Validate signal contract
    validate_signal(signal, panel)

    # 2. Convert signal scores to target portfolio weights
    target_weights = signal_to_weights(signal, config)

    # 3. AUDITABLE POINT-IN-TIME EXECUTION LAG:
    # Close(t) signal determines target position entered at Open(t+1)
    actual_weights = target_weights.shift(config.execution_lag_days).fillna(0.0)

    # 4. Compute daily gross portfolio returns
    gross_returns = (actual_weights * panel.returns).sum(axis=1)
    gross_returns.name = "gross_returns"

    # 5. Compute turnover (weight changes, including initial entry on day 0)
    weight_changes = actual_weights.diff()
    if not actual_weights.empty:
        weight_changes.iloc[0] = actual_weights.iloc[0]

    turnover = weight_changes.abs().sum(axis=1)
    turnover.name = "turnover"

    # 6. Compute transaction cost drag
    costs = compute_costs(weight_changes.abs(), cost_model)

    # 7. Compute net portfolio returns
    net_returns = gross_returns - costs
    net_returns.name = "net_returns"

    return BacktestResult(
        weights=actual_weights,
        turnover=turnover,
        gross_returns=gross_returns,
        net_returns=net_returns,
        costs=costs,
    )
