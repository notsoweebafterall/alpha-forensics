"""
Backtesting configuration and position sizing specification.

Methodological Details for `position_sizing = "rank_weighted"`:
    - On each rebalance date t, cross-sectionally rank signal scores across tickers.
    - NaNs are strictly excluded prior to ranking and demeaning (tickers with NaN signals
      are assigned a target weight of 0.0).
    - For valid (non-NaN) tickers:
        1. Compute fractional percentile ranks scaled from 1 to N_valid (or 1..N).
        2. If `dollar_neutral=True`, demean the ranks by subtracting the mean rank of valid tickers
           (mean(ranks_valid)), ensuring target weights sum to ~0.0.
        3. Scale weights such that total gross exposure sum(|weights_valid|) equals `gross_exposure`.
    - This ensures dollar-neutrality is strictly preserved across non-NaN assets without distortion
      from missing entries.
"""

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    execution_lag_days: int = 1
    rebalance_frequency: str = "daily"
    position_sizing: str = "rank_weighted"
    dollar_neutral: bool = True
    gross_exposure: float = 1.0
