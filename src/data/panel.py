from pathlib import Path
from typing import Union, Optional
import pandas as pd

from core.panel import Panel
from .loader import fetch_market_data


def build_panel(
    tickers: list[str],
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    missing_threshold: float = 0.05,
    cache_dir: Union[str, Path] = "data/cache",
    prices_df: Optional[pd.DataFrame] = None,
    volume_df: Optional[pd.DataFrame] = None,
) -> Panel:
    """
    Builds a Panel object containing prices, volume, and computed daily returns.

    Missing-Data Handling Policy:
        - Tickers are NEVER silently dropped from the requested universe.
        - The fraction of missing data per ticker is calculated as (NaN count / total dates).
        - If any ticker exceeds `missing_threshold` (default 5%, i.e. 0.05), a ValueError is raised.
        - Isolated single-day missing price gaps are forward-filled (`ffill(limit=1)`).
        - Daily returns are computed directly from the cleaned price series:
            returns = (prices - prices.shift(1)) / prices.shift(1).

    Args:
        tickers: List of ticker symbols comprising the universe.
        start_date: Start timestamp for the panel.
        end_date: End timestamp for the panel.
        missing_threshold: Maximum allowable fraction of missing data per ticker (0.0 to 1.0).
        cache_dir: Directory path for parquet caching.
        prices_df: Optional pre-loaded prices DataFrame (used for synthetic testing / bypass).
        volume_df: Optional pre-loaded volume DataFrame (used for synthetic testing / bypass).

    Returns:
        Panel: Fully constructed Panel object with matching date indices and ticker columns.

    Raises:
        ValueError: If any ticker exceeds the missing data threshold or date ranges are invalid.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if prices_df is None or volume_df is None:
        fetched_prices, fetched_volume = fetch_market_data(
            tickers=tickers,
            start_date=start_ts,
            end_date=end_ts,
            cache_dir=cache_dir,
        )
        prices = fetched_prices if prices_df is None else prices_df
        volume = fetched_volume if volume_df is None else volume_df
    else:
        prices = prices_df.copy()
        volume = volume_df.copy()

    # Ensure index alignment & slicing
    prices = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts), tickers]
    volume = volume.loc[(volume.index >= start_ts) & (volume.index <= end_ts), tickers]

    total_dates = len(prices)
    if total_dates == 0:
        raise ValueError(f"No dates available in range [{start_ts}, {end_ts}].")

    # Enforce missing-data threshold check per ticker
    missing_ratios = prices.isna().sum(axis=0) / total_dates
    violating_tickers = missing_ratios[missing_ratios > missing_threshold]

    if not violating_tickers.empty:
        violating_info = ", ".join(
            [f"{t}: {ratio:.2%}" for t, ratio in violating_tickers.items()]
        )
        raise ValueError(
            f"Tickers exceed missing data threshold ({missing_threshold:.2%}): {violating_info}"
        )

    # Forward-fill isolated single-day gaps (limit=1)
    prices_clean = prices.ffill(limit=1)
    volume_clean = volume.ffill(limit=1)

    # Compute daily returns from adjusted close prices
    returns_df = (prices_clean - prices_clean.shift(1)) / prices_clean.shift(1)

    return Panel(
        prices=prices_clean,
        volume=volume_clean,
        returns=returns_df,
        universe=tickers,
        start_date=start_ts,
        end_date=end_ts,
    )
