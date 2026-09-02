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
    drop_invalid_tickers: bool = True,
) -> Panel:
    """
    Builds a Panel object containing prices, volume, and computed daily returns.

    Missing-Data Handling Policy:
        - The fraction of missing data per ticker is calculated as (NaN count / total dates).
        - Tickers exceeding `missing_threshold` (default 5%, i.e. 0.05) are dropped with a warning if
          `drop_invalid_tickers=True` (default), allowing the panel to proceed with remaining valid tickers.
        - If `drop_invalid_tickers=False` or all tickers fail, a ValueError is raised.
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
        drop_invalid_tickers: If True, drop tickers exceeding missing threshold and continue with valid ones.

    Returns:
        Panel: Fully constructed Panel object with matching date indices and ticker columns.

    Raises:
        ValueError: If all tickers exceed the missing data threshold or date ranges are invalid.
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

    valid_tickers = list(tickers)
    if not violating_tickers.empty:
        violating_info = ", ".join(
            [f"{t}: {ratio:.2%}" for t, ratio in violating_tickers.items()]
        )
        if drop_invalid_tickers:
            drop_set = set(violating_tickers.index)
            valid_tickers = [t for t in tickers if t not in drop_set]
            print(
                f"Warning: Dropping {len(drop_set)} ticker(s) exceeding missing data threshold ({missing_threshold:.2%}): {violating_info}",
                flush=True,
            )
            if not valid_tickers:
                raise ValueError(
                    f"All tickers exceed missing data threshold ({missing_threshold:.2%}): {violating_info}"
                )
            prices = prices[valid_tickers]
            volume = volume[valid_tickers]
        else:
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
        universe=valid_tickers,
        start_date=start_ts,
        end_date=end_ts,
    )
