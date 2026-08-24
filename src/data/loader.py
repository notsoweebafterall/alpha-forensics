from pathlib import Path
from typing import Union
import pandas as pd
import yfinance as yf


def fetch_market_data(
    tickers: list[str],
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    cache_dir: Union[str, Path] = "data/cache",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch market data (Adj Close prices and Volume) for given tickers and date range.
    Uses Parquet file caching in `cache_dir`. Explicitly uses 'Adj Close' for prices.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        cache_dir: Directory path for saving/loading parquet caches.
        force_refresh: If True, bypass cache and re-download.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (prices_df, volume_df)
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    prices_cache_file = cache_path / "prices.parquet"
    volume_cache_file = cache_path / "volume.parquet"

    if (
        not force_refresh
        and prices_cache_file.exists()
        and volume_cache_file.exists()
    ):
        prices_df = pd.read_parquet(prices_cache_file)
        volume_df = pd.read_parquet(volume_cache_file)

        # Check if cache covers requested tickers and date range
        tickers_present = set(tickers).issubset(set(prices_df.columns))
        if tickers_present and not prices_df.empty:
            cache_start = prices_df.index.min()
            cache_end = prices_df.index.max()
            if cache_start <= start_ts and cache_end >= end_ts:
                # Slice requested tickers and date range
                sub_prices = prices_df.loc[start_ts:end_ts, tickers]
                sub_volume = volume_df.loc[start_ts:end_ts, tickers]
                return sub_prices, sub_volume

    # Download via yfinance if cache miss or insufficient
    raw_data = yf.download(
        tickers,
        start=start_ts.strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if raw_data.empty:
        raise ValueError(f"No market data returned from yfinance for tickers: {tickers}")

    # Extract explicitly 'Adj Close' for prices and 'Volume' for volume
    if isinstance(raw_data.columns, pd.MultiIndex):
        prices_df = raw_data["Adj Close"]
        volume_df = raw_data["Volume"]
    else:
        # Single ticker case
        prices_df = raw_data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        volume_df = raw_data[["Volume"]].rename(columns={"Volume": tickers[0]})

    # Ensure index is DatetimeIndex and columns ordered as requested
    prices_df.index = pd.to_datetime(prices_df.index)
    volume_df.index = pd.to_datetime(volume_df.index)

    prices_df = prices_df.reindex(columns=tickers)
    volume_df = volume_df.reindex(columns=tickers)

    # Filter to requested start/end range
    prices_df = prices_df.loc[(prices_df.index >= start_ts) & (prices_df.index <= end_ts)]
    volume_df = volume_df.loc[(volume_df.index >= start_ts) & (volume_df.index <= end_ts)]

    # Save to parquet cache
    try:
        prices_df.to_parquet(prices_cache_file)
        volume_df.to_parquet(volume_cache_file)
    except Exception as e:
        print(f"Warning: Could not save parquet cache: {e}")

    return prices_df, volume_df
