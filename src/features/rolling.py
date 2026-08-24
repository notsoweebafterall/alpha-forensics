from typing import Union
import pandas as pd

from core.panel import Panel


def _get_df(data: Union[Panel, pd.DataFrame], field: str = "prices") -> pd.DataFrame:
    if isinstance(data, Panel):
        return getattr(data, field)
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise TypeError(f"Expected Panel or DataFrame, got {type(data)}")


def rolling_mean(data: Union[Panel, pd.DataFrame], window: int) -> pd.DataFrame:
    """
    Computes rolling mean of prices over `window` periods.

    Args:
        data: Panel object or prices DataFrame.
        window: Rolling window size.

    Returns:
        pd.DataFrame: Rolling mean values shaped (date x ticker).
    """
    prices = _get_df(data, field="prices")
    return prices.rolling(window=window).mean()


def rolling_std(data: Union[Panel, pd.DataFrame], window: int) -> pd.DataFrame:
    """
    Computes rolling sample standard deviation of prices over `window` periods.

    Args:
        data: Panel object or prices DataFrame.
        window: Rolling window size.

    Returns:
        pd.DataFrame: Rolling standard deviation shaped (date x ticker).
    """
    prices = _get_df(data, field="prices")
    return prices.rolling(window=window).std()


def rolling_zscore(data: Union[Panel, pd.DataFrame], window: int) -> pd.DataFrame:
    """
    Computes rolling z-score: (P_t - rolling_mean(P_t, window)) / rolling_std(P_t, window).

    Args:
        data: Panel object or prices DataFrame.
        window: Rolling window size.

    Returns:
        pd.DataFrame: Rolling z-score shaped (date x ticker).
    """
    prices = _get_df(data, field="prices")
    mean = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    return (prices - mean) / std


def rolling_rank(data: Union[Panel, pd.DataFrame], window: int) -> pd.DataFrame:
    """
    Computes rolling cross-sectional rank.

    Definition:
        For each date t, computes the rolling mean over `window` per ticker,
        and then ranks tickers cross-sectionally on that date (percentile rank
        across all tickers, scaled from 0.0 to 1.0, not each ticker's rank against
        its own time-series history).

    Args:
        data: Panel object or prices DataFrame.
        window: Rolling window size for the mean computation.

    Returns:
        pd.DataFrame: Cross-sectional percentile rank of rolling mean (date x ticker).
    """
    prices = _get_df(data, field="prices")
    rmean = prices.rolling(window=window).mean()
    return rmean.rank(axis=1, pct=True)
