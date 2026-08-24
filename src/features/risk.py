from typing import Union
import pandas as pd

from core.panel import Panel


def _get_prices(data: Union[Panel, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, Panel):
        return data.prices
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise TypeError(f"Expected Panel or DataFrame, got {type(data)}")


def _get_returns(data: Union[Panel, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, Panel):
        return data.returns
    elif isinstance(data, pd.DataFrame):
        # Compute daily returns from price series if DataFrame passed
        return (data - data.shift(1)) / data.shift(1)
    else:
        raise TypeError(f"Expected Panel or DataFrame, got {type(data)}")


def realized_volatility(data: Union[Panel, pd.DataFrame], window: int) -> pd.DataFrame:
    """
    Computes realized volatility as rolling standard deviation of daily returns over `window` periods.

    Args:
        data: Panel object or returns/prices DataFrame.
        window: Rolling window size.

    Returns:
        pd.DataFrame: Realized volatility shaped (date x ticker).
    """
    rets = _get_returns(data)
    return rets.rolling(window=window).std()


def drawdown(data: Union[Panel, pd.DataFrame], window: int) -> pd.DataFrame:
    """
    Computes drawdown relative to rolling peak price over `window` periods:
    (P_t - max_{t-window..t}(P)) / max_{t-window..t}(P).

    Values are non-positive (<= 0.0).

    Args:
        data: Panel object or prices DataFrame.
        window: Rolling peak lookback window size.

    Returns:
        pd.DataFrame: Drawdown values shaped (date x ticker).
    """
    prices = _get_prices(data)
    peak = prices.rolling(window=window).max()
    return (prices - peak) / peak
