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


def simple_return(data: Union[Panel, pd.DataFrame], lag: int = 1) -> pd.DataFrame:
    """
    Computes simple return over `lag` periods: (P_t - P_{t-lag}) / P_{t-lag}.

    Args:
        data: Panel object or prices DataFrame.
        lag: Lookback lag (default 1).

    Returns:
        pd.DataFrame: Simple returns shaped (date x ticker).
    """
    prices = _get_df(data, field="prices")
    return (prices - prices.shift(lag)) / prices.shift(lag)


def momentum(data: Union[Panel, pd.DataFrame], lookback: int) -> pd.DataFrame:
    """
    Computes momentum as the cumulative return over `lookback` window:
    (P_t - P_{t-lookback}) / P_{t-lookback}.

    Args:
        data: Panel object or prices DataFrame.
        lookback: Rolling cumulative lookback window size.

    Returns:
        pd.DataFrame: Momentum values shaped (date x ticker).
    """
    prices = _get_df(data, field="prices")
    return (prices - prices.shift(lookback)) / prices.shift(lookback)
