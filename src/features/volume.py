from typing import Union
import pandas as pd

from core.panel import Panel


def _get_df(data: Union[Panel, pd.DataFrame], field: str = "volume") -> pd.DataFrame:
    if isinstance(data, Panel):
        return getattr(data, field)
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise TypeError(f"Expected Panel or DataFrame, got {type(data)}")


def volume_change(data: Union[Panel, pd.DataFrame], lookback: int = 1) -> pd.DataFrame:
    """
    Computes percentage volume change over `lookback` periods:
    (V_t - V_{t-lookback}) / V_{t-lookback}.

    Args:
        data: Panel object or volume DataFrame.
        lookback: Lookback window size (default 1).

    Returns:
        pd.DataFrame: Percentage volume change shaped (date x ticker).
    """
    vol = _get_df(data, field="volume")
    return (vol - vol.shift(lookback)) / vol.shift(lookback)


def turnover(data: Union[Panel, pd.DataFrame], lookback: int) -> pd.DataFrame:
    """
    Computes turnover defined as current volume relative to rolling average volume over `lookback`:
    V_t / rolling_mean(V_t, lookback).

    Args:
        data: Panel object or volume DataFrame.
        lookback: Rolling average volume window size.

    Returns:
        pd.DataFrame: Relative volume turnover shaped (date x ticker).
    """
    vol = _get_df(data, field="volume")
    avg_vol = vol.rolling(window=lookback).mean()
    return vol / avg_vol
