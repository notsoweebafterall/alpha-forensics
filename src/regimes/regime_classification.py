"""
Pure market regime classification logic based on Panel data.

Overview & Rationale:
    Market regimes are classified directly from the yfinance Panel data without using external
    index data or VIX feeds (strictly consistent with the project's locked self-constructed data policy).
    The market benchmark used is the equal-weighted daily return across all tickers in the Panel
    (the same MKT factor definition used in Phase 11 factor exposure analysis).

Regime Classification Scheme:
    1. Trend Regime (Trailing N-day Cumulative Return):
       - Trailing N-day cumulative market return = (1 + MKT).cumprod() / (1 + MKT).cumprod().shift(N) - 1.0.
       - Positive cumulative return -> 'bull'
       - Negative cumulative return -> 'bear'
       - Default trailing window N = 60 trading days (~1 calendar quarter), long enough to smooth
         daily noise while capturing major market trend shifts within a 1-2 year evaluation window.

    2. Volatility Regime (Trailing N-day Realized Volatility):
       - Trailing N-day realized volatility = std(MKT) over trailing N trading days (matching the VOL factor lookback).
       - Evaluated against its own median over the valid evaluation window:
         - Volatility >= median -> 'high_vol'
         - Volatility < median  -> 'low_vol'

    3. Combined Regime Label:
       - Formed as '{trend_regime}_{vol_regime}' (e.g. 'bull_high_vol', 'bear_low_vol').

EXPLICIT CAVEAT 1 (Ex-Post Structural Benchmark):
    While trailing N-day cumulative return and trailing N-day realized volatility are strictly backward-looking
    at each day t, the median volatility threshold is computed over the entire non-NaN evaluation window.
    This makes the volatility regime split an ex-post structural classification for audited regime analysis,
    rather than a real-time trading signal. This distinction is intentional for post-hoc generalization
    and robustness testing across evaluation samples.

EXPLICIT CAVEAT 2 (Tail OOS Window Coverage Limitation):
    Because the OOS holdout window sits chronologically at the end of the historical dataset by construction,
    regime robustness testing evaluates performance strictly across whichever market regimes happened to
    occur within that specific tail window. Regimes present earlier in the panel but absent in the OOS
    tail window are explicitly reported as skipped rather than evaluated or ignored.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from core.panel import Panel


@dataclass
class RegimeLabels:
    """Dataclass holding aligned regime classification Series for a Panel."""
    trend: pd.Series          # values: 'bull', 'bear', or NaN
    volatility: pd.Series     # values: 'high_vol', 'low_vol', or NaN
    combined: pd.Series       # values: 'bull_high_vol', 'bear_low_vol', etc., or NaN


def classify_regimes(
    panel: Panel,
    lookback_days: int = 60,
) -> RegimeLabels:
    """
    Classifies market trend and volatility regimes for each date in a Panel.

    Args:
        panel: Market data Panel.
        lookback_days: Trailing window in trading days for trend and realized volatility (default 60).

    Returns:
        RegimeLabels: Dataclass containing trend, volatility, and combined per-date regime Series.

    Raises:
        ValueError: If panel is empty or lookback_days is non-positive.
    """
    if lookback_days <= 0:
        raise ValueError(f"lookback_days must be > 0, got {lookback_days}.")

    returns = panel.returns
    if returns.empty or len(returns) == 0:
        raise ValueError("Panel returns DataFrame is empty.")

    # Equal-weighted universe market return (MKT)
    mkt_returns = returns.mean(axis=1)

    # 1. Trend Regime: trailing lookback_days cumulative return
    cum_returns = (1.0 + mkt_returns.fillna(0.0)).cumprod()
    trailing_mkt_ret = cum_returns / cum_returns.shift(lookback_days) - 1.0

    # First (lookback_days) dates do not have a full trailing window
    trend_labels = pd.Series(index=mkt_returns.index, dtype=object)
    valid_trend_mask = trailing_mkt_ret.notna()

    trend_labels.loc[valid_trend_mask & (trailing_mkt_ret >= 0.0)] = "bull"
    trend_labels.loc[valid_trend_mask & (trailing_mkt_ret < 0.0)] = "bear"

    # 2. Volatility Regime: trailing lookback_days realized volatility
    realized_vol = mkt_returns.rolling(window=lookback_days, min_periods=lookback_days).std()

    vol_labels = pd.Series(index=mkt_returns.index, dtype=object)
    valid_vol = realized_vol.dropna()

    if not valid_vol.empty:
        vol_median = float(valid_vol.median())
        vol_labels.loc[realized_vol >= vol_median] = "high_vol"
        vol_labels.loc[realized_vol < vol_median] = "low_vol"

    # 3. Combined Regime Label
    combined_labels = pd.Series(index=mkt_returns.index, dtype=object)
    valid_combined_mask = trend_labels.notna() & vol_labels.notna()
    combined_labels.loc[valid_combined_mask] = (
        trend_labels.loc[valid_combined_mask] + "_" + vol_labels.loc[valid_combined_mask]
    )

    return RegimeLabels(
        trend=trend_labels,
        volatility=vol_labels,
        combined=combined_labels,
    )
