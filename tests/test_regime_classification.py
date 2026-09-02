"""
Unit tests for pure market regime classification logic (Phase 12).
"""

import pytest
import numpy as np
import pandas as pd

from data.universe import UNIVERSE_60
from data.panel import build_panel
from regimes import classify_regimes, RegimeLabels


def test_regime_labels_partition_all_dates():
    """Verify that regime classification partitions all post-lookback dates with no gaps or overlaps."""
    dates = pd.date_range("2021-01-01", periods=200, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    np.random.seed(42)

    p_df = pd.DataFrame(100.0 + np.random.randn(200, 4).cumsum(axis=0), index=dates, columns=tickers)
    v_df = pd.DataFrame(1000, index=dates, columns=tickers)

    panel = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

    lookback = 30
    reg_labels = classify_regimes(panel, lookback_days=lookback)

    assert isinstance(reg_labels, RegimeLabels)
    assert len(reg_labels.trend) == len(dates)
    assert len(reg_labels.volatility) == len(dates)
    assert len(reg_labels.combined) == len(dates)

    # Post-lookback dates must all have valid trend, vol, and combined labels
    post_lookback_dates = dates[lookback:]
    assert reg_labels.trend.loc[post_lookback_dates].notna().all()
    assert reg_labels.volatility.loc[post_lookback_dates].notna().all()
    assert reg_labels.combined.loc[post_lookback_dates].notna().all()

    # Valid trend values must be either 'bull' or 'bear'
    trend_vals = set(reg_labels.trend.dropna().unique())
    assert trend_vals.issubset({"bull", "bear"})

    # Valid vol values must be either 'high_vol' or 'low_vol'
    vol_vals = set(reg_labels.volatility.dropna().unique())
    assert vol_vals.issubset({"high_vol", "low_vol"})

    # Combined labels must equal '{trend}_{vol}'
    for d in post_lookback_dates:
        t_lbl = reg_labels.trend.loc[d]
        v_lbl = reg_labels.volatility.loc[d]
        c_lbl = reg_labels.combined.loc[d]
        assert c_lbl == f"{t_lbl}_{v_lbl}"


def test_regime_labels_known_trend_case():
    """Verify known rising vs falling prices yield 'bull' and 'bear' labels as expected."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    tickers = ["AAPL", "MSFT"]

    # First 50 days: prices steadily increase (+1% per day) -> Bull
    # Next 50 days: prices steadily decrease (-1% per day) -> Bear
    prices_aapl = np.concatenate([
        100.0 * (1.01 ** np.arange(50)),
        100.0 * (1.01 ** 49) * (0.99 ** np.arange(1, 51))
    ])
    prices_msft = np.concatenate([
        200.0 * (1.01 ** np.arange(50)),
        200.0 * (1.01 ** 49) * (0.99 ** np.arange(1, 51))
    ])

    p_df = pd.DataFrame({"AAPL": prices_aapl, "MSFT": prices_msft}, index=dates)
    v_df = pd.DataFrame(1000, index=dates, columns=tickers)

    panel = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

    lookback = 20
    reg_labels = classify_regimes(panel, lookback_days=lookback)

    # Day 25 (inside first half, after lookback) must be 'bull'
    assert reg_labels.trend.iloc[25] == "bull"

    # Day 85 (inside second half, after lookback) must be 'bear'
    assert reg_labels.trend.iloc[85] == "bear"


def test_regime_lookback_nan_handling():
    """Verify that initial dates prior to lookback_days have NaN regime labels."""
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    tickers = ["AAPL", "MSFT"]
    p_df = pd.DataFrame(100.0, index=dates, columns=tickers)
    v_df = pd.DataFrame(1000, index=dates, columns=tickers)

    panel = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

    lookback = 15
    reg_labels = classify_regimes(panel, lookback_days=lookback)

    # First lookback-1 dates should have NaN
    assert reg_labels.trend.iloc[:lookback - 1].isna().all()
    assert reg_labels.volatility.iloc[:lookback - 1].isna().all()
    assert reg_labels.combined.iloc[:lookback - 1].isna().all()
