"""
Self-constructed factor model (MKT, MOM, VOL, SECTOR) from yfinance Panel data.

Why Factors Are Self-Constructed:
    Per the project's locked data policy, no external data sources (e.g. Fama-French, SPY, fundamental databases)
    are fetched. All factors are constructed from the existing 60-ticker yfinance Panel.

Factors Implemented & Rationale:
    1. MKT (Market Factor): Equal-weighted daily mean return across all 60 tickers in the panel.
       Proxies overall market movement without needing external index benchmark data.
    2. MOM (Momentum Factor): Long-short tercile return spread based on trailing 12-month minus 1-month
       returns, rebalanced monthly.
    3. VOL (Low-Volatility Factor): Long-short tercile return spread based on trailing 60-day realized
       volatility (long low-vol, short high-vol), rebalanced monthly.
    4. SECTOR (Sector Factors): Equal-weighted daily return series for each GICS sector present in SECTOR_MAP.

Why Value Factor Is NOT Implemented:
    No fundamental financial data (e.g. Price-to-Book, Price-to-Earnings, Balance Sheet metrics) exists
    in this codebase or yfinance pipeline. Fetching alternative fundamental datasets is explicitly deferred
    per the project specification's Future Extensions. This is a stated limitation, not a silent gap.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from core.panel import Panel
from .sector_map import SECTOR_MAP


@dataclass
class FactorPanel:
    mkt: pd.Series
    mom: pd.Series
    vol: pd.Series
    sector_returns: pd.DataFrame
    dates: pd.DatetimeIndex

    def to_dataframe(self) -> pd.DataFrame:
        """Combines MKT, MOM, VOL, and sector returns into a single DataFrame."""
        df = pd.DataFrame(
            {
                "MKT": self.mkt,
                "MOM": self.mom,
                "VOL": self.vol,
            },
            index=self.dates,
        )
        for col in self.sector_returns.columns:
            df[f"SECTOR_{col}"] = self.sector_returns[col]
        return df


def construct_market_factor(panel: Panel) -> pd.Series:
    """
    Constructs equal-weighted market factor (MKT) as daily mean return across all universe tickers.

    Args:
        panel: Market data Panel.

    Returns:
        pd.Series: Daily market factor returns.
    """
    mkt = panel.returns.mean(axis=1)
    mkt.name = "MKT"
    return mkt


def construct_momentum_factor(
    panel: Panel, formation_days: int = 252, skip_days: int = 21
) -> pd.Series:
    """
    Constructs long-short momentum factor (MOM) rebalanced monthly (vectorized).

    Strategy:
        Top tercile minus bottom tercile ranked by trailing formation_days - skip_days returns.
        Rebalanced at month-ends with zero lookahead (holdings set at close t apply to t+1).

    Args:
        panel: Market data Panel.
        formation_days: Trailing window for momentum (default 252 days = 12M).
        skip_days: Most recent window to skip (default 21 days = 1M).

    Returns:
        pd.Series: Daily momentum factor returns.
    """
    prices = panel.prices
    returns = panel.returns
    dates = returns.index

    ret_12m = prices.shift(skip_days) / prices.shift(formation_days) - 1.0

    # Identify month-end dates
    dates_ser = pd.Series(dates, index=dates)
    is_month_end = dates_ser.dt.to_period("M") != dates_ser.dt.to_period("M").shift(-1)
    month_end_dates = dates[is_month_end].tolist()

    weights_df = pd.DataFrame(np.nan, index=dates, columns=panel.universe)

    for me_date in month_end_dates:
        signal_row = ret_12m.loc[me_date].dropna()
        if len(signal_row) >= 6:
            q_low = signal_row.quantile(1.0 / 3.0)
            q_high = signal_row.quantile(2.0 / 3.0)

            longs = signal_row[signal_row >= q_high].index
            shorts = signal_row[signal_row <= q_low].index

            w = pd.Series(0.0, index=panel.universe)
            if len(longs) > 0:
                w[longs] = 1.0 / len(longs)
            if len(shorts) > 0:
                w[shorts] = -1.0 / len(shorts)
            weights_df.loc[me_date] = w

    # Forward-fill month-end weights across intervening days
    weights_filled = weights_df.ffill().fillna(0.0)

    # Zero-lookahead shift: weights determined at close t apply to daily return at t+1
    weights_applied = weights_filled.shift(1).fillna(0.0)

    # Daily factor returns
    mom_returns = (weights_applied * returns.fillna(0.0)).sum(axis=1)
    mom_returns.name = "MOM"
    return mom_returns


def construct_volatility_factor(
    panel: Panel, lookback_days: int = 60
) -> pd.Series:
    """
    Constructs low-volatility factor (VOL) rebalanced monthly (vectorized).

    Strategy:
        Long bottom tercile (lowest realized vol) minus short top tercile (highest realized vol)
        ranked by trailing 60-day realized volatility.

    Args:
        panel: Market data Panel.
        lookback_days: Trailing window for realized volatility (default 60 days).

    Returns:
        pd.Series: Daily low-volatility factor returns.
    """
    returns = panel.returns
    dates = returns.index

    realized_vol = returns.rolling(window=lookback_days, min_periods=20).std()

    dates_ser = pd.Series(dates, index=dates)
    is_month_end = dates_ser.dt.to_period("M") != dates_ser.dt.to_period("M").shift(-1)
    month_end_dates = dates[is_month_end].tolist()

    weights_df = pd.DataFrame(np.nan, index=dates, columns=panel.universe)

    for me_date in month_end_dates:
        vol_row = realized_vol.loc[me_date].dropna()
        if len(vol_row) >= 6:
            q_low = vol_row.quantile(1.0 / 3.0)
            q_high = vol_row.quantile(2.0 / 3.0)

            # Long low-volatility (bottom tercile), short high-volatility (top tercile)
            longs = vol_row[vol_row <= q_low].index
            shorts = vol_row[vol_row >= q_high].index

            w = pd.Series(0.0, index=panel.universe)
            if len(longs) > 0:
                w[longs] = 1.0 / len(longs)
            if len(shorts) > 0:
                w[shorts] = -1.0 / len(shorts)
            weights_df.loc[me_date] = w

    weights_filled = weights_df.ffill().fillna(0.0)
    weights_applied = weights_filled.shift(1).fillna(0.0)

    vol_returns = (weights_applied * returns.fillna(0.0)).sum(axis=1)
    vol_returns.name = "VOL"
    return vol_returns


def construct_sector_returns(
    panel: Panel, sector_map: Dict[str, str] = SECTOR_MAP
) -> pd.DataFrame:
    """
    Constructs equal-weighted daily return series for each sector in sector_map.

    Args:
        panel: Market data Panel.
        sector_map: Ticker-to-sector mapping.

    Returns:
        pd.DataFrame: Daily sector returns (columns = sector names).
    """
    returns = panel.returns
    unique_sectors = sorted(list(set(sector_map.values())))

    sector_df = pd.DataFrame(index=returns.index)
    for sec in unique_sectors:
        sec_tickers = [t for t in panel.universe if sector_map.get(t) == sec]
        if sec_tickers:
            sector_df[sec] = returns[sec_tickers].mean(axis=1)
        else:
            sector_df[sec] = 0.0

    return sector_df


def build_factor_panel(
    panel: Panel, sector_map: Dict[str, str] = SECTOR_MAP
) -> FactorPanel:
    """
    Orchestrates factor construction for MKT, MOM, VOL, and SECTOR factors.

    Args:
        panel: Market data Panel.
        sector_map: Ticker-to-sector mapping.

    Returns:
        FactorPanel: Dataclass containing all factor return series.
    """
    mkt = construct_market_factor(panel)
    mom = construct_momentum_factor(panel)
    vol = construct_volatility_factor(panel)
    sector_rets = construct_sector_returns(panel, sector_map=sector_map)

    return FactorPanel(
        mkt=mkt,
        mom=mom,
        vol=vol,
        sector_returns=sector_rets,
        dates=panel.returns.index,
    )
