"""
Performance and predictive metrics for backtest evaluation.
"""

from typing import Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core.panel import Panel
from core.signal import Signal
from .engine import BacktestResult, align_forward_returns


def economic_metrics(
    result: BacktestResult, annualization_factor: int = 252
) -> dict:
    """
    Computes summary economic performance metrics from a BacktestResult.

    Metrics:
        - annualized_return: Arithmetic mean daily return * annualization_factor
        - annualized_volatility: Daily return std * sqrt(annualization_factor)
        - sharpe_ratio: annualized_return / annualized_volatility
        - sortino_ratio: annualized_return / downside_volatility
        - max_drawdown: Maximum peak-to-trough decline (negative float)
        - calmar_ratio: annualized_return / abs(max_drawdown)
        - average_turnover: Daily mean turnover

    Args:
        result: BacktestResult object.
        annualization_factor: Trading days per year (default 252).

    Returns:
        dict: Performance metrics dictionary.
    """
    net_rets = result.net_returns

    if net_rets.empty or net_rets.isna().all():
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "average_turnover": 0.0,
        }

    mean_ret = net_rets.mean()
    std_ret = net_rets.std()

    ann_ret = mean_ret * annualization_factor
    ann_vol = std_ret * np.sqrt(annualization_factor) if std_ret > 0 else 0.0

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Downside volatility (using negative returns)
    neg_rets = net_rets[net_rets < 0]
    if len(neg_rets) > 1 and neg_rets.std() > 0:
        downside_vol = neg_rets.std() * np.sqrt(annualization_factor)
        sortino = ann_ret / downside_vol
    else:
        sortino = 0.0

    # Max Drawdown computation
    cum_rets = (1.0 + net_rets).cumprod()
    peaks = cum_rets.cummax()
    drawdowns = (cum_rets - peaks) / peaks
    max_dd = drawdowns.min() if not drawdowns.empty else 0.0

    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0
    avg_turnover = result.turnover.mean() if not result.turnover.empty else 0.0

    return {
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "average_turnover": float(avg_turnover),
    }


def predictive_metrics(
    signal: Signal,
    panel_or_returns: Union[Panel, pd.DataFrame],
    execution_lag_days: int = 1,
) -> dict:
    """
    Computes cross-sectional Information Coefficient (IC) and Rank IC metrics.

    Alignment Discipline:
        - Aligns forward returns using `align_forward_returns(returns, execution_lag_days)`
          so IC measures predictiveness over the exact horizon traded by the backtesting engine.

    Args:
        signal: Signal DataFrame (date x ticker).
        panel_or_returns: Panel object or asset returns DataFrame.
        execution_lag_days: Execution lag in days (default 1).

    Returns:
        dict: Containing 'ic_mean', 'ic_std', 'rank_ic_mean', 'rank_ic_std',
              'ic_series', and 'rank_ic_series'.
    """
    if isinstance(panel_or_returns, Panel):
        returns = panel_or_returns.returns
    elif isinstance(panel_or_returns, pd.DataFrame):
        returns = panel_or_returns
    else:
        raise TypeError(f"Expected Panel or DataFrame, got {type(panel_or_returns)}")

    # Align forward returns to signal dates
    fwd_returns = align_forward_returns(returns, execution_lag_days=execution_lag_days)

    common_dates = signal.index.intersection(fwd_returns.index)
    ic_dict = {}
    rank_ic_dict = {}

    for dt in common_dates:
        sig_row = signal.loc[dt]
        ret_row = fwd_returns.loc[dt]

        # Valid mask: non-NaN in both signal and forward return
        valid_mask = sig_row.notna() & ret_row.notna()
        if valid_mask.sum() > 1:
            s_valid = sig_row[valid_mask]
            r_valid = ret_row[valid_mask]

            # Pearson IC
            if s_valid.std() > 0 and r_valid.std() > 0:
                p_corr = np.corrcoef(s_valid, r_valid)[0, 1]
                ic_dict[dt] = p_corr
            else:
                ic_dict[dt] = np.nan

            # Spearman Rank IC
            s_corr, _ = spearmanr(s_valid, r_valid)
            rank_ic_dict[dt] = s_corr
        else:
            ic_dict[dt] = np.nan
            rank_ic_dict[dt] = np.nan

    ic_series = pd.Series(ic_dict, name="ic").dropna()
    rank_ic_series = pd.Series(rank_ic_dict, name="rank_ic").dropna()

    return {
        "ic_mean": float(ic_series.mean()) if not ic_series.empty else 0.0,
        "ic_std": float(ic_series.std()) if not ic_series.empty else 0.0,
        "rank_ic_mean": float(rank_ic_series.mean()) if not rank_ic_series.empty else 0.0,
        "rank_ic_std": float(rank_ic_series.std()) if not rank_ic_series.empty else 0.0,
        "ic_series": ic_series,
        "rank_ic_series": rank_ic_series,
    }
