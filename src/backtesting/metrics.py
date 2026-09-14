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
    if common_dates.empty:
        empty_ser = pd.Series(dtype=float)
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "rank_ic_mean": 0.0,
            "rank_ic_std": 0.0,
            "ic_series": empty_ser,
            "rank_ic_series": empty_ser,
        }

    sig = signal.loc[common_dates]
    ret = fwd_returns.loc[common_dates]

    # Valid mask for non-NaN in both signal and return
    valid_mask = sig.notna() & ret.notna()
    n_valid_per_row = valid_mask.sum(axis=1)
    sufficient_mask = n_valid_per_row > 1

    sig_valid = sig.where(valid_mask)
    ret_valid = ret.where(valid_mask)

    # 1. Vectorized Pearson IC
    sig_mean = sig_valid.mean(axis=1)
    ret_mean = ret_valid.mean(axis=1)

    sig_std = sig_valid.std(axis=1, ddof=1)
    ret_std = ret_valid.std(axis=1, ddof=1)

    valid_std_mask = sufficient_mask & (sig_std > 1e-8) & (ret_std > 1e-8)

    sig_demean = sig_valid.sub(sig_mean, axis=0)
    ret_demean = ret_valid.sub(ret_mean, axis=0)

    cov = (sig_demean * ret_demean).sum(axis=1) / (n_valid_per_row - 1).replace(0, np.nan)
    denom = sig_std * ret_std

    ic_series = (cov / denom).where(valid_std_mask).dropna()
    ic_series.name = "ic"

    # 2. Vectorized Spearman Rank IC
    sig_rank = sig_valid.rank(axis=1)
    ret_rank = ret_valid.rank(axis=1)

    sig_r_mean = sig_rank.mean(axis=1)
    ret_r_mean = ret_rank.mean(axis=1)

    sig_r_std = sig_rank.std(axis=1, ddof=1)
    ret_r_std = ret_rank.std(axis=1, ddof=1)

    valid_r_std_mask = sufficient_mask & (sig_r_std > 1e-8) & (ret_r_std > 1e-8)

    sig_r_demean = sig_rank.sub(sig_r_mean, axis=0)
    ret_r_demean = ret_rank.sub(ret_r_mean, axis=0)

    cov_rank = (sig_r_demean * ret_r_demean).sum(axis=1) / (n_valid_per_row - 1).replace(0, np.nan)
    denom_rank = sig_r_std * ret_r_std

    rank_ic_series = (cov_rank / denom_rank).where(valid_r_std_mask).dropna()
    rank_ic_series.name = "rank_ic"

    return {
        "ic_mean": float(ic_series.mean()) if not ic_series.empty else 0.0,
        "ic_std": float(ic_series.std()) if not ic_series.empty else 0.0,
        "rank_ic_mean": float(rank_ic_series.mean()) if not rank_ic_series.empty else 0.0,
        "rank_ic_std": float(rank_ic_series.std()) if not rank_ic_series.empty else 0.0,
        "ic_series": ic_series,
        "rank_ic_series": rank_ic_series,
    }
