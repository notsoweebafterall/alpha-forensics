"""
Shared helper for computing return-distribution statistics needed by
Probabilistic/Deflated Sharpe Ratio (PSR/DSR): skewness, excess-adjusted
kurtosis, and track record length (observation count).

Centralized here so every real log_trial() call site (Phase 7, 8, 9, 11)
computes these the same way, instead of each demo script reimplementing
its own scipy call with potentially different bias/ddof conventions.
"""

from typing import Optional, Tuple
import pandas as pd
from scipy import stats as scipy_stats


def compute_distribution_stats(
    returns: Optional[pd.Series],
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Computes (skew, kurtosis, track_record_length) from a returns series.

    Args:
        returns: A pandas Series of per-period returns (e.g. OOS gross or net
            returns). May be None or too short to compute distributional stats.

    Returns:
        Tuple of (skew, kurtosis, track_record_length). All three are None if
        returns is None or has fewer than 2 observations -- skew/kurtosis are
        undefined (or numerically unstable) below that, and it's more honest
        to store "unknown" than a garbage number.
    """
    if returns is None or len(returns) < 2:
        return None, None, None

    skew = float(scipy_stats.skew(returns, bias=False))
    kurtosis = float(scipy_stats.kurtosis(returns, fisher=False, bias=False))
    track_record_length = int(len(returns))
    return skew, kurtosis, track_record_length