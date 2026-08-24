"""
Statistical significance tests for alpha candidate evaluation.

Provides Probabilistic Sharpe Ratio (PSR) against fixed benchmarks and variance
computation across trial Sharpes.
"""

import math
from typing import List
import numpy as np
import scipy.stats as stats


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    track_record_length: int = 252,
) -> float:
    """
    Computes the Probabilistic Sharpe Ratio (PSR) against a fixed benchmark Sharpe ratio.

    PSR measures the probability that the true Sharpe ratio exceeds benchmark_sharpe,
    adjusting for non-normality (skewness and kurtosis) and sample size (T).
    Unlike DSR, PSR is independent of the number of evaluated candidate trials N.

    Formula:
        PSR = Phi( (SR_hat - SR_benchmark) * sqrt(T - 1) / sqrt(1 - gamma_3*SR_hat + (gamma_4 - 1)/4 * SR_hat^2) )

    Args:
        observed_sharpe: Observed Sharpe ratio of candidate (SR_hat).
        benchmark_sharpe: Target benchmark Sharpe ratio (default 0.0).
        skew: Return skewness (gamma_3). Default 0.0.
        kurtosis: Return non-excess kurtosis (gamma_4, normal = 3.0). Default 3.0.
        track_record_length: Number of return observations T. Default 252.

    Returns:
        float: Probabilistic Sharpe Ratio (probability between 0.0 and 1.0).
    """
    if track_record_length <= 1:
        raise ValueError(f"track_record_length T must be > 1, got {track_record_length}")

    gamma_4 = kurtosis + 3.0 if kurtosis < 1.0 else kurtosis
    gamma_3 = skew

    denom_sq = 1.0 - gamma_3 * observed_sharpe + ((gamma_4 - 1.0) / 4.0) * (observed_sharpe ** 2)
    if denom_sq <= 0.0:
        denom_sq = 1e-8

    se = math.sqrt(denom_sq / (track_record_length - 1))
    z = (observed_sharpe - benchmark_sharpe) / se

    return float(stats.norm.cdf(z))


def sharpe_variance_across_trials(sharpes: List[float]) -> float:
    """
    Computes the sample variance of Sharpe ratios across all evaluated trials (V[SR_hat]).

    Args:
        sharpes: List of Sharpe ratios from candidate trials.

    Returns:
        float: Sample variance of Sharpe ratios. Returns 0.0 if fewer than 2 Sharpes provided.
    """
    if len(sharpes) < 2:
        return 0.0
    return float(np.var(sharpes, ddof=1))
