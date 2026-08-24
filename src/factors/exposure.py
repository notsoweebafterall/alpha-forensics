"""
Factor exposure regression analysis with HAC-robust (Newey-West) standard errors.

Quantifies candidate alpha exposure to MKT, MOM, VOL, and candidate-specific sector factors.
Classifies strategies into a three-way alpha verdict:
  - 'distinct_alpha' (if raw_sharpe > 0, R^2 < 0.5, and residual_sharpe >= 0.5 * raw_sharpe)
  - 'factor_repackaging' (if raw_sharpe > 0, but R^2 >= 0.5 or residual_sharpe < 0.5 * raw_sharpe)
  - 'unprofitable' (if raw_sharpe <= 0)
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .factor_construction import FactorPanel


@dataclass
class FactorExposureResult:
    candidate_id: str
    betas: Dict[str, float]
    t_stats: Dict[str, float]
    r_squared: float
    residual_sharpe: float
    raw_sharpe: float
    factor_correlations: Dict[str, float]
    verdict: str  # "distinct_alpha", "factor_repackaging", or "unprofitable"


def run_factor_regression(
    candidate_returns: pd.Series,
    factor_panel: FactorPanel,
    candidate_sectors: List[str],
    candidate_id: str = "candidate",
    nw_lags: int = 5,
) -> FactorExposureResult:
    """
    Runs OLS factor exposure regression with Newey-West (HAC) standard errors.

    Design Matrix Formulation:
        Includes Constant (Intercept), MKT, MOM, VOL, and ONLY the sector return series
        corresponding to candidate_sectors to avoid degenerate design matrix collinearity
        (since all 10 sector returns sum back toward the market factor).

    Fail-Loud Guard:
        Raises ValueError if candidate_sectors is None or empty.

    Verdict Classification:
        - "unprofitable": If raw_sharpe <= 0.0 (no positive alpha to decompose).
        - "distinct_alpha": If raw_sharpe > 0.0, R^2 < 0.5, and residual_sharpe >= 0.5 * raw_sharpe.
        - "factor_repackaging": If raw_sharpe > 0.0, but R^2 >= 0.5 or residual_sharpe < 0.5 * raw_sharpe.

    Args:
        candidate_returns: Daily returns of candidate strategy.
        factor_panel: FactorPanel object containing MKT, MOM, VOL, and sector returns.
        candidate_sectors: Non-empty list of sector names relevant to the candidate's traded names.
        candidate_id: Unique candidate identifier string.
        nw_lags: Number of lags for Newey-West HAC covariance matrix (default 5).

    Returns:
        FactorExposureResult: OLS betas, HAC t-stats, R^2, residual Sharpe, raw Sharpe, and verdict.
    """
    if candidate_sectors is None or len(candidate_sectors) == 0:
        raise ValueError(
            "candidate_sectors must be explicitly provided as a non-empty list of sector names "
            "to avoid degenerate design matrix collinearity."
        )

    # 1. Align candidate returns and factor panel dates
    df_factors = factor_panel.to_dataframe()
    common_idx = candidate_returns.index.intersection(df_factors.index)

    if len(common_idx) < 30:
        raise ValueError(f"Insufficient overlapping date observations ({len(common_idx)}) for factor regression.")

    y_ser = candidate_returns.loc[common_idx].fillna(0.0)
    df_f = df_factors.loc[common_idx].fillna(0.0)

    # Build factor regression columns: MKT, MOM, VOL + selected sector columns
    cols_to_use = ["MKT", "MOM", "VOL"]
    for sec in candidate_sectors:
        sec_col = f"SECTOR_{sec}"
        if sec_col in df_f.columns:
            cols_to_use.append(sec_col)

    # 2. Compute simple factor correlations
    correlations: Dict[str, float] = {}
    for col in cols_to_use:
        corr_val = float(y_ser.corr(df_f[col]))
        correlations[col] = 0.0 if np.isnan(corr_val) else corr_val

    # 3. Construct Design Matrix X (including constant term).
    # Drop factor columns with near-zero variance to prevent rank deficiency
    # (e.g. MOM column all-zero when OOS window is shorter than the 252-day formation lookback).
    active_cols = []
    for col in cols_to_use:
        col_std = float(df_f[col].std())
        if col_std > 1e-8:
            active_cols.append(col)

    X_mat = np.column_stack([np.ones(len(common_idx)), df_f[active_cols].values])
    y_arr = y_ser.values
    N, K = X_mat.shape

    feature_names = ["Intercept"] + active_cols

    # 4. Fit OLS via numpy.linalg.lstsq
    betas_arr, residuals_sum, rank, s_vals = np.linalg.lstsq(X_mat, y_arr, rcond=None)
    residuals = y_arr - X_mat @ betas_arr

    # 5. Compute R^2
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # 6. Compute Newey-West (HAC) Variance-Covariance Matrix.
    # Use Moore-Penrose pseudoinverse (pinv) instead of inv to gracefully handle
    # rank-deficient design matrices (e.g. constant factor columns dropped above
    # still leave near-singular XtX when N is very small relative to K).
    xtx_inv = np.linalg.pinv(X_mat.T @ X_mat)

    # S_0 term
    S_hac = np.zeros((K, K))
    for t in range(N):
        x_t = X_mat[t, :, np.newaxis]
        e_t = residuals[t]
        S_hac += (e_t ** 2) * (x_t @ x_t.T)

    # Autocovariance terms with Bartlett weights
    for l in range(1, nw_lags + 1):
        w_l = 1.0 - (l / (nw_lags + 1.0))
        S_l = np.zeros((K, K))
        for t in range(l, N):
            x_t = X_mat[t, :, np.newaxis]
            x_tl = X_mat[t - l, :, np.newaxis]
            e_t = residuals[t]
            e_tl = residuals[t - l]
            S_l += e_t * e_tl * (x_t @ x_tl.T + x_tl @ x_t.T)
        S_hac += w_l * S_l

    var_hac = xtx_inv @ S_hac @ xtx_inv

    # 7. Compute t-statistics
    se_hac = np.sqrt(np.maximum(1e-12, np.diag(var_hac)))
    t_stats_arr = betas_arr / se_hac

    betas_dict = {name: float(betas_arr[i]) for i, name in enumerate(feature_names)}
    t_stats_dict = {name: float(t_stats_arr[i]) for i, name in enumerate(feature_names)}

    # 8. Compute residual Sharpe & raw Sharpe (annualized)
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals, ddof=1))
    residual_sharpe = (res_mean / res_std) * math.sqrt(252.0) if res_std > 1e-8 else 0.0

    raw_mean = float(np.mean(y_arr))
    raw_std = float(np.std(y_arr, ddof=1))
    raw_sharpe = (raw_mean / raw_std) * math.sqrt(252.0) if raw_std > 1e-8 else 0.0

    # 9. Evaluate Three-Way Alpha Verdict
    # Explicit Guard: If raw_sharpe <= 0, there is no positive alpha to be "distinct" from factor exposure.
    # The strategy never worked / is unprofitable.
    if raw_sharpe <= 0.0:
        verdict = "unprofitable"
    elif raw_sharpe > 0.0 and r_squared < 0.5 and (residual_sharpe >= 0.5 * raw_sharpe):
        verdict = "distinct_alpha"
    else:
        verdict = "factor_repackaging"

    return FactorExposureResult(
        candidate_id=candidate_id,
        betas=betas_dict,
        t_stats=t_stats_dict,
        r_squared=r_squared,
        residual_sharpe=residual_sharpe,
        raw_sharpe=raw_sharpe,
        factor_correlations=correlations,
        verdict=verdict,
    )
