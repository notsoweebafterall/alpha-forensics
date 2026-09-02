"""
Portfolio Construction Module — cross-candidate capital allocation & meta-portfolio weighting.

Single-Look Rationale:
    Combining already-logged OOS return series into a portfolio weighting is NOT a new single-look
    violation — each candidate's OOS data was already evaluated and logged exactly once during its
    original discovery/validation run. Computing a portfolio allocation from already-observed returns
    is a downstream synthesis, not a fresh query of held-out data.

Eligibility Pipeline Funnel:
    Stage 1: get_valid_non_redundant_candidates(db_path, top_level_only=True)
             (Filters out invalidated, redundant, and sub-slice candidates)
    Stage 2: Filter to candidates with stored daily net returns (has_returns_series == 1)
    Stage 3: Filter to candidates surviving DSR (dsr_verdict == 'survives_dsr') if require_dsr_survival=True

Weighting Discipline:
    1. equal_weight: Equal capital allocation 1/N across eligible constituents.
    2. inverse_volatility: Capital allocation proportional to 1 / std(returns).
    3. Strict Date Alignment: Return volatility and portfolio return series are computed strictly
       on the inner-joined common trading dates (aligned_matrix = matrix.dropna(how='any')). Both
       weights and realized portfolio returns use the exact same data window.
    4. Zero-Volatility Guard: Denominators are floored at 1e-8 to prevent division-by-zero or inf/NaN weights.
    5. Standard Annualized Sharpe: Sharpe ratio is annualized using mean / std * sqrt(252),
       matching Phase 7/8/9/12/13/14 convention throughout the codebase.
    6. Future Work Note: Mean-variance / Max-Sharpe optimization is deferred to future work due to
       singular covariance matrices and solver dependencies.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from registry import get_valid_non_redundant_candidates
from statistics.returns_store import load_returns_matrix


@dataclass
class PortfolioResult:
    """
    Result container for portfolio construction.
    """
    stage1_candidates: List[str]
    stage2_candidates: List[str]
    stage3_candidates: List[str]
    eligible_candidates: List[str]
    weighting_scheme: str
    weights: Dict[str, float] = field(default_factory=dict)
    portfolio_returns: Optional[pd.Series] = None
    portfolio_sharpe: Optional[float] = None
    message: str = ""


def build_portfolio(
    db_path: Union[str, Path] = "data/alpha_registry.db",
    returns_store_path: Union[str, Path] = "data/trial_returns.parquet",
    weighting: str = "equal_weight",
    require_dsr_survival: bool = True,
    min_overlap_days: int = 60,
) -> PortfolioResult:
    """
    Builds a meta-portfolio from eligible registry candidates.

    Args:
        db_path: Path to SQLite registry database.
        returns_store_path: Path to trial_returns.parquet.
        weighting: "equal_weight" or "inverse_volatility".
        require_dsr_survival: If True, filters candidates to dsr_verdict == 'survives_dsr'.
        min_overlap_days: Minimum required overlapping trading days for constituents (default 60).

    Returns:
        PortfolioResult: Dataclass containing stage filter counts, candidate weights,
                         and portfolio net return series/Sharpe.
    """
    if weighting not in ("equal_weight", "inverse_volatility"):
        raise ValueError(f"Unsupported weighting scheme '{weighting}'. Must be 'equal_weight' or 'inverse_volatility'.")

    db_path = Path(db_path)
    returns_store_path = Path(returns_store_path)

    # 1. Stage 1: Valid, non-redundant, top-level candidates
    try:
        df_stage1 = get_valid_non_redundant_candidates(db_path=db_path, top_level_only=True)
        stage1_ids = list(df_stage1["candidate_id"]) if not df_stage1.empty else []
    except Exception as e:
        return PortfolioResult(
            stage1_candidates=[],
            stage2_candidates=[],
            stage3_candidates=[],
            eligible_candidates=[],
            weighting_scheme=weighting,
            message=f"Error accessing registry database ({e}). Empty portfolio returned.",
        )

    if not stage1_ids:
        return PortfolioResult(
            stage1_candidates=[],
            stage2_candidates=[],
            stage3_candidates=[],
            eligible_candidates=[],
            weighting_scheme=weighting,
            message="Stage 1 filter (valid, non-redundant, top-level) yielded 0 candidates. Empty portfolio returned.",
        )

    # 2. Stage 2: Candidates with stored returns series (has_returns_series == 1)
    df_stage2 = df_stage1[df_stage1["has_returns_series"] == 1]
    stage2_ids = list(df_stage2["candidate_id"]) if not df_stage2.empty else []

    if not stage2_ids:
        return PortfolioResult(
            stage1_candidates=stage1_ids,
            stage2_candidates=[],
            stage3_candidates=[],
            eligible_candidates=[],
            weighting_scheme=weighting,
            message=f"Stage 2 filter (has_returns_series == 1) excluded all Stage 1 candidates. "
                    f"No stored returns series found in {returns_store_path.name}. Empty portfolio returned.",
        )

    # 3. Stage 3: Candidates surviving DSR (if require_dsr_survival=True)
    if require_dsr_survival:
        df_stage3 = df_stage2[df_stage2["dsr_verdict"] == "survives_dsr"]
        stage3_ids = list(df_stage3["candidate_id"]) if not df_stage3.empty else []
    else:
        stage3_ids = stage2_ids

    eligible_ids = stage3_ids

    if not eligible_ids:
        reason = "Stage 3 filter (dsr_verdict == 'survives_dsr')" if require_dsr_survival else "Candidate filtering"
        return PortfolioResult(
            stage1_candidates=stage1_ids,
            stage2_candidates=stage2_ids,
            stage3_candidates=stage3_ids,
            eligible_candidates=[],
            weighting_scheme=weighting,
            message=f"{reason} yielded 0 eligible candidates. Empty portfolio returned.",
        )

    # 4. Load returns matrix and perform inner-join date alignment
    returns_matrix = load_returns_matrix(eligible_ids, store_path=returns_store_path)
    if returns_matrix.empty:
        return PortfolioResult(
            stage1_candidates=stage1_ids,
            stage2_candidates=stage2_ids,
            stage3_candidates=stage3_ids,
            eligible_candidates=[],
            weighting_scheme=weighting,
            message=f"Could not load returns matrix from {returns_store_path.name}. Empty portfolio returned.",
        )

    # Filter to eligible IDs present in matrix
    present_ids = [cid for cid in eligible_ids if cid in returns_matrix.columns]
    if not present_ids:
        return PortfolioResult(
            stage1_candidates=stage1_ids,
            stage2_candidates=stage2_ids,
            stage3_candidates=stage3_ids,
            eligible_candidates=[],
            weighting_scheme=weighting,
            message="None of the eligible candidates were found in returns store matrix. Empty portfolio returned.",
        )

    # Inner-join date alignment: only dates where ALL eligible candidates have valid return data
    aligned_matrix = returns_matrix[present_ids].dropna(how="any")

    if len(aligned_matrix) < min_overlap_days:
        return PortfolioResult(
            stage1_candidates=stage1_ids,
            stage2_candidates=stage2_ids,
            stage3_candidates=stage3_ids,
            eligible_candidates=[],
            weighting_scheme=weighting,
            message=f"Insufficient overlapping trading dates ({len(aligned_matrix)} days < min_overlap_days {min_overlap_days}). Empty portfolio returned.",
        )

    n_eligible = len(present_ids)

    # 5. Compute weights strictly on aligned_matrix
    weights_dict: Dict[str, float] = {}

    if weighting == "equal_weight":
        eq_w = 1.0 / float(n_eligible)
        weights_dict = {cid: eq_w for cid in present_ids}

    elif weighting == "inverse_volatility":
        # Compute volatility strictly on aligned dates
        vols = aligned_matrix.std(axis=0)
        inv_vols: Dict[str, float] = {}

        for cid in present_ids:
            v_val = float(vols[cid])
            # Zero-volatility guard: floor denominator at 1e-8 to prevent division by zero or inf/NaN weights
            v_safe = max(v_val, 1e-8)
            inv_vols[cid] = 1.0 / v_safe

        sum_inv = sum(inv_vols.values())
        if sum_inv < 1e-12:
            # Fallback if total volatility sum is zero
            eq_w = 1.0 / float(n_eligible)
            weights_dict = {cid: eq_w for cid in present_ids}
        else:
            weights_dict = {cid: float(inv_vols[cid] / sum_inv) for cid in present_ids}

    # 6. Compute portfolio realized return series on aligned dates
    weights_series = pd.Series(weights_dict)[present_ids]
    portfolio_returns = (aligned_matrix[present_ids] * weights_series).sum(axis=1)
    portfolio_returns.name = f"portfolio_{weighting}"

    # 7. Compute annualized Sharpe ratio matching project standard (mean / std * sqrt(252))
    mean_r = float(portfolio_returns.mean())
    std_r = float(portfolio_returns.std())
    portfolio_sharpe = (mean_r / std_r * math.sqrt(252.0)) if std_r > 1e-8 else 0.0

    msg = f"Successfully constructed {weighting} portfolio across {n_eligible} constituents on {len(aligned_matrix)} aligned trading dates."

    return PortfolioResult(
        stage1_candidates=stage1_ids,
        stage2_candidates=stage2_ids,
        stage3_candidates=stage3_ids,
        eligible_candidates=present_ids,
        weighting_scheme=weighting,
        weights=weights_dict,
        portfolio_returns=portfolio_returns,
        portfolio_sharpe=portfolio_sharpe,
        message=msg,
    )
