"""
Transaction cost sensitivity analysis for alpha candidates.

Single-Look OOS Discipline & Methodological Note:
    To strictly preserve the out-of-sample holdout discipline, `run_cost_sensitivity`
    executes `run_walk_forward_validation` EXACTLY ONCE at cost_bps=0.0 to capture the raw
    OOS gross returns and turnover series.

    Performance across the entire cost grid [0, 5, 10, 25, 50] bps is derived via pure linear
    arithmetic post-processing:
        net_returns = oos_gross_returns - oos_turnover * (cost_bps / 10000.0)

    This guarantees that the OOS holdout is touched only once per candidate regardless of how
    many transaction cost scenarios are evaluated.

    Breakeven Bps Approximation Disclaimer:
        `breakeven_cost_bps` is an annualized-return breakeven approximation, computed by searching
        where derived net annualized return crosses 0.0. It is an approximation because Sharpe ratio
        volatility also shifts slightly with transaction costs; this calculation isolates the
        first-order linear return impact.

    Liquidity Constraints Note:
        Position limits, turnover caps, and non-linear market impact modeling are explicitly out of
        scope for this phase and documented for future extension.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from core.panel import Panel
from alpha.expressions.tree import Expression
from validation.folds import WalkForwardConfig
from .config import BacktestConfig
from .costs import CostModel

STANDARD_COST_GRID: List[float] = [0.0, 5.0, 10.0, 25.0, 50.0]
REALISTIC_COST_BPS: float = 10.0


@dataclass
class CostSensitivityResult:
    candidate_id: str
    cost_bps_grid: List[float]
    oos_sharpe_by_cost: Dict[float, float]
    oos_annualized_return_by_cost: Dict[float, float]
    breakeven_cost_bps: Optional[float]
    survives_realistic_costs: bool


def run_cost_sensitivity(
    candidate_id: str,
    expression: Expression,
    panel: Panel,
    wf_config: WalkForwardConfig = WalkForwardConfig(),
    backtest_config: BacktestConfig = BacktestConfig(),
    cost_grid: List[float] = STANDARD_COST_GRID,
    oos_fraction: float = 0.15,
) -> CostSensitivityResult:
    """
    Executes transaction cost sensitivity analysis on a candidate expression.

    Args:
        candidate_id: Unique candidate identifier string.
        expression: Expression tree.
        panel: Full Panel dataset.
        wf_config: WalkForwardConfig.
        backtest_config: BacktestConfig.
        cost_grid: List of cost levels in basis points (default [0, 5, 10, 25, 50]).
        oos_fraction: OOS holdout fraction.

    Returns:
        CostSensitivityResult: Sensitivity metrics across cost grid.
    """
    from validation.runner import run_walk_forward_validation

    # Single guarded OOS look at 0 bps
    base_candidate_id = f"{candidate_id}__cost_sweep_base"
    zero_cost_model = CostModel(cost_bps=0.0)

    val_res = run_walk_forward_validation(
        candidate_id=base_candidate_id,
        expression=expression,
        panel=panel,
        wf_config=wf_config,
        backtest_config=backtest_config,
        cost_model=zero_cost_model,
        oos_fraction=oos_fraction,
    )

    oos_gross = val_res.oos_gross_returns
    oos_turnover = val_res.oos_turnover

    if oos_gross is None or oos_turnover is None:
        raise RuntimeError("ValidationResult did not return raw oos_gross_returns or oos_turnover.")

    oos_sharpe_by_cost: Dict[float, float] = {}
    oos_ann_ret_by_cost: Dict[float, float] = {}

    # Derivation across cost grid via pure series arithmetic
    for bps in cost_grid:
        cost_drag = oos_turnover * (bps / 10000.0)
        net_rets = oos_gross - cost_drag

        mean_ret = float(net_rets.mean())
        std_ret = float(net_rets.std())

        ann_ret = mean_ret * 252.0
        ann_vol = std_ret * np.sqrt(252.0) if std_ret > 0 else 0.0
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        oos_sharpe_by_cost[bps] = float(sharpe)
        oos_ann_ret_by_cost[bps] = float(ann_ret)

    # Compute breakeven cost bps (annualized return 0-crossing approximation)
    breakeven: Optional[float] = None
    prev_bps = 0.0
    prev_ann_ret = float((oos_gross - oos_turnover * 0.0).mean() * 252.0)

    if prev_ann_ret <= 0.0:
        breakeven = 0.0
    else:
        # Fine-grained search up to 100 bps
        for step_bps in range(1, 101):
            cur_bps = float(step_bps)
            cur_ann_ret = float((oos_gross - oos_turnover * (cur_bps / 10000.0)).mean() * 252.0)

            if cur_ann_ret <= 0.0:
                slope = cur_ann_ret - prev_ann_ret
                if abs(slope) > 1e-12:
                    breakeven = float(prev_bps - (prev_ann_ret / slope))
                else:
                    breakeven = prev_bps
                break

            prev_bps = cur_bps
            prev_ann_ret = cur_ann_ret

    survives = oos_sharpe_by_cost.get(REALISTIC_COST_BPS, 0.0) > 0.0

    return CostSensitivityResult(
        candidate_id=candidate_id,
        cost_bps_grid=cost_grid,
        oos_sharpe_by_cost=oos_sharpe_by_cost,
        oos_annualized_return_by_cost=oos_ann_ret_by_cost,
        breakeven_cost_bps=breakeven,
        survives_realistic_costs=survives,
    )
