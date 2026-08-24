"""
Parameter landscape evaluation and stability classification for alpha candidates.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Any
import pandas as pd

from core.panel import Panel
from alpha.expressions.tree import Expression
from alpha.generator.variant_expansion import expand_variants
from validation.folds import WalkForwardConfig
from validation.runner import run_walk_forward_validation
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel

from .stability_metrics import classify_stability


@dataclass
class ParameterVariantResult:
    candidate_id: str
    params: Dict[str, Any]
    oos_sharpe: float
    oos_ic_mean: float
    fold_sharpe_mean: float


@dataclass
class ParameterLandscapeResult:
    base_candidate_id: str
    param_grid: Dict[str, List[Any]]
    variant_results: List[ParameterVariantResult]
    stability_label: str
    stability_rationale: str


def evaluate_parameter_landscape(
    base_candidate_id: str,
    build_fn: Callable[[Dict[str, Any]], Expression],
    param_grid: Dict[str, List[Any]],
    panel: Panel,
    wf_config: WalkForwardConfig = WalkForwardConfig(),
    backtest_config: BacktestConfig = BacktestConfig(),
    cost_model: CostModel = CostModel(),
    max_variants: int = 25,
) -> ParameterLandscapeResult:
    """
    Evaluates an alpha candidate across its pre-specified parameter landscape.

    Single-Look OOS Discipline:
        Parameter variation modifies the signal generator itself. Each unique parameter variant
        receives its own distinct, single guarded OOS look via run_walk_forward_validation.

    Safety Valve:
        Enforces max_variants limit (default 25) to prevent runaway OOS evaluations.

    Args:
        base_candidate_id: Parent candidate identifier string.
        build_fn: Strategy build function (params dict -> Expression).
        param_grid: Parameter grid dictionary.
        panel: Market data Panel.
        wf_config: WalkForwardConfig.
        backtest_config: BacktestConfig.
        cost_model: CostModel.
        max_variants: Maximum allowed unique parameter variants (default 25).

    Returns:
        ParameterLandscapeResult: Full landscape metrics and stability classification.
    """
    # 1. Expand parameter variants using Phase 6's expand_variants logic
    expanded = expand_variants(build_fn, param_grid, parent_family=base_candidate_id)

    # 2. Safety valve check
    if len(expanded) > max_variants:
        raise ValueError(
            f"Expanded parameter grid size ({len(expanded)}) exceeds max_variants limit ({max_variants}). "
            f"Please shrink the parameter grid bounds."
        )

    variant_results: List[ParameterVariantResult] = []

    # 3. Evaluate each deduplicated parameter variant
    for candidate in expanded:
        p_str = "_".join(f"{k}_{v}" for k, v in sorted(candidate.param_values.items()))
        variant_id = f"{base_candidate_id}__param_{p_str}" if p_str else base_candidate_id

        val_res = run_walk_forward_validation(
            candidate_id=variant_id,
            expression=candidate.expression,
            panel=panel,
            wf_config=wf_config,
            backtest_config=backtest_config,
            cost_model=cost_model,
        )

        oos_sharpe = float(val_res.oos_metrics.get("sharpe_ratio", 0.0))
        oos_ic = float(val_res.oos_metrics.get("ic_mean", 0.0))
        fold_sharpe = float(val_res.degradation.get("mean_train_sharpe", 0.0))

        vr = ParameterVariantResult(
            candidate_id=variant_id,
            params=candidate.param_values,
            oos_sharpe=oos_sharpe,
            oos_ic_mean=oos_ic,
            fold_sharpe_mean=fold_sharpe,
        )
        variant_results.append(vr)

    # 4. Classify stability across all variant OOS Sharpes
    oos_sharpes = [vr.oos_sharpe for vr in variant_results]
    label, rationale = classify_stability(oos_sharpes)

    return ParameterLandscapeResult(
        base_candidate_id=base_candidate_id,
        param_grid=param_grid,
        variant_results=variant_results,
        stability_label=label,
        stability_rationale=rationale,
    )
