"""
Parameter landscape evaluation and stability classification for alpha candidates.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional, Union
import pandas as pd

from core.panel import Panel
from alpha.expressions.tree import Expression
from alpha.generator.variant_expansion import expand_variants
from alpha.strategies.identity import build_candidate_id
from validation.folds import WalkForwardConfig
from validation.runner import run_walk_forward_validation
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from statistics import TrialRecord, load_trial_log, log_trial, compute_distribution_stats, save_returns

from .stability_metrics import classify_stability


@dataclass
class ParameterVariantResult:
    candidate_id: str
    params: Dict[str, Any]
    oos_sharpe: float
    oos_ic_mean: float
    fold_sharpe_mean: float
    reused_from_ledger: bool = False


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
    log_path: Optional[Union[str, Path]] = None,
    store_path: Optional[Union[str, Path]] = None,
    default_params: Optional[Dict[str, Any]] = None,
) -> ParameterLandscapeResult:
    """
    Evaluates an alpha candidate across its pre-specified parameter landscape.

    Single-Look OOS Discipline:
        Parameter variation modifies the signal generator itself. Each unique parameter variant
        receives its own distinct, single guarded OOS look via run_walk_forward_validation.

    Safety Valve:
        Enforces max_variants limit (default 25) to prevent runaway OOS evaluations.

    Persistent Trial Ledger (opt-in):
        If log_path is provided, each variant's candidate_id and oos_sharpe are checked
        against the ledger first. A variant already present is reused (its recorded
        oos_sharpe is used as-is, and it is NOT re-evaluated) rather than logged twice --
        the ledger is append-only and immutable per candidate_id. A variant not yet
        present is evaluated fresh and then logged. When log_path is None (the default),
        no ledger interaction happens at all -- this keeps unit tests (which use
        throwaway candidate_ids like "test_mom_grid") from ever touching the real,
        persistent ledger.

        NOTE: TrialRecord only persists oos_sharpe, not oos_ic_mean or
        fold_sharpe_mean. A reused variant will therefore have those two fields
        set to 0.0 with reused_from_ledger=True on its ParameterVariantResult,
        so callers can tell the difference between "genuinely near-zero" and
        "not available because this was reused from the ledger."

    Args:
        base_candidate_id: Parent candidate identifier string.
        build_fn: Strategy build function (params dict -> Expression).
        param_grid: Parameter grid dictionary.
        panel: Market data Panel.
        wf_config: WalkForwardConfig.
        backtest_config: BacktestConfig.
        cost_model: CostModel.
        max_variants: Maximum allowed unique parameter variants (default 25).
        log_path: Optional path to a persistent trial_log.jsonl ledger. See above.
        default_params: The strategy's default parameter values, if known. When a
            variant's param_values exactly equal default_params, its candidate_id is
            built WITHOUT a "__param_..." suffix (just base_candidate_id) -- the same
            convention other callers (e.g. Phase 10) use for a strategy's default-param
            candidate. Without this, the identical expression would get two different
            candidate_id strings depending on which script touched it first, and would
            be logged to the ledger twice -- silently inflating N for DSR purposes.

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

    existing_ledger_ids: Dict[str, TrialRecord] = {}
    if log_path is not None:
        existing_ledger_ids = {r.candidate_id: r for r in load_trial_log(log_path)}

    variant_results: List[ParameterVariantResult] = []

    # 3. Evaluate each deduplicated parameter variant
    for candidate in expanded:
        if default_params is not None and candidate.param_values == default_params:
            variant_id = build_candidate_id(base_candidate_id)
        else:
            variant_id = build_candidate_id(base_candidate_id, params=candidate.param_values)

        if log_path is not None and variant_id in existing_ledger_ids:
            rec = existing_ledger_ids[variant_id]
            variant_results.append(ParameterVariantResult(
                candidate_id=variant_id,
                params=candidate.param_values,
                oos_sharpe=rec.oos_sharpe,
                oos_ic_mean=0.0,
                fold_sharpe_mean=0.0,
                reused_from_ledger=True,
            ))
            continue

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
        # Use net returns (same series as oos_sharpe) for distribution stats and storage
        oos_net_returns = val_res.oos_net_returns
        skew, kurtosis, t_len = compute_distribution_stats(oos_net_returns)

        if log_path is not None:
            log_trial(
                TrialRecord(
                    candidate_id=variant_id,
                    phase="Phase 9 Parameter Landscape",
                    oos_sharpe=oos_sharpe,
                    timestamp=pd.Timestamp.now().isoformat(),
                    skew=skew,
                    kurtosis=kurtosis,
                    track_record_length=t_len,
                ),
                log_path=log_path,
            )
            if oos_net_returns is not None:
                target_store_path = store_path if store_path is not None else Path(log_path).parent / "trial_returns.parquet"
                try:
                    save_returns(variant_id, oos_net_returns, store_path=target_store_path)
                except ValueError:
                    pass  # Already stored; immutability discipline

        vr = ParameterVariantResult(
            candidate_id=variant_id,
            params=candidate.param_values,
            oos_sharpe=oos_sharpe,
            oos_ic_mean=oos_ic,
            fold_sharpe_mean=fold_sharpe,
            reused_from_ledger=False,
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