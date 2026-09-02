"""
Regime robustness evaluation and persistent trial logging for alpha candidates.

Scope & Design Principles:
    Evaluates out-of-sample (OOS) strategy performance across market trend and volatility regimes.
    Slices the candidate's single, continuous OOS returns series by regime labels generated
    by `classify_regimes()`.

    Why Slicing OOS Returns Is Used (No Gapped Walk-Forward Validation):
        Regime dates (e.g. 'bull_high_vol', 'bear_low_vol') interleave non-contiguously throughout time.
        Re-running walk-forward validation on a gapped date sub-panel would break daily return
        calculations (prices.shift(1) across multi-day time gaps) and fail fold/embargo indexing logic.
        Instead, the candidate's single guarded OOS evaluation returns series is evaluated once, and
        sliced directly by date per regime label. This maintains daily return integrity and single-look
        trial discipline.

EXPLICIT CAVEAT (Tail OOS Window Coverage Limitation):
    Because the OOS holdout window sits chronologically at the end of the historical dataset by construction,
    regime robustness testing evaluates candidate performance ONLY across whichever market regimes actually
    occurred within that specific tail window (e.g., 2 out of 4 potential regime types). Regimes not present
    in the OOS tail window (or with fewer than `min_regime_days` observations) are explicitly recorded in
    `skipped_regimes` with explanatory reasons rather than silently dropped, preventing callers from
    mistaking partial regime coverage for full-spectrum evaluation across all 4 market regimes.

Single-Look Guard & Distributional Statistics:
    Each evaluated regime subset is logged to the persistent trial ledger under a canonical candidate ID
    (built via build_candidate_id(candidate_id, suffix=f"regime_{label}")). If an entry already exists in the
    ledger, its logged Sharpe, skew, kurtosis, and track_record_length are retrieved directly without re-logging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

from core.panel import Panel
from alpha.expressions.tree import Expression
from alpha.strategies.identity import build_candidate_id
from validation.folds import WalkForwardConfig
from validation.runner import run_walk_forward_validation
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from statistics import TrialRecord, log_trial, load_trial_log, compute_distribution_stats, save_returns
from .regime_classification import classify_regimes, RegimeLabels

# Complete set of canonical 4 market regime combinations
ALL_REGIME_LABELS = (
    "bear_high_vol",
    "bear_low_vol",
    "bull_high_vol",
    "bull_low_vol",
)


@dataclass
class RegimeRobustnessResult:
    """Dataclass encapsulating per-regime Sharpe evaluations, skipped regimes, and robustness classification."""
    candidate_id: str
    regime_sharpes: Dict[str, float]
    regime_counts: Dict[str, int]
    skipped_regimes: Dict[str, str] = field(default_factory=dict)
    classification: str = "fragile"  # "regime_robust", "regime_dependent", "fragile"
    rationale: str = ""


def evaluate_regime_robustness(
    candidate_id: str,
    build_fn: Callable[[Dict[str, Any]], Expression],
    default_params: Dict[str, Any],
    panel: Panel,
    wf_config: WalkForwardConfig = WalkForwardConfig(),
    backtest_config: BacktestConfig = BacktestConfig(),
    cost_model: CostModel = CostModel(),
    log_path: Optional[Union[str, Path]] = None,
    store_path: Optional[Union[str, Path]] = None,
    min_regime_days: int = 20,
    regime_lookback: int = 60,
    oos_returns: Optional[pd.Series] = None,
) -> RegimeRobustnessResult:
    """
    Evaluates candidate alpha performance across market trend/volatility regime subsets.

    Args:
        candidate_id: Canonical base candidate ID.
        build_fn: Strategy expression builder function.
        default_params: Parameters dictionary for build_fn.
        panel: Market data Panel.
        wf_config: WalkForwardConfig for OOS evaluation.
        backtest_config: BacktestConfig.
        cost_model: CostModel.
        log_path: Path to trial_log.jsonl ledger (default None; opt-in).
        min_regime_days: Minimum trading days required for a regime subset to be evaluated (default 20).
        regime_lookback: Trailing window for regime classification (default 60).
        oos_returns: Optional pre-evaluated OOS net returns series. If None, run_walk_forward_validation is called once.

    Returns:
        RegimeRobustnessResult: Per-regime Sharpes, observation counts, skipped regime details, and robustness classification.
    """
    # 1. Classify regimes over the Panel
    reg_labels = classify_regimes(panel, lookback_days=regime_lookback)

    # 2. Retrieve base candidate's OOS returns series if not pre-supplied
    if oos_returns is None:
        expr = build_fn(default_params)
        val_res = run_walk_forward_validation(
            candidate_id=candidate_id,
            expression=expr,
            panel=panel,
            wf_config=wf_config,
            backtest_config=backtest_config,
            cost_model=cost_model,
        )
        oos_gross = val_res.oos_gross_returns
        oos_turnover = val_res.oos_turnover
        if oos_gross is not None and oos_turnover is not None:
            cost_drag = oos_turnover * (cost_model.cost_bps / 10000.0)
            net_rets = oos_gross - cost_drag
        else:
            net_rets = oos_gross
    else:
        net_rets = oos_returns

    if net_rets is None or len(net_rets) == 0:
        return RegimeRobustnessResult(
            candidate_id=candidate_id,
            regime_sharpes={},
            regime_counts={},
            skipped_regimes={lbl: "OOS returns series is empty or None." for lbl in ALL_REGIME_LABELS},
            classification="fragile",
            rationale="OOS candidate net returns series is empty or None.",
        )

    # 3. Load existing trial log ledger records if log_path is provided
    existing_records: Dict[str, TrialRecord] = {}
    if log_path is not None:
        records = load_trial_log(log_path)
        existing_records = {r.candidate_id: r for r in records}

    now_str = datetime.now().isoformat()

    # 4. Evaluate OOS performance within each potential regime subset
    regime_sharpes: Dict[str, float] = {}
    regime_counts: Dict[str, int] = {}
    skipped_regimes: Dict[str, str] = {}

    for label in ALL_REGIME_LABELS:
        # Match dates where combined label equals label and fall within net_rets date index
        regime_dates = reg_labels.combined[reg_labels.combined == label].index
        common_dates = net_rets.index.intersection(regime_dates)
        subset_rets = net_rets.loc[common_dates].dropna()
        n_days = len(subset_rets)

        if n_days < min_regime_days:
            # Explicitly record skipped regimes rather than silently dropping them
            if n_days == 0:
                skipped_regimes[label] = f"0 OOS trading days in holdout tail window (< min_regime_days {min_regime_days})"
            else:
                skipped_regimes[label] = f"Insufficient OOS trading days: {n_days} (< min_regime_days {min_regime_days})"
            continue

        var_id = build_candidate_id(candidate_id, suffix=f"regime_{label}")

        if var_id in existing_records:
            # Single-look guard hit: retrieve full TrialRecord from ledger (preserve distribution stats)
            rec = existing_records[var_id]
            oos_sh = rec.oos_sharpe
            regime_sharpes[label] = oos_sh
            regime_counts[label] = rec.track_record_length if rec.track_record_length is not None else n_days
        else:
            # Calculate Sharpe on sliced returns
            mean_ret = float(subset_rets.mean())
            std_ret = float(subset_rets.std())
            oos_sh = (mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 1e-8 else 0.0

            skew, kurtosis, t_len = compute_distribution_stats(subset_rets)

            regime_sharpes[label] = oos_sh
            regime_counts[label] = n_days

            if log_path is not None:
                try:
                    log_trial(
                        TrialRecord(
                            candidate_id=var_id,
                            phase=f"Phase 12 Regime {label}",
                            oos_sharpe=oos_sh,
                            timestamp=now_str,
                            backfilled=False,
                            skew=skew,
                            kurtosis=kurtosis,
                            track_record_length=t_len,
                        ),
                        log_path=log_path,
                    )
                    existing_records[var_id] = TrialRecord(
                        candidate_id=var_id,
                        phase=f"Phase 12 Regime {label}",
                        oos_sharpe=oos_sh,
                        timestamp=now_str,
                        backfilled=False,
                        skew=skew,
                        kurtosis=kurtosis,
                        track_record_length=t_len,
                    )
                    if log_path is not None:
                        target_store_path = store_path if store_path is not None else Path(log_path).parent / "trial_returns.parquet"
                        try:
                            save_returns(var_id, subset_rets, store_path=target_store_path)
                        except ValueError:
                            pass  # Already stored; immutability discipline
                except ValueError:
                    pass

    # 5. Classify Regime Robustness
    pos_regimes = [k for k, sh in regime_sharpes.items() if sh > 0.0]
    total_eval_regimes = len(regime_sharpes)
    pos_ratio = (len(pos_regimes) / total_eval_regimes) if total_eval_regimes > 0 else 0.0

    skipped_note = f" ({len(skipped_regimes)} regime types skipped due to tail-window coverage/min days)" if skipped_regimes else ""

    if pos_ratio >= 0.75 and total_eval_regimes >= 2:
        classification = "regime_robust"
        rationale = f"Positive OOS Sharpe across {len(pos_regimes)}/{total_eval_regimes} evaluated regime subsets ({pos_ratio:.0%}){skipped_note}."
    elif pos_ratio >= 0.50:
        classification = "regime_dependent"
        rationale = f"Positive OOS Sharpe in {len(pos_regimes)}/{total_eval_regimes} evaluated regime subsets ({pos_ratio:.0%}); performance varies by market regime{skipped_note}."
    else:
        classification = "fragile"
        rationale = f"Poor regime generalization: positive OOS Sharpe in only {len(pos_regimes)}/{total_eval_regimes} evaluated regime subsets ({pos_ratio:.0%}){skipped_note}."

    return RegimeRobustnessResult(
        candidate_id=candidate_id,
        regime_sharpes=regime_sharpes,
        regime_counts=regime_counts,
        skipped_regimes=skipped_regimes,
        classification=classification,
        rationale=rationale,
    )
