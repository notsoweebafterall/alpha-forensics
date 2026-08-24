"""
Chronological walk-forward fold generation with embargo enforcement.

Methodological Rationale for Embargo:
    Unlike machine learning models fit on training data, alpha expressions in this framework
    have fixed, a priori chosen parameters. Classic leakage through parameter fitting is not
    the primary risk here.

    The primary leakage risk is serial correlation across the train/validation boundaries
    arising from rolling-window features (e.g., momentum lookbacks up to 252 days, realized volatility
    windows, etc.). A validation date immediately following a train cutoff shares most of its
    underlying lookback window returns with the end of the train period.

    Enforcing an embargo gap (embargo_days) between train end and validation start isolates these
    time-series dependencies and ensures leak-free validation.
"""

from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

from core.panel import Panel


@dataclass
class WalkForwardConfig:
    mode: str = "expanding"          # "expanding" | "rolling"
    initial_train_window: int = 500  # trading days
    step_size: int = 60              # trading days
    val_window: int = 60             # trading days
    embargo_days: int = 10           # gap in trading days between train end and val start
    min_train_window: int = 500      # minimum required trading days for train set


@dataclass
class Fold:
    fold_id: int
    train_range: Tuple[pd.Timestamp, pd.Timestamp]
    val_range: Tuple[pd.Timestamp, pd.Timestamp]


def validate_no_overlap(
    folds: List[Fold], dates_index: pd.DatetimeIndex, embargo_days: int
) -> None:
    """
    Validates that folds contain valid chronological ranges and respect the embargo gap.

    Raises:
        ValueError: If any fold has invalid date bounds, overlapping train/val ranges,
                    or violates the required embargo gap.
    """
    for fold in folds:
        t_start, t_end = fold.train_range
        v_start, v_end = fold.val_range

        if t_start > t_end:
            raise ValueError(
                f"Fold {fold.fold_id}: Invalid train range [{t_start}, {t_end}]."
            )
        if v_start > v_end:
            raise ValueError(
                f"Fold {fold.fold_id}: Invalid val range [{v_start}, {v_end}]."
            )
        if v_start <= t_end:
            raise ValueError(
                f"Fold {fold.fold_id}: Overlap detected! Val start {v_start} <= Train end {t_end}."
            )

        # Map timestamps to trading day indices
        try:
            t_end_idx = dates_index.get_loc(t_end)
            v_start_idx = dates_index.get_loc(v_start)
        except KeyError as e:
            raise ValueError(
                f"Fold {fold.fold_id}: Timestamp {e} not found in panel dates index."
            )

        gap = v_start_idx - t_end_idx
        if gap < embargo_days:
            raise ValueError(
                f"Fold {fold.fold_id}: Embargo violation! Gap ({gap} trading days) < required embargo ({embargo_days} trading days)."
            )


def generate_folds(panel: Panel, config: WalkForwardConfig) -> List[Fold]:
    """
    Generates chronological walk-forward folds over the development panel date range.

    Args:
        panel: Development Panel object.
        config: WalkForwardConfig configuration.

    Returns:
        List[Fold]: List of validated Fold objects.
    """
    dates = panel.prices.index
    total_dates = len(dates)

    min_required = (
        config.initial_train_window + config.embargo_days + config.val_window
    )
    if total_dates < min_required:
        raise ValueError(
            f"Panel date range ({total_dates} days) is too short for configuration "
            f"(requires at least {min_required} trading days)."
        )

    folds: List[Fold] = []
    k = 0

    while True:
        if config.mode == "expanding":
            train_start_idx = 0
            train_end_idx = config.initial_train_window + k * config.step_size - 1
        elif config.mode == "rolling":
            train_start_idx = k * config.step_size
            train_end_idx = train_start_idx + config.initial_train_window - 1
        else:
            raise ValueError(f"Unsupported walk-forward mode: {config.mode}")

        val_start_idx = train_end_idx + config.embargo_days
        val_end_idx = val_start_idx + config.val_window - 1

        # Check if validation window exceeds total dates
        if val_end_idx >= total_dates:
            break

        train_range = (dates[train_start_idx], dates[train_end_idx])
        val_range = (dates[val_start_idx], dates[val_end_idx])

        folds.append(
            Fold(fold_id=k, train_range=train_range, val_range=val_range)
        )
        k += 1

    if not folds:
        raise ValueError("No valid walk-forward folds could be generated.")

    # Actively enforce no-overlap and embargo invariants
    validate_no_overlap(folds, dates, config.embargo_days)

    return folds
