"""
Statistics module for Alpha Forensics.

Provides statistical forensics, trial logging, Deflated Sharpe Ratio (DSR),
and Probabilistic Sharpe Ratio (PSR) analysis.
"""

from .trial_log import (
    TrialRecord,
    log_trial,
    load_trial_log,
    trial_count,
    backfill_from_phase_artifacts,
    StatusChangeRecord,
    append_status_change,
    effective_trial_records,
    VALID_STATUSES,
)
from .distribution_stats import compute_distribution_stats
from .dsr import (
    expected_max_sharpe_under_trials,
    deflated_sharpe_ratio,
    DSRResult,
    DSRLogRecord,
    log_dsr_result,
    load_dsr_log,
    load_latest_dsr_verdicts,
)
from .significance import (
    probabilistic_sharpe_ratio,
    sharpe_variance_across_trials,
)
from .returns_store import (
    save_returns,
    load_returns,
    load_returns_matrix,
)

__all__ = [
    "TrialRecord",
    "log_trial",
    "load_trial_log",
    "trial_count",
    "backfill_from_phase_artifacts",
    "StatusChangeRecord",
    "append_status_change",
    "effective_trial_records",
    "VALID_STATUSES",
    "compute_distribution_stats",
    "expected_max_sharpe_under_trials",
    "deflated_sharpe_ratio",
    "DSRResult",
    "DSRLogRecord",
    "log_dsr_result",
    "load_dsr_log",
    "load_latest_dsr_verdicts",
    "probabilistic_sharpe_ratio",
    "sharpe_variance_across_trials",
    "save_returns",
    "load_returns",
    "load_returns_matrix",
]