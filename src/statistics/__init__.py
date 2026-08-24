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
)
from .dsr import (
    expected_max_sharpe_under_trials,
    deflated_sharpe_ratio,
    DSRResult,
)
from .significance import (
    probabilistic_sharpe_ratio,
    sharpe_variance_across_trials,
)

__all__ = [
    "TrialRecord",
    "log_trial",
    "load_trial_log",
    "trial_count",
    "backfill_from_phase_artifacts",
    "expected_max_sharpe_under_trials",
    "deflated_sharpe_ratio",
    "DSRResult",
    "probabilistic_sharpe_ratio",
    "sharpe_variance_across_trials",
]
