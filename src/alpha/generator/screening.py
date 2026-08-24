"""
Fast triage screening for alpha candidates.

IMPORTANT DISCLAIMER:
    This screening function is a cheap triage filter designed to quickly eliminate broken,
    degenerate, or zero-predictiveness candidate signals. Passing this screen is NOT a
    validity claim or statistical significance test — full statistical validation and
    multiple testing corrections occur in Phase 10.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from core.panel import Panel
from backtesting.metrics import predictive_metrics
from .candidate import GeneratedCandidate


def screen_candidates(
    candidates: List[GeneratedCandidate],
    panel: Panel,
    ic_threshold: float = 0.005,
    min_coverage: float = 0.8,
) -> Tuple[List[GeneratedCandidate], List[Tuple[GeneratedCandidate, str]]]:
    """
    Screens generated candidates using fast signal quality & predictiveness checks.

    Rejection Reasons:
        - "invalid_signal": Signal evaluation failed or produced out-of-bounds/inf values.
        - "insufficient_coverage": Non-NaN entries fall below min_coverage fraction.
        - "degenerate_signal": Cross-sectional variance is zero/near-zero on majority of dates.
        - "weak_signal": Absolute mean Information Coefficient (|IC mean|) < ic_threshold.

    Args:
        candidates: List of GeneratedCandidate objects.
        panel: Market data Panel.
        ic_threshold: Minimum required absolute IC mean (default 0.005).
        min_coverage: Minimum required non-NaN coverage fraction (default 0.8).

    Returns:
        Tuple[List[GeneratedCandidate], List[Tuple[GeneratedCandidate, str]]]:
            (passed_candidates, rejected_candidates_with_reasons)
    """
    passed: List[GeneratedCandidate] = []
    rejected: List[Tuple[GeneratedCandidate, str]] = []

    total_cells = panel.prices.size

    for candidate in candidates:
        # 1. Signal evaluation & contract validation
        try:
            signal = candidate.expression.evaluate(panel)
        except Exception:
            rejected.append((candidate, "invalid_signal"))
            continue

        # 2. Coverage check
        non_nan_count = signal.notna().to_numpy().sum()
        coverage = non_nan_count / total_cells if total_cells > 0 else 0.0
        if coverage < min_coverage:
            rejected.append((candidate, "insufficient_coverage"))
            continue

        # 3. Degenerate signal check (cross-sectional variance)
        xs_stds = signal.std(axis=1)
        valid_stds = xs_stds.dropna()
        if valid_stds.empty or float(valid_stds.median()) < 1e-6 or (valid_stds < 1e-6).mean() > 0.5:
            rejected.append((candidate, "degenerate_signal"))
            continue

        # 4. Predictiveness check (quick full-sample IC)
        try:
            pred = predictive_metrics(signal, panel, execution_lag_days=1)
            ic_mean = pred.get("ic_mean", 0.0)
            if np.isnan(ic_mean) or abs(ic_mean) < ic_threshold:
                rejected.append((candidate, "weak_signal"))
                continue
        except Exception:
            rejected.append((candidate, "weak_signal"))
            continue

        # Passed all triage checks
        passed.append(candidate)

    return passed, rejected
