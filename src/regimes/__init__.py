"""
Regimes module for Alpha Forensics.

Provides pure market regime classification (trend and volatility) and cross-regime
alpha generalization / robustness analysis with persistent trial ledger tracking.
"""

from .regime_classification import (
    RegimeLabels,
    classify_regimes,
)
from .regime_analysis import (
    RegimeRobustnessResult,
    evaluate_regime_robustness,
)

__all__ = [
    "RegimeLabels",
    "classify_regimes",
    "RegimeRobustnessResult",
    "evaluate_regime_robustness",
]
