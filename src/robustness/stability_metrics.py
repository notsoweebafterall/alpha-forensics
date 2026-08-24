"""
Parameter stability metrics and heuristic classification logic.

Documented Heuristic Thresholds:
    - broad_robust: >= 70% of parameter variants exhibit OOS Sharpe > 0
    - moderately_sensitive: 40% - 70% of parameter variants exhibit OOS Sharpe > 0
    - highly_sensitive: 20% - 40% of parameter variants exhibit OOS Sharpe > 0
    - isolated_optimum: < 20% of parameter variants exhibit OOS Sharpe > 0

Disclaimers:
    These stability thresholds are empirical heuristic defaults, not formal theoretical derivations.
"""

from typing import List, Tuple
import numpy as np


def fraction_positive(sharpes: List[float]) -> float:
    """Computes the fraction of non-NaN Sharpe values strictly greater than 0.0."""
    valid = [s for s in sharpes if not np.isnan(s)]
    if not valid:
        return 0.0
    pos_count = sum(1 for s in valid if s > 0.0)
    return float(pos_count / len(valid))


def coefficient_of_variation(values: List[float]) -> float:
    """
    Computes the coefficient of variation (CV = std / |mean|) for a list of values.
    Returns 0.0 if mean is near-zero or values list is empty.
    """
    valid = [v for v in values if not np.isnan(v)]
    if not valid:
        return 0.0
    mean_val = float(np.mean(valid))
    std_val = float(np.std(valid))
    if abs(mean_val) < 1e-8:
        return 0.0
    return float(std_val / abs(mean_val))


def classify_stability(sharpes: List[float]) -> Tuple[str, str]:
    """
    Classifies the parameter stability of a candidate based on OOS Sharpe values.

    Args:
        sharpes: List of OOS Sharpe values across parameter variants.

    Returns:
        Tuple[str, str]: (stability_label, stability_rationale)
    """
    valid = [s for s in sharpes if not np.isnan(s)]
    total_count = len(valid)
    if total_count == 0:
        return "isolated_optimum", "0 of 0 variants showed valid OOS Sharpe — classified 'isolated_optimum'."

    pos_count = sum(1 for s in valid if s > 0.0)
    frac = float(pos_count / total_count)

    if frac >= 0.70:
        label = "broad_robust"
    elif frac >= 0.40:
        label = "moderately_sensitive"
    elif frac >= 0.20:
        label = "highly_sensitive"
    else:
        label = "isolated_optimum"

    rationale = (
        f"{pos_count} of {total_count} variants ({frac:.1%}) showed positive OOS Sharpe "
        f"— classified '{label}'."
    )

    return label, rationale
