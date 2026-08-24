"""
In-memory hypothesis registry catalog.
"""

from typing import List, Dict
from .spec import HypothesisSpec


HYPOTHESIS_CATALOG: Dict[str, HypothesisSpec] = {
    "hyp_momentum_vol_decay": HypothesisSpec(
        hypothesis_id="hyp_momentum_vol_decay",
        hypothesis_text="Momentum is stronger when volatility is declining.",
        economic_rationale="Declining volatility indicates regime stability, reducing noise and allowing trend persistence.",
        composition="Multiply(Momentum(126), Negate(Diff(RealizedVolatility(20), 5)))",
        source_features=["Momentum", "RealizedVolatility", "Diff", "Negate", "Multiply"],
        parent_family="momentum",
        author_notes="Phase 5 canonical hypothesis example 1",
    ),
    "hyp_reversal_volume_spike": HypothesisSpec(
        hypothesis_id="hyp_reversal_volume_spike",
        hypothesis_text="Short-term reversal is stronger following volume spikes.",
        economic_rationale="High volume extreme price moves often reflect forced liquidation or retail panic, leading to robust short-term mean-reversion.",
        composition="Multiply(Rank(Negate(Momentum(5))), Rank(VolumeChange(5)))",
        source_features=["Momentum", "VolumeChange", "Negate", "Rank", "Multiply"],
        parent_family="reversal",
        author_notes="Phase 5 canonical hypothesis example 2",
    ),
    "hyp_zscore_reversal_low_vol": HypothesisSpec(
        hypothesis_id="hyp_zscore_reversal_low_vol",
        hypothesis_text="Extreme price deviation reverts more reliably in low-volatility names.",
        economic_rationale="Overextended prices in stable low-volatility stocks represent temporary mispricings rather than fundamental structural shifts.",
        composition="Multiply(Rank(Negate(RollingZScore(20))), Rank(Negate(RealizedVolatility(60))))",
        source_features=["RollingZScore", "RealizedVolatility", "Negate", "Rank", "Multiply"],
        parent_family="reversal",
        author_notes="Phase 5 canonical hypothesis example 3",
    ),
}


def get_all_hypotheses() -> List[HypothesisSpec]:
    """
    Returns all registered hypothesis specifications in the catalog.
    """
    return list(HYPOTHESIS_CATALOG.values())


def get_hypothesis(hypothesis_id: str) -> HypothesisSpec:
    """
    Retrieves a hypothesis specification by ID.

    Args:
        hypothesis_id: Unique identifier for the hypothesis.

    Returns:
        HypothesisSpec: Target hypothesis specification.

    Raises:
        KeyError: If hypothesis_id is not found in the catalog.
    """
    if hypothesis_id not in HYPOTHESIS_CATALOG:
        available = sorted(list(HYPOTHESIS_CATALOG.keys()))
        raise KeyError(
            f"Hypothesis ID '{hypothesis_id}' not found in registry catalog. Available IDs: {available}"
        )
    return HYPOTHESIS_CATALOG[hypothesis_id]
