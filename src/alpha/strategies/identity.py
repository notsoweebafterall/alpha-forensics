"""
Canonical candidate identity construction.

Single source of truth for how a candidate_id string is built from a base
strategy/family name plus its parameter values and/or an evaluation-context
suffix (e.g. sector or period slice). Every module that constructs a
candidate_id (Phase 6 variant expansion, Phase 9 parameter landscape,
Phase 11 sector/period generalization, and any future one) must call this
function instead of re-implementing the string formatting locally.

Why this exists:
    A prior version of this codebase had candidate_id strings assembled
    independently in three different places, with no shared definition of
    what a candidate_id even meant. That let a fabricated ID
    (using parameter names — "vol_lookback"/"target" — that do not exist in
    any real strategy's parameter grid) get typed directly into a demo
    script and treated as if it referred to a real, previously-evaluated
    candidate. Centralizing construction here does not make fabrication
    impossible, but it removes the "reformat by hand" step where a fake ID
    is indistinguishable in shape from a real one.
"""

from typing import Any, Dict, Optional


def build_candidate_id(
    base: str,
    params: Optional[Dict[str, Any]] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    Builds a canonical candidate_id string.

    Args:
        base: Root strategy or parent family name (e.g. "cross_sectional_momentum").
        params: Parameter values dict for this variant, if any. Keys are
            sorted for determinism, so callers never need to pre-sort.
            The keys used here MUST match the real parameter names declared
            in the strategy's `variant_params` grid (see
            alpha.strategies.library) — this function does not invent or
            validate parameter names, it only formats what it is given.
        suffix: Optional evaluation-context suffix, e.g. "sector_Technology"
            or "period_h1", for Phase 11-style sub-evaluations of an
            already-identified candidate.

    Returns:
        str: Canonical candidate_id, e.g.
            "cross_sectional_momentum__param_lookback_20__sector_Technology"

    Raises:
        ValueError: If base is empty/blank.
    """
    if not base or not base.strip():
        raise ValueError("build_candidate_id requires a non-empty base name.")

    candidate_id = base

    if params:
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        if param_str:
            candidate_id = f"{candidate_id}__param_{param_str}"

    if suffix:
        candidate_id = f"{candidate_id}__{suffix}"

    return candidate_id