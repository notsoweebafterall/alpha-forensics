"""
Deterministic parameter variant expansion for strategy definitions.
"""

import itertools
from typing import Callable, Dict, List, Any, Set
import pandas as pd

from alpha.expressions.tree import Expression
from alpha.expressions.serialize import to_string
from .candidate import GeneratedCandidate


def expand_variants(
    build_fn: Callable[[Dict[str, Any]], Expression],
    param_grid: Dict[str, List[Any]],
    parent_family: str,
) -> List[GeneratedCandidate]:
    """
    Expands a strategy's declared parameter grid into a list of unique candidates.

    Deduplication:
        Calculates `canonical_hash()` for each built expression. If multiple parameter
        combinations produce structurally identical expression trees, only the first instance is kept.

    Args:
        build_fn: Factory function mapping params dict -> Expression.
        param_grid: Parameter grid dict (param_name -> list of values).
        parent_family: Parent strategy/hypothesis name for provenance tagging.

    Returns:
        List[GeneratedCandidate]: List of unique generated candidates with full provenance.
    """
    if not param_grid:
        keys = []
        combos = [{}]
    else:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    seen_hashes: Set[str] = set()
    candidates: List[GeneratedCandidate] = []
    now = pd.Timestamp.now()

    for params in combos:
        expr = build_fn(params)
        h = expr.canonical_hash()

        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        candidate = GeneratedCandidate(
            expression=expr,
            expression_string=to_string(expr),
            generation_method="variant_expansion",
            seed=None,
            parent_family=parent_family,
            param_values=params,
            generated_at=now,
        )
        candidates.append(candidate)

    return candidates
