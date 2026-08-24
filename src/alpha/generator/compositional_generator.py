"""
Reproducible random compositional candidate generator within the whitelisted vocabulary.
"""

import logging
import random
from typing import List, Optional, Set, Any, Dict
import pandas as pd

from alpha.expressions.tree import Expression, Leaf, UnaryNode, BinaryNode
from alpha.expressions.serialize import to_string
from .vocabulary import LEAF_PRIMITIVES, UNARY_OPS, BINARY_OPS
from .candidate import GeneratedCandidate

logger = logging.getLogger(__name__)


def _build_random_leaf(rng: random.Random) -> Leaf:
    """Helper to pick a random Leaf primitive and random parameters."""
    leaf_classes = list(LEAF_PRIMITIVES.keys())
    chosen_cls = rng.choice(leaf_classes)
    param_grid = LEAF_PRIMITIVES[chosen_cls]

    params: Dict[str, Any] = {}
    for p_name, p_vals in param_grid.items():
        params[p_name] = rng.choice(p_vals)

    return Leaf(chosen_cls(**params))


def _build_random_tree(
    current_depth: int, max_depth: int, rng: random.Random
) -> Expression:
    """
    Recursively builds a random expression tree subject to max_depth limit.
    Uses rng for every decision.
    """
    if current_depth >= max_depth:
        return _build_random_leaf(rng)

    # Choose node type: Leaf (0), Unary (1), Binary (2)
    node_type = rng.choice(["leaf", "unary", "binary"])

    if node_type == "leaf":
        return _build_random_leaf(rng)

    elif node_type == "unary":
        unary_classes = list(UNARY_OPS.keys())
        chosen_cls = rng.choice(unary_classes)
        param_grid = UNARY_OPS[chosen_cls]

        params: Dict[str, Any] = {}
        for p_name, p_vals in param_grid.items():
            params[p_name] = rng.choice(p_vals)

        op_instance = chosen_cls(**params)
        child = _build_random_tree(current_depth + 1, max_depth, rng)
        return UnaryNode(op_instance, child)

    else:  # binary
        chosen_cls = rng.choice(BINARY_OPS)
        op_instance = chosen_cls()
        left = _build_random_tree(current_depth + 1, max_depth, rng)
        right = _build_random_tree(current_depth + 1, max_depth, rng)
        return BinaryNode(op_instance, left, right)


def generate_random_candidates(
    budget: int,
    max_depth: int = 4,
    seed: int = 42,
    max_attempts: Optional[int] = None,
) -> List[GeneratedCandidate]:
    """
    Generates a reproducible set of unique random alpha candidates.

    Reproducibility:
        Uses an isolated random.Random(seed) instance for every choice.
        Calling this function with the same seed, budget, and max_depth produces
        identical candidate lists byte-for-byte.

    Budget Exhaustion:
        Attempts generation up to `max_attempts` (default 20 * budget). If the requested
        budget cannot be filled due to vocabulary/depth limitations, it returns all unique
        candidates found without infinite looping or raising an exception.

    Args:
        budget: Target number of unique candidates to generate.
        max_depth: Maximum tree depth constraint (default 4).
        seed: Random seed for reproducibility.
        max_attempts: Maximum generation attempts before stopping.

    Returns:
        List[GeneratedCandidate]: List of unique generated candidates.
    """
    rng = random.Random(seed)
    seen_hashes: Set[str] = set()
    candidates: List[GeneratedCandidate] = []

    limit_attempts = max_attempts if max_attempts is not None else (20 * budget)
    attempts = 0
    now = pd.Timestamp.now()

    while len(candidates) < budget and attempts < limit_attempts:
        attempts += 1
        expr = _build_random_tree(current_depth=1, max_depth=max_depth, rng=rng)
        h = expr.canonical_hash()

        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        candidate = GeneratedCandidate(
            expression=expr,
            expression_string=to_string(expr),
            generation_method="compositional_random",
            seed=seed,
            parent_family=None,
            param_values={},
            generated_at=now,
        )
        candidates.append(candidate)

    if len(candidates) < budget:
        logger.warning(
            f"Compositional generator requested budget ({budget}) could not be filled "
            f"after {attempts} attempts. Returning {len(candidates)} unique candidates."
        )

    return candidates
