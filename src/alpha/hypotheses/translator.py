"""
Deterministic translator converting a HypothesisSpec into a validated Expression tree.
"""

from typing import Set

from alpha.expressions.tree import Expression, Leaf, UnaryNode, BinaryNode
from alpha.expressions.serialize import parse
from alpha.expressions.constraints import check_depth
from .spec import HypothesisSpec


def _extract_node_type_names(expr: Expression) -> Set[str]:
    """
    Recursively extracts all primitive and operator class names present in an expression tree.
    """
    if isinstance(expr, Leaf):
        return {expr.primitive.__class__.__name__}
    elif isinstance(expr, UnaryNode):
        return {expr.operator.__class__.__name__} | _extract_node_type_names(expr.child)
    elif isinstance(expr, BinaryNode):
        return (
            {expr.operator.__class__.__name__}
            | _extract_node_type_names(expr.left)
            | _extract_node_type_names(expr.right)
        )
    else:
        raise TypeError(f"Unknown expression node type: {type(expr)}")


def translate(spec: HypothesisSpec) -> Expression:
    """
    Translates a HypothesisSpec into a validated Expression tree.

    Checks:
        1. Parses composition string using Phase 3 `parse()`.
        2. Provenance Consistency Check: Asserts every primitive and operator type used in the
           expression tree is explicitly listed in `spec.source_features`.
        3. Depth Constraint: Enforces `check_depth()` (MAX_DEPTH=4).

    Args:
        spec: HypothesisSpec definition.

    Returns:
        Expression: Validated expression tree ready for evaluation and backtesting.

    Raises:
        ValueError: If composition syntax is invalid, undeclared features are used, or depth limit is exceeded.
    """
    # 1. Parse composition string
    try:
        expr = parse(spec.composition)
    except Exception as e:
        raise ValueError(
            f"Failed to parse composition string for hypothesis '{spec.hypothesis_id}': {e}"
        )

    # 2. Provenance consistency check
    used_types = _extract_node_type_names(expr)
    declared_set = set(spec.source_features)
    undeclared = used_types - declared_set

    if undeclared:
        sorted_undeclared = sorted(list(undeclared))
        raise ValueError(
            f"Provenance consistency check failed for hypothesis '{spec.hypothesis_id}': "
            f"Primitive/operator types {sorted_undeclared} used in composition but not declared in source_features."
        )

    # 3. Depth constraint enforcement
    check_depth(expr)

    return expr
