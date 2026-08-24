"""
Expression tree constraints and structural validation checks.
"""

from .tree import Expression

# Maximum allowed expression tree depth.
# Bounded depth is a research-efficiency control to avoid isolated overfit trees.
MAX_DEPTH: int = 4


def check_depth(expression: Expression, max_depth: int = MAX_DEPTH) -> None:
    """
    Enforces maximum tree depth constraint.

    Raises:
        ValueError: If expression depth exceeds max_depth.
    """
    d = expression.depth()
    if d > max_depth:
        raise ValueError(
            f"Expression depth ({d}) exceeds maximum allowed depth limit ({max_depth})."
        )


def check_valid_structure(expression: Expression) -> None:
    """
    Validates structural integrity of an expression tree.

    Raises:
        ValueError: If any structural constraint is violated.
    """
    if expression is None:
        raise ValueError("Expression cannot be None.")

    # Enforce depth check
    check_depth(expression)
