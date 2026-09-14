"""
Canonical structural deduplication hashing for expression trees.
"""

import hashlib
import json


def compute_canonical_string(expr: "Expression") -> str:  # type: ignore # noqa: F821
    """
    Recursively builds a canonical string representation for an expression node.

    Normalizes commutative operators (Add, Multiply) by sorting child hashes.
    Explicitly incorporates primitive and operator parameter values.
    """
    from .tree import Leaf, UnaryNode, BinaryNode, ConstantNode

    if isinstance(expr, Leaf):
        params_json = json.dumps(expr.primitive.params_dict(), sort_keys=True)
        return f"Leaf:{expr.primitive.__class__.__name__}:{params_json}"

    elif isinstance(expr, UnaryNode):
        child_str = compute_canonical_string(expr.child)
        op_params = json.dumps(expr.operator.params_dict(), sort_keys=True)
        return f"Unary:{expr.operator.__class__.__name__}:{op_params}({child_str})"

    elif isinstance(expr, BinaryNode):
        left_str = compute_canonical_string(expr.left)
        right_str = compute_canonical_string(expr.right)
        op_params = json.dumps(expr.operator.params_dict(), sort_keys=True)

        if expr.operator.is_commutative:
            # Sort child canonical strings to normalize A+B and B+A
            left_str, right_str = sorted([left_str, right_str])

        return f"Binary:{expr.operator.__class__.__name__}:{op_params}({left_str},{right_str})"

    elif isinstance(expr, ConstantNode):
        # repr() preserves type distinction: int 1 → "1", float 1.0 → "1.0"
        return f"Constant:{repr(expr.value)}"

    else:
        raise TypeError(f"Unknown expression node type: {type(expr)}")


def canonical_hash(expr: "Expression") -> str:  # type: ignore # noqa: F821
    """
    Returns SHA-256 hex digest of the canonical string representation of an expression.
    """
    canonical_str = compute_canonical_string(expr)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
