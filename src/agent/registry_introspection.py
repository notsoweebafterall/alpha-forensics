"""
Primitive & Operator Registry Introspection for Phase 5.5 Autonomous Research Agent.

Pulls the real primitive/operator vocabulary from src/alpha/generator/vocabulary.py
and formats it into a prompt-ready specification for the LLM proposal engine.
Also provides expression tree vocabulary validation.
"""

from typing import List, Set, Dict, Any, Type
import alpha.generator.vocabulary as vocab
from alpha.expressions.tree import Expression, Leaf, UnaryNode, BinaryNode, ConstantNode
from alpha.expressions.tree import validate_constant_value
from alpha.primitives.inputs import (
    PriceReturn, Momentum, RollingMean, RollingStd, RollingZScore,
    RollingRank, VolumeChange, Turnover, RealizedVolatility, Drawdown
)
from alpha.primitives.operators import (
    Rank, ZScore, Lag, Diff, Negate, Abs, SignedPower,
    Add, Subtract, Multiply, Divide
)


def get_registry_summary() -> str:
    """
    Returns a prompt-ready formatted summary of the whitelisted primitive & operator vocabulary.
    This ensures the LLM receives exact parameter grids and signature specifications.
    """
    lines = [
        "================================================================================",
        "OFFICIAL ALPHA PRIMITIVE & OPERATOR REGISTRY (VOCABULARY WHITELIST)",
        "================================================================================",
        "You MUST combine ONLY the primitives and operators listed below.",
        "Do NOT invent new primitive names, functions, or unlisted parameter values.",
        "",
        "1. LEAF PRIMITIVES (Input Features):",
        "  - PriceReturn(lag): Price return over lag days. Allowed lag: [1, 5, 10]",
        "  - Momentum(lookback): Cumulative return over lookback days. Allowed lookback: [20, 60, 126, 252]",
        "  - RollingMean(window): Moving average price return over window. Allowed window: [10, 20, 60]",
        "  - RollingStd(window): Moving standard deviation of price return. Allowed window: [10, 20, 60]",
        "  - RollingZScore(window): Time-series Z-score over window. Allowed window: [10, 20, 60]",
        "  - RollingRank(window): Time-series percentile rank over window. Allowed window: [10, 20, 60]",
        "  - VolumeChange(lookback): Volume percentage change over lookback days. Allowed lookback: [1, 5, 20]",
        "  - Turnover(lookback): Dollar volume / turnover over lookback days. Allowed lookback: [5, 20, 60]",
        "  - RealizedVolatility(window): Rolling return volatility over window. Allowed window: [20, 60]",
        "  - Drawdown(window): Peak-to-trough price drawdown over window (non-positive). Allowed window: [20, 60]",
        "",
        "2. UNARY OPERATORS (Single-child Transformations):",
        "  - Rank(expr): Cross-sectional percentile rank across tickers (values between 0.0 and 1.0).",
        "  - ZScore(expr): Cross-sectional Z-score normalization across tickers.",
        "  - Negate(expr): Elementwise negation (-expr).",
        "  - Abs(expr): Elementwise absolute value (|expr|).",
        "  - Lag(expr, periods): Time-series shift by periods days. Allowed periods: [1, 5]",
        "  - Diff(expr, periods): Time-series difference (expr_t - expr_{t-periods}). Allowed periods: [1, 5]",
        "  - SignedPower(expr, power): Elementwise sign-preserving power sign(x) * |x|^power. Allowed power: [0.5, 2.0]",
        "",
        "3. BINARY OPERATORS (Two-child Combinations):",
        "  - Add(left, right): Elementwise addition (left + right).",
        "  - Subtract(left, right): Elementwise subtraction (left - right).",
        "  - Multiply(left, right): Elementwise multiplication (left * right).",
        "  - Divide(left, right): Elementwise division (left / right, handles div-by-zero safely).",
        "================================================================================",
    ]
    return "\n".join(lines)


def get_known_primitive_names() -> Set[str]:
    """Returns set of all valid primitive and operator class names in the whitelist."""
    leaf_names = {cls.__name__ for cls in vocab.LEAF_PRIMITIVES.keys()}
    unary_names = {cls.__name__ for cls in vocab.UNARY_OPS.keys()}
    binary_names = {cls.__name__ for cls in vocab.BINARY_OPS}
    return leaf_names | unary_names | binary_names


def validate_expression_vocabulary(expr: Expression) -> List[str]:
    """
    Recursively validates an Expression tree against the official vocabulary whitelist.
    
    Checks:
      1. Every primitive/operator class is in vocabulary.py.
      2. Every parameter value matches the whitelisted parameter grid.

    Returns:
        List[str]: List of violation messages. Empty if 100% valid.
    """
    violations: List[str] = []
    _check_node(expr, violations)
    return violations


def _check_node(node: Expression, violations: List[str]) -> None:
    if isinstance(node, Leaf):
        prim = node.primitive
        prim_cls = prim.__class__
        if prim_cls not in vocab.LEAF_PRIMITIVES:
            violations.append(f"Unknown leaf primitive class: '{prim_cls.__name__}'")
            return
        
        allowed_grid = vocab.LEAF_PRIMITIVES[prim_cls]
        params = prim.params_dict()
        for p_name, p_val in params.items():
            if p_name in allowed_grid:
                if p_val not in allowed_grid[p_name]:
                    violations.append(
                        f"Invalid parameter value for {prim_cls.__name__}.{p_name}: got {p_val}, "
                        f"allowed: {allowed_grid[p_name]}"
                    )
            else:
                violations.append(f"Unexpected parameter '{p_name}' for {prim_cls.__name__}")

    elif isinstance(node, UnaryNode):
        op = node.operator
        op_cls = op.__class__
        if op_cls not in vocab.UNARY_OPS:
            violations.append(f"Unknown unary operator class: '{op_cls.__name__}'")
        else:
            allowed_grid = vocab.UNARY_OPS[op_cls]
            params = op.params_dict()
            for p_name, p_val in params.items():
                if p_name in allowed_grid:
                    if p_val not in allowed_grid[p_name]:
                        violations.append(
                            f"Invalid parameter value for {op_cls.__name__}.{p_name}: got {p_val}, "
                            f"allowed: {allowed_grid[p_name]}"
                        )
        _check_node(node.child, violations)

    elif isinstance(node, BinaryNode):
        op = node.operator
        op_cls = op.__class__
        if op_cls not in vocab.BINARY_OPS:
            violations.append(f"Unknown binary operator class: '{op_cls.__name__}'")
        _check_node(node.left, violations)
        _check_node(node.right, violations)

    elif isinstance(node, ConstantNode):
        # Re-run bounds check so validation catches any ConstantNode that
        # bypassed the constructor (e.g. direct instantiation in mutation code).
        violations.extend(validate_constant_value(node.value))

    else:
        violations.append(f"Unknown expression node type: {type(node)}")
