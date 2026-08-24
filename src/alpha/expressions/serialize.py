"""
String serialization and AST-based parsing for expression trees.
"""

import ast
from typing import Any

import alpha.primitives.inputs as inputs
import alpha.primitives.operators as operators
from .tree import Expression, Leaf, UnaryNode, BinaryNode


def to_string(expression: Expression) -> str:
    """Converts expression tree to string representation."""
    return expression.to_string()


def _eval_ast_val(node: ast.AST) -> Any:
    """Extracts constant literal value from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast_val(node.operand)
    else:
        raise ValueError(f"Cannot evaluate AST literal: {ast.dump(node)}")


def _parse_ast_node(node: ast.AST) -> Expression:
    """Recursively parses AST node into Expression tree."""
    if isinstance(node, ast.Expression):
        return _parse_ast_node(node.body)

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Invalid expression AST node: {ast.dump(node)}")

    func_name = node.func.id
    args = node.args

    # 1. Check Leaf Primitives
    if func_name == "PriceReturn":
        lag = _eval_ast_val(args[0]) if len(args) > 0 else 1
        return Leaf(inputs.PriceReturn(lag=lag))
    elif func_name == "Momentum":
        lookback = _eval_ast_val(args[0])
        return Leaf(inputs.Momentum(lookback=lookback))
    elif func_name == "RollingMean":
        window = _eval_ast_val(args[0])
        return Leaf(inputs.RollingMean(window=window))
    elif func_name == "RollingStd":
        window = _eval_ast_val(args[0])
        return Leaf(inputs.RollingStd(window=window))
    elif func_name == "RollingZScore":
        window = _eval_ast_val(args[0])
        return Leaf(inputs.RollingZScore(window=window))
    elif func_name == "RollingRank":
        window = _eval_ast_val(args[0])
        return Leaf(inputs.RollingRank(window=window))
    elif func_name == "VolumeChange":
        lookback = _eval_ast_val(args[0]) if len(args) > 0 else 1
        return Leaf(inputs.VolumeChange(lookback=lookback))
    elif func_name == "Turnover":
        lookback = _eval_ast_val(args[0])
        return Leaf(inputs.Turnover(lookback=lookback))
    elif func_name == "RealizedVolatility":
        window = _eval_ast_val(args[0])
        return Leaf(inputs.RealizedVolatility(window=window))
    elif func_name == "Drawdown":
        window = _eval_ast_val(args[0])
        return Leaf(inputs.Drawdown(window=window))

    # 2. Check Unary Operators
    elif func_name == "Rank":
        child = _parse_ast_node(args[0])
        return UnaryNode(operators.Rank(), child)
    elif func_name == "ZScore":
        child = _parse_ast_node(args[0])
        return UnaryNode(operators.ZScore(), child)
    elif func_name == "Negate":
        child = _parse_ast_node(args[0])
        return UnaryNode(operators.Negate(), child)
    elif func_name == "Abs":
        child = _parse_ast_node(args[0])
        return UnaryNode(operators.Abs(), child)
    elif func_name == "Lag":
        child = _parse_ast_node(args[0])
        periods = _eval_ast_val(args[1]) if len(args) > 1 else 1
        return UnaryNode(operators.Lag(periods=periods), child)
    elif func_name == "Diff":
        child = _parse_ast_node(args[0])
        periods = _eval_ast_val(args[1]) if len(args) > 1 else 1
        return UnaryNode(operators.Diff(periods=periods), child)
    elif func_name == "SignedPower":
        child = _parse_ast_node(args[0])
        power = _eval_ast_val(args[1])
        return UnaryNode(operators.SignedPower(power=float(power)), child)

    # 3. Check Binary Operators
    elif func_name == "Add":
        left = _parse_ast_node(args[0])
        right = _parse_ast_node(args[1])
        return BinaryNode(operators.Add(), left, right)
    elif func_name == "Subtract":
        left = _parse_ast_node(args[0])
        right = _parse_ast_node(args[1])
        return BinaryNode(operators.Subtract(), left, right)
    elif func_name == "Multiply":
        left = _parse_ast_node(args[0])
        right = _parse_ast_node(args[1])
        return BinaryNode(operators.Multiply(), left, right)
    elif func_name == "Divide":
        left = _parse_ast_node(args[0])
        right = _parse_ast_node(args[1])
        return BinaryNode(operators.Divide(), left, right)

    else:
        raise ValueError(f"Unknown primitive or operator in expression: '{func_name}'")


def parse(expression_str: str) -> Expression:
    """
    Parses string representation back into an Expression tree.

    Args:
        expression_str: String formatted by expression.to_string().

    Returns:
        Expression: Constructed expression tree.
    """
    try:
        parsed_ast = ast.parse(expression_str, mode="eval")
    except Exception as e:
        raise ValueError(f"Failed to parse expression string '{expression_str}': {e}")

    return _parse_ast_node(parsed_ast)
