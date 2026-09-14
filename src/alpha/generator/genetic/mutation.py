"""
Mutation operators for Genetic Programming on Alpha Expression Trees.

Modifies expression trees while respecting:
  1. Vocabulary whitelist in alpha.generator.vocabulary.
  2. Maximum depth constraints to prevent bloat.
  3. Single-look & valid AST contract checks.
"""

import copy
import random
from typing import List, Dict, Any, Type, Optional, Tuple

from alpha.expressions.tree import Expression, Leaf, UnaryNode, BinaryNode
from alpha.expressions.serialize import to_string, parse
from alpha.generator.vocabulary import LEAF_PRIMITIVES, UNARY_OPS, BINARY_OPS
from agent.registry_introspection import validate_expression_vocabulary


def get_all_subtrees(expr: Expression) -> List[Expression]:
    """Returns a list of all node/subtree references in an Expression tree."""
    nodes = [expr]
    if isinstance(expr, UnaryNode):
        nodes.extend(get_all_subtrees(expr.child))
    elif isinstance(expr, BinaryNode):
        nodes.extend(get_all_subtrees(expr.left))
        nodes.extend(get_all_subtrees(expr.right))
    return nodes


def replace_subtree(
    root: Expression, target: Expression, replacement: Expression
) -> Expression:
    """
    Creates a deep copy of root with the target node reference replaced by replacement.
    """
    if root is target:
        return copy.deepcopy(replacement)

    if isinstance(root, Leaf):
        return copy.deepcopy(root)

    elif isinstance(root, UnaryNode):
        new_child = replace_subtree(root.child, target, replacement)
        return UnaryNode(copy.deepcopy(root.operator), new_child)

    elif isinstance(root, BinaryNode):
        new_left = replace_subtree(root.left, target, replacement)
        new_right = replace_subtree(root.right, target, replacement)
        return BinaryNode(copy.deepcopy(root.operator), new_left, new_right)

    return copy.deepcopy(root)


def _random_leaf(rng: random.Random) -> Leaf:
    """Helper to generate a random Leaf node from whitelisted vocabulary."""
    leaf_classes = list(LEAF_PRIMITIVES.keys())
    chosen_cls = rng.choice(leaf_classes)
    param_grid = LEAF_PRIMITIVES[chosen_cls]

    params: Dict[str, Any] = {}
    for p_name, p_vals in param_grid.items():
        params[p_name] = rng.choice(p_vals)

    return Leaf(chosen_cls(**params))


def _random_subtree(current_depth: int, max_depth: int, rng: random.Random) -> Expression:
    """Recursively generates a random expression subtree up to max_depth."""
    if current_depth >= max_depth:
        return _random_leaf(rng)

    node_type = rng.choice(["leaf", "unary", "binary"])

    if node_type == "leaf":
        return _random_leaf(rng)

    elif node_type == "unary":
        unary_classes = list(UNARY_OPS.keys())
        chosen_cls = rng.choice(unary_classes)
        param_grid = UNARY_OPS[chosen_cls]

        params: Dict[str, Any] = {}
        for p_name, p_vals in param_grid.items():
            params[p_name] = rng.choice(p_vals)

        op_instance = chosen_cls(**params)
        child = _random_subtree(current_depth + 1, max_depth, rng)
        return UnaryNode(op_instance, child)

    else:  # binary
        chosen_cls = rng.choice(BINARY_OPS)
        op_instance = chosen_cls()
        left = _random_subtree(current_depth + 1, max_depth, rng)
        right = _random_subtree(current_depth + 1, max_depth, rng)
        return BinaryNode(op_instance, left, right)


def mutate_expression(
    expr: Expression,
    max_depth: int = 6,
    rng: Optional[random.Random] = None,
    max_retries: int = 10,
) -> Expression:
    """
    Mutates an Expression tree by modifying a randomly selected node or replacing a subtree.

    Mutation Modes:
      1. Point mutation: Replace an operator or primitive with another of same arity.
      2. Parameter mutation: Change parameter value for a leaf/unary operator.
      3. Subtree mutation: Replace a target node with a newly generated random subtree.

    Guarantees:
      - Tree depth <= max_depth (prevents bloat).
      - Output passes validate_expression_vocabulary().
      - If mutation fails or exceeds depth, returns a valid mutated tree or copy of parent.
    """
    if rng is None:
        rng = random.Random()

    for _ in range(max_retries):
        subtrees = get_all_subtrees(expr)
        if not subtrees:
            break

        target_node = rng.choice(subtrees)

        # Decide mutation type
        mutation_mode = rng.choice(["point", "parameter", "subtree"])

        replacement: Expression

        if mutation_mode == "point":
            if isinstance(target_node, Leaf):
                replacement = _random_leaf(rng)
            elif isinstance(target_node, UnaryNode):
                unary_classes = list(UNARY_OPS.keys())
                chosen_cls = rng.choice(unary_classes)
                param_grid = UNARY_OPS[chosen_cls]
                params = {p_name: rng.choice(p_vals) for p_name, p_vals in param_grid.items()}
                new_op = chosen_cls(**params)
                replacement = UnaryNode(new_op, copy.deepcopy(target_node.child))
            else:  # BinaryNode
                chosen_cls = rng.choice(BINARY_OPS)
                new_op = chosen_cls()
                replacement = BinaryNode(new_op, copy.deepcopy(target_node.left), copy.deepcopy(target_node.right))

        elif mutation_mode == "parameter":
            if isinstance(target_node, Leaf):
                prim_cls = target_node.primitive.__class__
                if prim_cls in LEAF_PRIMITIVES:
                    param_grid = LEAF_PRIMITIVES[prim_cls]
                    curr_params = target_node.primitive.params_dict()
                    new_params = copy.deepcopy(curr_params)
                    for p_name, p_vals in param_grid.items():
                        if len(p_vals) > 1:
                            new_params[p_name] = rng.choice(p_vals)
                    try:
                        replacement = Leaf(prim_cls(**new_params))
                    except Exception:
                        replacement = _random_leaf(rng)
                else:
                    replacement = _random_leaf(rng)
            elif isinstance(target_node, UnaryNode):
                op_cls = target_node.operator.__class__
                if op_cls in UNARY_OPS:
                    param_grid = UNARY_OPS[op_cls]
                    curr_params = target_node.operator.params_dict()
                    new_params = copy.deepcopy(curr_params)
                    for p_name, p_vals in param_grid.items():
                        if len(p_vals) > 1:
                            new_params[p_name] = rng.choice(p_vals)
                    try:
                        new_op = op_cls(**new_params)
                        replacement = UnaryNode(new_op, copy.deepcopy(target_node.child))
                    except Exception:
                        replacement = copy.deepcopy(target_node)
                else:
                    replacement = copy.deepcopy(target_node)
            else:
                replacement = copy.deepcopy(target_node)

        else:  # subtree replacement
            allowed_depth = max(1, max_depth - 1)
            replacement = _random_subtree(current_depth=1, max_depth=allowed_depth, rng=rng)

        # Apply replacement
        mutated_expr = replace_subtree(expr, target_node, replacement)

        # Check contract validation & max depth constraint
        if mutated_expr.depth() <= max_depth:
            violations = validate_expression_vocabulary(mutated_expr)
            if not violations:
                return mutated_expr

    # Fallback to copy of original expression if retries exhausted
    return copy.deepcopy(expr)
