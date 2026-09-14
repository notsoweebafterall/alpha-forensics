"""
Crossover operators for Genetic Programming on Alpha Expression Trees.

Swaps subtrees between two parent expression trees while respecting depth limits
and vocabulary whitelist contracts.
"""

import copy
import random
from typing import Tuple, Optional

from alpha.expressions.tree import Expression
from alpha.generator.genetic.mutation import get_all_subtrees, replace_subtree
from agent.registry_introspection import validate_expression_vocabulary


def crossover_expressions(
    parent1: Expression,
    parent2: Expression,
    max_depth: int = 6,
    rng: Optional[random.Random] = None,
    max_retries: int = 10,
) -> Tuple[Expression, Expression]:
    """
    Performs subtree crossover between parent1 and parent2.

    Algorithm:
      1. Collects all subtree locations from parent1 and parent2.
      2. Chooses random subtrees sub1 in parent1 and sub2 in parent2.
      3. Swaps sub1 and sub2 to create child1 and child2.
      4. Validates max_depth constraint and vocabulary whitelist.
      5. Retries up to max_retries times; falls back to parent copies if constraints violated.

    Returns:
        Tuple[Expression, Expression]: Valid child1 and child2 expression trees.
    """
    if rng is None:
        rng = random.Random()

    p1_copy = copy.deepcopy(parent1)
    p2_copy = copy.deepcopy(parent2)

    subtrees1 = get_all_subtrees(p1_copy)
    subtrees2 = get_all_subtrees(p2_copy)

    if not subtrees1 or not subtrees2:
        return p1_copy, p2_copy

    for _ in range(max_retries):
        sub1 = rng.choice(subtrees1)
        sub2 = rng.choice(subtrees2)

        child1 = replace_subtree(p1_copy, sub1, sub2)
        child2 = replace_subtree(p2_copy, sub2, sub1)

        # Verify depth and vocabulary constraints
        valid1 = (child1.depth() <= max_depth) and (len(validate_expression_vocabulary(child1)) == 0)
        valid2 = (child2.depth() <= max_depth) and (len(validate_expression_vocabulary(child2)) == 0)

        if valid1 and valid2:
            return child1, child2
        elif valid1:
            return child1, p2_copy
        elif valid2:
            return p1_copy, child2

    return p1_copy, p2_copy
