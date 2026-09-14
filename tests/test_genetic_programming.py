"""
Unit tests for Genetic Programming operators (mutation, crossover, selection).
"""

import random
import pytest
from alpha.expressions.serialize import parse, to_string
from alpha.generator.genetic.mutation import mutate_expression, get_all_subtrees
from alpha.generator.genetic.crossover import crossover_expressions
from alpha.generator.genetic.population import (
    initialize_gp_population,
    Individual,
    select_tournament,
)
from agent.registry_introspection import validate_expression_vocabulary


def test_subtree_extraction():
    """Verify get_all_subtrees extracts all nodes from tree."""
    expr = parse("Rank(Multiply(Momentum(20), VolumeChange(5)))")
    subtrees = get_all_subtrees(expr)
    assert len(subtrees) == 4  # Rank, Multiply, Momentum, VolumeChange


def test_mutation_produces_valid_tree():
    """Verify mutate_expression produces valid expression trees respecting max_depth."""
    rng = random.Random(42)
    expr = parse("Rank(Multiply(Momentum(20), VolumeChange(5)))")

    for _ in range(20):
        mutated = mutate_expression(expr, max_depth=6, rng=rng)
        assert mutated.depth() <= 6
        violations = validate_expression_vocabulary(mutated)
        assert len(violations) == 0, f"Vocabulary violations: {violations}"


def test_crossover_produces_valid_trees():
    """Verify crossover_expressions produces valid offspring respecting max_depth."""
    rng = random.Random(42)
    p1 = parse("Rank(Multiply(Momentum(20), VolumeChange(5)))")
    p2 = parse("ZScore(Divide(RollingZScore(10), RealizedVolatility(20)))")

    for _ in range(20):
        c1, c2 = crossover_expressions(p1, p2, max_depth=6, rng=rng)
        assert c1.depth() <= 6
        assert c2.depth() <= 6
        assert len(validate_expression_vocabulary(c1)) == 0
        assert len(validate_expression_vocabulary(c2)) == 0


def test_tournament_selection():
    """Verify tournament selection picks individual with highest fitness."""
    pop = [
        Individual(candidate=None, expression_string="A", canonical_hash="h1", fitness=0.01, ic_mean=0.01, coverage=1.0, passed_screen=True, generation=1),
        Individual(candidate=None, expression_string="B", canonical_hash="h2", fitness=0.05, ic_mean=0.05, coverage=1.0, passed_screen=True, generation=1),
        Individual(candidate=None, expression_string="C", canonical_hash="h3", fitness=0.02, ic_mean=0.02, coverage=1.0, passed_screen=True, generation=1),
    ]

    selected = select_tournament(pop, k=3, rng=random.Random(42))
    assert selected.expression_string == "B"
    assert selected.fitness == 0.05


def test_behavioral_fingerprint_deduplication():
    """Verify compute_behavioral_fingerprint identifies monotonic rank equivalents."""
    from pathlib import Path
    from data.universe import UNIVERSE_60
    from data.panel import build_panel
    from alpha.generator.genetic.population import compute_behavioral_fingerprint

    panel = build_panel(tickers=UNIVERSE_60, start_date="2020-01-01", end_date="2023-12-31", cache_dir=Path("data/cache"))
    e1 = parse("Momentum(20)").evaluate(panel)
    e2 = parse("Rank(Momentum(20))").evaluate(panel)
    e3 = parse("SignedPower(Rank(Momentum(20)), 2.0)").evaluate(panel)
    e4 = parse("RealizedVolatility(20)").evaluate(panel)

    fp1 = compute_behavioral_fingerprint(e1)
    fp2 = compute_behavioral_fingerprint(e2)
    fp3 = compute_behavioral_fingerprint(e3)
    fp4 = compute_behavioral_fingerprint(e4)

    assert fp1 == fp2, "Momentum(20) and Rank(Momentum(20)) must produce identical behavioral fingerprints"
    assert fp2 == fp3, "Rank(Momentum(20)) and SignedPower(Rank(Momentum(20)), 2.0) must produce identical behavioral fingerprints"
    assert fp1 != fp4, "Momentum(20) and RealizedVolatility(20) must produce different behavioral fingerprints"


# ---------------------------------------------------------------------------
# Part 1: Constant-Literal Tests
# ---------------------------------------------------------------------------

def test_constant_parse_target_expression():
    """
    The exact expression that was previously rejected now parses successfully.
    Divide(Rank(Drawdown(20)), Add(1, RealizedVolatility(60))) contains the bare
    integer literal 1 in an expression-argument position inside Add.
    """
    expr_str = "Divide(Rank(Drawdown(20)), Add(1, RealizedVolatility(60)))"
    expr = parse(expr_str)
    assert expr is not None
    # Confirm round-trip
    rt = expr.to_string()
    # The integer 1 round-trips as Constant(1) wrapped in Add
    assert "Constant(1" in rt or "Constant(1.0" in rt, f"Unexpected round-trip: {rt}"


def test_constant_canonical_hash_deterministic():
    """canonical_hash() must be identical for the same expression parsed twice."""
    expr_str = "Divide(Rank(Drawdown(20)), Add(Constant(1.0), RealizedVolatility(60)))"
    h1 = parse(expr_str).canonical_hash()
    h2 = parse(expr_str).canonical_hash()
    assert h1 == h2, "canonical_hash must be deterministic across two parse calls"


def test_constant_distinct_from_different_value():
    """Two Constant nodes with different values must produce different hashes."""
    h1 = parse("Add(Constant(1.0), Momentum(20))").canonical_hash()
    h2 = parse("Add(Constant(2.0), Momentum(20))").canonical_hash()
    assert h1 != h2, "Constant(1.0) and Constant(2.0) must yield different hashes"


def test_constant_bounds_rejection():
    """ConstantNode must reject zero and out-of-range values."""
    from alpha.expressions.tree import ConstantNode
    import pytest as _pytest
    with _pytest.raises(ValueError, match="0.0 is not allowed"):
        ConstantNode(0.0)
    with _pytest.raises(ValueError, match="near-zero"):
        ConstantNode(1e-8)
    with _pytest.raises(ValueError, match="unreasonably large"):
        ConstantNode(1e6)
    # In-bounds values must not raise
    ConstantNode(1.0)
    ConstantNode(-0.5)
    ConstantNode(999.0)


def test_constant_vocabulary_validation():
    """validate_expression_vocabulary must accept in-bounds constants and reject out-of-bounds."""
    valid_expr = parse("Add(Constant(1.0), Momentum(20))")
    violations = validate_expression_vocabulary(valid_expr)
    assert len(violations) == 0, f"Valid constant expression has violations: {violations}"

