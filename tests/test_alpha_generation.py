import pytest
import numpy as np
import pandas as pd

from core.panel import Panel
from core.signal import validate_signal
from data.universe import UNIVERSE_60
from data.panel import build_panel

import alpha.primitives.inputs as in_prim
import alpha.primitives.operators as op_prim
from alpha.expressions.tree import Expression, Leaf, UnaryNode, BinaryNode
from alpha.expressions.serialize import to_string

from alpha.generator.vocabulary import LEAF_PRIMITIVES, UNARY_OPS, BINARY_OPS
from alpha.generator.candidate import GeneratedCandidate
from alpha.generator.variant_expansion import expand_variants
from alpha.generator.compositional_generator import generate_random_candidates
from alpha.generator.screening import screen_candidates
from alpha.strategies.registry import get_strategy


@pytest.fixture
def synthetic_panel_phase6():
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    prices_data = {
        "AAPL": [100.0 + i * 0.5 for i in range(15)],
        "MSFT": [200.0 - i * 0.2 for i in range(15)],
        "GOOGL": [300.0 + (i % 3) * 2.0 for i in range(15)],
    }
    volume_data = {
        "AAPL": [1000 + i * 50 for i in range(15)],
        "MSFT": [2000 - i * 30 for i in range(15)],
        "GOOGL": [1500 + (i % 2) * 100 for i in range(15)],
    }
    prices_df = pd.DataFrame(prices_data, index=dates)
    volume_df = pd.DataFrame(volume_data, index=dates)

    return build_panel(
        tickers=tickers,
        start_date=dates[0],
        end_date=dates[-1],
        missing_threshold=0.05,
        prices_df=prices_df,
        volume_df=volume_df,
    )


# -------------------------------------------------------------------
# 1. Vocabulary Alignment Test
# -------------------------------------------------------------------

def test_vocabulary_alignment():
    # Verify LEAF_PRIMITIVES contains valid subclasses of PrimitiveInput
    for cls in LEAF_PRIMITIVES.keys():
        assert issubclass(cls, in_prim.PrimitiveInput)

    # Verify UNARY_OPS contains valid subclasses of Operator
    for cls in UNARY_OPS.keys():
        assert issubclass(cls, op_prim.Operator)

    # Verify BINARY_OPS contains valid subclasses of Operator
    for cls in BINARY_OPS:
        assert issubclass(cls, op_prim.Operator)


# -------------------------------------------------------------------
# 2. Variant Expansion Tests
# -------------------------------------------------------------------

def test_variant_expansion():
    def mock_build(params):
        return UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(params["lookback"])))

    param_grid = {"lookback": [20, 60]}
    candidates = expand_variants(mock_build, param_grid, parent_family="momentum")

    assert len(candidates) == 2
    hashes = [c.canonical_hash() for c in candidates]
    assert len(set(hashes)) == 2
    assert candidates[0].generation_method == "variant_expansion"
    assert candidates[0].parent_family == "momentum"


def test_variant_expansion_dedup_collapse():
    # Mock build function where parameters do not affect the tree structure
    def mock_collapsing_build(params):
        return UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(20)))

    param_grid = {"dummy_param": [1, 2, 3, 4]}
    candidates = expand_variants(mock_collapsing_build, param_grid, parent_family="test")

    # Should collapse all 4 parameter combos into 1 unique candidate
    assert len(candidates) == 1


# -------------------------------------------------------------------
# 3. Compositional Generation Reproducibility & Constraint Tests
# -------------------------------------------------------------------

def test_compositional_reproducibility():
    run1 = generate_random_candidates(budget=20, max_depth=3, seed=7)
    run2 = generate_random_candidates(budget=20, max_depth=3, seed=7)

    assert len(run1) == 20
    assert len(run2) == 20

    for c1, c2 in zip(run1, run2):
        assert c1.expression_string == c2.expression_string
        assert c1.canonical_hash() == c2.canonical_hash()


def test_compositional_max_depth():
    candidates = generate_random_candidates(budget=15, max_depth=3, seed=123)
    for c in candidates:
        assert c.expression.depth() <= 3


def test_compositional_no_duplicates():
    candidates = generate_random_candidates(budget=30, max_depth=4, seed=99)
    hashes = [c.canonical_hash() for c in candidates]
    assert len(hashes) == len(set(hashes))


def test_compositional_budget_exhaustion(caplog):
    # Set impossible/tight parameters: max_depth=1 with large budget
    candidates = generate_random_candidates(budget=500, max_depth=1, seed=42, max_attempts=50)

    # Max unique depth=1 expressions is limited by number of Leaf primitives & params
    assert len(candidates) < 500
    assert len(candidates) > 0


# -------------------------------------------------------------------
# 4. Screening Tests
# -------------------------------------------------------------------

def test_screening_rejections(synthetic_panel_phase6):
    panel = synthetic_panel_phase6

    # 1. Degenerate candidate (constant signal)
    # PriceReturn(0) if supported, or Rank of constant
    class ConstantExpr(Expression):
        def _evaluate_raw(self, p):
            return pd.DataFrame(1.0, index=p.prices.index, columns=p.prices.columns)
        def depth(self): return 1
        def to_string(self): return "Constant"

    const_cand = GeneratedCandidate(
        expression=ConstantExpr(),
        expression_string="Constant",
        generation_method="test",
        seed=None,
        parent_family=None,
        param_values={},
        generated_at=pd.Timestamp.now(),
    )

    # 2. Insufficient coverage candidate (all NaNs)
    class NaNExpr(Expression):
        def _evaluate_raw(self, p):
            return pd.DataFrame(np.nan, index=p.prices.index, columns=p.prices.columns)
        def depth(self): return 1
        def to_string(self): return "NaN"

    nan_cand = GeneratedCandidate(
        expression=NaNExpr(),
        expression_string="NaN",
        generation_method="test",
        seed=None,
        parent_family=None,
        param_values={},
        generated_at=pd.Timestamp.now(),
    )

    passed, rejected = screen_candidates([const_cand, nan_cand], panel, min_coverage=0.8)

    assert len(passed) == 0
    assert len(rejected) == 2

    reasons = [reason for _, reason in rejected]
    assert "degenerate_signal" in reasons or "weak_signal" in reasons
    assert "insufficient_coverage" in reasons or "invalid_signal" in reasons


def test_screening_pass(synthetic_panel_phase6):
    panel = synthetic_panel_phase6
    strat = get_strategy("cross_sectional_momentum")
    expr = strat.build({"lookback": 2})

    cand = GeneratedCandidate(
        expression=expr,
        expression_string=to_string(expr),
        generation_method="variant_expansion",
        seed=None,
        parent_family="cross_sectional_momentum",
        param_values={"lookback": 2},
        generated_at=pd.Timestamp.now(),
    )

    passed, rejected = screen_candidates([cand], panel, ic_threshold=-1.0)  # low threshold to pass
    assert len(passed) == 1
    assert len(rejected) == 0
