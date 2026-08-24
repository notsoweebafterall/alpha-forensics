import pytest
import numpy as np
import pandas as pd

from core.signal import validate_signal
from data.universe import UNIVERSE_60
from data.panel import build_panel

from alpha.expressions.tree import Leaf, UnaryNode, BinaryNode
from alpha.expressions.constraints import MAX_DEPTH
import alpha.primitives.inputs as in_prim
import alpha.primitives.operators as op_prim
from alpha.hypotheses.spec import HypothesisSpec
from alpha.hypotheses.translator import translate
from alpha.hypotheses.registry import get_all_hypotheses, get_hypothesis
from backtesting.engine import run_backtest, BacktestConfig


@pytest.fixture
def synthetic_panel_phase5():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    prices_data = {
        "AAPL": [100.0, 102.0, 105.0, 100.0, 104.0, 108.0, 106.0, 110.0, 109.0, 112.0],
        "MSFT": [200.0, 204.0, 200.0, 210.0, 208.0, 212.0, 215.0, 220.0, 218.0, 225.0],
        "GOOGL": [300.0, 297.0, 300.0, 303.0, 306.0, 300.0, 309.0, 312.0, 315.0, 320.0],
    }
    volume_data = {
        "AAPL": [1000, 1100, 1200, 900, 1300, 1400, 1150, 1500, 1250, 1600],
        "MSFT": [2000, 2100, 1900, 2200, 2000, 2300, 2400, 2500, 2350, 2600],
        "GOOGL": [1500, 1450, 1500, 1600, 1550, 1400, 1650, 1700, 1600, 1750],
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
# 1. Spec Validation Tests
# -------------------------------------------------------------------

def test_spec_validation_empty_fields():
    # Empty hypothesis_text
    with pytest.raises(ValueError, match="hypothesis_text cannot be empty"):
        HypothesisSpec(
            hypothesis_id="hyp_0",
            hypothesis_text="  ",
            economic_rationale="Rationale text",
            composition="Momentum(10)",
            source_features=["Momentum"],
        )

    # Empty economic_rationale
    with pytest.raises(ValueError, match="economic_rationale cannot be empty"):
        HypothesisSpec(
            hypothesis_id="hyp_0",
            hypothesis_text="Valid text",
            economic_rationale="",
            composition="Momentum(10)",
            source_features=["Momentum"],
        )


# -------------------------------------------------------------------
# 2. Registry Catalog & Lookup Tests
# -------------------------------------------------------------------

def test_registry_catalog_and_lookup():
    hyps = get_all_hypotheses()
    assert len(hyps) == 3

    h1 = get_hypothesis("hyp_momentum_vol_decay")
    assert h1.hypothesis_id == "hyp_momentum_vol_decay"

    with pytest.raises(KeyError, match="not found in registry catalog"):
        get_hypothesis("unknown_hyp_id")


# -------------------------------------------------------------------
# 3. Translator Hash Equality Tests
# -------------------------------------------------------------------

def test_translate_hash_equality():
    # 1. hyp_momentum_vol_decay: Multiply(Momentum(126), Negate(Diff(RealizedVolatility(20), 5)))
    spec1 = get_hypothesis("hyp_momentum_vol_decay")
    expr1_translated = translate(spec1)
    expr1_manual = BinaryNode(
        op_prim.Multiply(),
        Leaf(in_prim.Momentum(126)),
        UnaryNode(
            op_prim.Negate(),
            UnaryNode(op_prim.Diff(5), Leaf(in_prim.RealizedVolatility(20))),
        ),
    )
    assert expr1_translated.canonical_hash() == expr1_manual.canonical_hash()

    # 2. hyp_reversal_volume_spike: Multiply(Rank(Negate(Momentum(5))), Rank(VolumeChange(5)))
    spec2 = get_hypothesis("hyp_reversal_volume_spike")
    expr2_translated = translate(spec2)
    expr2_manual = BinaryNode(
        op_prim.Multiply(),
        UnaryNode(
            op_prim.Rank(),
            UnaryNode(op_prim.Negate(), Leaf(in_prim.Momentum(5))),
        ),
        UnaryNode(op_prim.Rank(), Leaf(in_prim.VolumeChange(5))),
    )
    assert expr2_translated.canonical_hash() == expr2_manual.canonical_hash()

    # 3. hyp_zscore_reversal_low_vol: Multiply(Rank(Negate(RollingZScore(20))), Rank(Negate(RealizedVolatility(60))))
    spec3 = get_hypothesis("hyp_zscore_reversal_low_vol")
    expr3_translated = translate(spec3)
    expr3_manual = BinaryNode(
        op_prim.Multiply(),
        UnaryNode(
            op_prim.Rank(),
            UnaryNode(op_prim.Negate(), Leaf(in_prim.RollingZScore(20))),
        ),
        UnaryNode(
            op_prim.Rank(),
            UnaryNode(op_prim.Negate(), Leaf(in_prim.RealizedVolatility(60))),
        ),
    )
    assert expr3_translated.canonical_hash() == expr3_manual.canonical_hash()


# -------------------------------------------------------------------
# 4. Provenance Consistency Check Test
# -------------------------------------------------------------------

def test_translate_provenance_check():
    # Composition uses 'RollingRank', but source_features leaves it out
    bad_spec = HypothesisSpec(
        hypothesis_id="bad_provenance",
        hypothesis_text="Undeclared primitive test",
        economic_rationale="Test rationale",
        composition="Rank(RollingRank(20))",
        source_features=["Rank"],  # Missing 'RollingRank'
    )
    with pytest.raises(ValueError, match="Provenance consistency check failed"):
        translate(bad_spec)


# -------------------------------------------------------------------
# 5. Malformed Syntax & Depth Limit Tests
# -------------------------------------------------------------------

def test_translate_malformed_syntax():
    # Unbalanced parens
    bad_syntax_spec = HypothesisSpec(
        hypothesis_id="bad_syntax",
        hypothesis_text="Malformed syntax test",
        economic_rationale="Test rationale",
        composition="Rank(Momentum(20",
        source_features=["Rank", "Momentum"],
    )
    with pytest.raises(ValueError, match="Failed to parse composition string"):
        translate(bad_syntax_spec)

    # Unknown primitive
    unknown_prim_spec = HypothesisSpec(
        hypothesis_id="unknown_prim",
        hypothesis_text="Unknown primitive test",
        economic_rationale="Test rationale",
        composition="NonExistentPrimitive(20)",
        source_features=["NonExistentPrimitive"],
    )
    with pytest.raises(ValueError, match="Failed to parse composition string"):
        translate(unknown_prim_spec)


def test_translate_depth_limit():
    # Construct composition string exceeding MAX_DEPTH (depth 5 > 4)
    deep_composition = "Rank(ZScore(Rank(ZScore(Momentum(10)))))"
    deep_spec = HypothesisSpec(
        hypothesis_id="deep_hyp",
        hypothesis_text="Deep expression test",
        economic_rationale="Test rationale",
        composition=deep_composition,
        source_features=["Rank", "ZScore", "Momentum"],
    )
    with pytest.raises(ValueError, match="exceeds maximum allowed depth"):
        translate(deep_spec)


# -------------------------------------------------------------------
# 6. End-to-End Backtest Test
# -------------------------------------------------------------------

def test_hypotheses_end_to_end_backtest(synthetic_panel_phase5):
    panel = synthetic_panel_phase5
    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)

    for hyp in get_all_hypotheses():
        expr = translate(hyp)
        signal = expr.evaluate(panel)

        # Output shape matches panel
        assert signal.shape == panel.prices.shape

        # Signal passes validation
        validate_signal(signal, panel)

        # Run through Phase 2 backtest engine
        res = run_backtest(signal, panel, config)

        assert res.weights.shape == panel.prices.shape
        assert len(res.net_returns) == len(panel.prices)
