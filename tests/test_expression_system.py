import pytest
import numpy as np
import pandas as pd

from core.panel import Panel
from core.signal import validate_signal
from data.universe import UNIVERSE_60
from data.panel import build_panel

import features.returns as f_returns
import features.rolling as f_rolling
import features.volume as f_volume
import features.risk as f_risk

import alpha.primitives.inputs as in_prim
import alpha.primitives.operators as op_prim
from alpha.expressions.tree import Leaf, UnaryNode, BinaryNode
from alpha.expressions.constraints import MAX_DEPTH, check_depth
from alpha.expressions.dedup import canonical_hash
from alpha.expressions.serialize import to_string, parse
from backtesting.engine import run_backtest, BacktestConfig


@pytest.fixture
def toy_panel():
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
# 1. Leaf Evaluation Tests
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "leaf_obj, feature_fn, kwargs",
    [
        (Leaf(in_prim.PriceReturn(1)), f_returns.simple_return, {"lag": 1}),
        (Leaf(in_prim.Momentum(3)), f_returns.momentum, {"lookback": 3}),
        (Leaf(in_prim.RollingMean(3)), f_rolling.rolling_mean, {"window": 3}),
        (Leaf(in_prim.RollingStd(3)), f_rolling.rolling_std, {"window": 3}),
        (Leaf(in_prim.RollingZScore(3)), f_rolling.rolling_zscore, {"window": 3}),
        (Leaf(in_prim.RollingRank(3)), f_rolling.rolling_rank, {"window": 3}),
        (Leaf(in_prim.VolumeChange(1)), f_volume.volume_change, {"lookback": 1}),
        (Leaf(in_prim.Turnover(3)), f_volume.turnover, {"lookback": 3}),
        (Leaf(in_prim.RealizedVolatility(3)), f_risk.realized_volatility, {"window": 3}),
        (Leaf(in_prim.Drawdown(3)), f_risk.drawdown, {"window": 3}),
    ],
)
def test_leaf_evaluation(toy_panel, leaf_obj, feature_fn, kwargs):
    expr_signal = leaf_obj.evaluate(toy_panel)
    expected_df = feature_fn(toy_panel, **kwargs)
    pd.testing.assert_frame_equal(expr_signal, expected_df)


# -------------------------------------------------------------------
# 2. Operator Correctness Tests
# -------------------------------------------------------------------

def test_operator_hand_calculated(toy_panel):
    df_a = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0]})
    df_b = pd.DataFrame({"X": [10.0, 20.0, 30.0], "Y": [40.0, 50.0, 60.0]})

    # Unary
    pd.testing.assert_frame_equal(op_prim.Negate().apply(df_a), -df_a)
    pd.testing.assert_frame_equal(op_prim.Abs().apply(-df_a), df_a)
    pd.testing.assert_frame_equal(op_prim.Lag(1).apply(df_a), df_a.shift(1))
    pd.testing.assert_frame_equal(op_prim.Diff(1).apply(df_a), df_a.diff(1))
    pd.testing.assert_frame_equal(
        op_prim.SignedPower(2.0).apply(pd.DataFrame({"X": [-2.0, 3.0]})),
        pd.DataFrame({"X": [-4.0, 9.0]}),
    )

    # Binary
    pd.testing.assert_frame_equal(op_prim.Add().apply(df_a, df_b), df_a + df_b)
    pd.testing.assert_frame_equal(op_prim.Subtract().apply(df_b, df_a), df_b - df_a)
    pd.testing.assert_frame_equal(op_prim.Multiply().apply(df_a, df_b), df_a * df_b)


# -------------------------------------------------------------------
# 3. Safe Division Test
# -------------------------------------------------------------------

def test_safe_division(toy_panel):
    # Construct denominator with exact 0.0 values
    zero_df = pd.DataFrame(0.0, index=toy_panel.prices.index, columns=toy_panel.prices.columns)
    numer_df = pd.DataFrame(1.0, index=toy_panel.prices.index, columns=toy_panel.prices.columns)

    div_op = op_prim.Divide()
    res = div_op.apply(numer_df, zero_df)

    # All outputs must be NaN (never inf or -inf)
    assert res.isna().all().all()
    assert not np.isinf(res.to_numpy()).any()

    # Wrap in expression and evaluate
    node = BinaryNode(
        op_prim.Divide(),
        Leaf(in_prim.Momentum(2)),
        Leaf(in_prim.PriceReturn(1)),  # Has zeros/NaNs
    )
    signal = node.evaluate(toy_panel)

    # Must pass validate_signal without raising inf error
    validate_signal(signal, toy_panel)


# -------------------------------------------------------------------
# 4. Depth Limit Enforcement Tests
# -------------------------------------------------------------------

def test_depth_limit():
    l1 = Leaf(in_prim.Momentum(2))
    l2 = Leaf(in_prim.PriceReturn(1))

    # Depth 1
    assert l1.depth() == 1
    check_depth(l1)

    # Depth 2
    d2 = BinaryNode(op_prim.Add(), l1, l2)
    assert d2.depth() == 2
    check_depth(d2)

    # Depth 3
    d3 = UnaryNode(op_prim.Rank(), d2)
    assert d3.depth() == 3
    check_depth(d3)

    # Depth 4
    d4 = UnaryNode(op_prim.ZScore(), d3)
    assert d4.depth() == 4
    check_depth(d4)  # Passes exactly at MAX_DEPTH=4

    # Depth 5 -> Should raise ValueError
    d5 = UnaryNode(op_prim.Abs(), d4)
    assert d5.depth() == 5
    with pytest.raises(ValueError, match="exceeds maximum allowed depth"):
        check_depth(d5)


# -------------------------------------------------------------------
# 5. Canonical Hash Deduplication Tests
# -------------------------------------------------------------------

def test_canonical_hash_commutative():
    a = Leaf(in_prim.Momentum(20))
    b = Leaf(in_prim.RealizedVolatility(10))

    # Add(A, B) vs Add(B, A)
    add_ab = BinaryNode(op_prim.Add(), a, b)
    add_ba = BinaryNode(op_prim.Add(), b, a)
    assert add_ab.canonical_hash() == add_ba.canonical_hash()

    # Multiply(A, B) vs Multiply(B, A)
    mul_ab = BinaryNode(op_prim.Multiply(), a, b)
    mul_ba = BinaryNode(op_prim.Multiply(), b, a)
    assert mul_ab.canonical_hash() == mul_ba.canonical_hash()


def test_canonical_hash_non_commutative():
    a = Leaf(in_prim.Momentum(20))
    b = Leaf(in_prim.RealizedVolatility(10))

    # Subtract(A, B) vs Subtract(B, A)
    sub_ab = BinaryNode(op_prim.Subtract(), a, b)
    sub_ba = BinaryNode(op_prim.Subtract(), b, a)
    assert sub_ab.canonical_hash() != sub_ba.canonical_hash()

    # Divide(A, B) vs Divide(B, A)
    div_ab = BinaryNode(op_prim.Divide(), a, b)
    div_ba = BinaryNode(op_prim.Divide(), b, a)
    assert div_ab.canonical_hash() != div_ba.canonical_hash()


def test_canonical_hash_distinct_parameters():
    # Momentum(20) vs Momentum(252) MUST produce distinct hashes!
    m20 = Leaf(in_prim.Momentum(20))
    m252 = Leaf(in_prim.Momentum(252))

    assert m20.canonical_hash() != m252.canonical_hash()

    # Diff(1) vs Diff(5)
    d1 = UnaryNode(op_prim.Diff(1), m20)
    d5 = UnaryNode(op_prim.Diff(5), m20)
    assert d1.canonical_hash() != d5.canonical_hash()


def test_canonical_hash_distinct_structures():
    e1 = UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(20)))
    e2 = UnaryNode(op_prim.ZScore(), Leaf(in_prim.Momentum(20)))
    assert e1.canonical_hash() != e2.canonical_hash()


# -------------------------------------------------------------------
# 6. Serialization Round-Trip Tests
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "expr",
    [
        Leaf(in_prim.Momentum(126)),
        Leaf(in_prim.PriceReturn(1)),
        UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(20))),
        UnaryNode(op_prim.SignedPower(0.5), Leaf(in_prim.Turnover(5))),
        UnaryNode(op_prim.Lag(2), Leaf(in_prim.RealizedVolatility(10))),
        BinaryNode(op_prim.Divide(), Leaf(in_prim.Momentum(126)), Leaf(in_prim.RealizedVolatility(20))),
        BinaryNode(
            op_prim.Add(),
            UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(10))),
            UnaryNode(op_prim.ZScore(), Leaf(in_prim.Turnover(5))),
        ),
    ],
)
def test_serialization_roundtrip(toy_panel, expr):
    s = to_string(expr)
    parsed_expr = parse(s)

    # Evaluate both original and parsed trees
    sig_orig = expr.evaluate(toy_panel)
    sig_parsed = parsed_expr.evaluate(toy_panel)

    pd.testing.assert_frame_equal(sig_orig, sig_parsed)
    assert canonical_hash(expr) == canonical_hash(parsed_expr)


# -------------------------------------------------------------------
# 7. End-to-End Real Data Pipeline Test
# -------------------------------------------------------------------

def test_end_to_end_real_data():
    # Construct moderately complex expression: Rank(Divide(Momentum(5), RealizedVolatility(3)))
    expr = UnaryNode(
        op_prim.Rank(),
        BinaryNode(
            op_prim.Divide(),
            Leaf(in_prim.Momentum(3)),
            Leaf(in_prim.RealizedVolatility(3)),
        ),
    )

    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    tickers = UNIVERSE_60

    np.random.seed(42)
    p_data = np.random.uniform(50.0, 150.0, size=(10, 60))
    v_data = np.random.uniform(1000, 5000, size=(10, 60))

    prices_df = pd.DataFrame(p_data, index=dates, columns=tickers)
    volume_df = pd.DataFrame(v_data, index=dates, columns=tickers)

    panel = build_panel(
        tickers=tickers,
        start_date=dates[0],
        end_date=dates[-1],
        missing_threshold=0.05,
        prices_df=prices_df,
        volume_df=volume_df,
    )

    # 1. Evaluate expression tree
    signal = expr.evaluate(panel)
    assert signal.shape == (10, 60)

    # 2. Pass signal to Phase 2 Backtester Engine
    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    res = run_backtest(signal, panel, config)

    assert res.weights.shape == (10, 60)
    assert len(res.net_returns) == 10
