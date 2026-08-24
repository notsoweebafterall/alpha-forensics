import pytest
import numpy as np
import pandas as pd

from core.signal import validate_signal
from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.expressions.constraints import MAX_DEPTH
from alpha.strategies.registry import get_all_strategies, get_strategy
from backtesting.engine import run_backtest, BacktestConfig


@pytest.fixture
def synthetic_panel_phase4():
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
# 1. Catalog & Registry Tests
# -------------------------------------------------------------------

def test_registry_catalog():
    strategies = get_all_strategies()
    assert len(strategies) == 5

    names = [s.name for s in strategies]
    assert "cross_sectional_momentum" in names
    assert "short_term_reversal" in names
    assert "volatility_adjusted_momentum" in names
    assert "volume_confirmed_momentum" in names
    assert "low_volatility" in names

    for strat in strategies:
        assert isinstance(strat.hypothesis_text, str)
        assert len(strat.hypothesis_text.strip()) > 10
        assert isinstance(strat.default_params, dict)
        assert isinstance(strat.variant_params, dict)


def test_registry_unknown_strategy():
    with pytest.raises(KeyError, match="Strategy 'unknown_strat' not found"):
        get_strategy("unknown_strat")


# -------------------------------------------------------------------
# 2. Expression Depth Tests
# -------------------------------------------------------------------

def test_strategy_expression_depth():
    strategies = get_all_strategies()
    for strat in strategies:
        expr = strat.build(strat.default_params)
        depth = expr.depth()
        assert depth <= MAX_DEPTH, f"Strategy {strat.name} depth ({depth}) exceeds MAX_DEPTH ({MAX_DEPTH})"


# -------------------------------------------------------------------
# 3. Signal Validation Tests
# -------------------------------------------------------------------

def test_strategy_signal_validation(synthetic_panel_phase4):
    panel = synthetic_panel_phase4
    strategies = get_all_strategies()

    for strat in strategies:
        expr = strat.build(strat.default_params)
        signal = expr.evaluate(panel)

        # Output shape matches panel
        assert signal.shape == panel.prices.shape

        # Root evaluate() automatically validates signal, but call explicitly as well
        validate_signal(signal, panel)


# -------------------------------------------------------------------
# 4. Variant Hashing Distinction Tests
# -------------------------------------------------------------------

def test_strategy_variant_hashes():
    strategies = get_all_strategies()

    for strat in strategies:
        expr1 = strat.build(strat.default_params)

        # Build modified variant params
        mod_params = strat.default_params.copy()
        first_key = list(strat.variant_params.keys())[0]
        # Pick a variant value different from default
        alt_val = [v for v in strat.variant_params[first_key] if v != mod_params[first_key]][0]
        mod_params[first_key] = alt_val

        expr2 = strat.build(mod_params)

        hash1 = expr1.canonical_hash()
        hash2 = expr2.canonical_hash()

        assert hash1 != hash2, f"Strategy {strat.name} variants produced identical canonical hash!"


# -------------------------------------------------------------------
# 5. End-to-End Backtest Integration Tests
# -------------------------------------------------------------------

def test_strategy_end_to_end_backtest(synthetic_panel_phase4):
    panel = synthetic_panel_phase4
    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    strategies = get_all_strategies()

    for strat in strategies:
        expr = strat.build(strat.default_params)
        signal = expr.evaluate(panel)
        res = run_backtest(signal, panel, config)

        assert res.weights.shape == panel.prices.shape
        assert len(res.net_returns) == len(panel.prices)
        assert len(res.turnover) == len(panel.prices)
