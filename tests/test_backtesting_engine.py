import pytest
import numpy as np
import pandas as pd

from core.panel import Panel
from core.signal import validate_signal
from data.universe import UNIVERSE_60
from data.panel import build_panel
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel, compute_costs
from backtesting.engine import (
    BacktestResult,
    signal_to_weights,
    align_forward_returns,
    run_backtest,
)
from backtesting.metrics import economic_metrics, predictive_metrics


@pytest.fixture
def synthetic_panel_phase2():
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    prices_data = {
        "AAPL": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        "MSFT": [200.0, 198.0, 202.0, 200.0, 204.0, 206.0],
        "GOOGL": [300.0, 303.0, 300.0, 306.0, 303.0, 309.0],
    }
    prices_df = pd.DataFrame(prices_data, index=dates)

    volume_data = {
        "AAPL": [1000, 1000, 1000, 1000, 1000, 1000],
        "MSFT": [2000, 2000, 2000, 2000, 2000, 2000],
        "GOOGL": [1500, 1500, 1500, 1500, 1500, 1500],
    }
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
# 1. Point-in-Time & Execution Timing Tests
# -------------------------------------------------------------------

def test_execution_timing(synthetic_panel_phase2):
    panel = synthetic_panel_phase2
    # Create signal where Day 0 = [AAPL=1, MSFT=2, GOOGL=3]
    # Target weights on Day 0 (dollar neutral, ranks [1, 2, 3], mean 2):
    # AAPL = (1-2)/2 = -0.5, MSFT = (2-2)/2 = 0.0, GOOGL = (3-2)/2 = 0.5
    signal = pd.DataFrame(
        {
            "AAPL": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "MSFT": [2.0, 1.0, 2.0, 2.0, 1.0, 2.0],
            "GOOGL": [3.0, 3.0, 1.0, 3.0, 3.0, 1.0],
        },
        index=panel.prices.index,
    )

    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    res = run_backtest(signal, panel, config)

    # On Day 0 (start date), actual weights held must be 0.0 because of execution lag
    assert (res.weights.iloc[0] == 0.0).all()

    # On Day 1, actual weights held must match target weights from Day 0
    target_weights_day0 = signal_to_weights(signal, config).iloc[0]
    pd.testing.assert_series_equal(res.weights.iloc[1], target_weights_day0, check_names=False)


def test_no_lookahead(synthetic_panel_phase2):
    panel_base = synthetic_panel_phase2
    dates = panel_base.prices.index
    tickers = panel_base.universe

    # Signal computed up to date t=3
    signal = pd.DataFrame(
        {
            "AAPL": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "MSFT": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            "GOOGL": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        },
        index=dates,
    )

    # Base backtest
    res_base = run_backtest(signal, panel_base)

    # Future modified panel: Extreme price jump on Day 4 & Day 5
    future_prices = panel_base.prices.copy()
    future_prices.iloc[4:] = 99999.0

    panel_future = build_panel(
        tickers=tickers,
        start_date=dates[0],
        end_date=dates[-1],
        missing_threshold=0.05,
        prices_df=future_prices,
        volume_df=panel_base.volume,
    )

    res_future = run_backtest(signal, panel_future)

    # Up to Day 4 open (weights at Day 4), positions held must be identical
    pd.testing.assert_frame_equal(res_base.weights.iloc[:4], res_future.weights.iloc[:4])


# -------------------------------------------------------------------
# 2. Costs & Turnover Tests
# -------------------------------------------------------------------

def test_costs_zero_turnover(synthetic_panel_phase2):
    panel = synthetic_panel_phase2
    # Constant signal across all days
    signal = pd.DataFrame(
        {
            "AAPL": [1.0] * 6,
            "MSFT": [2.0] * 6,
            "GOOGL": [3.0] * 6,
        },
        index=panel.prices.index,
    )

    config = BacktestConfig(execution_lag_days=1)
    cost_model = CostModel(cost_bps=10.0)
    res = run_backtest(signal, panel, config, cost_model)

    # Day 0: weight=0
    # Day 1: initial entry (turnover > 0, cost > 0)
    # Day 2 to Day 5: no rebalancing -> turnover=0, costs=0
    assert (res.turnover.iloc[2:] == 0.0).all()
    assert (res.costs.iloc[2:] == 0.0).all()


def test_costs_scale_with_turnover(synthetic_panel_phase2):
    panel = synthetic_panel_phase2
    dates = panel.prices.index

    # Day 0: AAPL=1, MSFT=2, GOOGL=NaN -> Target Day 0: AAPL=-0.5, MSFT=0.5, GOOGL=0.0
    # Day 1: AAPL=2, MSFT=1, GOOGL=NaN -> Target Day 1: AAPL=0.5, MSFT=-0.5, GOOGL=0.0
    signal = pd.DataFrame(
        {
            "AAPL": [1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "MSFT": [2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "GOOGL": [np.nan] * 6,
        },
        index=dates,
    )

    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True, gross_exposure=1.0)
    cost_model = CostModel(cost_bps=10.0)  # 10 bps = 0.001 fraction

    res = run_backtest(signal, panel, config, cost_model)

    # Actual weights:
    # Day 0: [0, 0, 0]
    # Day 1: [-0.5, 0.5, 0.0]  -> weight change = |-0.5| + |0.5| = 1.0 -> cost = 1.0 * 0.001 = 0.001
    # Day 2: [0.5, -0.5, 0.0]   -> weight change = |0.5 - (-0.5)| + |-0.5 - 0.5| = 2.0 -> cost = 2.0 * 0.001 = 0.002

    assert pytest.approx(res.costs.iloc[1]) == 0.001
    assert pytest.approx(res.costs.iloc[2]) == 0.002


# -------------------------------------------------------------------
# 3. Position Sizing, Dollar Neutrality, and NaN Handling Tests
# -------------------------------------------------------------------

def test_dollar_neutral(synthetic_panel_phase2):
    panel = synthetic_panel_phase2
    signal = pd.DataFrame(
        {
            "AAPL": [10.0, 5.0, 1.0, 8.0, 3.0, 6.0],
            "MSFT": [20.0, 15.0, 2.0, 4.0, 9.0, 1.0],
            "GOOGL": [30.0, 25.0, 3.0, 12.0, 1.0, 10.0],
        },
        index=panel.prices.index,
    )

    config = BacktestConfig(dollar_neutral=True)
    target_w = signal_to_weights(signal, config)

    # Sum of weights across tickers on each date must be ~0.0
    for dt in target_w.index:
        assert pytest.approx(target_w.loc[dt].sum(), abs=1e-7) == 0.0


def test_gross_exposure(synthetic_panel_phase2):
    panel = synthetic_panel_phase2
    signal = pd.DataFrame(
        {
            "AAPL": [10.0, 5.0, 1.0, 8.0, 3.0, 6.0],
            "MSFT": [20.0, 15.0, 2.0, 4.0, 9.0, 1.0],
            "GOOGL": [30.0, 25.0, 3.0, 12.0, 1.0, 10.0],
        },
        index=panel.prices.index,
    )

    config = BacktestConfig(gross_exposure=1.5)
    target_w = signal_to_weights(signal, config)

    # Sum of abs weights across tickers on each date must equal gross_exposure
    for dt in target_w.index:
        assert pytest.approx(target_w.loc[dt].abs().sum(), abs=1e-7) == 1.5


def test_missing_signal_handling(synthetic_panel_phase2):
    panel = synthetic_panel_phase2
    # Signal with NaNs in GOOGL
    signal = pd.DataFrame(
        {
            "AAPL": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "MSFT": [2.0, 1.0, 2.0, 2.0, 1.0, 2.0],
            "GOOGL": [np.nan, np.nan, 5.0, np.nan, 1.0, np.nan],
        },
        index=panel.prices.index,
    )

    config = BacktestConfig(dollar_neutral=True, gross_exposure=1.0)
    target_w = signal_to_weights(signal, config)

    # On Day 0: GOOGL is NaN. Non-NaN tickers are AAPL and MSFT.
    # AAPL rank=1, MSFT rank=2. Mean rank = 1.5. Demeaned: AAPL=-0.5, MSFT=0.5.
    # GOOGL weight must be exactly 0.0.
    # Non-NaN weights must sum to 0.0 (dollar neutral).
    assert target_w.iloc[0]["GOOGL"] == 0.0
    assert pytest.approx(target_w.iloc[0]["AAPL"]) == -0.5
    assert pytest.approx(target_w.iloc[0]["MSFT"]) == 0.5
    assert pytest.approx(target_w.iloc[0].sum(), abs=1e-7) == 0.0


# -------------------------------------------------------------------
# 4. Metrics Hand-Calculated Tests
# -------------------------------------------------------------------

def test_metrics_hand_calculated():
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    weights = pd.DataFrame(0.0, index=dates, columns=["AAPL"])
    turnover = pd.Series([1.0, 0.0, 0.0, 0.0], index=dates)
    gross_returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=dates)
    costs = pd.Series([0.0, 0.0, 0.0, 0.0], index=dates)
    net_returns = gross_returns - costs

    res = BacktestResult(
        weights=weights,
        turnover=turnover,
        gross_returns=gross_returns,
        net_returns=net_returns,
        costs=costs,
    )

    metrics = economic_metrics(res, annualization_factor=252)

    # Mean return = (0.01 + 0.02 - 0.01 + 0.03) / 4 = 0.0125
    # Ann return = 0.0125 * 252 = 3.15
    assert pytest.approx(metrics["annualized_return"]) == 3.15
    assert pytest.approx(metrics["average_turnover"]) == 0.25


def test_ic_hand_calculated(synthetic_panel_phase2):
    panel = synthetic_panel_phase2
    dates = panel.prices.index

    # Perfectly aligned signal for returns at t+1
    # returns.shift(-1) is the forward return
    fwd_rets = panel.returns.shift(-1)

    signal = fwd_rets.copy()

    metrics = predictive_metrics(signal, panel, execution_lag_days=1)

    # Perfect correlation (signal == forward returns) -> Pearson IC == 1.0, Rank IC == 1.0
    assert pytest.approx(metrics["ic_mean"], abs=1e-6) == 1.0
    assert pytest.approx(metrics["rank_ic_mean"], abs=1e-6) == 1.0


# -------------------------------------------------------------------
# 5. End-to-End Dummy Signal Backtest on Real 60-Ticker Panel
# -------------------------------------------------------------------

def test_end_to_end_dummy_signal():
    # Build Panel on real 60-ticker universe using synthetic prices for deterministic test
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    tickers = UNIVERSE_60

    # Seeded random price matrix
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

    # Dummy signal: random scores
    np.random.seed(123)
    sig_data = np.random.randn(10, 60)
    signal = pd.DataFrame(sig_data, index=dates, columns=tickers)

    config = BacktestConfig(execution_lag_days=1, dollar_neutral=True, gross_exposure=1.0)
    cost_model = CostModel(cost_bps=5.0)

    # Run 1
    res1 = run_backtest(signal, panel, config, cost_model)

    # Assert output shapes
    assert res1.weights.shape == (10, 60)
    assert len(res1.gross_returns) == 10
    assert len(res1.net_returns) == 10
    assert len(res1.turnover) == 10
    assert len(res1.costs) == 10

    # Check dollar neutrality across non-empty days
    for dt in dates[1:]:
        assert pytest.approx(res1.weights.loc[dt].sum(), abs=1e-6) == 0.0
        assert pytest.approx(res1.weights.loc[dt].abs().sum(), abs=1e-6) == 1.0

    # Economic & Predictive metrics calculation
    econ = economic_metrics(res1)
    pred = predictive_metrics(signal, panel, execution_lag_days=1)

    assert "sharpe_ratio" in econ
    assert "ic_mean" in pred

    # Determinism check (Run 2 produces identical output)
    res2 = run_backtest(signal, panel, config, cost_model)
    pd.testing.assert_frame_equal(res1.weights, res2.weights)
    pd.testing.assert_series_equal(res1.net_returns, res2.net_returns)
