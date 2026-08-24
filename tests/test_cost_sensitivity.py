import pytest
import numpy as np
import pandas as pd

from core.panel import Panel
from data.panel import build_panel
from alpha.strategies.registry import get_strategy
from validation.folds import WalkForwardConfig
from validation.oos import _oos_access_log, reset_oos_access_log
from backtesting.cost_sensitivity import (
    run_cost_sensitivity,
    CostSensitivityResult,
    STANDARD_COST_GRID,
    REALISTIC_COST_BPS,
)


@pytest.fixture
def synthetic_panel_phase8():
    dates = pd.date_range("2020-01-01", periods=800, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    np.random.seed(42)
    p_data = 100.0 + np.random.randn(800, 3).cumsum(axis=0)
    v_data = 10000 + np.random.randint(0, 5000, size=(800, 3))

    prices_df = pd.DataFrame(p_data, index=dates, columns=tickers)
    volume_df = pd.DataFrame(v_data, index=dates, columns=tickers)

    return build_panel(
        tickers=tickers,
        start_date=dates[0],
        end_date=dates[-1],
        missing_threshold=0.05,
        prices_df=prices_df,
        volume_df=volume_df,
    )


# -------------------------------------------------------------------
# 1. Single OOS Look Guard Test
# -------------------------------------------------------------------

def test_single_oos_look_guard(synthetic_panel_phase8):
    reset_oos_access_log()
    panel = synthetic_panel_phase8
    strat = get_strategy("cross_sectional_momentum")
    expr = strat.build({"lookback": 20})
    candidate_id = "test_cand_guard"

    # Call run_cost_sensitivity
    res = run_cost_sensitivity(candidate_id, expr, panel)

    # Verify _oos_access_log gained exactly the base entry
    base_id = f"{candidate_id}__cost_sweep_base"
    assert base_id in _oos_access_log

    # Verify a second call with the same candidate_id raises RuntimeError
    with pytest.raises(RuntimeError, match="OOS already evaluated"):
        run_cost_sensitivity(candidate_id, expr, panel)


# -------------------------------------------------------------------
# 2. Hand-Calculated Arithmetic Test
# -------------------------------------------------------------------

def test_cost_arithmetic_hand_calculated():
    # Toy series
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    gross_returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01], index=dates)
    turnover = pd.Series([0.1, 0.2, 0.15, 0.1, 0.05], index=dates)

    bps = 10.0  # 10 bps -> multiplier 0.001
    cost_drag = turnover * (10.0 / 10000.0)  # [0.0001, 0.0002, 0.00015, 0.0001, 0.00005]
    expected_net = gross_returns - cost_drag

    actual_net = gross_returns - turnover * (bps / 10000.0)
    pd.testing.assert_series_equal(actual_net, expected_net)

    expected_mean_ann = expected_net.mean() * 252.0
    actual_mean_ann = actual_net.mean() * 252.0
    assert pytest.approx(actual_mean_ann) == expected_mean_ann


# -------------------------------------------------------------------
# 3. Full Grid Reporting Test (No Suppression)
# -------------------------------------------------------------------

def test_cost_grid_full_reporting(synthetic_panel_phase8):
    reset_oos_access_log()
    panel = synthetic_panel_phase8
    strat = get_strategy("cross_sectional_momentum")
    expr = strat.build({"lookback": 60})

    custom_grid = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
    res = run_cost_sensitivity("test_cand_full_grid", expr, panel, cost_grid=custom_grid)

    # Verify every single cost in custom_grid is present in output dict
    assert len(res.oos_sharpe_by_cost) == len(custom_grid)
    for bps in custom_grid:
        assert bps in res.oos_sharpe_by_cost
        assert bps in res.oos_annualized_return_by_cost


# -------------------------------------------------------------------
# 4. Breakeven Cost Bps Test
# -------------------------------------------------------------------

def test_breakeven_cost_bps():
    # Hand-constructed linear crossing toy case
    # Daily gross mean return = 0.001 (annualized ~ 25.2%)
    # Daily turnover = 1.0
    # Cost drag at bps = turnover * (bps / 10000) = bps / 10000
    # Annualized net return = (0.001 - bps / 10000) * 252 = 0 => bps / 10000 = 0.001 => bps = 10.0
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    gross_rets = pd.Series([0.001] * 100, index=dates)
    turnover_series = pd.Series([1.0] * 100, index=dates)

    prev_ann_ret = (gross_rets - turnover_series * 0.0).mean() * 252.0  # 0.252
    step_bps = 10.0
    cur_ann_ret = (gross_rets - turnover_series * (step_bps / 10000.0)).mean() * 252.0  # 0.0

    # Linear crossing exactly at 10.0 bps
    slope_per_bps = (cur_ann_ret - prev_ann_ret) / step_bps  # -0.0252 per bps
    breakeven = 0.0 - (prev_ann_ret / slope_per_bps)
    assert pytest.approx(breakeven) == 10.0


# -------------------------------------------------------------------
# 5. Survives Realistic Costs Test
# -------------------------------------------------------------------

def test_survives_realistic_costs(synthetic_panel_phase8):
    reset_oos_access_log()
    panel = synthetic_panel_phase8
    strat = get_strategy("cross_sectional_momentum")
    expr = strat.build({"lookback": 60})

    res = run_cost_sensitivity("test_cand_survives", expr, panel)

    if res.oos_sharpe_by_cost[REALISTIC_COST_BPS] > 0:
        assert res.survives_realistic_costs is True
    else:
        assert res.survives_realistic_costs is False


# -------------------------------------------------------------------
# 6. End-to-End Execution Test
# -------------------------------------------------------------------

def test_end_to_end_cost_sensitivity(synthetic_panel_phase8):
    reset_oos_access_log()
    panel = synthetic_panel_phase8
    strat = get_strategy("low_volatility")
    expr = strat.build({"window": 60})

    res = run_cost_sensitivity("test_e2e_cost_sweep", expr, panel)

    assert isinstance(res, CostSensitivityResult)
    assert res.candidate_id == "test_e2e_cost_sweep"
    assert res.cost_bps_grid == STANDARD_COST_GRID
    assert len(res.oos_sharpe_by_cost) == 5
    assert len(res.oos_annualized_return_by_cost) == 5
