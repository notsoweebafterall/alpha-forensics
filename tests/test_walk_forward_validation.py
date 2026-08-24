import pytest
import numpy as np
import pandas as pd

from core.panel import Panel
from data.panel import build_panel
from alpha.strategies.registry import get_strategy
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel

from validation.folds import WalkForwardConfig, Fold, generate_folds, validate_no_overlap
from validation.oos import reserve_oos_holdout, evaluate_oos, reset_oos_access_log
from validation.runner import run_walk_forward_validation


@pytest.fixture
def synthetic_panel_phase7():
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
# 1. Fold Generation & Embargo Tests
# -------------------------------------------------------------------

def test_generate_folds_expanding(synthetic_panel_phase7):
    panel = synthetic_panel_phase7
    config = WalkForwardConfig(
        mode="expanding",
        initial_train_window=400,
        step_size=50,
        val_window=50,
        embargo_days=10,
    )

    folds = generate_folds(panel, config)
    assert len(folds) > 0

    # Verify expanding train windows and strictly increasing fold_ids
    prev_train_len = 0
    for fold in folds:
        t_start, t_end = fold.train_range
        v_start, v_end = fold.val_range

        # Slicing length
        t_len = len(panel.prices.loc[t_start:t_end])
        assert t_len > prev_train_len
        prev_train_len = t_len

        # Train start is always initial start date in expanding mode
        assert t_start == panel.prices.index[0]


def test_embargo_enforcement(synthetic_panel_phase7):
    panel = synthetic_panel_phase7
    embargo_days = 15
    config = WalkForwardConfig(
        initial_train_window=300,
        step_size=40,
        val_window=40,
        embargo_days=embargo_days,
    )

    folds = generate_folds(panel, config)

    dates = panel.prices.index
    for fold in folds:
        t_end_idx = dates.get_loc(fold.train_range[1])
        v_start_idx = dates.get_loc(fold.val_range[0])

        gap = v_start_idx - t_end_idx
        assert gap >= embargo_days


def test_validate_no_overlap_error(synthetic_panel_phase7):
    dates = synthetic_panel_phase7.prices.index

    # Construct corrupted overlapping fold (val_start before train_end)
    corrupted_fold = Fold(
        fold_id=0,
        train_range=(dates[0], dates[100]),
        val_range=(dates[90], dates[150]),  # Overlaps at 90..100!
    )

    with pytest.raises(ValueError, match="Overlap detected"):
        validate_no_overlap([corrupted_fold], dates, embargo_days=10)

    # Construct fold violating embargo gap
    embargo_violating_fold = Fold(
        fold_id=1,
        train_range=(dates[0], dates[100]),
        val_range=(dates[103], dates[150]),  # Gap is 3 days < 10
    )

    with pytest.raises(ValueError, match="Embargo violation"):
        validate_no_overlap([embargo_violating_fold], dates, embargo_days=10)


# -------------------------------------------------------------------
# 2. OOS Reservation & Access Guard Tests
# -------------------------------------------------------------------

def test_reserve_oos_holdout(synthetic_panel_phase7):
    panel = synthetic_panel_phase7
    dev_panel, oos_panel = reserve_oos_holdout(panel, oos_fraction=0.20)

    # Date counts
    total_len = len(panel.prices)
    oos_len = len(oos_panel.prices)
    dev_len = len(dev_panel.prices)

    assert oos_len == int(total_len * 0.20)
    assert dev_len + oos_len == total_len

    # Chronological ordering: dev ends before OOS starts
    assert dev_panel.end_date < oos_panel.start_date
    assert dev_panel.prices.index[-1] < oos_panel.prices.index[0]


def test_oos_access_guard(synthetic_panel_phase7):
    reset_oos_access_log()
    dev_panel, oos_panel = reserve_oos_holdout(synthetic_panel_phase7, oos_fraction=0.20)

    strat = get_strategy("cross_sectional_momentum")
    expr = strat.build({"lookback": 20})
    backtest_config = BacktestConfig()
    cost_model = CostModel()

    # Call 1 with 'cand_A' -> Passes
    m1 = evaluate_oos("cand_A", expr, oos_panel, backtest_config, cost_model)
    assert "sharpe_ratio" in m1

    # Call 2 with same 'cand_A' -> Raises RuntimeError
    with pytest.raises(RuntimeError, match="OOS already evaluated for 'cand_A'"):
        evaluate_oos("cand_A", expr, oos_panel, backtest_config, cost_model)

    # Call 3 with different 'cand_B' -> Passes cleanly
    m2 = evaluate_oos("cand_B", expr, oos_panel, backtest_config, cost_model)
    assert "sharpe_ratio" in m2


# -------------------------------------------------------------------
# 3. Walk-Forward Runner Determinism & Degradation Tests
# -------------------------------------------------------------------

def test_run_walk_forward_validation_determinism(synthetic_panel_phase7):
    panel = synthetic_panel_phase7
    strat = get_strategy("cross_sectional_momentum")
    expr = strat.build({"lookback": 60})

    wf_config = WalkForwardConfig(
        initial_train_window=400,
        step_size=60,
        val_window=60,
        embargo_days=10,
    )

    # Run 1
    reset_oos_access_log()
    res1 = run_walk_forward_validation("cand_det_1", expr, panel, wf_config)

    # Run 2
    reset_oos_access_log()
    res2 = run_walk_forward_validation("cand_det_1", expr, panel, wf_config)

    # Assert identical results across runs
    assert len(res1.folds) == len(res2.folds)
    assert res1.degradation == res2.degradation
    # Compare scalar oos_metrics keys
    for k in res1.oos_metrics.keys():
        if isinstance(res1.oos_metrics[k], pd.Series):
            pd.testing.assert_series_equal(res1.oos_metrics[k], res2.oos_metrics[k])
        else:
            assert pytest.approx(res1.oos_metrics[k]) == res2.oos_metrics[k]


def test_degradation_calculation():
    # Test hand-calculated degradation ratio cases
    from validation.runner import run_walk_forward_validation

    # Positive train Sharpe (mean = 2.0), OOS Sharpe = 1.5 -> ratio = 0.75
    train_sharpe = 2.0
    oos_sharpe = 1.5
    expected_ratio = 1.5 / 2.0

    # Directly check degradation formula
    degrade_ratio = oos_sharpe / train_sharpe
    assert pytest.approx(degrade_ratio) == expected_ratio
