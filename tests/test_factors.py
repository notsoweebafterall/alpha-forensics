"""
Unit tests for self-constructed factor model, factor exposure regression, and generalization testing.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.strategies.registry import get_strategy
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from backtesting.engine import run_backtest
from statistics import load_trial_log, trial_count
from factors import (
    SECTOR_MAP,
    build_factor_panel,
    construct_market_factor,
    construct_momentum_factor,
    construct_volatility_factor,
    run_factor_regression,
    evaluate_generalization,
)


def test_sector_map_completeness():
    """Verify every ticker in UNIVERSE_60 has a valid SECTOR_MAP entry."""
    missing = set(UNIVERSE_60) - set(SECTOR_MAP.keys())
    assert len(missing) == 0, f"Missing tickers in SECTOR_MAP: {missing}"
    assert len(SECTOR_MAP) >= 60


def test_market_factor_equals_universe_mean():
    """Verify construct_market_factor matches hand-calculated equal-weighted mean return."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    p_df = pd.DataFrame(
        {
            "AAPL": [100, 102, 101, 103, 105, 104, 106, 108, 107, 110],
            "MSFT": [200, 204, 202, 206, 210, 208, 212, 216, 214, 220],
            "GOOGL": [50, 51, 50.5, 51.5, 52.5, 52, 53, 54, 53.5, 55],
        },
        index=dates,
    )
    v_df = pd.DataFrame(1000, index=dates, columns=tickers)

    panel = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

    mkt = construct_market_factor(panel)
    expected_mkt = panel.returns.mean(axis=1)

    pd.testing.assert_series_equal(mkt, expected_mkt, check_names=False)


def test_momentum_factor_no_lookahead():
    """Verify perturbing future prices after a rebalance date does not change current momentum factor values."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]

    np.random.seed(42)
    p1 = pd.DataFrame(100.0 + np.random.randn(300, 6).cumsum(axis=0), index=dates, columns=tickers)
    v_df = pd.DataFrame(1000, index=dates, columns=tickers)

    panel1 = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p1, volume_df=v_df)
    mom1 = construct_momentum_factor(panel1, formation_days=100, skip_days=10)

    # Create p2 by perturbing prices in the last 50 days only
    p2 = p1.copy()
    p2.iloc[-50:, :] = p2.iloc[-50:, :] * 2.0
    panel2 = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p2, volume_df=v_df)
    mom2 = construct_momentum_factor(panel2, formation_days=100, skip_days=10)

    # Momentum returns before the perturbation point (first 240 days) must be identical
    pd.testing.assert_series_equal(mom1.iloc[:240], mom2.iloc[:240])


def test_volatility_factor_tercile_spread():
    """Verify volatility factor yields long low-vol / short high-vol tercile spread."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    tickers = ["LOW_VOL_1", "LOW_VOL_2", "HIGH_VOL_1", "HIGH_VOL_2", "MID_VOL_1", "MID_VOL_2"]

    # Low vol tickers have tiny daily changes; high vol tickers have huge daily swings
    np.random.seed(123)
    p_dict = {}
    p_dict["LOW_VOL_1"] = 100.0 + np.random.normal(0, 0.01, 100).cumsum()
    p_dict["LOW_VOL_2"] = 100.0 + np.random.normal(0, 0.01, 100).cumsum()
    p_dict["MID_VOL_1"] = 100.0 + np.random.normal(0, 0.10, 100).cumsum()
    p_dict["MID_VOL_2"] = 100.0 + np.random.normal(0, 0.10, 100).cumsum()
    p_dict["HIGH_VOL_1"] = 100.0 + np.random.normal(0, 1.00, 100).cumsum()
    p_dict["HIGH_VOL_2"] = 100.0 + np.random.normal(0, 1.00, 100).cumsum()

    p_df = pd.DataFrame(p_dict, index=dates)
    v_df = pd.DataFrame(1000, index=dates, columns=tickers)

    panel = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)
    vol_factor = construct_volatility_factor(panel, lookback_days=20)

    assert isinstance(vol_factor, pd.Series)
    assert len(vol_factor) == len(dates)


def test_factor_regression_known_case():
    """Verify OLS HAC regression reproduces known true beta and R^2 on synthetic data."""
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    np.random.seed(42)

    mkt = pd.Series(np.random.normal(0.0005, 0.01, 200), index=dates, name="MKT")
    mom = pd.Series(np.random.normal(0.0002, 0.008, 200), index=dates, name="MOM")
    vol = pd.Series(np.random.normal(0.0001, 0.005, 200), index=dates, name="VOL")
    sec_tech = pd.Series(np.random.normal(0.0003, 0.009, 200), index=dates, name="Technology")

    sector_returns = pd.DataFrame({"Technology": sec_tech}, index=dates)
    f_panel = build_factor_panel(
        build_panel(
            tickers=["AAPL", "MSFT", "NVDA"],
            start_date=dates[0],
            end_date=dates[-1],
            prices_df=pd.DataFrame(100.0 + np.random.randn(200, 3).cumsum(axis=0), index=dates, columns=["AAPL", "MSFT", "NVDA"]),
            volume_df=pd.DataFrame(1000, index=dates, columns=["AAPL", "MSFT", "NVDA"]),
        ),
        sector_map={"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology"},
    )
    f_panel.mkt = mkt
    f_panel.mom = mom
    f_panel.vol = vol
    f_panel.sector_returns = sector_returns

    # True synthetic return: y = 0.001 + 1.2 * MKT + 0.5 * MOM + noise (positive mean -> raw_sharpe > 0)
    noise = np.random.normal(0.0, 0.001, 200)
    y_pos = 0.005 + 1.2 * mkt + 0.5 * mom + noise

    res_pos = run_factor_regression(candidate_returns=y_pos, factor_panel=f_panel, candidate_sectors=["Technology"])

    assert pytest.approx(res_pos.betas["MKT"], abs=0.15) == 1.2
    assert pytest.approx(res_pos.betas["MOM"], abs=0.15) == 0.5
    assert res_pos.r_squared > 0.80
    assert res_pos.verdict in ("distinct_alpha", "factor_repackaging")

    # Negative mean return -> raw_sharpe <= 0 must yield "unprofitable" verdict
    y_neg = -0.01 + 1.2 * mkt + noise
    res_neg = run_factor_regression(candidate_returns=y_neg, factor_panel=f_panel, candidate_sectors=["Technology"])
    assert res_neg.raw_sharpe <= 0.0
    assert res_neg.verdict == "unprofitable"


def test_factor_regression_negative_raw_sharpe_guard():
    """Verify that negative raw Sharpe NEVER yields 'distinct_alpha' even if residual_sharpe >= 0.5 * raw_sharpe."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(99)

    mkt = pd.Series(np.random.normal(0.0001, 0.01, 100), index=dates, name="MKT")
    mom = pd.Series(np.random.normal(0.0001, 0.005, 100), index=dates, name="MOM")
    vol = pd.Series(np.random.normal(0.0001, 0.005, 100), index=dates, name="VOL")
    sec_tech = pd.Series(np.random.normal(0.0001, 0.005, 100), index=dates, name="Technology")

    sector_returns = pd.DataFrame({"Technology": sec_tech}, index=dates)
    f_panel = build_factor_panel(
        build_panel(
            tickers=["AAPL", "MSFT", "NVDA"],
            start_date=dates[0],
            end_date=dates[-1],
            prices_df=pd.DataFrame(100.0 + np.random.randn(100, 3).cumsum(axis=0), index=dates, columns=["AAPL", "MSFT", "NVDA"]),
            volume_df=pd.DataFrame(1000, index=dates, columns=["AAPL", "MSFT", "NVDA"]),
        ),
        sector_map={"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology"},
    )
    f_panel.mkt = mkt
    f_panel.mom = mom
    f_panel.vol = vol
    f_panel.sector_returns = sector_returns

    # Construct negative return series where raw_sharpe = -0.5, residual_sharpe = -0.1
    # residual_sharpe (-0.1) >= 0.5 * raw_sharpe (-0.25) is mathematically True, but raw_sharpe <= 0
    y_neg = pd.Series(-0.002 + np.random.normal(0, 0.01, 100), index=dates)
    res = run_factor_regression(candidate_returns=y_neg, factor_panel=f_panel, candidate_sectors=["Technology"])

    assert res.raw_sharpe < 0.0
    assert res.verdict == "unprofitable"
    assert res.verdict != "distinct_alpha"


def test_factor_regression_candidate_sectors_validation():
    """Verify passing candidate_sectors=None or empty raises ValueError (fail loud)."""
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    y = pd.Series(np.random.randn(50), index=dates)

    p_df = pd.DataFrame(100.0 + np.random.randn(50, 3).cumsum(axis=0), index=dates, columns=["AAPL", "MSFT", "NVDA"])
    v_df = pd.DataFrame(1000, index=dates, columns=["AAPL", "MSFT", "NVDA"])
    panel = build_panel(tickers=["AAPL", "MSFT", "NVDA"], start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)
    f_panel = build_factor_panel(panel, sector_map={"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology"})

    with pytest.raises(ValueError, match="candidate_sectors must be explicitly provided"):
        run_factor_regression(candidate_returns=y, factor_panel=f_panel, candidate_sectors=None)

    with pytest.raises(ValueError, match="candidate_sectors must be explicitly provided"):
        run_factor_regression(candidate_returns=y, factor_panel=f_panel, candidate_sectors=[])


def test_generalization_single_guarded_look():
    """Verify sub-evaluations get unique guarded IDs and are logged to trial log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "trial_log.jsonl"

        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        tickers = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "WFC"]
        p_df = pd.DataFrame(100.0 + np.random.randn(500, 6).cumsum(axis=0), index=dates, columns=tickers)
        v_df = pd.DataFrame(1000, index=dates, columns=tickers)
        panel = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

        sec_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "NVDA": "Technology",
            "JPM": "Financials",
            "BAC": "Financials",
            "WFC": "Financials",
        }

        strat = get_strategy("cross_sectional_momentum")

        res = evaluate_generalization(
            candidate_id="cs_mom_test",
            build_fn=strat.build,
            default_params=strat.default_params,
            panel=panel,
            sector_map=sec_map,
            log_path=log_file,
        )

        assert res.candidate_id == "cs_mom_test"
        assert res.classification in ("universal", "sector_specific", "period_specific", "fragile")

        # Verify records written to trial log
        n_logged = trial_count(log_file)
        assert n_logged >= 3  # At least 2 sectors + 2 period halves
        records = load_trial_log(log_file)
        c_ids = {r.candidate_id for r in records}
        assert "cs_mom_test__sector_Technology" in c_ids
        assert "cs_mom_test__period_h1" in c_ids


def test_end_to_end_factor_and_generalization():
    """Verify end-to-end factor regression and generalization pipeline on a real strategy candidate."""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    tickers = UNIVERSE_60
    np.random.seed(42)
    p_df = pd.DataFrame(100.0 + np.random.randn(500, 60).cumsum(axis=0), index=dates, columns=tickers)
    v_df = pd.DataFrame(1000, index=dates, columns=tickers)

    panel = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)
    f_panel = build_factor_panel(panel, sector_map=SECTOR_MAP)

    strat = get_strategy("cross_sectional_momentum")
    expr = strat.build(strat.default_params)
    signal = expr.evaluate(panel)
    res = run_backtest(signal, panel, BacktestConfig(execution_lag_days=1), CostModel())

    reg_res = run_factor_regression(
        candidate_returns=res.gross_returns,
        factor_panel=f_panel,
        candidate_sectors=["Technology", "Financials"],
        candidate_id="cs_mom_e2e",
    )

    assert reg_res.candidate_id == "cs_mom_e2e"
    assert "MKT" in reg_res.betas
    assert "MOM" in reg_res.betas
    assert "VOL" in reg_res.betas
    assert "SECTOR_Technology" in reg_res.betas
    assert "SECTOR_Financials" in reg_res.betas
    assert 0.0 <= reg_res.r_squared <= 1.0
