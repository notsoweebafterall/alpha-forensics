"""
Unit tests for regime robustness analysis and trial logging (Phase 12).
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
from validation.folds import WalkForwardConfig
from statistics import load_trial_log, trial_count
from regimes import evaluate_regime_robustness, RegimeRobustnessResult


def test_regime_single_guarded_look():
    """Verify regime sub-evaluations create unique IDs, compute distribution stats, and reuse existing ledger records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "trial_log.jsonl"

        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        tickers = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "WFC"]
        np.random.seed(42)
        p_df = pd.DataFrame(100.0 + np.random.randn(300, 6).cumsum(axis=0), index=dates, columns=tickers)
        v_df = pd.DataFrame(1000, index=dates, columns=tickers)
        panel = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

        strat = get_strategy("cross_sectional_momentum")
        wf_cfg = WalkForwardConfig(
            mode="expanding",
            initial_train_window=100,
            step_size=30,
            val_window=30,
            embargo_days=5,
        )

        # Pre-generate synthetic OOS net returns series matching panel dates
        oos_rets = pd.Series(np.random.normal(0.0005, 0.01, 150), index=dates[150:])

        # 1. First execution — writes regime trials to temp log file
        res1 = evaluate_regime_robustness(
            candidate_id="cs_mom_reg_test",
            build_fn=strat.build,
            default_params=strat.default_params,
            panel=panel,
            log_path=log_file,
            min_regime_days=10,
            regime_lookback=20,
            oos_returns=oos_rets,
        )

        assert isinstance(res1, RegimeRobustnessResult)
        assert res1.candidate_id == "cs_mom_reg_test"
        assert res1.classification in ("regime_robust", "regime_dependent", "fragile")
        assert len(res1.regime_sharpes) > 0

        initial_count = trial_count(log_file)
        assert initial_count > 0

        # Verify logged records contain skew, kurtosis, and track_record_length
        records = load_trial_log(log_file)
        for r in records:
            assert r.candidate_id.startswith("cs_mom_reg_test__regime_")
            assert r.phase.startswith("Phase 12 Regime ")
            assert r.skew is not None or r.track_record_length is not None

        # 2. Second execution — single-look guard should reuse logged entries without re-logging duplicates
        res2 = evaluate_regime_robustness(
            candidate_id="cs_mom_reg_test",
            build_fn=strat.build,
            default_params=strat.default_params,
            panel=panel,
            log_path=log_file,
            min_regime_days=10,
            regime_lookback=20,
            oos_returns=oos_rets,
        )

        second_count = trial_count(log_file)
        assert second_count == initial_count, f"Expected {initial_count} logged trials, but got {second_count} after second run."
        assert res1.regime_sharpes == res2.regime_sharpes


def test_regime_end_to_end():
    """Verify end-to-end regime robustness pipeline with pre-evaluated OOS returns."""
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    tickers = UNIVERSE_60
    np.random.seed(42)
    p_df = pd.DataFrame(100.0 + np.random.randn(400, 60).cumsum(axis=0), index=dates, columns=tickers)
    v_df = pd.DataFrame(1000, index=dates, columns=tickers)

    panel = build_panel(tickers=tickers, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

    strat = get_strategy("volatility_adjusted_momentum")
    
    # Pre-generate synthetic OOS net returns series matching panel dates
    oos_rets = pd.Series(np.random.normal(0.0005, 0.01, 200), index=dates[200:])

    res = evaluate_regime_robustness(
        candidate_id="vol_mom_e2e",
        build_fn=strat.build,
        default_params=strat.default_params,
        panel=panel,
        min_regime_days=15,
        regime_lookback=30,
        oos_returns=oos_rets,
    )

    assert res.candidate_id == "vol_mom_e2e"
    assert res.classification in ("regime_robust", "regime_dependent", "fragile")
    assert isinstance(res.rationale, str) and len(res.rationale) > 0
    assert len(res.regime_sharpes) > 0
    assert len(res.regime_counts) > 0
