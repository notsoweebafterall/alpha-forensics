import pytest
import numpy as np
import pandas as pd

from core.panel import Panel
from data.panel import build_panel
from alpha.strategies.registry import get_strategy
from alpha.expressions.tree import UnaryNode, Leaf
import alpha.primitives.inputs as in_prim
import alpha.primitives.operators as op_prim
from validation.folds import WalkForwardConfig
from validation.oos import _oos_access_log, reset_oos_access_log

from robustness.stability_metrics import (
    fraction_positive,
    coefficient_of_variation,
    classify_stability,
)
from robustness.parameter_landscape import (
    evaluate_parameter_landscape,
    ParameterLandscapeResult,
)


@pytest.fixture
def synthetic_panel_phase9():
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
# 1. Stability Classification Tests
# -------------------------------------------------------------------

def test_classify_stability():
    # 1. broad_robust: >= 70% positive (e.g. 8 of 10 positive)
    sharpes_broad = [1.2, 0.8, 1.5, 0.5, 0.9, 1.1, 0.2, 0.4, -0.1, -0.3]
    label, rat = classify_stability(sharpes_broad)
    assert label == "broad_robust"
    assert "8 of 10 variants (80.0%)" in rat

    # 2. moderately_sensitive: 40% - 70% positive (e.g. 5 of 10 positive)
    sharpes_mod = [1.2, 0.8, 1.5, 0.5, 0.9, -0.2, -0.1, -0.4, -0.3, -0.5]
    label, rat = classify_stability(sharpes_mod)
    assert label == "moderately_sensitive"
    assert "5 of 10 variants (50.0%)" in rat

    # 3. highly_sensitive: 20% - 40% positive (e.g. 3 of 10 positive)
    sharpes_high = [1.2, 0.8, 0.5, -0.2, -0.1, -0.4, -0.3, -0.5, -0.2, -0.8]
    label, rat = classify_stability(sharpes_high)
    assert label == "highly_sensitive"
    assert "3 of 10 variants (30.0%)" in rat

    # 4. isolated_optimum: < 20% positive (e.g. 1 of 10 positive)
    sharpes_iso = [2.5, -0.1, -0.4, -0.3, -0.5, -0.2, -0.8, -0.3, -0.4, -0.6]
    label, rat = classify_stability(sharpes_iso)
    assert label == "isolated_optimum"
    assert "1 of 10 variants (10.0%)" in rat


def test_fraction_and_cv():
    sharpes = [1.0, 2.0, -1.0, np.nan]
    assert pytest.approx(fraction_positive(sharpes)) == 2.0 / 3.0

    vals = [10.0, 10.0, 10.0]
    assert coefficient_of_variation(vals) == 0.0


# -------------------------------------------------------------------
# 2. Parameter Landscape Single-Look Guard & Deduplication Tests
# -------------------------------------------------------------------

def test_evaluate_parameter_landscape_single_look_guard(synthetic_panel_phase9):
    reset_oos_access_log()
    panel = synthetic_panel_phase9

    def mock_build(params):
        return UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(params["lookback"])))

    param_grid = {"lookback": [10, 20, 30, 40]}
    base_id = "test_mom_grid"

    res = evaluate_parameter_landscape(
        base_candidate_id=base_id,
        build_fn=mock_build,
        param_grid=param_grid,
        panel=panel,
        max_variants=10,
    )

    # Verify exactly 4 variant results produced with 4 distinct candidate_ids
    assert len(res.variant_results) == 4
    ids = [vr.candidate_id for vr in res.variant_results]
    assert len(set(ids)) == 4

    # Verify _oos_access_log contains all 4 variant candidate IDs
    for var_id in ids:
        assert var_id in _oos_access_log


def test_grid_deduplication(synthetic_panel_phase9):
    reset_oos_access_log()
    panel = synthetic_panel_phase9

    # Mock build function ignoring dummy_param -> produces identical canonical hashes
    def mock_collapsing_build(params):
        return UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(20)))

    param_grid = {"dummy_param": [1, 2, 3, 4, 5]}
    res = evaluate_parameter_landscape(
        base_candidate_id="test_dedup_grid",
        build_fn=mock_collapsing_build,
        param_grid=param_grid,
        panel=panel,
    )

    # Should collapse 5 raw parameter combinations into 1 unique variant
    assert len(res.variant_results) == 1


# -------------------------------------------------------------------
# 3. Safety Valve & No Silent Filtering Tests
# -------------------------------------------------------------------

def test_max_variants_safety_valve(synthetic_panel_phase9):
    panel = synthetic_panel_phase9

    def mock_build(params):
        return UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(params["lookback"])))

    # Grid size = 6 > max_variants (5)
    param_grid = {"lookback": [10, 20, 30, 40, 50, 60]}

    # Safety valve must raise ValueError BEFORE executing any OOS evaluations
    with pytest.raises(ValueError, match="exceeds max_variants limit"):
        evaluate_parameter_landscape(
            base_candidate_id="test_safety_valve",
            build_fn=mock_build,
            param_grid=param_grid,
            panel=panel,
            max_variants=5,
        )


def test_end_to_end_parameter_landscape(synthetic_panel_phase9):
    reset_oos_access_log()
    panel = synthetic_panel_phase9
    strat = get_strategy("cross_sectional_momentum")

    res = evaluate_parameter_landscape(
        base_candidate_id=strat.name,
        build_fn=strat.build,
        param_grid=strat.variant_params,
        panel=panel,
        max_variants=10,
    )

    assert isinstance(res, ParameterLandscapeResult)
    assert res.base_candidate_id == strat.name
    assert len(res.variant_results) == len(strat.variant_params["lookback"])
    assert res.stability_label in ["broad_robust", "moderately_sensitive", "highly_sensitive", "isolated_optimum"]
    assert len(res.stability_rationale) > 0
