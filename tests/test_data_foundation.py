import pytest
import numpy as np
import pandas as pd

from core.panel import Panel
from core.signal import validate_signal
from core.experiment import ExperimentMetadata
from data.universe import UNIVERSE_60
from data.panel import build_panel
from features.returns import simple_return, momentum
from features.rolling import rolling_mean, rolling_std, rolling_zscore, rolling_rank
from features.volume import volume_change, turnover
from features.risk import realized_volatility, drawdown


@pytest.fixture
def synthetic_data():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    # Hand-crafted price series
    prices_data = {
        "AAPL": [100.0, 102.0, 105.0, 100.0, 104.0, 108.0, 106.0, 110.0, 109.0, 112.0],
        "MSFT": [200.0, 204.0, 200.0, 210.0, 208.0, 212.0, 215.0, 220.0, 218.0, 225.0],
        "GOOGL": [300.0, 297.0, 300.0, 303.0, 306.0, 300.0, 309.0, 312.0, 315.0, 320.0],
    }
    prices_df = pd.DataFrame(prices_data, index=dates)

    # Hand-crafted volume series
    volume_data = {
        "AAPL": [1000, 1100, 1200, 900, 1300, 1400, 1150, 1500, 1250, 1600],
        "MSFT": [2000, 2100, 1900, 2200, 2000, 2300, 2400, 2500, 2350, 2600],
        "GOOGL": [1500, 1450, 1500, 1600, 1550, 1400, 1650, 1700, 1600, 1750],
    }
    volume_df = pd.DataFrame(volume_data, index=dates)

    return prices_df, volume_df, tickers, dates


@pytest.fixture
def synthetic_panel(synthetic_data):
    prices_df, volume_df, tickers, dates = synthetic_data
    return build_panel(
        tickers=tickers,
        start_date=dates[0],
        end_date=dates[-1],
        missing_threshold=0.05,
        prices_df=prices_df,
        volume_df=volume_df,
    )


# -------------------------------------------------------------------
# 1. Core Contracts & Panel Construction Tests
# -------------------------------------------------------------------

def test_panel_construction(synthetic_panel, synthetic_data):
    prices_df, volume_df, tickers, dates = synthetic_data
    panel = synthetic_panel

    assert isinstance(panel, Panel)
    assert list(panel.prices.columns) == tickers
    assert list(panel.volume.columns) == tickers
    assert list(panel.returns.columns) == tickers
    assert (panel.prices.index == dates).all()
    assert (panel.volume.index == dates).all()
    assert (panel.returns.index == dates).all()
    assert panel.start_date == dates[0]
    assert panel.end_date == dates[-1]
    assert panel.universe == tickers


def test_experiment_metadata_dataclass():
    meta = ExperimentMetadata(
        experiment_id="exp_001",
        source="hypothesis",
        hypothesis_text="Momentum predicts returns",
        expression="momentum(panel, 10)",
        parent_family=None,
        generation_method="manual",
        universe=["AAPL", "MSFT"],
        date_range=(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
        config_snapshot={"param": 1},
        created_at=pd.Timestamp("2023-01-01 10:00:00"),
    )
    assert meta.experiment_id == "exp_001"
    assert meta.source == "hypothesis"


# -------------------------------------------------------------------
# 2. Hand-Calculated Returns Calculation Test
# -------------------------------------------------------------------

def test_returns_computation(synthetic_panel):
    panel = synthetic_panel
    # Day 0 return should be NaN
    assert np.isnan(panel.returns.iloc[0]["AAPL"])

    # Day 1 AAPL: (102 - 100) / 100 = 0.02
    assert pytest.approx(panel.returns.iloc[1]["AAPL"], abs=1e-6) == 0.02
    # Day 1 MSFT: (204 - 200) / 200 = 0.02
    assert pytest.approx(panel.returns.iloc[1]["MSFT"], abs=1e-6) == 0.02
    # Day 1 GOOGL: (297 - 300) / 300 = -0.01
    assert pytest.approx(panel.returns.iloc[1]["GOOGL"], abs=1e-6) == -0.01

    # Day 3 AAPL: (100 - 105) / 105 = -0.0476190476...
    assert pytest.approx(panel.returns.iloc[3]["AAPL"], abs=1e-6) == (100.0 - 105.0) / 105.0


# -------------------------------------------------------------------
# 3. Missing-Data Threshold & Ffill Enforcement Tests
# -------------------------------------------------------------------

def test_missing_data_threshold_trigger(synthetic_data):
    prices_df, volume_df, tickers, dates = synthetic_data

    # Introduce 20% missing data in AAPL (2 out of 10 dates)
    corrupted_prices = prices_df.copy()
    corrupted_prices.iloc[2, 0] = np.nan
    corrupted_prices.iloc[3, 0] = np.nan

    with pytest.raises(ValueError, match="exceed missing data threshold"):
        build_panel(
            tickers=tickers,
            start_date=dates[0],
            end_date=dates[-1],
            missing_threshold=0.05,  # 5% max allowed
            prices_df=corrupted_prices,
            volume_df=volume_df,
        )


def test_missing_data_ffill_single_gap(synthetic_data):
    prices_df, volume_df, tickers, dates = synthetic_data

    # Introduce single-day gap in AAPL (1 out of 10 dates = 10%, but set threshold to 15%)
    corrupted_prices = prices_df.copy()
    corrupted_prices.iloc[2, 0] = np.nan  # Day 2 AAPL = NaN (was 105.0)

    panel = build_panel(
        tickers=tickers,
        start_date=dates[0],
        end_date=dates[-1],
        missing_threshold=0.15,
        prices_df=corrupted_prices,
        volume_df=volume_df,
    )

    # AAPL Day 2 should be forward-filled from Day 1 (102.0)
    assert panel.prices.iloc[2]["AAPL"] == 102.0


# -------------------------------------------------------------------
# 4. validate_signal Tests
# -------------------------------------------------------------------

def test_validate_signal_valid(synthetic_panel):
    panel = synthetic_panel
    # Valid signal with floats and NaNs
    valid_sig = pd.DataFrame(
        {
            "AAPL": [0.5, np.nan, 0.2, -0.1, np.nan, 0.8, 0.0, np.nan, 0.1, -0.5],
            "MSFT": [np.nan, 0.1, 0.3, np.nan, -0.2, 0.5, 0.1, np.nan, 0.2, 0.4],
            "GOOGL": [0.1, 0.2, np.nan, 0.4, 0.5, np.nan, 0.7, 0.8, 0.9, 1.0],
        },
        index=panel.prices.index,
    )
    # Should pass without raising exception
    validate_signal(valid_sig, panel)


def test_validate_signal_rejects_inf(synthetic_panel):
    panel = synthetic_panel
    inf_sig = pd.DataFrame(
        {"AAPL": [0.5, np.inf, 0.2, 0.1], "MSFT": [0.1, 0.2, 0.3, 0.4], "GOOGL": [0.1, 0.2, 0.3, 0.4]},
        index=panel.prices.index[:4],
    )
    with pytest.raises(ValueError, match="infinite"):
        validate_signal(inf_sig, panel)


def test_validate_signal_rejects_out_of_universe(synthetic_panel):
    panel = synthetic_panel
    bad_universe_sig = pd.DataFrame(
        {"AAPL": [0.5, 0.2], "INVALID_TICKER": [0.1, 0.3]},
        index=panel.prices.index[:2],
    )
    with pytest.raises(ValueError, match="out-of-universe tickers"):
        validate_signal(bad_universe_sig, panel)


def test_validate_signal_rejects_out_of_range_dates(synthetic_panel):
    panel = synthetic_panel
    out_of_range_dates = pd.date_range("2023-01-01", periods=12, freq="D")
    bad_date_sig = pd.DataFrame(
        {"AAPL": np.random.randn(12), "MSFT": np.random.randn(12), "GOOGL": np.random.randn(12)},
        index=out_of_range_dates,
    )
    with pytest.raises(ValueError, match="outside panel date range"):
        validate_signal(bad_date_sig, panel)


# -------------------------------------------------------------------
# 5. Features Output Shape Tests
# -------------------------------------------------------------------

ALL_FEATURE_FUNCS = [
    (simple_return, {"lag": 1}),
    (momentum, {"lookback": 3}),
    (rolling_mean, {"window": 3}),
    (rolling_std, {"window": 3}),
    (rolling_zscore, {"window": 3}),
    (rolling_rank, {"window": 3}),
    (volume_change, {"lookback": 1}),
    (turnover, {"lookback": 3}),
    (realized_volatility, {"window": 3}),
    (drawdown, {"window": 3}),
]


@pytest.mark.parametrize("func, kwargs", ALL_FEATURE_FUNCS)
def test_features_output_shape(synthetic_panel, func, kwargs):
    out = func(synthetic_panel, **kwargs)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == synthetic_panel.prices.shape
    assert (out.index == synthetic_panel.prices.index).all()
    assert list(out.columns) == list(synthetic_panel.prices.columns)


# -------------------------------------------------------------------
# 6. Features No-Lookahead Tests
# -------------------------------------------------------------------

@pytest.mark.parametrize("func, kwargs", ALL_FEATURE_FUNCS)
def test_features_no_lookahead(synthetic_data, func, kwargs):
    prices_df, volume_df, tickers, dates = synthetic_data

    panel_base = build_panel(
        tickers=tickers,
        start_date=dates[0],
        end_date=dates[-1],
        missing_threshold=0.05,
        prices_df=prices_df,
        volume_df=volume_df,
    )
    res_base = func(panel_base, **kwargs)

    # Inject extreme future price and volume spikes at Day 7
    future_prices = prices_df.copy()
    future_volume = volume_df.copy()
    future_prices.iloc[7, :] = 99999.0
    future_volume.iloc[7, :] = 9999999.0

    panel_future = build_panel(
        tickers=tickers,
        start_date=dates[0],
        end_date=dates[-1],
        missing_threshold=0.05,
        prices_df=future_prices,
        volume_df=future_volume,
    )
    res_future = func(panel_future, **kwargs)

    # Compare results up to Day 5 (strictly before spike at Day 7)
    cutoff = dates[5]
    pd.testing.assert_frame_equal(
        res_base.loc[:cutoff],
        res_future.loc[:cutoff],
        check_exact=False,
        atol=1e-7,
    )


# -------------------------------------------------------------------
# 7. Hand-Calculated Correctness Tests for Every Feature Function
# -------------------------------------------------------------------

def test_hand_calculated_simple_return(synthetic_panel):
    res = simple_return(synthetic_panel, lag=1)
    # Day 1 AAPL: (102 - 100) / 100 = 0.02
    assert pytest.approx(res.loc["2023-01-02", "AAPL"]) == 0.02


def test_hand_calculated_momentum(synthetic_panel):
    res = momentum(synthetic_panel, lookback=2)
    # Day 2 AAPL: (105 - 100) / 100 = 0.05
    assert pytest.approx(res.loc["2023-01-03", "AAPL"]) == 0.05


def test_hand_calculated_rolling_mean(synthetic_panel):
    res = rolling_mean(synthetic_panel, window=3)
    # AAPL Day 0..2: [100, 102, 105] -> mean = 102.3333333
    assert pytest.approx(res.loc["2023-01-03", "AAPL"]) == 102.33333333333333


def test_hand_calculated_rolling_std(synthetic_panel):
    res = rolling_std(synthetic_panel, window=3)
    # AAPL Day 0..2: sample std([100, 102, 105]) = 2.516611478423583
    expected_std = np.std([100.0, 102.0, 105.0], ddof=1)
    assert pytest.approx(res.loc["2023-01-03", "AAPL"]) == expected_std


def test_hand_calculated_rolling_zscore(synthetic_panel):
    res = rolling_zscore(synthetic_panel, window=3)
    # AAPL Day 2: (105 - 102.3333333) / 2.5166114784 = 1.059441097...
    mean_val = 102.33333333333333
    std_val = np.std([100.0, 102.0, 105.0], ddof=1)
    expected_z = (105.0 - mean_val) / std_val
    assert pytest.approx(res.loc["2023-01-03", "AAPL"]) == expected_z


def test_hand_calculated_rolling_rank(synthetic_panel):
    res = rolling_rank(synthetic_panel, window=3)
    # Day 2 rolling means (window=3):
    # AAPL: (100+102+105)/3 = 102.333
    # MSFT: (200+204+200)/3 = 201.333
    # GOOGL: (300+297+300)/3 = 298.333
    # Cross-sectional percentile rank across tickers:
    # AAPL rank 1/3 = 0.33333333...
    # MSFT rank 2/3 = 0.66666666...
    # GOOGL rank 3/3 = 1.0
    day2_res = res.loc["2023-01-03"]
    assert pytest.approx(day2_res["AAPL"]) == 1.0 / 3.0
    assert pytest.approx(day2_res["MSFT"]) == 2.0 / 3.0
    assert pytest.approx(day2_res["GOOGL"]) == 1.0


def test_hand_calculated_volume_change(synthetic_panel):
    res = volume_change(synthetic_panel, lookback=1)
    # AAPL Day 1: (1100 - 1000) / 1000 = 0.10
    assert pytest.approx(res.loc["2023-01-02", "AAPL"]) == 0.10


def test_hand_calculated_turnover(synthetic_panel):
    res = turnover(synthetic_panel, lookback=3)
    # AAPL Day 0..2 volume: [1000, 1100, 1200] -> mean = 1100
    # Day 2 turnover: 1200 / 1100 = 1.09090909...
    assert pytest.approx(res.loc["2023-01-03", "AAPL"]) == 1200.0 / 1100.0


def test_hand_calculated_realized_volatility(synthetic_panel):
    res = realized_volatility(synthetic_panel, window=3)
    # Returns for AAPL Days 1, 2, 3:
    rets_aapl = synthetic_panel.returns["AAPL"].iloc[1:4].values
    expected_vol = np.std(rets_aapl, ddof=1)
    assert pytest.approx(res.iloc[3]["AAPL"]) == expected_vol


def test_hand_calculated_drawdown(synthetic_panel):
    res = drawdown(synthetic_panel, window=3)
    # AAPL Days 1..3 prices: [102.0, 105.0, 100.0] -> rolling peak = 105.0
    # Day 3 drawdown: (100.0 - 105.0) / 105.0 = -5 / 105 = -0.0476190476...
    assert pytest.approx(res.iloc[3]["AAPL"]) == (100.0 - 105.0) / 105.0


def test_universe_60_import():
    assert isinstance(UNIVERSE_60, list)
    assert len(UNIVERSE_60) == 60
    assert "AAPL" in UNIVERSE_60
    assert "MSFT" in UNIVERSE_60
