"""
Unit tests for statistical forensics, trial log ledger, DSR, and PSR functions.
"""

import pytest
import math
from pathlib import Path
import tempfile

from statistics import (
    TrialRecord,
    log_trial,
    load_trial_log,
    trial_count,
    backfill_from_phase_artifacts,
    expected_max_sharpe_under_trials,
    deflated_sharpe_ratio,
    DSRResult,
    probabilistic_sharpe_ratio,
    sharpe_variance_across_trials,
)


def test_trial_log_append_only():
    """Verify duplicate candidate_id raises ValueError (immutability & single-look discipline)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "trial_log.jsonl"

        rec1 = TrialRecord(candidate_id="strat_alpha", phase="Phase 4", oos_sharpe=0.5, timestamp="2026-08-24T12:00:00")
        log_trial(rec1, log_path=log_file)

        assert trial_count(log_file) == 1

        # Attempt to log same candidate_id again
        rec2 = TrialRecord(candidate_id="strat_alpha", phase="Phase 4", oos_sharpe=0.6, timestamp="2026-08-24T12:05:00")
        with pytest.raises(ValueError, match="already exists in trial log"):
            log_trial(rec2, log_path=log_file)

        # Confirm ledger count remains 1
        assert trial_count(log_file) == 1


def test_trial_count_matches_backfill():
    """Verify backfilling trial ledger produces expected total N distinct records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "trial_log.jsonl"

        backfilled = backfill_from_phase_artifacts(log_path=log_file)
        n = trial_count(log_file)

        assert n == len(backfilled)
        assert n > 0

        # Check all backfilled records have backfilled=True
        records = load_trial_log(log_file)
        assert all(r.backfilled for r in records)

        # Backfilling again does not duplicate records
        backfilled_again = backfill_from_phase_artifacts(log_path=log_file)
        assert len(backfilled_again) == 0
        assert trial_count(log_file) == n


def test_expected_max_sharpe_increases_with_n():
    """Sanity check that expected max Sharpe SR_0 increases strictly monotonically with trial count N."""
    var_sr = 0.25

    sr_0_n2 = expected_max_sharpe_under_trials(var_sr, n_trials=2)
    sr_0_n10 = expected_max_sharpe_under_trials(var_sr, n_trials=10)
    sr_0_n100 = expected_max_sharpe_under_trials(var_sr, n_trials=100)
    sr_0_n1000 = expected_max_sharpe_under_trials(var_sr, n_trials=1000)

    assert sr_0_n2 > 0.0
    assert sr_0_n2 < sr_0_n10 < sr_0_n100 < sr_0_n1000


def test_dsr_known_values():
    """Validate DSR against Bailey & López de Prado (2014) theoretical reference values."""
    # Worked example: N = 100 trials, V[SR] = 0.25, T = 1250 days, observed Sharpe = 1.2
    n_trials = 100
    var_sr = 0.25
    t_len = 1250
    observed_sr = 1.2

    sr_0 = expected_max_sharpe_under_trials(var_sr, n_trials)
    # Expected max Sharpe for N=100, var=0.25 is ~1.265
    assert math.isclose(sr_0, 1.2652, abs_tol=0.01)

    dsr_val = deflated_sharpe_ratio(
        observed_sharpe=observed_sr,
        sharpe_variance=var_sr,
        n_trials=n_trials,
        skew=0.0,
        kurtosis=3.0,
        track_record_length=t_len,
    )

    # Since observed SR (1.20) < SR_0 (~1.265), DSR must be < 0.50 (specifically ~0.039)
    assert dsr_val < 0.50
    assert math.isclose(dsr_val, 0.0395, abs_tol=0.02)

    # Higher observed Sharpe (e.g. 2.50 > SR_0) should yield high DSR (~1.0)
    high_dsr = deflated_sharpe_ratio(
        observed_sharpe=2.50,
        sharpe_variance=var_sr,
        n_trials=n_trials,
        skew=0.0,
        kurtosis=3.0,
        track_record_length=t_len,
    )
    assert high_dsr > 0.99


def test_dsr_verdict_thresholding():
    """Verify DSRResult verdict assignment: dsr >= 0.95 -> 'survives_dsr', else 'fails_dsr'."""
    res_pass = DSRResult.create(
        candidate_id="top_candidate",
        observed_sharpe=2.5,
        sharpe_variance=0.1,
        n_trials=10,
        track_record_length=252,
    )
    assert res_pass.verdict == "survives_dsr"
    assert res_pass.dsr >= 0.95

    res_fail = DSRResult.create(
        candidate_id="weak_candidate",
        observed_sharpe=0.3,
        sharpe_variance=0.25,
        n_trials=100,
        track_record_length=252,
    )
    assert res_fail.verdict == "fails_dsr"
    assert res_fail.dsr < 0.95


def test_end_to_end_demo_candidates():
    """Verify DSR evaluation on Phase 9 candidates produces well-formed DSRResult objects."""
    sharpes = [0.45, 0.38, 0.12, 0.52, 0.28, -0.10, 0.05, 0.42]
    var_sr = sharpe_variance_across_trials(sharpes)
    assert var_sr > 0.0

    psr = probabilistic_sharpe_ratio(observed_sharpe=0.52, benchmark_sharpe=0.0, track_record_length=252)
    assert 0.0 <= psr <= 1.0

    res = DSRResult.create(
        candidate_id="volatility_adjusted_momentum",
        observed_sharpe=0.52,
        sharpe_variance=var_sr,
        n_trials=len(sharpes),
        skew=0.05,
        kurtosis=3.1,
        track_record_length=252,
    )

    assert res.candidate_id == "volatility_adjusted_momentum"
    assert res.n_trials == len(sharpes)
    assert res.expected_max_sharpe >= 0.0
    assert 0.0 <= res.dsr <= 1.0
    assert res.verdict in ("survives_dsr", "fails_dsr")
    assert len(res.caveats) >= 3
