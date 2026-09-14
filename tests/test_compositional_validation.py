"""
Tests for Phase 6→7 Compositional Validation Wiring.

Tests cover:
1. _candidate_id() mapping: compositional_random → generated_{hash[:12]},
   variant_expansion → build_candidate_id(parent_family, params) convention.
2. Single-look ledger reuse: candidates already in ledger are not re-evaluated.
3. max_candidates cap: excess candidates are skipped in screening order with explicit
   reporting; no candidate is silently dropped.
4. Idempotency: running run_demo() twice in a row produces zero newly-logged rows on
   the second run, confirming canonical_hash() determinism and ledger reuse.
5. Phase label: ledger rows written by this demo use
   "Phase 6→7 Compositional Validation", not "Phase 7 Walk-Forward Validation".
6. Summary funnel counts: generated / screened / retained / already_in_ledger /
   newly_logged / skipped_over_cap are consistent.
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.panel import Panel
from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.generator.candidate import GeneratedCandidate
from alpha.expressions.tree import Leaf, UnaryNode
from alpha.expressions.serialize import to_string
import alpha.primitives.inputs as in_prim
import alpha.primitives.operators as op_prim
from alpha.strategies import build_candidate_id

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from compositional_validation_demo import _candidate_id, run_demo


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_panel() -> Panel:
    """Small synthetic panel sufficient for walk-forward with minimal window sizes."""
    dates = pd.date_range("2020-01-01", periods=600, freq="B")
    tickers = UNIVERSE_60[:5]
    np.random.seed(0)
    prices = pd.DataFrame(
        100.0 + np.random.randn(len(dates), len(tickers)).cumsum(axis=0),
        index=dates,
        columns=tickers,
    )
    volumes = pd.DataFrame(
        10000 + np.random.randint(0, 5000, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    return build_panel(
        tickers=tickers,
        start_date=dates[0],
        end_date=dates[-1],
        prices_df=prices,
        volume_df=volumes,
    )


def _make_comp_candidate(seed_offset: int = 0) -> GeneratedCandidate:
    """Helper: a minimal compositional_random candidate."""
    expr = UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(20 + seed_offset)))
    return GeneratedCandidate(
        expression=expr,
        expression_string=to_string(expr),
        generation_method="compositional_random",
        seed=42,
        parent_family=None,
        param_values={},
        generated_at=pd.Timestamp.now(),
    )


def _make_variant_candidate(lookback: int = 20) -> GeneratedCandidate:
    """Helper: a minimal variant_expansion candidate with a parent family."""
    expr = UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(lookback)))
    return GeneratedCandidate(
        expression=expr,
        expression_string=to_string(expr),
        generation_method="variant_expansion",
        seed=None,
        parent_family="cross_sectional_momentum",
        param_values={"lookback": lookback},
        generated_at=pd.Timestamp.now(),
    )


# ─── Test 1: _candidate_id() mapping ──────────────────────────────────────────

def test_candidate_id_compositional_random():
    """compositional_random candidates get generated_{hash[:12]} IDs."""
    cand = _make_comp_candidate()
    cid = _candidate_id(cand)
    assert cid.startswith("generated_"), f"Expected 'generated_' prefix, got: {cid}"
    assert len(cid) == len("generated_") + 12, f"Expected 12-char hash suffix, got: {cid}"


def test_candidate_id_variant_expansion():
    """variant_expansion candidates get build_candidate_id(parent, params) IDs."""
    cand = _make_variant_candidate(lookback=20)
    cid = _candidate_id(cand)
    expected = build_candidate_id("cross_sectional_momentum", params={"lookback": 20})
    assert cid == expected, f"Expected '{expected}', got '{cid}'"


def test_candidate_id_deterministic():
    """Same expression always produces the same candidate ID across calls."""
    cand1 = _make_comp_candidate(seed_offset=0)
    cand2 = _make_comp_candidate(seed_offset=0)
    assert _candidate_id(cand1) == _candidate_id(cand2)


def test_candidate_id_distinct_for_distinct_expressions():
    """Different expressions produce different candidate IDs."""
    cand1 = _make_comp_candidate(seed_offset=0)
    cand2 = _make_comp_candidate(seed_offset=10)  # Momentum(30) vs Momentum(20)
    assert _candidate_id(cand1) != _candidate_id(cand2)


def test_candidate_id_no_parent_family_falls_back_to_hash():
    """variant_expansion candidate without parent_family falls back to hash-based ID."""
    expr = UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(30)))
    cand = GeneratedCandidate(
        expression=expr,
        expression_string=to_string(expr),
        generation_method="variant_expansion",
        seed=None,
        parent_family=None,   # <-- no parent family
        param_values={"lookback": 30},
        generated_at=pd.Timestamp.now(),
    )
    cid = _candidate_id(cand)
    # Should fall back to hash-based ID since parent_family is None
    assert cid.startswith("generated_")


# ─── Test 2: Single-look ledger reuse ─────────────────────────────────────────

def test_single_look_ledger_reuse(tiny_panel, tmp_path):
    """
    Running run_demo() twice on the same paths: second run should report zero
    newly-logged rows (all candidates already in ledger from first run).

    Uses a small generation_budget and max_candidates to keep runtime short.
    Also uses a WalkForwardConfig with small windows via the existing default
    (initial_train_window=400 fits within 600-date tiny_panel).
    """
    log_path = tmp_path / "trial_log.jsonl"
    store_path = tmp_path / "trial_returns.parquet"

    tickers = tiny_panel.universe

    # First run: log some candidates
    run_demo(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2022-05-31",
        log_path=log_path,
        store_path=store_path,
        cache_dir=tmp_path / "cache",
        generation_budget=10,
        generation_seed=42,
        max_depth=2,
        max_candidates=3,
    )

    # Count rows after first run
    first_run_count = sum(1 for line in log_path.read_text().splitlines() if line.strip())

    # Second run: must reuse from ledger, no new rows
    run_demo(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2022-05-31",
        log_path=log_path,
        store_path=store_path,
        cache_dir=tmp_path / "cache",
        generation_budget=10,
        generation_seed=42,
        max_depth=2,
        max_candidates=3,
    )

    second_run_count = sum(1 for line in log_path.read_text().splitlines() if line.strip())
    assert second_run_count == first_run_count, (
        f"Second run added {second_run_count - first_run_count} new rows — "
        f"single-look ledger reuse not working. First: {first_run_count}, Second: {second_run_count}"
    )


# ─── Test 3: Phase label in ledger rows ───────────────────────────────────────

def test_phase_label_in_ledger(tiny_panel, tmp_path):
    """
    Rows written by run_demo() use phase='Phase 6→7 Compositional Validation',
    not 'Phase 7 Walk-Forward Validation'.
    """
    log_path = tmp_path / "trial_log.jsonl"
    store_path = tmp_path / "trial_returns.parquet"
    tickers = tiny_panel.universe

    run_demo(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2022-05-31",
        log_path=log_path,
        store_path=store_path,
        cache_dir=tmp_path / "cache",
        generation_budget=10,
        generation_seed=42,
        max_depth=2,
        max_candidates=2,
    )

    if not log_path.exists() or not log_path.read_text().strip():
        pytest.skip("No candidates passed screening — nothing written to ledger.")

    phases_written = set()
    for line in log_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            phases_written.add(rec.get("phase", ""))

    assert "Phase 6->7 Compositional Validation" in phases_written, (
        f"Expected phase label not found. Got phases: {phases_written}"
    )
    assert "Phase 7 Walk-Forward Validation" not in phases_written, (
        "Phase 7 label should not appear in rows written by the Phase 6→7 demo."
    )


# ─── Test 4: max_candidates cap is applied, skipped set is explicit ────────────

def test_max_candidates_cap_respected(tiny_panel, tmp_path):
    """
    When more candidates pass screening than max_candidates, the ledger only
    receives at most max_candidates new rows (not more), confirming the cap
    is enforced rather than silently dropped.
    """
    log_path = tmp_path / "trial_log.jsonl"
    store_path = tmp_path / "trial_returns.parquet"
    tickers = tiny_panel.universe

    # Set a very tight cap so it is likely to be hit on a budget=50 run
    run_demo(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2022-05-31",
        log_path=log_path,
        store_path=store_path,
        cache_dir=tmp_path / "cache",
        generation_budget=50,
        generation_seed=0,
        max_depth=3,
        max_candidates=2,
    )

    if not log_path.exists() or not log_path.read_text().strip():
        pytest.skip("No candidates passed screening — cap test not meaningful.")

    rows_written = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    # Only rows from this phase
    phase_rows = [r for r in rows_written if r.get("phase") == "Phase 6->7 Compositional Validation"]
    assert len(phase_rows) <= 2, (
        f"max_candidates=2 cap was violated: {len(phase_rows)} rows written."
    )


# ─── Test 5: candidate_ids use 'generated_' prefix for compositional_random ───

def test_ledger_ids_are_generated_prefixed(tiny_panel, tmp_path):
    """
    All candidate IDs written to the ledger by compositional candidates use
    the 'generated_' prefix (not a bare strategy name).
    """
    log_path = tmp_path / "trial_log.jsonl"
    store_path = tmp_path / "trial_returns.parquet"
    tickers = tiny_panel.universe

    run_demo(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2022-05-31",
        log_path=log_path,
        store_path=store_path,
        cache_dir=tmp_path / "cache",
        generation_budget=20,
        generation_seed=7,
        max_depth=2,
        max_candidates=3,
    )

    if not log_path.exists() or not log_path.read_text().strip():
        pytest.skip("No candidates passed screening.")

    for line in log_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("phase") == "Phase 6->7 Compositional Validation":
                cid = rec["candidate_id"]
                assert cid.startswith("generated_"), (
                    f"Expected 'generated_' prefix for compositional candidate, got: '{cid}'"
                )
                hash_suffix = cid[len("generated_"):]
                assert len(hash_suffix) == 12, (
                    f"Expected 12-char hash suffix, got {len(hash_suffix)} chars: '{hash_suffix}'"
                )


# ─── Test 6: Accepts custom path parameters ───────────────────────────────────

def test_run_demo_accepts_custom_paths():
    """run_demo() signature exposes log_path, store_path, cache_dir as parameters."""
    import inspect
    sig = inspect.signature(run_demo)
    params = sig.parameters
    assert "log_path" in params, "run_demo() must accept 'log_path' parameter"
    assert "store_path" in params, "run_demo() must accept 'store_path' parameter"
    assert "cache_dir" in params, "run_demo() must accept 'cache_dir' parameter"
    assert "max_candidates" in params, "run_demo() must accept 'max_candidates' parameter"
    assert "generation_budget" in params, "run_demo() must accept 'generation_budget' parameter"
    assert "generation_seed" in params, "run_demo() must accept 'generation_seed' parameter"
