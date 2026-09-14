"""
Unit tests for Phase 5.5 Autonomous Research Agent Component 1 (Proposer & Validation Gate).
"""

import json
import pytest
from pathlib import Path

from alpha.expressions.serialize import parse
from agent.registry_introspection import (
    get_registry_summary,
    get_known_primitive_names,
    validate_expression_vocabulary,
)
from agent.research_proposer import (
    summarize_trial_history,
    build_proposal_prompt,
    validate_proposal,
    run_research_round,
    ResearchRound,
    Proposal,
)


def test_registry_summary_contains_all_primitives():
    """Verify registry summary text includes key primitives and parameter rules."""
    summary = get_registry_summary()
    assert "PriceReturn(lag)" in summary
    assert "Momentum(lookback)" in summary
    assert "RealizedVolatility(window)" in summary
    assert "Rank(expr)" in summary
    assert "ZScore(expr)" in summary
    assert "Multiply(left, right)" in summary


def test_known_primitive_names():
    """Verify get_known_primitive_names returns all core leaf and operator names."""
    names = get_known_primitive_names()
    assert "Momentum" in names
    assert "RealizedVolatility" in names
    assert "Rank" in names
    assert "Divide" in names
    assert len(names) >= 15


def test_validate_expression_vocabulary_valid():
    """Verify valid expression tree returns no vocabulary violations."""
    expr = parse("Rank(Multiply(Momentum(20), VolumeChange(5)))")
    violations = validate_expression_vocabulary(expr)
    assert len(violations) == 0


def test_validate_expression_vocabulary_invalid_parameter():
    """Verify unlisted parameter value triggers a vocabulary violation."""
    expr = parse("Momentum(999)")
    violations = validate_expression_vocabulary(expr)
    assert len(violations) > 0
    assert "Invalid parameter value for Momentum.lookback" in violations[0]


def test_validation_gate_rejection_reasons():
    """Verify validation gate catches parse errors, invalid vocabulary, duplicates, and missing rationales."""
    existing_hashes = set()

    # 1. Missing rationale
    p1 = validate_proposal(
        {"expression_string": "Rank(Momentum(20))", "rationale": ""},
        existing_hashes,
        proposal_idx=1,
    )
    assert p1.status == "REJECTED"
    assert p1.rejection_reason == "missing_rationale"

    # 2. Parse error (unknown primitive)
    p2 = validate_proposal(
        {"expression_string": "FakePrim(10)", "rationale": "Valid rationale"},
        existing_hashes,
        proposal_idx=2,
    )
    assert p2.status == "REJECTED"
    assert "parse_error" in p2.rejection_reason

    # 3. Invalid parameter grid
    p3 = validate_proposal(
        {"expression_string": "PriceReturn(99)", "rationale": "Valid rationale"},
        existing_hashes,
        proposal_idx=3,
    )
    assert p3.status == "REJECTED"
    assert "invalid_vocabulary" in p3.rejection_reason

    # 4. Valid proposal
    p4 = validate_proposal(
        {"expression_string": "Rank(Momentum(20))", "rationale": "Valid rationale"},
        existing_hashes,
        proposal_idx=4,
    )
    assert p4.status == "ACCEPTED"
    assert p4.rejection_reason is None

    # 5. Duplicate hash check
    p5 = validate_proposal(
        {"expression_string": "Rank(Momentum(20))", "rationale": "Another rationale"},
        existing_hashes,
        proposal_idx=5,
    )
    assert p5.status == "REJECTED"
    assert "duplicate_hash" in p5.rejection_reason


def test_run_research_round_creates_output_file(tmp_path):
    """Verify run_research_round creates valid JSON output file in output_dir."""
    round_res = run_research_round(
        round_number=1,
        n_proposals=4,
        output_dir=tmp_path / "rounds",
    )

    out_file = Path(round_res.output_filepath)
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["round_number"] == 1
    assert "accepted_proposals" in data
    assert "rejected_proposals" in data
    assert len(data["accepted_proposals"]) + len(data["rejected_proposals"]) >= 4
