"""
Phase 5.5 Autonomous Research Agent — Component 1 Demo Script.

Manual execution entry point for testing the research proposal engine in isolation.
Runs exactly one round:
  1. Introspects primitive registry vocabulary.
  2. Queries trial ledger history (handles cold start gracefully).
  3. Prompts LLM (or deterministic fallback / test mode).
  4. Passes all proposals through the Validation Gate (parsing, vocabulary, deduplication).
  5. Saves output to data/agent/research_rounds/round_001.json and prints results.

Usage:
  python experiments/research_round_demo.py
  python experiments/research_round_demo.py --test-invalid
"""

import argparse
import sys
from pathlib import Path
import json

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.research_proposer import run_research_round, summarize_trial_history
from agent.registry_introspection import get_registry_summary
from groq_caller import groq_llm_caller

def run_demo(test_invalid: bool = False, round_number: int = 1, n_proposals: int = 8):
    print("=" * 90)
    print("ALPHA FORENSICS — PHASE 5.5 AUTONOMOUS RESEARCH AGENT (COMPONENT 1 DEMO)")
    print("=" * 90)

    # 1. Print Registry Introspection Summary
    print("\n[Step 1] Introspecting Official Primitive & Operator Vocabulary...")
    registry_text = get_registry_summary()
    print(registry_text[:600] + "\n... [registry specification truncated for console] ...\n")

    # 2. Print Trial History Summary
    print("[Step 2] Summarizing Current Trial Ledger History...")
    history_info = summarize_trial_history()
    print(history_info["text_summary"])

    # 3. Test Invalid Proposal Mode (if requested via --test-invalid)
    extra_test_proposals = None
    if test_invalid:
        print("\n[TEST MODE] Injecting custom invalid & valid test proposals into Validation Gate...")
        extra_test_proposals = [
            {
                "expression_string": "Rank(Multiply(Momentum(20), VolumeChange(5)))",
                "rationale": "Valid proposal: Cross-sectional rank of momentum weighted by volume change.",
            },
            {
                "expression_string": "NonExistentPrimitive(60)",
                "rationale": "Invalid proposal: Uses non-existent primitive name 'NonExistentPrimitive'.",
            },
            {
                "expression_string": "Momentum(999)",
                "rationale": "Invalid proposal: Uses lookback 999 not in allowed parameter grid [20, 60, 126, 252].",
            },
            {
                "expression_string": "Divide(RollingZScore(10), RealizedVolatility(20))",
                "rationale": "",  # Empty rationale
            },
        ]

    # 4. Run Component 1 Proposal Round
    print(f"\n[Step 3] Running Research Proposal Engine (Round {round_number}, N={n_proposals})...")
    res_round = run_research_round(
        round_number=round_number,
        n_proposals=n_proposals,
        extra_test_proposals=extra_test_proposals,
        llm_caller=groq_llm_caller,
    )

    # 5. Print Results
    print("\n" + "=" * 90)
    print(f"RESEARCH ROUND {res_round.round_number:03d} RESULTS")
    print("=" * 90)
    print(f"Saved Output File : {res_round.output_filepath}")
    print(f"Accepted Proposals: {len(res_round.accepted_proposals)}")
    print(f"Rejected Proposals: {len(res_round.rejected_proposals)}")
    print("-" * 90)

    print("\nACCEPTED PROPOSALS:")
    for prop in res_round.accepted_proposals:
        print(f"  [ACCEPTED] ID: {prop['proposal_id']} | Hash: {prop['canonical_hash'][:12]}")
        print(f"    Expression : {prop['expression_string']}")
        print(f"    Rationale  : {prop['rationale']}")
        print()

    if res_round.rejected_proposals:
        print("REJECTED PROPOSALS (VALIDATION GATE):")
        for prop in res_round.rejected_proposals:
            print(f"  [REJECTED] ID: {prop['proposal_id']} | Reason: {prop['rejection_reason']}")
            print(f"    Expression : {prop['expression_string']}")
            if prop['rationale']:
                print(f"    Rationale  : {prop['rationale']}")
            print()

    print("=" * 90)
    print("COMPONENT 1 EXECUTION COMPLETE.")
    print("Output available for manual inspection at:", res_round.output_filepath)
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5.5 Research Agent Component 1 Demo")
    parser.add_argument("--test-invalid", action="store_true", help="Inject malformed/invalid test proposals into the validation gate")
    parser.add_argument("--round", type=int, default=1, help="Research round number (default: 1)")
    parser.add_argument("--proposals", type=int, default=8, help="Number of proposals to request (default: 8)")
    args = parser.parse_args()

    run_demo(test_invalid=args.test_invalid, round_number=args.round, n_proposals=args.proposals)
