"""
Phase 5.5 Autonomous Research Agent — Component 1 (Reasoning & Proposal Engine).

Responsibility:
  1. Queries existing trial ledgers (trial_log.jsonl, dsr_results.jsonl, trial_status_log.jsonl)
     and builds a structured summary of evaluated alpha history.
  2. Constructs a prompt containing the history summary and the official primitive/operator
     vocabulary spec from registry_introspection.py.
  3. Prompts the LLM (or mock fallback) to propose N candidate expressions with rationale.
  4. Runs a strict Validation Gate on every proposal:
     - Syntax/AST parsing via serialize.parse()
     - Primitive vocabulary & parameter grid validation via validate_expression_vocabulary()
     - Single-look / deduplication check against already-tried canonical_hash() values
  5. Outputs a ResearchRound object containing round metadata, history summary, accepted proposals,
     and rejected proposals with explicit rejection reasons, saving to data/agent/research_rounds/.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable, Union

from alpha.expressions.serialize import parse, to_string
from alpha.expressions.tree import Expression
from statistics import (
    load_trial_log,
    load_latest_dsr_verdicts,
    effective_trial_records,
    TrialRecord,
)
from agent.registry_introspection import (
    get_registry_summary,
    validate_expression_vocabulary,
    get_known_primitive_names,
)

logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    proposal_id: str
    expression_string: str
    canonical_hash: str
    rationale: str
    status: str  # "ACCEPTED" or "REJECTED"
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchRound:
    round_number: int
    timestamp: str
    num_requested: int
    history_summary: str
    accepted_proposals: List[Dict[str, Any]]
    rejected_proposals: List[Dict[str, Any]]
    output_filepath: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_trial_history(
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    dsr_log_path: Union[str, Path] = "data/dsr_results.jsonl",
) -> Dict[str, Any]:
    """
    Queries trial ledger files and constructs a summary dictionary of past evaluations.

    Handles cold-start / empty ledgers gracefully.
    """
    log_p = Path(log_path)
    status_p = Path(status_log_path)
    dsr_p = Path(dsr_log_path)

    if not log_p.exists() or not log_p.read_text().strip():
        return {
            "has_history": False,
            "total_trials": 0,
            "tried_hashes": set(),
            "tried_ids": [],
            "text_summary": (
                "No historical trials logged yet. This is Round 1 (Cold Start).\n"
                "Focus on proposing diverse, foundational exploratory candidates across momentum, "
                "volatility, volume, and rolling trend primitives."
            ),
        }

    all_records = load_trial_log(log_p)
    effective_records = effective_trial_records(log_path=log_p, status_log_path=status_p)
    dsr_verdicts = load_latest_dsr_verdicts(log_path=dsr_p)

    tried_hashes: Set[str] = set()
    tried_ids: List[str] = []

    # Map candidate IDs and hashes
    for rec in all_records:
        tried_ids.append(rec.candidate_id)
        # Attempt to parse candidate_id if it contains generated hash or standard formula
        # We also attempt to parse expression if recorded
        try:
            expr = parse(rec.candidate_id)
            tried_hashes.add(expr.canonical_hash())
        except Exception:
            pass

    # Build text summary for prompt
    lines = [
        f"Historical Trial Summary (Total Trials Logged: {len(all_records)}, Active Effective: {len(effective_records)}):",
        "",
        "Top Evaluated Candidates & Outcomes:",
    ]

    top_sorted = sorted(effective_records, key=lambda x: x.oos_sharpe, reverse=True)[:10]
    for r in top_sorted:
        dsr_info = dsr_verdicts.get(r.candidate_id)
        dsr_str = f"DSR={dsr_info.dsr_score:.4f} ({dsr_info.verdict})" if dsr_info else "DSR=Not Evaluated"
        lines.append(
            f"  - ID: {r.candidate_id:<40} | Phase: {r.phase:<35} | "
            f"OOS Sharpe: {r.oos_sharpe:>7.4f} | {dsr_str} | Status: {r.status}"
        )

    lines.append("")
    lines.append("Key Research Directives:")
    lines.append("  - Pure momentum (PriceReturn/Momentum) often fails DSR or factor tests when unadjusted.")
    lines.append("  - RealizedVolatility long-high-vol candidates load heavily on the VOL factor.")
    lines.append("  - Search for non-linear combinations (VolumeChange, Turnover, Drawdown, ZScore, Rank) with robust risk controls.")

    return {
        "has_history": True,
        "total_trials": len(all_records),
        "tried_hashes": tried_hashes,
        "tried_ids": tried_ids,
        "text_summary": "\n".join(lines),
    }


def build_proposal_prompt(history_summary_text: str, n_proposals: int = 8) -> str:
    """Builds the exact prompt instructing the LLM to generate N validated candidate proposals."""
    vocab_summary = get_registry_summary()
    
    prompt = f"""You are the Lead Quantitative Research Agent for Alpha Forensics.
Your objective is to propose {n_proposals} novel, hypothesis-driven alpha candidate expressions.

{vocab_summary}

CURRENT TRIAL LEDGER SUMMARY:
{history_summary_text}

INSTRUCTIONS & RULES:
1. Propose EXACTLY {n_proposals} candidate expressions in valid JSON format.
2. Every proposal MUST contain two keys:
   - "expression_string": A parseable string using ONLY the primitives/operators in the Vocabulary Whitelist above.
   - "rationale": A clear, non-empty plain-English quantitative hypothesis explaining why this expression is expected to generate non-redundant alpha.
3. Syntax Rules:
   - Leaf primitives must specify parameters, e.g. PriceReturn(1), Momentum(20), RollingMean(10), VolumeChange(5), Turnover(20), RealizedVolatility(20), Drawdown(20).
   - Unary operators wrap a child, e.g. Rank(Momentum(20)), ZScore(RealizedVolatility(60)), Negate(PriceReturn(5)), Lag(Turnover(20), 5).
   - Binary operators take two children, e.g. Multiply(Rank(Momentum(20)), ZScore(VolumeChange(5))), Divide(RollingZScore(10), RealizedVolatility(60)).
4. DO NOT include any markdown code blocks, explanation text, or commentary outside the JSON array.

OUTPUT FORMAT (JSON ARRAY ONLY):
[
  {{
    "expression_string": "Rank(Multiply(Momentum(20), VolumeChange(5)))",
    "rationale": "Cross-sectional ranking of momentum amplified by volume expansion to isolate high-conviction institutional flow."
  }}
]
"""
    return prompt


def _default_mock_proposals(n_proposals: int = 8) -> List[Dict[str, str]]:
    """
    Default structured proposals used when running in offline/test mode without LLM credentials.
    Ensures manual testing and cold-start execution work flawlessly.
    """
    templates = [
        (
            "Rank(Multiply(Momentum(20), VolumeChange(5)))",
            "Cross-sectional ranking of 20-day momentum weighted by 5-day volume expansion to capture volume-confirmed price trends."
        ),
        (
            "Divide(RollingZScore(10), RealizedVolatility(20))",
            "Short-term price Z-score normalized by 20-day realized volatility to isolate risk-adjusted mean-reverting signals."
        ),
        (
            "Subtract(Rank(PriceReturn(5)), Rank(Turnover(20)))",
            "Long high 5-day return assets that have low 20-day turnover, capturing quiet price momentum under institutional accumulation."
        ),
        (
            "Multiply(ZScore(Drawdown(60)), Negate(SignedPower(RollingZScore(20), 2.0)))",
            "Combines deep drawdown recovery potential with non-linear suppression of recent extreme Z-score volatility."
        ),
        (
            "Lag(Rank(VolumeChange(20)), 1)",
            "Lagged 1-day cross-sectional rank of 20-day volume change to prevent execution lag mismatch."
        ),
        (
            "ZScore(RealizedVolatility(60))",
            "Cross-sectional standardization of 60-day realized volatility for factor exposure testing."
        ),
        (
            "Diff(Multiply(RollingStd(10), Abs(RollingZScore(10))), 1)",
            "One-day change in volatility-scaled magnitude of short-term price deviation."
        ),
        (
            "Add(Turnover(5), Subtract(Turnover(60), RollingZScore(60)))",
            "Multi-scale turnover divergence combining short-term volume spikes with long-term trend baseline."
        ),
        (
            "FakePrimitive(20)",  # Intentional invalid test candidate if n > 8 requested
            "Test invalid primitive for validation gate testing."
        ),
    ]
    
    results = []
    for i in range(min(n_proposals, len(templates))):
        expr_str, rat = templates[i]
        results.append({"expression_string": expr_str, "rationale": rat})
    return results


def call_llm(prompt: str, llm_caller: Optional[Callable[[str], str]] = None) -> str:
    """
    Executes LLM call using:
      1. Provided custom llm_caller function, if passed.
      2. google-genai Client if GEMINI_API_KEY is present in environment.
      3. Fallback to structured mock JSON if no credentials/caller available.
    """
    if llm_caller is not None:
        return llm_caller(prompt)

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            # Try available models in order of speed/availability
            for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if response.text:
                        return response.text
                except Exception as e:
                    logger.warning(f"LLM model {model_name} failed: {e}. Trying next model...")
        except Exception as e:
            logger.warning(f"Failed to initialize google-genai Client ({e}). Using mock proposal fallback.")

    logger.info("No active GEMINI_API_KEY or LLM caller provided. Using deterministic research proposal generator.")
    return json.dumps(_default_mock_proposals())


def validate_proposal(
    prop_raw: Dict[str, Any],
    existing_hashes: Set[str],
    proposal_idx: int,
) -> Proposal:
    """
    Validation Gate for a single raw proposal:
      1. Check non-empty rationale string.
      2. Parse AST via serialize.parse(expression_string).
      3. Validate primitive vocabulary & parameter values via validate_expression_vocabulary().
      4. Check canonical_hash() duplicate collision against existing_hashes.

    Returns:
        Proposal object marked ACCEPTED or REJECTED with explicit rejection_reason.
    """
    prop_id = f"proposal_{proposal_idx:03d}"
    expr_str = str(prop_raw.get("expression_string", "")).strip()
    rationale = str(prop_raw.get("rationale", "")).strip()

    if not expr_str:
        return Proposal(
            proposal_id=prop_id,
            expression_string="",
            canonical_hash="",
            rationale=rationale,
            status="REJECTED",
            rejection_reason="missing_expression_string",
        )

    if not rationale:
        return Proposal(
            proposal_id=prop_id,
            expression_string=expr_str,
            canonical_hash="",
            rationale="",
            status="REJECTED",
            rejection_reason="missing_rationale",
        )

    # 1. Parse AST
    try:
        expr = parse(expr_str)
    except Exception as exc:
        return Proposal(
            proposal_id=prop_id,
            expression_string=expr_str,
            canonical_hash="",
            rationale=rationale,
            status="REJECTED",
            rejection_reason=f"parse_error: {exc}",
        )

    # 2. Vocabulary & Parameter Grid Validation
    violations = validate_expression_vocabulary(expr)
    if violations:
        return Proposal(
            proposal_id=prop_id,
            expression_string=expr_str,
            canonical_hash=expr.canonical_hash(),
            rationale=rationale,
            status="REJECTED",
            rejection_reason=f"invalid_vocabulary: {'; '.join(violations)}",
        )

    # 3. Canonical Hash Duplicate Check
    c_hash = expr.canonical_hash()
    if c_hash in existing_hashes:
        return Proposal(
            proposal_id=prop_id,
            expression_string=expr_str,
            canonical_hash=c_hash,
            rationale=rationale,
            status="REJECTED",
            rejection_reason=f"duplicate_hash: candidate with hash '{c_hash[:12]}' already exists in trial history",
        )

    # All checks passed! Register hash to prevent duplicates within the same round
    existing_hashes.add(c_hash)

    return Proposal(
        proposal_id=prop_id,
        expression_string=to_string(expr),
        canonical_hash=c_hash,
        rationale=rationale,
        status="ACCEPTED",
        rejection_reason=None,
    )


def run_research_round(
    round_number: int = 1,
    n_proposals: int = 8,
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    dsr_log_path: Union[str, Path] = "data/dsr_results.jsonl",
    output_dir: Union[str, Path] = "data/agent/research_rounds",
    llm_caller: Optional[Callable[[str], str]] = None,
    extra_test_proposals: Optional[List[Dict[str, Any]]] = None,
) -> ResearchRound:
    """
    Executes Component 1 Research Proposal Round:
      1. Summarizes trial history from ledgers.
      2. Builds prompt and calls LLM (or caller/mock).
      3. Runs Validation Gate on all proposals (logging rejections).
      4. Saves ResearchRound result to data/agent/research_rounds/round_NNN.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summarize History
    history_info = summarize_trial_history(
        log_path=log_path,
        status_log_path=status_log_path,
        dsr_log_path=dsr_log_path,
    )

    existing_hashes: Set[str] = set(history_info.get("tried_hashes", set()))

    # 2. Build Prompt & Get Raw Proposals
    if extra_test_proposals is not None:
        raw_proposals = extra_test_proposals
    else:
        prompt = build_proposal_prompt(history_summary_text=history_info["text_summary"], n_proposals=n_proposals)
        response_text = call_llm(prompt, llm_caller=llm_caller)

        # Parse JSON response
        try:
            # Strip markdown code blocks if model wrapped output in ```json ... ```
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            
            raw_proposals = json.loads(cleaned)
        except Exception as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}. Raw response:\n{response_text}")
            raw_proposals = _default_mock_proposals(n_proposals=n_proposals)

    if not isinstance(raw_proposals, list):
        raw_proposals = [raw_proposals]

    # 3. Validation Gate
    accepted_list: List[Dict[str, Any]] = []
    rejected_list: List[Dict[str, Any]] = []

    for idx, prop_raw in enumerate(raw_proposals, start=1):
        prop_res = validate_proposal(prop_raw, existing_hashes, idx)
        if prop_res.status == "ACCEPTED":
            accepted_list.append(prop_res.to_dict())
        else:
            rejected_list.append(prop_res.to_dict())

    # 4. Save Research Round Output
    out_file = output_dir / f"round_{round_number:03d}.json"
    res_round = ResearchRound(
        round_number=round_number,
        timestamp=datetime.now().isoformat(),
        num_requested=n_proposals,
        history_summary=history_info["text_summary"],
        accepted_proposals=accepted_list,
        rejected_proposals=rejected_list,
        output_filepath=str(out_file),
    )

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(res_round.to_dict(), f, indent=2)

    logger.info(
        f"Research Round {round_number:03d} complete: "
        f"{len(accepted_list)} accepted, {len(rejected_list)} rejected. "
        f"Saved to {out_file}"
    )

    return res_round
