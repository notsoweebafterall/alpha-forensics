"""
Alpha Forensics Report Builder — Phase 16.

What This Module Does:
    Reads already-computed, durable artifacts (alpha_registry.db, trial_returns.parquet,
    dsr_results.jsonl) and formats them into a durable, readable report document.

Single-Look Discipline — Why This Phase Needs No New Scrutiny:
    Phase 16 does NOT call run_walk_forward_validation(), run_dsr_analysis(), or any
    function that queries held-out OOS data for the first time. It reads exclusively from
    data already logged and observed during earlier phases:
      - data/alpha_registry.db     (built from trial_log.jsonl + trial_status_log.jsonl)
      - data/trial_returns.parquet (already-logged OOS net return series)
      - data/dsr_results.jsonl     (already-logged DSR verdicts)
    This is a pure read/aggregate/format layer. No single-look concerns apply here beyond
    those already exercised in Phases 7–15.

Freshness Design:
    build_report() calls build_registry() at the start by default (skip_rebuild=False),
    guaranteeing the report reflects the current on-disk state of trial_log.jsonl,
    trial_status_log.jsonl, and dsr_results.jsonl. Set skip_rebuild=True only when the
    caller has already just rebuilt the registry in the same session.

Honesty Principle:
    The report must state the data's actual current state — whether encouraging or not.
    An Executive Summary that reads like a positive pitch when the underlying numbers say
    "no viable alpha found yet" would be a form of misrepresentation, the same category of
    problem this project exists to prevent.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional, Union
import math

import pandas as pd

from registry.build_registry import build_registry
from registry.query import get_all_candidates, get_valid_non_redundant_candidates
from portfolio.portfolio_construction import build_portfolio


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------

@dataclass
class ReportResult:
    """
    Summary of a completed build_report() call.
    Allows callers and tests to inspect key numbers without re-parsing the markdown.
    """
    markdown_path: Path
    json_path: Optional[Path]          # None if write_json=False
    timestamp: str
    total_candidates: int
    valid_top_level_candidates: int
    candidates_with_returns: int
    candidates_failing_dsr: int        # explicit fails_dsr verdict
    candidates_not_dsr_evaluated: int  # NULL dsr_verdict among has-returns candidates
    candidates_surviving_dsr: int      # survives_dsr
    portfolio_rigorous_eligible: int   # 0 = empty under require_dsr_survival=True
    portfolio_relaxed_eligible: int
    executive_summary: str


# ---------------------------------------------------------------------------
# Caveat Text (quoted from source module docstrings — do not paraphrase divergently)
# ---------------------------------------------------------------------------

_CAVEAT_DSR = """\
**DSR Limitations** (from `src/statistics/dsr.py`):

DSR corrects for selection bias and data mining bias resulting from evaluating multiple
candidate alpha strategies (N trials) before selecting the top-performing strategy. It
computes the probability that the observed Sharpe ratio exceeds the expected maximum
Sharpe ratio under the null hypothesis that all trials are uninformative.

DSR specifically addresses selection bias across multiple trial evaluations. It does
**NOT** protect against:
- Data snooping or lookahead bias embedded directly into feature engineering or universe definition.
- Structural non-stationarity or macroeconomic regime shifts.
- Mismodeling of execution friction, market impact, or transaction costs.
- Overfitting within parameter landscapes when parameter tuning is unconstrained.
"""

_CAVEAT_REGIME = """\
**Regime Coverage Limitation** (from `src/regimes/regime_analysis.py`):

Because the OOS holdout window sits chronologically at the end of the historical dataset
by construction, regime robustness testing evaluates candidate performance ONLY across
whichever market regimes actually occurred within that specific tail window. Regimes not
present in the OOS tail window (or with fewer than `min_regime_days` observations) are
explicitly recorded in `skipped_regimes` with explanatory reasons rather than silently
dropped, preventing callers from mistaking partial regime coverage for full-spectrum
evaluation across all 4 market regimes.
"""

_CAVEAT_REDUNDANCY = """\
**Redundancy Analysis Scope Restriction** (from `src/redundancy/correlation_analysis.py`):

Redundancy analysis is strictly restricted to top-level candidates evaluated over the full
OOS window. Sub-slice trials containing `__sector_`, `__period_`, `__regime_`, or `__cost_`
in their candidate_id are filtered out because comparing a sub-slice to a full-window
candidate mixes different underlying populations and date ranges.

Candidates logged to `trial_log.jsonl` before Phase 13 was deployed do not have a stored
returns series in `trial_returns.parquet`. They are reported with an explicit reason
("no stored returns series") rather than silently dropped or crashed on.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_sharpe(v) -> str:
    """Format a Sharpe value, handling None/NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{float(v):+.4f}"


def _fmt_verdict(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NULL (not evaluated)"
    return str(v)


def _fmt_redundant_with(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return str(v)


def _md_table(df: pd.DataFrame, columns: dict) -> str:
    """
    Render a subset of df columns as a markdown table.
    columns: {src_col: header_label}
    """
    headers = list(columns.values())
    src_cols = list(columns.keys())
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in src_cols:
            v = row.get(c)
            cells.append(str(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Builder
# ---------------------------------------------------------------------------

def build_report(
    db_path: Union[str, Path] = "data/alpha_registry.db",
    returns_store_path: Union[str, Path] = "data/trial_returns.parquet",
    dsr_log_path: Union[str, Path] = "data/dsr_results.jsonl",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    output_dir: Union[str, Path] = "reports/",
    skip_rebuild: bool = False,
    write_json: bool = True,
    top_n_candidates: int = 15,
) -> ReportResult:
    """
    Build a durable alpha forensics report from the current registry and portfolio state.

    Args:
        db_path: Path to SQLite registry database.
        returns_store_path: Path to trial_returns.parquet.
        dsr_log_path: Path to dsr_results.jsonl.
        log_path: Path to trial_log.jsonl (for registry rebuild).
        status_log_path: Path to trial_status_log.jsonl (for registry rebuild).
        output_dir: Directory where report files will be written (created if missing).
        skip_rebuild: If False (default), calls build_registry() before reading.
                      Set True only if caller has already rebuilt the registry this session.
        write_json: If True, also writes a companion JSON report file.
        top_n_candidates: Number of top-level candidates to show in the per-candidate table.

    Returns:
        ReportResult dataclass with paths written and key summary numbers.
    """
    db_path = Path(db_path)
    returns_store_path = Path(returns_store_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ------------------------------------------------------------------
    # Step 1: Rebuild registry (default) for freshness guarantee
    # ------------------------------------------------------------------
    if not skip_rebuild:
        build_registry(
            db_path=db_path,
            log_path=log_path,
            status_log_path=status_log_path,
            store_path=returns_store_path,
            dsr_log_path=dsr_log_path,
        )

    # ------------------------------------------------------------------
    # Step 2: Read registry data
    # ------------------------------------------------------------------
    df_all = get_all_candidates(db_path=db_path)
    df_top = get_valid_non_redundant_candidates(db_path=db_path, top_level_only=True)

    # Funnel counts
    total_candidates = len(df_all)
    # All VALID candidates (any scope)
    valid_all = len(df_all[df_all["effective_status"] == "VALID"])
    # Valid, non-redundant, top-level
    valid_top_level = len(df_top)
    # + has returns
    df_top_ret = df_top[df_top["has_returns_series"] == 1]
    candidates_with_returns = len(df_top_ret)
    # DSR breakdown among has-returns candidates
    candidates_surviving_dsr = int((df_top_ret["dsr_verdict"] == "survives_dsr").sum())
    candidates_failing_dsr = int((df_top_ret["dsr_verdict"] == "fails_dsr").sum())
    candidates_not_evaluated = int(df_top_ret["dsr_verdict"].isna().sum())

    # ------------------------------------------------------------------
    # Step 3: Run portfolio construction (both modes)
    # ------------------------------------------------------------------
    port_rigorous = build_portfolio(
        db_path=db_path,
        returns_store_path=returns_store_path,
        weighting="equal_weight",
        require_dsr_survival=True,
    )
    port_relaxed_eq = build_portfolio(
        db_path=db_path,
        returns_store_path=returns_store_path,
        weighting="equal_weight",
        require_dsr_survival=False,
    )
    port_relaxed_iv = build_portfolio(
        db_path=db_path,
        returns_store_path=returns_store_path,
        weighting="inverse_volatility",
        require_dsr_survival=False,
    )

    port_rigorous_n = len(port_rigorous.eligible_candidates)
    port_relaxed_n = len(port_relaxed_eq.eligible_candidates)

    # ------------------------------------------------------------------
    # Step 4: Build Executive Summary (honest, matches data)
    # ------------------------------------------------------------------
    exec_summary = _build_executive_summary(
        total_candidates=total_candidates,
        valid_top_level=valid_top_level,
        candidates_with_returns=candidates_with_returns,
        candidates_surviving_dsr=candidates_surviving_dsr,
        candidates_failing_dsr=candidates_failing_dsr,
        candidates_not_evaluated=candidates_not_evaluated,
        port_rigorous_n=port_rigorous_n,
    )

    # ------------------------------------------------------------------
    # Step 5: Build top-N candidate table (all top-level, by oos_sharpe)
    # ------------------------------------------------------------------
    # Include all top-level candidates (not just valid) for honest picture
    _TOP_LEVEL_EXCLUSIONS = ["__sector_", "__period_", "__regime_", "__cost_"]
    mask_top = df_all["candidate_id"].apply(
        lambda cid: all(exc not in cid for exc in _TOP_LEVEL_EXCLUSIONS)
    )
    df_top_all = df_all[mask_top].sort_values("oos_sharpe", ascending=False).head(top_n_candidates).copy()
    df_top_all["oos_sharpe_fmt"] = df_top_all["oos_sharpe"].apply(_fmt_sharpe)
    df_top_all["dsr_verdict_fmt"] = df_top_all["dsr_verdict"].apply(_fmt_verdict)
    df_top_all["redundant_with_fmt"] = df_top_all["redundant_with"].apply(_fmt_redundant_with)

    # ------------------------------------------------------------------
    # Step 6: Assemble markdown
    # ------------------------------------------------------------------
    md = _assemble_markdown(
        timestamp=timestamp,
        exec_summary=exec_summary,
        total_candidates=total_candidates,
        valid_all=valid_all,
        valid_top_level=valid_top_level,
        candidates_with_returns=candidates_with_returns,
        candidates_surviving_dsr=candidates_surviving_dsr,
        candidates_failing_dsr=candidates_failing_dsr,
        candidates_not_evaluated=candidates_not_evaluated,
        df_top_all=df_top_all,
        port_rigorous=port_rigorous,
        port_relaxed_eq=port_relaxed_eq,
        port_relaxed_iv=port_relaxed_iv,
    )

    # ------------------------------------------------------------------
    # Step 7: Write markdown file
    # ------------------------------------------------------------------
    md_filename = f"alpha_forensics_report_{timestamp}.md"
    md_path = output_dir / md_filename
    md_path.write_text(md, encoding="utf-8")

    # ------------------------------------------------------------------
    # Step 8: Optionally write JSON companion
    # ------------------------------------------------------------------
    json_path: Optional[Path] = None
    if write_json:
        json_data = {
            "timestamp": timestamp,
            "funnel": {
                "total_logged": total_candidates,
                "valid_all": valid_all,
                "valid_top_level_non_redundant": valid_top_level,
                "with_returns": candidates_with_returns,
                "survives_dsr": candidates_surviving_dsr,
                "fails_dsr": candidates_failing_dsr,
                "not_dsr_evaluated": candidates_not_evaluated,
            },
            "portfolio": {
                "rigorous_eligible_count": port_rigorous_n,
                "rigorous_message": port_rigorous.message,
                "relaxed_eligible_count": port_relaxed_n,
                "relaxed_equal_weight_sharpe": port_relaxed_eq.portfolio_sharpe,
                "relaxed_inverse_vol_sharpe": port_relaxed_iv.portfolio_sharpe,
                "relaxed_constituents": port_relaxed_eq.eligible_candidates,
                "relaxed_equal_weights": port_relaxed_eq.weights,
                "relaxed_inverse_vol_weights": port_relaxed_iv.weights,
            },
            "top_candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "phase": row.get("phase"),
                    "oos_sharpe": row["oos_sharpe"],
                    "effective_status": row["effective_status"],
                    "dsr_verdict": None if pd.isna(row.get("dsr_verdict")) else row["dsr_verdict"],
                    "redundant_with": None if pd.isna(row.get("redundant_with")) else row["redundant_with"],
                }
                for _, row in df_top_all.iterrows()
            ],
        }
        json_filename = f"alpha_forensics_report_{timestamp}.json"
        json_path = output_dir / json_filename
        json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")

    return ReportResult(
        markdown_path=md_path,
        json_path=json_path,
        timestamp=timestamp,
        total_candidates=total_candidates,
        valid_top_level_candidates=valid_top_level,
        candidates_with_returns=candidates_with_returns,
        candidates_failing_dsr=candidates_failing_dsr,
        candidates_not_dsr_evaluated=candidates_not_evaluated,
        candidates_surviving_dsr=candidates_surviving_dsr,
        portfolio_rigorous_eligible=port_rigorous_n,
        portfolio_relaxed_eligible=port_relaxed_n,
        executive_summary=exec_summary,
    )


# ---------------------------------------------------------------------------
# Section Builders
# ---------------------------------------------------------------------------

def _build_executive_summary(
    total_candidates: int,
    valid_top_level: int,
    candidates_with_returns: int,
    candidates_surviving_dsr: int,
    candidates_failing_dsr: int,
    candidates_not_evaluated: int,
    port_rigorous_n: int,
) -> str:
    """
    3-4 sentence plain statement of current state. Written to match the data, not to flatter.
    """
    lines = []
    lines.append(
        f"The project has evaluated {total_candidates} total candidate alpha strategies "
        f"across all phases, of which {valid_top_level} are currently valid, non-redundant, "
        f"top-level candidates."
    )
    if candidates_with_returns == 0:
        lines.append(
            "None of those candidates have stored OOS daily net returns, "
            "so portfolio construction and DSR evaluation are not possible from current data."
        )
    else:
        dsr_line_parts = []
        if candidates_failing_dsr > 0:
            dsr_line_parts.append(f"{candidates_failing_dsr} explicitly fail DSR")
        if candidates_not_evaluated > 0:
            dsr_line_parts.append(f"{candidates_not_evaluated} have not been DSR-evaluated")
        if candidates_surviving_dsr > 0:
            dsr_line_parts.append(f"{candidates_surviving_dsr} survive DSR")

        dsr_summary = " and ".join(dsr_line_parts) if dsr_line_parts else "none have been DSR-evaluated"
        lines.append(
            f"Of the {candidates_with_returns} candidates with stored returns, {dsr_summary}."
        )

    if port_rigorous_n == 0:
        lines.append(
            "Under rigorous portfolio eligibility criteria (must survive DSR), "
            "the current portfolio is empty — no candidate meets all criteria. "
            "This is the honest, correct result given present alpha quality, not a reporting gap."
        )
    else:
        lines.append(
            f"A {port_rigorous_n}-candidate portfolio is constructible under rigorous DSR criteria."
        )

    return " ".join(lines)


def _assemble_markdown(
    timestamp: str,
    exec_summary: str,
    total_candidates: int,
    valid_all: int,
    valid_top_level: int,
    candidates_with_returns: int,
    candidates_surviving_dsr: int,
    candidates_failing_dsr: int,
    candidates_not_evaluated: int,
    df_top_all: pd.DataFrame,
    port_rigorous,
    port_relaxed_eq,
    port_relaxed_iv,
) -> str:
    sections = []

    # ---- Header ----
    sections.append(f"# Alpha Forensics Report\n\n**Generated:** {timestamp} UTC\n")

    # ---- 1. Executive Summary ----
    sections.append("## 1. Executive Summary\n\n" + exec_summary + "\n")

    # ---- 2. Candidate Funnel ----
    funnel_lines = [
        "## 2. Candidate Eligibility Funnel\n",
        "| Stage | Filter | Count |",
        "| --- | --- | --- |",
        f"| Stage 0 | Total logged candidates (all phases) | {total_candidates} |",
        f"| Stage 0b | Currently VALID (any scope) | {valid_all} |",
        f"| Stage 1 | Valid + non-redundant + top-level only | {valid_top_level} |",
        f"| Stage 2 | + Has stored daily net returns | {candidates_with_returns} |",
        f"| Stage 3a | + Explicitly fails DSR | {candidates_failing_dsr} |",
        f"| Stage 3b | + Not yet DSR-evaluated (verdict NULL) | {candidates_not_evaluated} |",
        f"| Stage 3c | + **Survives DSR** (eligible for rigorous portfolio) | **{candidates_surviving_dsr}** |",
        "",
        "> Stage 3a and 3b candidates are excluded from the rigorous portfolio. "
        "Stage 3b reflects candidates with stored returns that have not yet been explicitly "
        "run through Phase 10 DSR evaluation — not that they passed.",
    ]
    sections.append("\n".join(funnel_lines) + "\n")

    # ---- 3. Per-Candidate Table ----
    if df_top_all.empty:
        cand_section = "## 3. Per-Candidate Summary (Top-Level Candidates)\n\n*No top-level candidates in registry.*\n"
    else:
        table_rows = ["## 3. Per-Candidate Summary (Top-Level Candidates, by OOS Sharpe)\n",
                      "| Candidate ID | Phase | OOS Sharpe | Status | DSR Verdict | Redundant With |",
                      "| --- | --- | --- | --- | --- | --- |"]
        for _, row in df_top_all.iterrows():
            table_rows.append(
                f"| {row['candidate_id']} "
                f"| {row.get('phase', '—')} "
                f"| {row['oos_sharpe_fmt']} "
                f"| {row['effective_status']} "
                f"| {row['dsr_verdict_fmt']} "
                f"| {row['redundant_with_fmt']} |"
            )
        cand_section = "\n".join(table_rows) + "\n"
    sections.append(cand_section)

    # ---- 4. Portfolio Section ----
    port_lines = ["## 4. Portfolio Construction Results\n"]

    # 4a. Rigorous
    port_lines.append("### 4a. Rigorous Mode (`require_dsr_survival=True`)\n")
    port_lines.append(_format_funnel_stage_counts(port_rigorous))
    if port_rigorous.eligible_candidates:
        port_lines.append(_format_weights_table(port_rigorous, None))
        port_lines.append(f"\n**Portfolio OOS Sharpe:** {_fmt_sharpe(port_rigorous.portfolio_sharpe)}\n")
    else:
        port_lines.append(f"\n**Result:** {port_rigorous.message}\n")

    # 4b. Relaxed
    port_lines.append("### 4b. Relaxed Mode (`require_dsr_survival=False`)\n")
    port_lines.append(_format_funnel_stage_counts(port_relaxed_eq))
    if port_relaxed_eq.eligible_candidates:
        port_lines.append(_format_weights_table(port_relaxed_eq, port_relaxed_iv))
        port_lines.append(f"\n**Equal-Weight Portfolio OOS Sharpe:** {_fmt_sharpe(port_relaxed_eq.portfolio_sharpe)}\n")
        port_lines.append(f"**Inverse-Volatility Portfolio OOS Sharpe:** {_fmt_sharpe(port_relaxed_iv.portfolio_sharpe)}\n")
    else:
        port_lines.append(f"\n**Result:** {port_relaxed_eq.message}\n")

    sections.append("\n".join(port_lines) + "\n")

    # ---- 5. Consolidated Caveats ----
    sections.append(
        "## 5. Consolidated Caveats\n\n"
        + _CAVEAT_DSR + "\n"
        + _CAVEAT_REGIME + "\n"
        + _CAVEAT_REDUNDANCY + "\n"
    )

    return "\n---\n\n".join(sections)


def _format_funnel_stage_counts(port_result) -> str:
    lines = [
        "**Eligibility Funnel:**\n",
        f"- Stage 1 (Valid, non-redundant, top-level): {len(port_result.stage1_candidates)} candidates",
        f"- Stage 2 (+ has stored returns): {len(port_result.stage2_candidates)} candidates",
        f"- Stage 3 (DSR filter): {len(port_result.stage3_candidates)} candidates",
        f"- **Final eligible constituents: {len(port_result.eligible_candidates)} candidates**",
        "",
    ]
    return "\n".join(lines)


def _format_weights_table(port_eq, port_iv) -> str:
    """Format constituent weights table. port_iv may be None."""
    if port_iv is not None:
        header = "| Candidate ID | Equal Weight | Inverse-Vol Weight |"
        sep =    "| --- | --- | --- |"
    else:
        header = "| Candidate ID | Weight |"
        sep =    "| --- | --- |"

    rows = [header, sep]
    for cid in port_eq.eligible_candidates:
        eq_w = f"{port_eq.weights.get(cid, 0.0):.4f}"
        if port_iv is not None:
            iv_w = f"{port_iv.weights.get(cid, 0.0):.4f}"
            rows.append(f"| {cid} | {eq_w} | {iv_w} |")
        else:
            rows.append(f"| {cid} | {eq_w} |")
    return "\n".join(rows) + "\n"
