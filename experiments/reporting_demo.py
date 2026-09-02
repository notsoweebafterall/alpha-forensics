"""
Reporting Demo (Phase 16).

Calls build_report() against the current live registry, prints the report file path
and the executive summary inline, so the state is visible without opening the file.

Pre-condition: experiments/statistical_forensics_demo.py must have been run at least
once since Phase 14 was deployed (to populate data/dsr_results.jsonl). If dsr_results.jsonl
is absent, DSR verdicts will be NULL for all candidates; the report will state this plainly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reporting import build_report


from typing import Optional, Union


def run_demo(
    db_path: Union[str, Path] = "data/alpha_registry.db",
    returns_store_path: Union[str, Path] = "data/trial_returns.parquet",
    dsr_log_path: Union[str, Path] = "data/dsr_results.jsonl",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    output_dir: Union[str, Path] = "reports/",
    skip_rebuild: bool = False,
    write_json: bool = True,
) -> None:
    print("=" * 100, flush=True)
    print("ALPHA FORENSICS — PHASE 16 REPORTING DEMO", flush=True)
    print("=" * 100, flush=True)
    print("Building durable report from current registry and portfolio state...", flush=True)
    print("(build_registry() is called automatically — skip_rebuild=False by default)", flush=True)
    print("-" * 100, flush=True)

    result = build_report(
        db_path=db_path,
        returns_store_path=returns_store_path,
        dsr_log_path=dsr_log_path,
        log_path=log_path,
        status_log_path=status_log_path,
        output_dir=output_dir,
        skip_rebuild=skip_rebuild,
        write_json=write_json,
    )

    print(f"\nReport written to: {result.markdown_path}", flush=True)
    if result.json_path:
        print(f"JSON companion:    {result.json_path}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("EXECUTIVE SUMMARY (inline):", flush=True)
    print("=" * 100, flush=True)
    print(result.executive_summary, flush=True)

    print("\n" + "=" * 100, flush=True)
    print("CANDIDATE FUNNEL COUNTS:", flush=True)
    print("=" * 100, flush=True)
    print(f"  Total logged candidates              : {result.total_candidates}", flush=True)
    print(f"  Valid, non-redundant, top-level      : {result.valid_top_level_candidates}", flush=True)
    print(f"  + Has stored returns                 : {result.candidates_with_returns}", flush=True)
    print(f"  + Explicitly fails DSR               : {result.candidates_failing_dsr}", flush=True)
    print(f"  + Not yet DSR-evaluated (NULL)       : {result.candidates_not_dsr_evaluated}", flush=True)
    print(f"  + Survives DSR                       : {result.candidates_surviving_dsr}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("PORTFOLIO ELIGIBILITY:", flush=True)
    print("=" * 100, flush=True)
    print(f"  Rigorous (require_dsr_survival=True) : {result.portfolio_rigorous_eligible} eligible constituents", flush=True)
    print(f"  Relaxed  (require_dsr_survival=False): {result.portfolio_relaxed_eligible} eligible constituents", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    run_demo()
