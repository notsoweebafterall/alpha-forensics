"""
Alpha Registry Build & Query Demo (Phase 14).

Consolidates data scattered across trial_log.jsonl, trial_status_log.jsonl,
trial_returns.parquet, and dsr_results.jsonl into a derived, queryable SQLite catalog
(data/alpha_registry.db).

Key Design Discipline:
    1. Rebuildable Derived Index:
       The SQLite database is recreated from scratch on every run. Raw JSONL/Parquet files
       remain authoritative.
    2. Zero Duplicate Returns Data:
       Daily return series stay in trial_returns.parquet; registry stores has_returns_series flag.
    3. Explicit NULL DSR Verdict Notice:
       Candidates that have not undergone Phase 10 DSR analysis will have dsr_score = NULL and
       dsr_verdict = NULL. Most candidates in the trial ledger have not undergone Phase 10 DSR
       evaluation; NULL values are expected and correct, not a data gap.
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from registry import (
    build_registry,
    get_all_candidates,
    get_valid_non_redundant_candidates,
    get_candidates_by_verdict,
    get_candidate_by_id,
    get_status_history,
)


from typing import Optional, Union


def run_demo(
    db_path: Union[str, Path] = "data/alpha_registry.db",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    dsr_log_path: Union[str, Path] = "data/dsr_results.jsonl",
) -> None:
    db_path = Path(db_path)
    log_path = Path(log_path)
    status_log_path = Path(status_log_path)
    store_path = Path(store_path)
    dsr_log_path = Path(dsr_log_path)

    print("=" * 100, flush=True)
    print("ALPHA FORENSICS — PHASE 14 ALPHA REGISTRY BUILD & QUERY DEMO", flush=True)
    print("=" * 100, flush=True)
    print(f"Building derived SQLite candidate catalog ({db_path})...", flush=True)
    print("-" * 100, flush=True)

    # 1. Build SQLite Registry from authoritative log files
    build_registry(
        db_path=db_path,
        log_path=log_path,
        status_log_path=status_log_path,
        store_path=store_path,
        dsr_log_path=dsr_log_path,
    )
    print(f"Registry rebuilt successfully at [{db_path}]!\n", flush=True)

    # 2. Database Statistics & Summary Counts
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM candidates;")
    total_candidates = cursor.fetchone()[0]

    cursor.execute("SELECT effective_status, COUNT(*) FROM candidates GROUP BY effective_status;")
    status_counts = dict(cursor.fetchall())

    cursor.execute("SELECT COALESCE(dsr_verdict, 'NULL (not evaluated for DSR)'), COUNT(*) FROM candidates GROUP BY dsr_verdict;")
    dsr_counts = dict(cursor.fetchall())

    cursor.execute("SELECT COUNT(*) FROM candidates WHERE redundant_with IS NOT NULL;")
    redundant_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM candidates WHERE has_returns_series = 1;")
    returns_store_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM status_history;")
    history_count = cursor.fetchone()[0]

    conn.close()

    print("=" * 100, flush=True)
    print("CATALOG SUMMARY AUDIT", flush=True)
    print("=" * 100, flush=True)
    print(f"Total Logged Candidates               : {total_candidates}", flush=True)
    print(f"Candidates by Effective Status         : {status_counts}", flush=True)
    print(f"Candidates with Stored Net Returns     : {returns_store_count}", flush=True)
    print(f"Candidates Marked Redundant            : {redundant_count}", flush=True)
    print(f"Status Change Event Log Entries        : {history_count}", flush=True)
    print(f"\nCandidates by DSR Verdict Breakdown:", flush=True)
    for v_name, count in dsr_counts.items():
        print(f"  * {v_name:<38} : {count}", flush=True)

    print("\n" + "-" * 100, flush=True)
    print("NOTE: Most candidate rows legitimately have dsr_score = NULL and dsr_verdict = NULL.", flush=True)
    print("Phase 10 DSR analysis is evaluated specifically for target candidates, not every historical trial.", flush=True)
    print("NULL values reflect unevaluated candidates — this is expected and correct, not a data gap.", flush=True)
    print("-" * 100, flush=True)

    # 3. Demonstrate Query API Helpers
    print("\n" + "=" * 100, flush=True)
    print("QUERY API HELPER EXAMPLES", flush=True)
    print("=" * 100, flush=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}" if pd.notna(x) else " NULL ")

    # Query 1: Top 5 VALID, non-redundant candidates ranked by OOS Sharpe
    print("\nQuery 1: get_valid_non_redundant_candidates() (Top 5 by OOS Sharpe):", flush=True)
    df_valid = get_valid_non_redundant_candidates(db_path)
    cols_to_show = ["candidate_id", "phase", "oos_sharpe", "effective_status", "has_returns_series", "dsr_verdict"]
    print(df_valid[cols_to_show].head(5).to_string(index=False), flush=True)

    # Query 2: Candidates that fail DSR
    print("\nQuery 2: get_candidates_by_verdict('fails_dsr'):", flush=True)
    df_fails = get_candidates_by_verdict("fails_dsr", db_path)
    if not df_fails.empty:
        print(df_fails[["candidate_id", "oos_sharpe", "dsr_score", "dsr_verdict"]].to_string(index=False), flush=True)
    else:
        print("  No candidates currently marked 'fails_dsr' in registry.", flush=True)

    # Query 3: Inspect specific redundant candidate detail
    cursor_conn = sqlite3.connect(str(db_path))
    cursor_conn.row_factory = sqlite3.Row
    c_red = cursor_conn.execute("SELECT * FROM candidates WHERE redundant_with IS NOT NULL LIMIT 1;").fetchone()
    cursor_conn.close()

    if c_red:
        red_cid = c_red["candidate_id"]
        print(f"\nQuery 3: get_candidate_by_id('{red_cid}') [Redundant Candidate Detail]:", flush=True)
        c_detail = get_candidate_by_id(red_cid, db_path)
        for k, v in c_detail.items():
            print(f"  * {k:<30} : {v}", flush=True)

        print(f"\nQuery 4: get_status_history('{red_cid}') [Status Audit Trail]:", flush=True)
        df_hist = get_status_history(red_cid, db_path)
        print(df_hist.to_string(index=False), flush=True)

    print("\n" + "=" * 100, flush=True)
    print("SUMMARY CONCLUSION:", flush=True)
    print("Phase 14 Alpha Registry successfully consolidates raw JSONL and Parquet trial logs into a", flush=True)
    print("clean, queryable SQLite catalog (data/alpha_registry.db). Downstream Phase 15 (portfolio) and", flush=True)
    print("Phase 16 (reporting) can now execute structured SQL queries over candidate state.", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    run_demo()
