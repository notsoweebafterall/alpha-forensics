"""
Alpha Registry Builder — builds derived SQLite catalog from authoritative JSONL/Parquet files.

Design Principles:
    1. Derived Index, Never Source of Truth:
       The SQLite database (data/alpha_registry.db by default) is a rebuildable derived index.
       All underlying JSONL and Parquet files (trial_log.jsonl, trial_status_log.jsonl,
       trial_returns.parquet, dsr_results.jsonl) remain authoritative. build_registry()
       drops and recreates its tables from scratch on every execution — there is zero drift
       or staleness risk between runs.
    2. Canonical Effective Status Resolution:
       build_registry() calls effective_trial_records(..., include_invalidated=True) directly
       to resolve candidate records and their effective status. It does NOT reimplement
       status resolution logic, guaranteeing 100% agreement with the rest of the codebase.
    3. Precise Latest Invalidation Metadata Semantics:
       redundant_with and correlation_with_redundant are populated ONLY if the candidate's
       current effective status is "INVALIDATED" AND the specific latest status-change event
       that produced that effective status carries the redundancy metadata. If a candidate was
       re-validated (VALID) or invalidated for another reason, old stale redundancy metadata
       is ignored and stored as NULL.
    4. Legitimate NULL DSR Verdicts:
       Candidates that have not been evaluated under Phase 10 DSR analysis will have
       dsr_score = NULL and dsr_verdict = NULL. Most candidates in the trial ledger have not
       undergone DSR evaluation; NULL is expected and correct, not a data gap.
"""

import json
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Union

import pandas as pd

from statistics import (
    effective_trial_records,
    load_latest_dsr_verdicts,
    load_trial_log,
    StatusChangeRecord,
)


def _load_status_history_raw(status_log_path: Path) -> List[StatusChangeRecord]:
    """Loads raw StatusChangeRecord events from file order."""
    if not status_log_path.exists():
        return []
    records: List[StatusChangeRecord] = []
    with status_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(StatusChangeRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return records


def build_registry(
    db_path: Union[str, Path] = "data/alpha_registry.db",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    dsr_log_path: Union[str, Path] = "data/dsr_results.jsonl",
) -> None:
    """
    Rebuilds the SQLite alpha candidate catalog database from authoritative log files.

    Drops and recreates 'candidates' and 'status_history' tables from scratch.

    Args:
        db_path: Path to SQLite database file (default data/alpha_registry.db).
        log_path: Path to trial_log.jsonl.
        status_log_path: Path to trial_status_log.jsonl.
        store_path: Path to trial_returns.parquet.
        dsr_log_path: Path to dsr_results.jsonl.
    """
    db_path = Path(db_path)
    log_path = Path(log_path)
    status_log_path = Path(status_log_path)
    store_path = Path(store_path)
    dsr_log_path = Path(dsr_log_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fetch effective trial records via canonical latest-wins resolution
    eff_records = effective_trial_records(
        log_path=log_path, status_log_path=status_log_path, include_invalidated=True
    )

    # 2. Fetch raw status history and map latest event per candidate
    raw_status_history = _load_status_history_raw(status_log_path)
    latest_status_event: Dict[str, StatusChangeRecord] = {}
    for event in raw_status_history:
        latest_status_event[event.candidate_id] = event

    # 3. Fetch candidate presence in trial_returns.parquet
    stored_candidate_ids: set = set()
    if store_path.exists():
        try:
            df_returns = pd.read_parquet(store_path, columns=["candidate_id"])
            stored_candidate_ids = set(df_returns["candidate_id"].unique())
        except Exception:
            pass

    # 4. Fetch latest DSR verdicts from dsr_results.jsonl
    dsr_verdicts = load_latest_dsr_verdicts(dsr_log_path)

    # Connect to SQLite DB
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Drop existing tables to enforce idempotent full rebuild
        cursor.execute("DROP TABLE IF EXISTS candidates;")
        cursor.execute("DROP TABLE IF EXISTS status_history;")

        # Create schema
        cursor.execute("""
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                oos_sharpe REAL NOT NULL,
                skew REAL,
                kurtosis REAL,
                track_record_length INTEGER,
                backfilled INTEGER NOT NULL,
                effective_status TEXT NOT NULL,
                has_returns_series INTEGER NOT NULL,
                dsr_score REAL,
                dsr_verdict TEXT,
                redundant_with TEXT,
                correlation_with_redundant REAL,
                logged_at TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                new_status TEXT NOT NULL,
                reason TEXT NOT NULL,
                metadata_json TEXT,
                timestamp TEXT NOT NULL
            );
        """)

        # Insert into candidates table
        candidate_rows = []
        for rec in eff_records:
            cid = rec.candidate_id
            effective_status = rec.status
            has_returns = 1 if cid in stored_candidate_ids else 0

            # DSR info
            dsr_score: Optional[float] = None
            dsr_verdict: Optional[str] = None
            if cid in dsr_verdicts:
                dsr_rec = dsr_verdicts[cid]
                dsr_score = dsr_rec.dsr_score
                dsr_verdict = dsr_rec.verdict

            # Redundancy metadata rule: ONLY populate if effective_status is INVALIDATED
            # AND the latest status-change event that set it to INVALIDATED carries redundancy metadata
            redundant_with: Optional[str] = None
            corr_with_redundant: Optional[float] = None

            if effective_status == "INVALIDATED" and cid in latest_status_event:
                latest_evt = latest_status_event[cid]
                if latest_evt.new_status == "INVALIDATED" and latest_evt.metadata:
                    meta = latest_evt.metadata
                    if isinstance(meta, dict) and "redundant_with" in meta:
                        redundant_with = str(meta["redundant_with"])
                        corr_val = meta.get("correlation")
                        corr_with_redundant = float(corr_val) if corr_val is not None else None

            candidate_rows.append((
                cid,
                rec.phase,
                float(rec.oos_sharpe),
                float(rec.skew) if rec.skew is not None else None,
                float(rec.kurtosis) if rec.kurtosis is not None else None,
                int(rec.track_record_length) if rec.track_record_length is not None else None,
                1 if rec.backfilled else 0,
                effective_status,
                has_returns,
                dsr_score,
                dsr_verdict,
                redundant_with,
                corr_with_redundant,
                rec.timestamp,
            ))

        cursor.executemany("""
            INSERT INTO candidates (
                candidate_id, phase, oos_sharpe, skew, kurtosis, track_record_length,
                backfilled, effective_status, has_returns_series, dsr_score, dsr_verdict,
                redundant_with, correlation_with_redundant, logged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, candidate_rows)

        # Insert into status_history table
        history_rows = []
        for evt in raw_status_history:
            meta_json = json.dumps(evt.metadata) if evt.metadata is not None else None
            history_rows.append((
                evt.candidate_id,
                evt.new_status,
                evt.reason,
                meta_json,
                evt.timestamp,
            ))

        cursor.executemany("""
            INSERT INTO status_history (
                candidate_id, new_status, reason, metadata_json, timestamp
            ) VALUES (?, ?, ?, ?, ?);
        """, history_rows)

        conn.commit()

    finally:
        conn.close()
