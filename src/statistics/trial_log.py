"""
Persistent trial log ledger for out-of-sample (OOS) strategy evaluation tracking.

Provides an append-only, on-disk ledger (data/trial_log.jsonl) that records every guarded OOS trial.
This ensures the total number of candidate trials N used in Deflated Sharpe Ratio (DSR) calculations
is an accurate, reproducible count across execution runs.

Status Lifecycle:
    A trial's measured oos_sharpe is immutable once logged (see log_trial). But a trial can later
    be found to be invalid for reasons unrelated to its number -- e.g. a bug in the expression that
    produced it, a data quality issue discovered after the fact, or a duplicate under a different
    candidate_id. Rather than edit or delete the original record (which would break the append-only
    audit trail), status changes are themselves append-only events in a separate log
    (data/trial_status_log.jsonl, by default). See append_status_change() / effective_trial_records().
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from datetime import datetime

VALID_STATUSES = ("VALID", "INVALIDATED")


@dataclass
class TrialRecord:
    candidate_id: str
    phase: str
    oos_sharpe: float
    timestamp: str
    backfilled: bool = False
    status: str = "VALID"
    skew: Optional[float] = None
    kurtosis: Optional[float] = None
    track_record_length: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialRecord":
        skew = data.get("skew")
        kurtosis = data.get("kurtosis")
        t_len = data.get("track_record_length")
        return cls(
            candidate_id=str(data["candidate_id"]),
            phase=str(data["phase"]),
            oos_sharpe=float(data["oos_sharpe"]),
            timestamp=str(data["timestamp"]),
            backfilled=bool(data.get("backfilled", False)),
            status=str(data.get("status", "VALID")),
            skew=float(skew) if skew is not None else None,
            kurtosis=float(kurtosis) if kurtosis is not None else None,
            track_record_length=int(t_len) if t_len is not None else None,
        )


def load_trial_log(log_path: Union[str, Path] = "data/trial_log.jsonl") -> List[TrialRecord]:
    """
    Loads all trial records from the on-disk trial log.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        List[TrialRecord]: List of trial records. Returns empty list if file does not exist.
    """
    path = Path(log_path)
    if not path.exists():
        return []

    records: List[TrialRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(TrialRecord.from_dict(data))
            except (json.JSONDecodeError, KeyError) as e:
                continue
    return records


def log_trial(record: TrialRecord, log_path: Union[str, Path] = "data/trial_log.jsonl") -> None:
    """
    Appends a new trial record to the on-disk trial log.

    Immutability & Single-Look Discipline:
        Raises ValueError if candidate_id already exists in the ledger.

    Args:
        record: TrialRecord to append.
        log_path: Path to the JSONL log file.

    Raises:
        ValueError: If candidate_id has already been logged.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_records = load_trial_log(path)
    existing_ids = {r.candidate_id for r in existing_records}

    if record.candidate_id in existing_ids:
        raise ValueError(
            f"Candidate ID '{record.candidate_id}' already exists in trial log ({log_path}). "
            f"The trial ledger is append-only and immutable per candidate."
        )

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def trial_count(
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Returns the total count of distinct candidate trials logged in the ledger.
    This count serves as N for Deflated Sharpe Ratio (DSR) calculations.

    Args:
        log_path: Path to the JSONL log file.
        status_log_path: Optional path to a status-change event log (see
            append_status_change). When provided, trials whose latest status is
            "INVALIDATED" are excluded from the count. When omitted (the default),
            behavior is unchanged from before status tracking existed -- every
            logged trial counts, regardless of status. This keeps existing callers
            that don't know about status tracking working exactly as before.

    Returns:
        int: Number of logged trials (optionally excluding invalidated ones).
    """
    if status_log_path is None:
        return len(load_trial_log(log_path))
    return len(effective_trial_records(log_path, status_log_path, include_invalidated=False))


@dataclass
class StatusChangeRecord:
    candidate_id: str
    new_status: str
    reason: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatusChangeRecord":
        return cls(
            candidate_id=str(data["candidate_id"]),
            new_status=str(data["new_status"]),
            reason=str(data["reason"]),
            timestamp=str(data["timestamp"]),
            metadata=data.get("metadata"),
        )


def append_status_change(
    candidate_id: str,
    new_status: str,
    reason: str,
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Appends a status-change event for a candidate_id. This does NOT touch the
    original trial_log.jsonl record -- the measured oos_sharpe stays immutable and
    the original entry is never edited or removed. Multiple status-change events
    for the same candidate_id are allowed (e.g. INVALIDATED, then later reversed);
    the most recent one (by file order) is authoritative -- see effective_trial_records.

    Args:
        candidate_id: The trial's candidate_id, as it appears in trial_log.jsonl.
        new_status: One of VALID_STATUSES ("VALID", "INVALIDATED").
        reason: Free-text justification, e.g. "expression had a lookahead bug".
        status_log_path: Path to the status-change event log.
        metadata: Optional structured metadata dictionary (e.g. {"redundant_with": ..., "correlation": ...}).

    Raises:
        ValueError: If new_status is not a recognized status.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(f"new_status must be one of {VALID_STATUSES}, got '{new_status}'.")

    path = Path(status_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = StatusChangeRecord(
        candidate_id=candidate_id,
        new_status=new_status,
        reason=reason,
        timestamp=datetime.now().isoformat(),
        metadata=metadata,
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_dict()) + "\n")


def effective_trial_records(
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    include_invalidated: bool = True,
) -> List[TrialRecord]:
    """
    Loads trial records with their status field updated to reflect the latest
    status-change event for each candidate_id, if any. The original oos_sharpe,
    phase, and timestamp are never altered -- only .status is overridden in the
    returned (in-memory) objects.

    Args:
        log_path: Path to the JSONL trial log.
        status_log_path: Path to the JSONL status-change event log.
        include_invalidated: If False, trials whose effective status is
            "INVALIDATED" are omitted from the returned list entirely.

    Returns:
        List[TrialRecord]: Trial records with effective status applied.
    """
    records = load_trial_log(log_path)

    status_path = Path(status_log_path)
    latest_status: Dict[str, str] = {}
    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    change = StatusChangeRecord.from_dict(json.loads(line))
                    latest_status[change.candidate_id] = change.new_status
                except (json.JSONDecodeError, KeyError):
                    continue

    result = []
    for r in records:
        if r.candidate_id in latest_status:
            r.status = latest_status[r.candidate_id]
        if not include_invalidated and r.status == "INVALIDATED":
            continue
        result.append(r)
    return result


def backfill_from_phase_artifacts(
    records_to_backfill: List[TrialRecord],
    log_path: Union[str, Path] = "data/trial_log.jsonl",
) -> List[TrialRecord]:
    """
    Performs a one-time backfill of historical candidates from earlier phases into the ledger.

    Records ingested via this function are marked with backfilled=True. Duplicate candidate IDs
    already present in the ledger are skipped to preserve ledger integrity.

    Single-Look / No-Fabrication Discipline:
        There is no default or synthetic candidate set. Callers MUST supply real
        TrialRecord objects derived from actually re-executed evaluations (e.g. by
        re-running the Phase 4/6/9 demo scripts and capturing their genuine output).
        This function only performs deduplication and disk persistence — it never
        invents data.

    Args:
        records_to_backfill: List of real TrialRecord objects to backfill. Must be
            non-empty; there is deliberately no default.
        log_path: Path to the JSONL log file.

    Returns:
        List[TrialRecord]: List of newly added backfilled records.

    Raises:
        ValueError: If records_to_backfill is empty.
    """
    if not records_to_backfill:
        raise ValueError(
            "backfill_from_phase_artifacts requires a non-empty list of real "
            "TrialRecord objects. There is no synthetic/default fallback."
        )

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_records = load_trial_log(path)
    existing_ids = {r.candidate_id for r in existing_records}

    added: List[TrialRecord] = []
    with path.open("a", encoding="utf-8") as f:
        for rec in records_to_backfill:
            if rec.candidate_id in existing_ids:
                continue
            rec.backfilled = True
            f.write(json.dumps(rec.to_dict()) + "\n")
            existing_ids.add(rec.candidate_id)
            added.append(rec)

    return added