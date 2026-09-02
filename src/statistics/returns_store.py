"""
Returns Store: persistent OOS net returns series for Phase 13 redundancy analysis.

Stores per-candidate daily OOS net returns in a long-format Parquet file
(data/trial_returns.parquet by default) with schema:
    candidate_id (string), date (datetime64[ns]), return (float64)

Design Decisions:
    - Separate file from trial_log.jsonl: keeps the scalar ledger lightweight and
      avoids bloating JSONL with full time series.
    - Immutability per candidate_id: save_returns() raises ValueError if
      candidate_id already exists, mirroring log_trial()'s discipline. Callers
      guard with try/except ValueError: pass.
    - Net returns only: only cost-adjusted net daily returns are stored, matching
      the net Sharpe and net-based skew/kurtosis in TrialRecord. Gross returns are
      NOT stored; doing so would recreate the gross-vs-net inconsistency this
      module was designed to eliminate.

Honest Limitation:
    Candidates logged to trial_log.jsonl BEFORE Phase 13 was deployed do not
    have a stored returns series in trial_returns.parquet. They appear as
    SKIPPED (no stored returns series) in redundancy analysis until re-evaluated
    under Phase 13 logging. Same honest-gap convention used throughout this project.

Scope Restriction (enforced in correlation_analysis.py, documented here):
    Redundancy analysis is restricted to top-level candidates — those whose
    candidate_id does NOT contain __sector_, __period_, __regime_, or __cost_.
    Sub-slice candidates represent different sub-populations and comparing them to
    a full-OOS-window candidate conflates different populations.
"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def save_returns(
    candidate_id: str,
    returns: pd.Series,
    store_path: Union[str, Path] = "data/trial_returns.parquet",
) -> None:
    """
    Persists a candidate's OOS daily net returns series to the returns store.

    Immutability Discipline:
        Raises ValueError if candidate_id already exists in store_path. This mirrors
        log_trial()'s single-write discipline. OOS returns are a one-time byproduct
        of the single guarded look and must not be silently overwritten.

    Args:
        candidate_id: Unique candidate identifier (must match trial_log.jsonl entry).
        returns: pd.Series of daily OOS net returns, indexed by date.
        store_path: Path to the Parquet store file (default: data/trial_returns.parquet).

    Raises:
        ValueError: If candidate_id already exists in the returns store.
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    # Immutability guard: reject duplicate candidate_id writes
    if store_path.exists():
        try:
            existing = pd.read_parquet(store_path, columns=["candidate_id"])
            if candidate_id in existing["candidate_id"].values:
                raise ValueError(
                    f"Candidate ID '{candidate_id}' already exists in returns store "
                    f"({store_path}). The returns store is immutable per candidate_id, "
                    f"same as trial_log.jsonl. Re-running OOS and overwriting stored "
                    f"returns would violate single-look discipline."
                )
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            # Parquet unreadable or schema mismatch — treat as new store

    clean = returns.dropna()
    if clean.empty:
        return  # Nothing meaningful to store

    new_rows = pd.DataFrame({
        "candidate_id": candidate_id,
        "date": pd.to_datetime(clean.index),
        "return": clean.values,
    })

    if store_path.exists():
        try:
            combined = pd.concat([pd.read_parquet(store_path), new_rows], ignore_index=True)
        except Exception:
            combined = new_rows
    else:
        combined = new_rows

    combined["date"] = pd.to_datetime(combined["date"])
    combined.to_parquet(store_path, index=False)


def load_returns(
    candidate_id: str,
    store_path: Union[str, Path] = "data/trial_returns.parquet",
) -> Optional[pd.Series]:
    """
    Loads the OOS daily net returns series for a single candidate.

    Args:
        candidate_id: Unique candidate identifier.
        store_path: Path to the Parquet store file.

    Returns:
        pd.Series indexed by date (datetime64[ns]) if found, None otherwise.
    """
    store_path = Path(store_path)
    if not store_path.exists():
        return None
    try:
        df = pd.read_parquet(store_path)
    except Exception:
        return None
    subset = df[df["candidate_id"] == candidate_id]
    if subset.empty:
        return None
    series = subset.set_index("date")["return"].sort_index()
    series.index = pd.to_datetime(series.index)
    series.name = candidate_id
    return series


def load_returns_matrix(
    candidate_ids: List[str],
    store_path: Union[str, Path] = "data/trial_returns.parquet",
) -> pd.DataFrame:
    """
    Loads OOS daily net returns for multiple candidates into a wide-format matrix.

    Only candidates with stored returns produce a column. Missing candidates are
    excluded silently — callers are responsible for detecting and reporting them
    as skipped (compare returned columns against requested candidate_ids).

    Args:
        candidate_ids: List of candidate identifiers to load.
        store_path: Path to the Parquet store file.

    Returns:
        pd.DataFrame with index=date (datetime64[ns]) and columns=candidate_id.
        Returns empty DataFrame if store does not exist or no candidates found.
    """
    store_path = Path(store_path)
    if not store_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(store_path)
    except Exception:
        return pd.DataFrame()
    subset = df[df["candidate_id"].isin(candidate_ids)]
    if subset.empty:
        return pd.DataFrame()
    matrix = (
        subset
        .pivot_table(index="date", columns="candidate_id", values="return", aggfunc="first")
        .sort_index()
    )
    matrix.index = pd.to_datetime(matrix.index)
    matrix.columns.name = None
    return matrix
