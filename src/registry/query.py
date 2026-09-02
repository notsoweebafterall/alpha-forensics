"""
Alpha Registry Query API — read and filter helpers over data/alpha_registry.db.
"""

from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Union
import pandas as pd


def _connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Alpha registry database not found at {path}. Run build_registry() first.")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def get_all_candidates(db_path: Union[str, Path] = "data/alpha_registry.db") -> pd.DataFrame:
    """
    Returns all candidates logged in the registry as a pandas DataFrame.
    """
    conn = _connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM candidates ORDER BY oos_sharpe DESC;", conn)
        return df
    finally:
        conn.close()


def get_valid_non_redundant_candidates(
    db_path: Union[str, Path] = "data/alpha_registry.db",
    top_level_only: bool = True,
) -> pd.DataFrame:
    """
    Queries candidates that have effective_status == 'VALID' and are non-redundant
    (redundant_with IS NULL), ranked by oos_sharpe descending.

    Args:
        db_path: Path to SQLite database file.
        top_level_only: If True (default), excludes sub-slice candidate IDs
                        containing __sector_, __period_, __regime_, or __cost_.
    """
    conn = _connect(db_path)
    try:
        if top_level_only:
            query = """
                SELECT * FROM candidates
                WHERE effective_status = 'VALID'
                  AND redundant_with IS NULL
                  AND candidate_id NOT LIKE '%__sector_%'
                  AND candidate_id NOT LIKE '%__period_%'
                  AND candidate_id NOT LIKE '%__regime_%'
                  AND candidate_id NOT LIKE '%__cost_%'
                ORDER BY oos_sharpe DESC;
            """
        else:
            query = """
                SELECT * FROM candidates
                WHERE effective_status = 'VALID' AND redundant_with IS NULL
                ORDER BY oos_sharpe DESC;
            """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def get_candidates_by_verdict(
    verdict: str,
    db_path: Union[str, Path] = "data/alpha_registry.db",
) -> pd.DataFrame:
    """
    Queries candidates matching a specific DSR verdict (e.g. 'survives_dsr' or 'fails_dsr'),
    ranked by oos_sharpe descending.
    """
    conn = _connect(db_path)
    try:
        query = """
            SELECT * FROM candidates
            WHERE dsr_verdict = ?
            ORDER BY oos_sharpe DESC;
        """
        return pd.read_sql_query(query, conn, params=(verdict,))
    finally:
        conn.close()


def get_candidate_by_id(
    candidate_id: str,
    db_path: Union[str, Path] = "data/alpha_registry.db",
) -> Optional[Dict[str, str]]:
    """
    Fetches a single candidate record by candidate_id as a dictionary.
    Returns None if candidate_id is not found.
    """
    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates WHERE candidate_id = ?;", (candidate_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def get_status_history(
    candidate_id: str,
    db_path: Union[str, Path] = "data/alpha_registry.db",
) -> pd.DataFrame:
    """
    Returns the complete status-change history for a candidate_id from the status_history table.
    """
    conn = _connect(db_path)
    try:
        query = """
            SELECT candidate_id, new_status, reason, metadata_json, timestamp
            FROM status_history
            WHERE candidate_id = ?
            ORDER BY id ASC;
        """
        return pd.read_sql_query(query, conn, params=(candidate_id,))
    finally:
        conn.close()
