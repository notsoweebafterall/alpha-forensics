"""
Alpha Registry Package — persistent, queryable SQLite catalog for alpha candidates.
"""

from .build_registry import build_registry
from .query import (
    get_all_candidates,
    get_valid_non_redundant_candidates,
    get_candidates_by_verdict,
    get_candidate_by_id,
    get_status_history,
)

__all__ = [
    "build_registry",
    "get_all_candidates",
    "get_valid_non_redundant_candidates",
    "get_candidates_by_verdict",
    "get_candidate_by_id",
    "get_status_history",
]
