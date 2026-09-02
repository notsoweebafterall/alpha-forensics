"""
Redundancy module for Alpha Forensics.

Provides pairwise return correlation analysis, graph-based redundancy clustering,
and status log invalidation for redundant alpha candidates.
"""

from .correlation_analysis import (
    RedundancyCluster,
    RedundancyAnalysisResult,
    is_top_level_candidate_id,
    run_redundancy_analysis,
)

__all__ = [
    "RedundancyCluster",
    "RedundancyAnalysisResult",
    "is_top_level_candidate_id",
    "run_redundancy_analysis",
]
