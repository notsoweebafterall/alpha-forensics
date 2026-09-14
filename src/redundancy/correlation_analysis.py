"""
Redundancy Detection & Return Correlation Analysis for Alpha Candidates.

Goal:
    Phase 3 performs structural deduplication (identical expression trees hash the same).
    Phase 13 catches the harder case: two structurally different expressions whose OOS daily net
    returns are highly correlated — i.e., redundant alpha performance, not redundant code.

Architectural Storage & Net-Returns Discipline:
    Requires date-aligned daily OOS net returns from data/trial_returns.parquet (saved
    alongside log_trial()). All calculations use net (cost-adjusted) daily returns, matching
    TrialRecord's oos_sharpe.

Scope Restriction:
    Redundancy analysis is strictly restricted to top-level candidates evaluated over the full OOS window.
    Sub-slice trials containing __sector_, __period_, __regime_, or __cost_ in their candidate_id are
    filtered out because comparing a sub-slice (e.g., sector subset or regime window) to a full-window candidate
    mixes different underlying populations and date ranges.

Honest Limitation:
    Candidates logged to trial_log.jsonl before Phase 13 was deployed do not have a stored returns series in
    trial_returns.parquet. They are reported in skipped_candidates with an explicit reason ("no stored returns series")
    rather than silently dropped or crashed on. Re-evaluating such candidates under Phase 13 logging populates
    their returns store.

Threshold & Clustering Discipline:
    1. Minimum Overlap Days (default: 60): Pairs with fewer than min_overlap_days overlapping OOS trading dates
       are excluded from correlation analysis and reported in skipped_candidates with reason.
    2. Pairwise Correlation Threshold (default: 0.90): Daily net return correlations >= threshold define
       redundancy edges in an undirected graph.
    3. Union-Find Clustering: Connected components on the thresholded graph form redundancy clusters.
    4. Representative Selection: Within each cluster, the candidate with the highest logged oos_sharpe is
       kept as the representative (VALID). All other cluster members are marked INVALIDATED.
    5. P0.3 Status Log Lifecycle Integration: Uses append_status_change(candidate_id, "INVALIDATED", reason=...)
       to mark redundant candidates without altering immutable historical trial_log.jsonl entries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from statistics import (
    load_trial_log,
    effective_trial_records,
    append_status_change,
    load_returns_matrix,
    TrialRecord,
)

# Sub-slice ID tags that mark candidate_ids as out-of-scope for top-level redundancy comparison
SUB_SLICE_TAGS = ("__sector_", "__period_", "__regime_", "__cost_")


def is_top_level_candidate_id(candidate_id: str) -> bool:
    """
    Returns True if candidate_id represents a top-level, full-OOS-window trial.
    Returns False if it contains any sub-slice identifier tag.
    """
    return not any(tag in candidate_id for tag in SUB_SLICE_TAGS)


@dataclass
class RedundancyCluster:
    """
    Represents a cluster of highly correlated alpha candidates.
    """
    cluster_id: int
    representative_id: str
    redundant_ids: List[str]
    pairwise_correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)


@dataclass
class RedundancyAnalysisResult:
    """
    Output of Phase 13 redundancy analysis.
    """
    evaluated_candidates: List[str]
    correlation_matrix: pd.DataFrame
    clusters: List[RedundancyCluster]
    invalidated_candidates: List[str]
    skipped_candidates: Dict[str, str]


class UnionFind:
    """Simple Disjoint Set Union (DSU) structure for connected-components clustering."""
    def __init__(self, elements: List[str]):
        self.parent: Dict[str, str] = {x: x for x in elements}

    def find(self, i: str) -> str:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: str, j: str) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def run_redundancy_analysis(
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    status_log_path: Union[str, Path] = "data/trial_status_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    correlation_threshold: float = 0.90,
    min_overlap_days: int = 60,
    apply_status_changes: bool = True,
) -> RedundancyAnalysisResult:
    """
    Executes Phase 13 redundancy analysis across active top-level candidates.

    Args:
        log_path: Path to trial_log.jsonl.
        status_log_path: Path to trial_status_log.jsonl.
        store_path: Path to trial_returns.parquet.
        correlation_threshold: Correlation cutoff for grouping candidates into clusters (default 0.90).
        min_overlap_days: Minimum required overlapping daily returns observations (default 60).
        apply_status_changes: If True, appends INVALIDATED status records for redundant candidates.

    Returns:
        RedundancyAnalysisResult: Evaluated candidates, correlation matrix, clusters, and skipped log.
    """
    log_path = Path(log_path)
    status_log_path = Path(status_log_path)
    store_path = Path(store_path)

    # 1. Fetch effective VALID records from ledger & status log
    records_list = effective_trial_records(log_path=log_path, status_log_path=status_log_path)
    skipped_candidates: Dict[str, str] = {}
    candidate_records: Dict[str, TrialRecord] = {}

    # 2. Filter for in-scope top-level candidates
    for rec in records_list:
        cid = rec.candidate_id
        if not is_top_level_candidate_id(cid):
            skipped_candidates[cid] = "Out of scope: sub-slice candidate ID (contains __sector_, __period_, __regime_, or __cost_)"
        else:
            candidate_records[cid] = rec

    all_in_scope_ids = list(candidate_records.keys())

    # 3. Load daily net returns matrix from returns store
    returns_matrix = load_returns_matrix(all_in_scope_ids, store_path=store_path)

    evaluated_candidates: List[str] = []
    for cid in all_in_scope_ids:
        if returns_matrix.empty or cid not in returns_matrix.columns or returns_matrix[cid].dropna().empty:
            skipped_candidates[cid] = f"No stored returns series found in {store_path.name} (logged before Phase 13 or missing store)"
        else:
            evaluated_candidates.append(cid)

    if len(evaluated_candidates) < 2:
        # Fewer than 2 candidates with stored returns — no pairwise correlation possible
        empty_corr = pd.DataFrame(index=evaluated_candidates, columns=evaluated_candidates, dtype=float)
        if len(evaluated_candidates) == 1:
            empty_corr.iloc[0, 0] = 1.0
        return RedundancyAnalysisResult(
            evaluated_candidates=evaluated_candidates,
            correlation_matrix=empty_corr,
            clusters=[],
            invalidated_candidates=[],
            skipped_candidates=skipped_candidates,
        )

    # Filter matrix to evaluated candidates
    matrix = returns_matrix[evaluated_candidates]

    # 4. Compute pairwise correlation matrix and track overlap exclusions
    n = len(evaluated_candidates)
    corr_df = pd.DataFrame(1.0, index=evaluated_candidates, columns=evaluated_candidates, dtype=float)
    pairwise_corrs: Dict[Tuple[str, str], float] = {}
    edges: List[Tuple[str, str, float]] = []

    for i in range(n):
        cid1 = evaluated_candidates[i]
        s1 = matrix[cid1]
        for j in range(i + 1, n):
            cid2 = evaluated_candidates[j]
            s2 = matrix[cid2]

            # Common valid dates
            valid_mask = s1.notna() & s2.notna()
            overlap_count = int(valid_mask.sum())

            if overlap_count < min_overlap_days:
                corr_df.loc[cid1, cid2] = float("nan")
                corr_df.loc[cid2, cid1] = float("nan")
                pair_key = f"pair ({cid1}, {cid2})"
                skipped_candidates[pair_key] = f"Insufficient date overlap ({overlap_count} days < min_overlap_days {min_overlap_days})"
            else:
                c1_clean = s1[valid_mask]
                c2_clean = s2[valid_mask]
                std1 = float(c1_clean.std())
                std2 = float(c2_clean.std())
                if std1 < 1e-8 or std2 < 1e-8:
                    val = 0.0
                else:
                    val = float(c1_clean.corr(c2_clean))
                    if pd.isna(val):
                        val = 0.0

                corr_df.loc[cid1, cid2] = val
                corr_df.loc[cid2, cid1] = val
                pairwise_corrs[(cid1, cid2)] = val
                pairwise_corrs[(cid2, cid1)] = val

                if val >= correlation_threshold:
                    edges.append((cid1, cid2, val))

    # 5. Connected-components clustering via Union-Find
    uf = UnionFind(evaluated_candidates)
    for u, v, _ in edges:
        uf.union(u, v)

    components: Dict[str, List[str]] = {}
    for cid in evaluated_candidates:
        root = uf.find(cid)
        components.setdefault(root, []).append(cid)

    # 6. Resolve clusters: select representative with highest oos_sharpe
    clusters: List[RedundancyCluster] = []
    invalidated_candidates: List[str] = []
    cluster_counter = 1

    for root, members in components.items():
        if len(members) > 1:
            # Sort members by oos_sharpe descending, then candidate_id ascending for determinism
            sorted_members = sorted(
                members,
                key=lambda x: (candidate_records[x].oos_sharpe, x),
                reverse=True,
            )
            representative = sorted_members[0]
            redundant = sorted_members[1:]

            cluster_corrs: Dict[Tuple[str, str], float] = {}
            for m1 in members:
                for m2 in members:
                    if m1 != m2:
                        cluster_corrs[(m1, m2)] = corr_df.loc[m1, m2]

            cluster = RedundancyCluster(
                cluster_id=cluster_counter,
                representative_id=representative,
                redundant_ids=redundant,
                pairwise_correlations=cluster_corrs,
            )
            clusters.append(cluster)
            cluster_counter += 1

            for red_id in redundant:
                invalidated_candidates.append(red_id)
                corr_val = corr_df.loc[representative, red_id]
                if apply_status_changes:
                    # Idempotency discipline: skip duplicate append if candidate is already INVALIDATED
                    if candidate_records[red_id].status != "INVALIDATED":
                        if pd.notna(corr_val) and corr_val >= correlation_threshold:
                            reason = f"redundant with {representative}, corr={corr_val:.3f} (>= {correlation_threshold:.2f})"
                            is_transitive = False
                        else:
                            display_corr = 0.0 if pd.isna(corr_val) else corr_val
                            reason = (
                                f"transitively redundant via cluster containing {representative}; "
                                f"direct correlation to {representative} is {display_corr:.3f}, below threshold {correlation_threshold:.2f}"
                            )
                            is_transitive = True
                        metadata = {
                            "redundant_with": representative,
                            "correlation": float(corr_val) if pd.notna(corr_val) else 0.0,
                            "transitive": is_transitive,
                        }
                        try:
                            append_status_change(
                                candidate_id=red_id,
                                new_status="INVALIDATED",
                                reason=reason,
                                status_log_path=status_log_path,
                                metadata=metadata,
                            )
                        except Exception:
                            pass

    return RedundancyAnalysisResult(
        evaluated_candidates=evaluated_candidates,
        correlation_matrix=corr_df,
        clusters=clusters,
        invalidated_candidates=invalidated_candidates,
        skipped_candidates=skipped_candidates,
    )
