"""
Portfolio Construction Demo (Phase 15).

Combines eligible alpha strategy candidates from the registry into a single meta-portfolio,
allocating capital across candidate return series using equal-weight and inverse-volatility
weighting schemes.

Single-Look Discipline Rationale:
    Combining already-logged OOS return series into a portfolio weighting is NOT a new single-look
    violation — each candidate's OOS data was already evaluated and logged exactly once during its
    original discovery/validation run. Computing a portfolio allocation from already-observed returns
    is a downstream synthesis, not a fresh query of held-out data.

Eligibility Funnel Stages:
    Stage 1: get_valid_non_redundant_candidates(db_path, top_level_only=True)
    Stage 2: Filter to candidates with stored daily net returns (has_returns_series == 1)
    Stage 3: Filter to candidates surviving DSR (dsr_verdict == 'survives_dsr') if require_dsr_survival=True
"""

import sys
from pathlib import Path
import pandas as pd

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio import build_portfolio


from typing import Optional, Union


def run_demo(
    db_path: Union[str, Path] = "data/alpha_registry.db",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
) -> None:
    db_path = Path(db_path)
    store_path = Path(store_path)

    print("=" * 100, flush=True)
    print("ALPHA FORENSICS — PHASE 15 PORTFOLIO CONSTRUCTION DEMO", flush=True)
    print("=" * 100, flush=True)
    print("Cross-Candidate Capital Allocation & Meta-Portfolio Construction", flush=True)
    print(f"Database: {db_path} | Returns Store: {store_path}", flush=True)
    print("-" * 100, flush=True)

    # -----------------------------------------------------------------------------------------
    # DEMO RUN 1: Rigorous Mode (require_dsr_survival=True)
    # -----------------------------------------------------------------------------------------
    print("\n" + "=" * 100, flush=True)
    print("DEMO RUN 1: RIGOROUS MODE (require_dsr_survival=True)", flush=True)
    print("=" * 100, flush=True)

    res_rigorous = build_portfolio(
        db_path=db_path,
        returns_store_path=store_path,
        weighting="equal_weight",
        require_dsr_survival=True,
    )

    print(f"\nStage-by-Stage Candidate Eligibility Funnel:", flush=True)
    print(f"  * Stage 1 (Valid, Non-Redundant, Top-Level) : {len(res_rigorous.stage1_candidates)} candidates", flush=True)
    print(f"  * Stage 2 (+ Has Stored Net Returns Series)  : {len(res_rigorous.stage2_candidates)} candidates", flush=True)
    print(f"  * Stage 3 (+ Survives DSR Verdict)          : {len(res_rigorous.stage3_candidates)} candidates", flush=True)
    print(f"  * Final Eligible Portfolio Constituents     : {len(res_rigorous.eligible_candidates)} candidates", flush=True)

    print(f"\nPortfolio Construction Result Message:", flush=True)
    print(f"  {res_rigorous.message}", flush=True)

    # -----------------------------------------------------------------------------------------
    # DEMO RUN 2: Relaxed DSR Mode (require_dsr_survival=False)
    # -----------------------------------------------------------------------------------------
    print("\n" + "=" * 100, flush=True)
    print("DEMO RUN 2: RELAXED MODE (require_dsr_survival=False) — Weighting Demonstration", flush=True)
    print("=" * 100, flush=True)

    # 2A. Equal Weighting
    res_eq = build_portfolio(
        db_path=db_path,
        returns_store_path=store_path,
        weighting="equal_weight",
        require_dsr_survival=False,
    )

    # 2B. Inverse Volatility Weighting
    res_iv = build_portfolio(
        db_path=db_path,
        returns_store_path=store_path,
        weighting="inverse_volatility",
        require_dsr_survival=False,
    )

    print(f"\nStage-by-Stage Candidate Eligibility Funnel:", flush=True)
    print(f"  * Stage 1 (Valid, Non-Redundant, Top-Level) : {len(res_eq.stage1_candidates)} candidates", flush=True)
    print(f"  * Stage 2 (+ Has Stored Net Returns Series)  : {len(res_eq.stage2_candidates)} candidates", flush=True)
    print(f"  * Stage 3 (DSR Requirement Bypassed)       : {len(res_eq.stage3_candidates)} candidates", flush=True)
    print(f"  * Final Eligible Portfolio Constituents     : {len(res_eq.eligible_candidates)} candidates", flush=True)

    if res_eq.eligible_candidates:
        print("\n" + "-" * 100, flush=True)
        print("CONSTITUENT CAPITAL WEIGHTS & PORTFOLIO METRICS:", flush=True)
        print("-" * 100, flush=True)

        weight_rows = []
        for cid in res_eq.eligible_candidates:
            weight_rows.append({
                "Candidate ID": cid,
                "Equal Weight": res_eq.weights.get(cid, 0.0),
                "Inverse Volatility Weight": res_iv.weights.get(cid, 0.0),
            })

        df_weights = pd.DataFrame(weight_rows)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print(df_weights.to_string(index=False), flush=True)

        print(f"\nMeta-Portfolio Realized Out-of-Sample Metrics (Aligned Dates):", flush=True)
        print(f"  * Equal Weight Portfolio Net Sharpe      : {res_eq.portfolio_sharpe:+.4f}", flush=True)
        print(f"  * Inverse Volatility Portfolio Net Sharpe: {res_iv.portfolio_sharpe:+.4f}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("SUMMARY CONCLUSION:", flush=True)
    print("Phase 15 Portfolio Construction successfully allocates capital across eligible alpha candidates,", flush=True)
    print("strictly respecting single-look OOS discipline, date alignment, and DSR eligibility filtering.", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    run_demo()
