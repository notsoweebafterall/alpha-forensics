"""
Statistical Forensics & Multiple Testing Demo (Phase 10).

Demonstrates persistent trial ledger backfilling, Probabilistic Sharpe Ratio (PSR),
and Deflated Sharpe Ratio (DSR) evaluation for Phase 9 alpha candidates.

DISCLAIMER:
    DSR evaluates selection bias across multiple candidate trials. It does not replace
    parameter landscape stability checks (Phase 9), transaction cost modeling (Phase 8),
    or regime robustness testing (Phase 11/12).
"""

import sys
from pathlib import Path
import pandas as pd

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from statistics import (
    TrialRecord,
    load_trial_log,
    trial_count,
    backfill_from_phase_artifacts,
    expected_max_sharpe_under_trials,
    deflated_sharpe_ratio,
    DSRResult,
    probabilistic_sharpe_ratio,
    sharpe_variance_across_trials,
)


def run_demo():
    print("=" * 100)
    print("ALPHA FORENSICS — PHASE 10 STATISTICAL FORENSICS & MULTIPLE TESTING DEMO")
    print("=" * 100)
    print("Persistent Trial Ledger & Deflated Sharpe Ratio (DSR) Evaluation (Bailey & López de Prado, 2014)")
    print("-" * 100)

    log_path = Path("data/trial_log.jsonl")

    # 1. Backfill persistent trial log with Phase 4-9 candidates
    print(f"Backfilling trial log ledger at '{log_path}'...")
    added_records = backfill_from_phase_artifacts(log_path=log_path)
    total_n = trial_count(log_path=log_path)
    print(f"Ingested {len(added_records)} backfilled trials. Total distinct trial count N = {total_n}")

    # 2. Extract Sharpes across all logged trials to compute trial variance V[SR_hat]
    all_records = load_trial_log(log_path=log_path)
    all_sharpes = [r.oos_sharpe for r in all_records]
    var_sr = sharpe_variance_across_trials(all_sharpes)
    expected_max_sr = expected_max_sharpe_under_trials(var_sr, total_n)

    print(f"Sample Sharpe Variance V[SR_hat] across N={total_n} trials: {var_sr:.6f}")
    print(f"Expected Max Sharpe SR_0 under N={total_n} trials: {expected_max_sr:.4f}")
    print("-" * 100)

    # 3. Evaluate Phase 9 candidates
    candidates = [
        {
            "id": "cross_sectional_momentum",
            "name": "Cross-Sectional Momentum",
            "oos_sharpe": 0.45,
            "skew": -0.15,
            "kurtosis": 3.4,
            "t_len": 252,
            "phase9_landscape": "33% positive Sharpes across 1D parameter grid (Isolated Peak / Fragile)",
        },
        {
            "id": "volatility_adjusted_momentum",
            "name": "Volatility-Adjusted Momentum",
            "oos_sharpe": 0.52,
            "skew": 0.05,
            "kurtosis": 3.1,
            "t_len": 252,
            "phase9_landscape": "50% positive Sharpes across 2D parameter grid (Moderate Neighborhood Robustness)",
        },
    ]

    results = []
    for c in candidates:
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=c["oos_sharpe"],
            benchmark_sharpe=0.0,
            skew=c["skew"],
            kurtosis=c["kurtosis"],
            track_record_length=c["t_len"],
        )

        dsr_res = DSRResult.create(
            candidate_id=c["id"],
            observed_sharpe=c["oos_sharpe"],
            sharpe_variance=var_sr,
            n_trials=total_n,
            skew=c["skew"],
            kurtosis=c["kurtosis"],
            track_record_length=c["t_len"],
        )

        results.append({
            "Candidate Name": c["name"],
            "OOS Sharpe": c["oos_sharpe"],
            "PSR (vs 0)": f"{psr:.4f}",
            "N (Trials)": total_n,
            "Expected Max SR_0": f"{dsr_res.expected_max_sharpe:.4f}",
            "DSR": f"{dsr_res.dsr:.4f}",
            "Verdict": dsr_res.verdict,
            "Phase 9 Landscape Finding": c["phase9_landscape"],
        })

    # 4. Display summary comparison table
    df = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\nSTATISTICAL FORENSICS & DSR SUMMARY REPORT")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    # 5. Display detailed DSR Result & Caveats
    print("\nDETAILED DSR CAVEATS & FORENSIC RATIONALE:")
    for c in candidates:
        dsr_res = DSRResult.create(
            candidate_id=c["id"],
            observed_sharpe=c["oos_sharpe"],
            sharpe_variance=var_sr,
            n_trials=total_n,
            skew=c["skew"],
            kurtosis=c["kurtosis"],
            track_record_length=c["t_len"],
        )
        print(f"\nCandidate: [{c['name']}] (ID: {c['id']})")
        print(f"  - Observed OOS Sharpe : {dsr_res.observed_sharpe:.4f}")
        print(f"  - DSR Score           : {dsr_res.dsr:.4f} ({dsr_res.verdict})")
        print(f"  - Phase 9 Finding     : {c['phase9_landscape']}")
        print("  - Auto-Generated Caveats:")
        for cav in dsr_res.caveats:
            print(f"      * {cav}")

    print("\n" + "=" * 100)
    print("SUMMARY CONCLUSION:")
    print("PSR vs zero measures isolated statistical significance, whereas DSR accounts for the total number of")
    print("trials N evaluated during discovery. Combining DSR with Phase 9 parameter landscape robustness")
    print("prevents overestimating strategy viability prior to production deployment.")
    print("=" * 100)


if __name__ == "__main__":
    run_demo()
