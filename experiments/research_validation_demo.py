"""
Groq Research Round Validation Demo.

Takes the accepted proposals from a Phase 5.5 research round (data/agent/research_rounds/round_NNN.json)
and runs them through the exact same Phase 6 screening + Phase 7 validation gauntlet as
compositional_validation_demo.py — reusing every function and config unmodified.

Candidates from a research round are structurally identical to compositional_random
candidates (no registered strategy name), so they use the same generated_{hash[:12]}
ID convention.
"""

import json
import sys
from pathlib import Path
from typing import List, Union
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.expressions.serialize import parse
from alpha.generator.screening import screen_candidates
from alpha.generator.candidate import GeneratedCandidate
from validation.folds import WalkForwardConfig
from validation.oos import reset_oos_access_log
from validation.runner import run_walk_forward_validation
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from statistics import TrialRecord, load_trial_log, log_trial, compute_distribution_stats, save_returns


def _candidate_id(candidate: GeneratedCandidate) -> str:
    return f"generated_{candidate.canonical_hash()[:12]}"


def load_round_as_candidates(round_path: Union[str, Path]) -> List[GeneratedCandidate]:
    """Loads accepted proposals from a research round JSON and wraps them as GeneratedCandidate objects."""
    with open(round_path, "r", encoding="utf-8") as f:
        round_data = json.load(f)

    candidates = []
    for prop in round_data["accepted_proposals"]:
        expr = parse(prop["expression_string"])
        candidates.append(GeneratedCandidate(
            expression=expr,
            expression_string=prop["expression_string"],
            generation_method="research_agent_proposed",
            seed=None,
            parent_family=None,
            param_values={},
            generated_at=pd.Timestamp.now(),
        ))
    return candidates


def run_demo(
    round_path: Union[str, Path] = "data/agent/research_rounds/round_001.json",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    cache_dir: Union[str, Path] = "data/cache",
) -> None:
    print("=" * 90)
    print("ALPHA FORENSICS -- RESEARCH AGENT PROPOSAL VALIDATION DEMO")
    print("=" * 90)
    print(f"Round File   : {round_path}")

    reset_oos_access_log()
    log_path = Path(log_path)
    store_path = Path(store_path)
    cache_dir = Path(cache_dir)

    print(f"\nLoading {len(UNIVERSE_60)}-ticker Panel [{start_date} to {end_date}]...")
    panel = build_panel(
        tickers=UNIVERSE_60,
        start_date=start_date,
        end_date=end_date,
        missing_threshold=0.05,
        cache_dir=cache_dir,
    )
    print(f"Loaded Panel successfully! Dates: {len(panel.prices)}, Tickers: {len(panel.universe)}")

    print(f"\nLoading proposals from {round_path}...")
    raw_candidates = load_round_as_candidates(round_path)
    print(f"Loaded {len(raw_candidates)} accepted proposals as candidates.")

    print("\nApplying fast triage screening (IC threshold=0.005, coverage=0.8)...")
    passed, rejected = screen_candidates(raw_candidates, panel, ic_threshold=0.005, min_coverage=0.8)

    from collections import Counter
    rejection_counts = Counter(reason for _, reason in rejected)
    print(f"Passed screening : {len(passed)}")
    print(f"Rejected         : {len(rejected)}")
    for reason, cnt in rejection_counts.items():
        print(f"  - {reason:<22}: {cnt}")
    for cand, reason in rejected:
        print(f"  REJECTED | {reason:<22} | {cand.expression_string}")

    if not passed:
        print("\nNo candidates passed screening. Nothing to evaluate. Exiting.")
        return

    print("\n--- Walk-Forward Validation (single-look, with ledger reuse) ---")

    wf_config = WalkForwardConfig(
        mode="expanding",
        initial_train_window=400,
        step_size=60,
        val_window=60,
        embargo_days=10,
    )
    backtest_config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    cost_model = CostModel(cost_bps=5.0)

    existing_ledger = {r.candidate_id: r for r in load_trial_log(log_path)}
    summary_rows = []

    for candidate in passed:
        cid = _candidate_id(candidate)
        print(f"\n{'-' * 80}\nCandidate ID : {cid}\nExpression   : {candidate.expression_string}\n{'-' * 80}")

        if cid in existing_ledger:
            rec = existing_ledger[cid]
            print(f"'{cid}' already logged (oos_sharpe={rec.oos_sharpe:.4f}) -- reusing per single-look discipline.")
            summary_rows.append({"Candidate ID": cid, "Expression": candidate.expression_string,
                                  "OOS Sharpe": rec.oos_sharpe, "Source": "ledger_reuse"})
            continue

        val_res = run_walk_forward_validation(
            candidate_id=cid, expression=candidate.expression, panel=panel,
            wf_config=wf_config, backtest_config=backtest_config, cost_model=cost_model,
            oos_fraction=0.15,
        )
        deg = val_res.degradation
        skew, kurtosis, t_len = compute_distribution_stats(val_res.oos_net_returns)
        print(f"  OOS Sharpe: {deg['oos_sharpe']:.4f} | Degradation Ratio: {deg['degradation_ratio']:.4f}")

        log_trial(
            TrialRecord(
                candidate_id=cid,
                phase="Phase 5.5 Research Agent Proposal",
                oos_sharpe=float(deg["oos_sharpe"]),
                timestamp=pd.Timestamp.now().isoformat(),
                skew=skew, kurtosis=kurtosis, track_record_length=t_len,
            ),
            log_path=log_path,
        )
        if val_res.oos_net_returns is not None:
            try:
                save_returns(cid, val_res.oos_net_returns, store_path=store_path)
            except ValueError:
                pass

        summary_rows.append({"Candidate ID": cid, "Expression": candidate.expression_string,
                              "OOS Sharpe": deg["oos_sharpe"], "Source": "newly_evaluated"})

    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df.to_string(index=False))


if __name__ == "__main__":
    run_demo()