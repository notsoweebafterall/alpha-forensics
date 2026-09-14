"""
Compositional Validation Demo (Phase 6→7 Wiring).

Closes the gap between Phase 6 (systematic candidate generation + triage screening) and
Phase 7–13 (the rigorous validation gauntlet). Phase 6 already produces a `passed` list of
GeneratedCandidate objects that clear the cheap IC/coverage pre-filter; until this module,
nothing downstream ever consumed that output.

This script is additive orchestration only:
    - No new validation machinery is introduced.
    - `run_walk_forward_validation()`, `log_trial()`, and every Phase 7–13 internal is called
      unmodified, exactly as the existing per-phase demos already call them.
    - The only new logic here is (a) mapping GeneratedCandidate → canonical_id and (b)
      applying the max_candidates cap with explicit skipped-over-cap reporting.

Single-look discipline:
    Same rule as all other phases. Each candidate is checked against the trial ledger before
    running walk-forward validation. If already logged, its stored result is reused and the
    candidate is never re-evaluated against held-out data.

Candidate ID convention:
    - `compositional_random` candidates have no registered strategy name. They are identified
      by expression structure: `generated_{canonical_hash[:12]}`.
    - `variant_expansion` candidates (have a real `parent_family`) reuse the existing
      `build_candidate_id(parent_family, params=param_values)` convention so their IDs
      match what Phase 9 variant expansion would assign for the same expression.
    - Caveat (documented, not fixed here): Neither scheme encodes universe or date range.
      Two runs on different panels can collide on the same ID and reuse a stale ledger
      result. Not a live bug under the current single-panel setup, but noted so it is not
      rediscovered as a surprise if this is run against a different universe or date window.

Cap tie-break:
    `screen_candidates()` returns pass/fail only — no per-candidate predictiveness score is
    exposed. When `len(passed) > max_candidates`, the cap is applied in screening/generation
    order (i.e. the order candidates passed the triage filter), which is explicitly arbitrary
    — not a principled ranking. The skipped candidates and their expression strings are always
    logged to stdout.

Phase label in ledger:
    Rows written here use `phase = "Phase 6→7 Compositional Validation"` to cleanly
    distinguish them from Phase 7's `"Phase 7 Walk-Forward Validation"` rows in the shared
    trial_log.jsonl.
"""

import sys
from pathlib import Path
from typing import Optional, Union, List, Tuple
import pandas as pd

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.strategies import build_candidate_id
from alpha.generator.compositional_generator import generate_random_candidates
from alpha.generator.screening import screen_candidates
from alpha.generator.candidate import GeneratedCandidate
from validation.folds import WalkForwardConfig
from validation.oos import reset_oos_access_log
from validation.runner import run_walk_forward_validation
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from statistics import TrialRecord, load_trial_log, log_trial, compute_distribution_stats, save_returns


def _candidate_id(candidate: GeneratedCandidate) -> str:
    """
    Returns the canonical candidate_id for a GeneratedCandidate.

    - variant_expansion: delegate to build_candidate_id(parent_family, params) — same
      convention Phase 9 uses, so IDs are consistent across phases for the same expression.
    - compositional_random: structure-identified as generated_{hash[:12]} since there is
      no registered strategy name to anchor the ID to.
    """
    if candidate.generation_method == "variant_expansion" and candidate.parent_family:
        return build_candidate_id(candidate.parent_family, params=candidate.param_values or None)
    else:
        return f"generated_{candidate.canonical_hash()[:12]}"


def run_demo(
    tickers: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    log_path: Union[str, Path] = "data/trial_log.jsonl",
    store_path: Union[str, Path] = "data/trial_returns.parquet",
    cache_dir: Union[str, Path] = "data/cache",
    generation_budget: int = 200,
    generation_seed: int = 42,
    max_depth: int = 4,
    max_candidates: int = 25,
) -> None:
    """
    Runs Phase 6 generation + screening, then evaluates each passed candidate through
    the full Phase 7 walk-forward validation gauntlet with single-look ledger discipline.

    Args:
        tickers: Ticker list. Defaults to UNIVERSE_60.
        start_date: Panel start date.
        end_date: Panel end date.
        log_path: Path to trial_log.jsonl ledger.
        store_path: Path to trial_returns.parquet store.
        cache_dir: Price cache directory.
        generation_budget: Number of unique compositional candidates to generate (Phase 6).
        generation_seed: RNG seed for reproducible generation.
        max_depth: Maximum expression tree depth for compositional generation.
        max_candidates: Maximum candidates to run through Phase 7 after screening.
            When len(passed) > max_candidates, excess candidates are skipped in
            screening/generation order (explicitly arbitrary — no score is available
            from screen_candidates() to rank them) and reported to stdout.
    """
    print("=" * 90)
    print("ALPHA FORENSICS -- PHASE 6->7 COMPOSITIONAL VALIDATION DEMO")
    print("=" * 90)
    print(f"Log Path     : {log_path}")
    print(f"Returns Store: {store_path}")
    print(f"Generation   : budget={generation_budget}, seed={generation_seed}, max_depth={max_depth}")
    print(f"Cap          : max_candidates={max_candidates} (tie-break: screening/generation order -- arbitrary)")
    print("-" * 90)

    reset_oos_access_log()
    log_path = Path(log_path)
    store_path = Path(store_path)
    cache_dir = Path(cache_dir)
    target_tickers = UNIVERSE_60 if tickers is None else tickers

    # ── Step 1: Load panel ─────────────────────────────────────────────────────────────
    print(f"\nLoading {len(target_tickers)}-ticker Panel [{start_date} to {end_date}]...")
    try:
        panel = build_panel(
            tickers=target_tickers,
            start_date=start_date,
            end_date=end_date,
            missing_threshold=0.05,
            cache_dir=cache_dir,
        )
        print(f"Loaded Panel successfully! Dates: {len(panel.prices)}, Tickers: {len(panel.universe)}")
    except Exception as e:
        print(f"Warning: Could not fetch live data ({e}). Creating synthetic demo panel...")
        import numpy as np
        dates = pd.date_range(start_date, periods=800, freq="B")
        np.random.seed(42)
        p_df = pd.DataFrame(
            100.0 + np.random.randn(len(dates), len(target_tickers)).cumsum(axis=0),
            index=dates,
            columns=target_tickers,
        )
        v_df = pd.DataFrame(
            10000 + np.random.randint(0, 5000, size=(len(dates), len(target_tickers))),
            index=dates,
            columns=target_tickers,
        )
        panel = build_panel(
            tickers=target_tickers,
            start_date=dates[0],
            end_date=dates[-1],
            prices_df=p_df,
            volume_df=v_df,
        )

    # ── Step 2: Phase 6 — Generate + Screen ────────────────────────────────────────────
    print(f"\n--- PHASE 6: Compositional Generation (budget={generation_budget}, seed={generation_seed}) ---")
    raw_candidates = generate_random_candidates(
        budget=generation_budget,
        max_depth=max_depth,
        seed=generation_seed,
    )
    print(f"Generated {len(raw_candidates)} unique compositional candidates.")

    print("Applying fast triage screening (IC threshold=0.005, coverage=0.8)...")
    passed, rejected = screen_candidates(raw_candidates, panel, ic_threshold=0.005, min_coverage=0.8)

    from collections import Counter
    rejection_counts = Counter(reason for _, reason in rejected)
    print(f"Passed screening : {len(passed)}")
    print(f"Rejected         : {len(rejected)}")
    for reason, cnt in rejection_counts.items():
        print(f"  - {reason:<22}: {cnt}")

    # ── Step 3: Apply max_candidates cap ───────────────────────────────────────────────
    skipped_over_cap: List[GeneratedCandidate] = []
    if len(passed) > max_candidates:
        print(
            f"\nCap applied: {len(passed)} passed candidates exceed max_candidates={max_candidates}. "
            f"Retaining first {max_candidates} in screening/generation order (arbitrary — "
            f"screen_candidates() returns no score)."
        )
        skipped_over_cap = passed[max_candidates:]
        passed = passed[:max_candidates]
        print(f"Skipped over cap ({len(skipped_over_cap)} candidates):")
        for sc in skipped_over_cap:
            print(f"  SKIPPED_OVER_CAP | {_candidate_id(sc)} | {sc.expression_string}")

    retained = passed  # may be empty if all failed screening
    print(f"\nCandidates retained for Phase 7 evaluation: {len(retained)}")

    if not retained:
        print("\nNo candidates passed screening. Nothing to evaluate. Exiting.")
        _print_summary(
            generated=len(raw_candidates),
            screened=len(passed) + len(skipped_over_cap),
            retained=0,
            already_in_ledger=0,
            newly_logged=0,
            skipped_over_cap=len(skipped_over_cap),
        )
        return

    # ── Step 4: Phase 7 — Walk-Forward Validation with single-look ledger discipline ──
    print("\n--- PHASE 7: Walk-Forward Validation (single-look, with ledger reuse) ---")

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

    already_in_ledger = 0
    newly_logged = 0
    summary_rows = []

    for candidate in retained:
        cid = _candidate_id(candidate)
        expr = candidate.expression

        print(f"\n" + "-" * 80)
        print(f"Candidate ID : {cid}")
        print(f"Method       : {candidate.generation_method}")
        print(f"Expression   : {candidate.expression_string}")
        print("-" * 80)

        # Single-look guard: reuse from ledger if already logged
        if cid in existing_ledger:
            rec = existing_ledger[cid]
            print(
                f"'{cid}' already logged (oos_sharpe={rec.oos_sharpe:.4f}) "
                f"-- reusing per single-look discipline, not re-evaluating."
            )
            already_in_ledger += 1
            summary_rows.append({
                "Candidate ID": cid,
                "Expression": candidate.expression_string,
                "OOS Sharpe": rec.oos_sharpe,
                "Source": "ledger_reuse",
            })
            continue

        # Fresh evaluation
        val_res = run_walk_forward_validation(
            candidate_id=cid,
            expression=expr,
            panel=panel,
            wf_config=wf_config,
            backtest_config=backtest_config,
            cost_model=cost_model,
            oos_fraction=0.15,
        )

        deg = val_res.degradation
        oos_net_returns = val_res.oos_net_returns
        skew, kurtosis, t_len = compute_distribution_stats(oos_net_returns)

        print(f"  OOS Sharpe          : {deg['oos_sharpe']:.4f}")
        print(f"  Mean Train Sharpe   : {deg['mean_train_sharpe']:.4f}")
        print(f"  Degradation Ratio   : {deg['degradation_ratio']:.4f}")

        log_trial(
            TrialRecord(
                candidate_id=cid,
                phase="Phase 6->7 Compositional Validation",
                oos_sharpe=float(deg["oos_sharpe"]),
                timestamp=pd.Timestamp.now().isoformat(),
                skew=skew,
                kurtosis=kurtosis,
                track_record_length=t_len,
            ),
            log_path=log_path,
        )

        if oos_net_returns is not None:
            try:
                save_returns(cid, oos_net_returns, store_path=store_path)
            except ValueError:
                pass  # Already stored; immutability discipline

        newly_logged += 1
        summary_rows.append({
            "Candidate ID": cid,
            "Expression": candidate.expression_string,
            "OOS Sharpe": deg["oos_sharpe"],
            "Source": "newly_evaluated",
        })

    # ── Step 5: Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("COMPOSITIONAL VALIDATION SUMMARY TABLE")
    print("=" * 90)
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print(df.to_string(index=False))
    print("=" * 90)

    _print_summary(
        generated=len(raw_candidates),
        screened=len(passed) + len(skipped_over_cap),
        retained=len(retained),
        already_in_ledger=already_in_ledger,
        newly_logged=newly_logged,
        skipped_over_cap=len(skipped_over_cap),
    )


def _print_summary(
    generated: int,
    screened: int,
    retained: int,
    already_in_ledger: int,
    newly_logged: int,
    skipped_over_cap: int,
) -> None:
    print("\n" + "=" * 90)
    print("PHASE 6->7 FUNNEL SUMMARY")
    print("=" * 90)
    print(f"  Generated (Phase 6)         : {generated}")
    print(f"  Passed screening            : {screened}")
    print(f"  Skipped over cap            : {skipped_over_cap}")
    print(f"  Retained for Phase 7        : {retained}")
    print(f"    Already in ledger (reused): {already_in_ledger}")
    print(f"    Newly logged              : {newly_logged}")
    print("=" * 90)
    print("NOTE: Phase label in ledger rows is 'Phase 6->7 Compositional Validation',")
    print("  distinct from Phase 7's 'Phase 7 Walk-Forward Validation' rows.")
    print("NOTE: Results from this phase feed Phase 10 (DSR), Phase 13 (redundancy),")
    print("  Phase 14 (registry), Phase 15 (portfolio), and Phase 16 (reporting)")
    print("  via the same shared trial_log.jsonl and trial_returns.parquet used by all phases.")
    print("=" * 90)


if __name__ == "__main__":
    run_demo()
