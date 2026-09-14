"""
Autonomous Genetic Programming Alpha Discovery Loop (Unattended End-to-End Runner).

Runs evolutionary search over expression trees to discover non-redundant, factor-distinct alpha:
  - Stage 1: Initial Population (300 candidates)
  - Stage 2: Cheap Fitness Evaluation (IC magnitude on dev panel)
  - Stage 3: Selection (Elitism top 20%, Tournament selection k=3)
  - Stage 4: Breeding (Subtree Crossover + Mutation)
  - Stage 5: Generation Loop (up to 40 generations with per-generation IC & diversity diagnostics)
  - Stage 6: Final Gauntlet (Top candidates evaluated via walk-forward, 10bps costs, DSR, HAC OLS factor exposure, redundancy)
  - Stage 7: Stop Condition (Stops early when 5 genuine survivors are found or max_generations reached)

Usage:
  # Full production run (300 pop, 40 max generations):
  python experiments/genetic_discovery_loop.py

  # Fast smoke test (20 pop, 3 generations):
  python experiments/genetic_discovery_loop.py --smoke-test
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import pandas as pd
import numpy as np

# Add src/ to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_60, UNIVERSE_150
from data.panel import build_panel
from validation.oos import reserve_oos_holdout, evaluate_oos, reset_oos_access_log
from backtesting.config import BacktestConfig
from backtesting.costs import CostModel
from statistics import compute_distribution_stats
from statistics import (
    TrialRecord,
    load_trial_log,
    log_trial,
    effective_trial_records,
    save_returns,
    load_returns,
    sharpe_variance_across_trials,
    expected_max_sharpe_under_trials,
    DSRResult,
    log_dsr_result,
)
from factors.factor_construction import build_factor_panel
from factors.exposure import run_factor_regression
from redundancy.correlation_analysis import run_redundancy_analysis
from registry import build_registry
from reporting import build_report

from alpha.strategies import build_candidate_id
from alpha.generator.genetic.population import (
    Individual,
    initialize_gp_population,
    evaluate_population_fitness,
    breed_next_generation,
    compute_behavioral_fingerprint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("genetic_discovery_loop")


def _candidate_id(cand) -> str:
    if cand.generation_method == "variant_expansion" and cand.parent_family:
        return build_candidate_id(cand.parent_family, params=cand.param_values or None)
    else:
        return f"generated_{cand.canonical_hash()[:12]}"


def run_full_gauntlet_for_candidates(
    candidates: List[Any],
    panel: Any,
    dev_panel: Any,
    oos_panel: Any,
    log_p: Path,
    status_p: Path,
    store_p: Path,
    dsr_p: Path,
    db_p: Path,
    reports_dir: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Runs Stage 6 gauntlet for candidate set:
      1. Walk-forward OOS evaluation
      2. 10bps cost sensitivity sweep
      3. Phase 10 DSR re-run
      4. Phase 11 Factor Exposure HAC OLS regression
      5. Phase 13 Redundancy Analysis

    Returns:
        Tuple[List[Dict], List[Dict]]: (all_gauntlet_results, survivors)
    """
    backtest_config = BacktestConfig(execution_lag_days=1, dollar_neutral=True)
    cost_model_5bps = CostModel(cost_bps=5.0)
    cost_model_10bps = CostModel(cost_bps=10.0)

    evaluated_results: List[Dict[str, Any]] = []

    print(f"\n[Stage 6 Gauntlet] Evaluating {len(candidates)} top candidates...")

    # Pre-load existing trial ledger once (canonical_id AND behavioral fingerprints)
    existing_ledger = {r.candidate_id: r for r in load_trial_log(log_p)}

    # Build a set of behavioral fingerprints already in the ledger so that a
    # behaviorally-identical candidate with a *different* canonical hash cannot
    # inflate N_trials in DSR.
    #
    # We cannot recompute signals for already-logged candidates (no stored signal),
    # so we seed from fingerprints carried in the current gauntlet batch itself.
    # Any fingerprint seen within this batch is marked as duplicate on its second
    # appearance, matching the same logic used during fitness evaluation.
    seen_gauntlet_fingerprints: Set[str] = set()

    # 1. Walk-Forward OOS Evaluation (5bps & 10bps)
    for cand in candidates:
        cid = _candidate_id(cand)
        cost_cid = f"{cid}__cost_10bps"

        # --- Behavioral fingerprint gate (Issue 2 fix) ---
        # Compute signal on oos_panel (same panel used for evaluation below) and
        # derive its behavioral fingerprint.  If we've already sent an identical
        # signal through this gauntlet batch, skip log_trial entirely — it's a
        # behavioral duplicate that would inflate N_trials in DSR.
        try:
            raw_signal = cand.expression.evaluate(oos_panel)
            fp = compute_behavioral_fingerprint(raw_signal)
        except Exception:
            fp = "invalid"

        is_behavioral_dup = fp in seen_gauntlet_fingerprints
        if fp not in ("invalid", "empty_signal"):
            seen_gauntlet_fingerprints.add(fp)

        if is_behavioral_dup:
            logger.info(
                "[Gauntlet] Skipping behavioral duplicate before log_trial: %s (fp=%s…)",
                cid, fp[:8],
            )
            evaluated_results.append({
                "candidate_id": cid,
                "expression_string": cand.expression_string,
                "sharpe_5bps": 0.0,
                "sharpe_10bps": 0.0,
                "cost_survives": False,
                "behavioral_duplicate": True,
            })
            continue

        # Check existing ledger to preserve single-look discipline

        if cid not in existing_ledger:
            reset_oos_access_log()
            oos_res = evaluate_oos(
                candidate_id=cid,
                expression=cand.expression,
                oos_panel=oos_panel,
                config=backtest_config,
                cost_model=cost_model_5bps,
            )
            sharpe_5bps = float(oos_res.get("sharpe_ratio", 0.0))
            net_rets_5bps = oos_res.get("oos_net_returns")
            skew_5, kurt_5, t_len_5 = compute_distribution_stats(net_rets_5bps)

            log_trial(
                TrialRecord(
                    candidate_id=cid,
                    phase="Phase 6->7 Compositional Validation",
                    oos_sharpe=sharpe_5bps,
                    timestamp=pd.Timestamp.now().isoformat(),
                    skew=skew_5,
                    kurtosis=kurt_5,
                    track_record_length=t_len_5,
                ),
                log_path=log_p,
            )
            if net_rets_5bps is not None:
                try:
                    save_returns(cid, net_rets_5bps, store_path=store_p)
                except ValueError:
                    pass
        else:
            sharpe_5bps = existing_ledger[cid].oos_sharpe

        # 10bps cost model evaluation
        if cost_cid not in existing_ledger:
            reset_oos_access_log()
            oos_res_10 = evaluate_oos(
                candidate_id=cost_cid,
                expression=cand.expression,
                oos_panel=oos_panel,
                config=backtest_config,
                cost_model=cost_model_10bps,
            )
            sharpe_10bps = float(oos_res_10.get("sharpe_ratio", 0.0))
            net_rets_10 = oos_res_10.get("oos_net_returns")
            skew_10, kurt_10, t_len_10 = compute_distribution_stats(net_rets_10)

            log_trial(
                TrialRecord(
                    candidate_id=cost_cid,
                    phase="Phase 8 Cost Sensitivity",
                    oos_sharpe=sharpe_10bps,
                    timestamp=pd.Timestamp.now().isoformat(),
                    skew=skew_10,
                    kurtosis=kurt_10,
                    track_record_length=t_len_10,
                ),
                log_path=log_p,
            )
            if net_rets_10 is not None:
                try:
                    save_returns(cost_cid, net_rets_10, store_path=store_p)
                except ValueError:
                    pass
        else:
            sharpe_10bps = existing_ledger[cost_cid].oos_sharpe

        evaluated_results.append({
            "candidate_id": cid,
            "expression_string": cand.expression_string,
            "sharpe_5bps": sharpe_5bps,
            "sharpe_10bps": sharpe_10bps,
            "cost_survives": sharpe_10bps > 0.0,
        })

    # 2. Phase 10 DSR Re-Run
    active_records = effective_trial_records(log_path=log_p, status_log_path=status_p)
    active_sharpes = [r.oos_sharpe for r in active_records]
    N_trials = len(active_records)
    v_sr = sharpe_variance_across_trials(active_sharpes)

    dsr_map = {}
    for rec in active_records:
        dsr_res = DSRResult.create(
            candidate_id=rec.candidate_id,
            observed_sharpe=rec.oos_sharpe,
            sharpe_variance=v_sr,
            n_trials=N_trials,
            skew=rec.skew,
            kurtosis=rec.kurtosis,
            track_record_length=rec.track_record_length or 150,
        )
        log_dsr_result(dsr_res, log_path=dsr_p)
        dsr_map[rec.candidate_id] = dsr_res

    # 3. Phase 11 Factor Exposure Regression
    factor_panel = build_factor_panel(oos_panel)
    all_sectors = list(factor_panel.sector_returns.columns)

    factor_map = {}
    for item in evaluated_results:
        cid = item["candidate_id"]
        if item["sharpe_5bps"] > 0.0:
            ret_series = load_returns(cid, store_path=store_p)
            if ret_series is not None and not ret_series.empty:
                f_res = run_factor_regression(
                    candidate_returns=ret_series,
                    factor_panel=factor_panel,
                    candidate_sectors=all_sectors,
                    candidate_id=cid,
                    nw_lags=5,
                )
                factor_map[cid] = f_res

    # 4. Phase 13 Redundancy Analysis
    red_res = run_redundancy_analysis(
        log_path=log_p,
        status_log_path=status_p,
        store_path=store_p,
        correlation_threshold=0.90,
        min_overlap_days=60,
        apply_status_changes=True,
    )
    invalidated_ids = set(red_res.invalidated_candidates)

    # Rebuild SQLite Registry & Report
    build_registry(log_path=log_p, status_log_path=status_p, dsr_log_path=dsr_p, store_path=store_p, db_path=db_p)
    build_report(db_path=db_p, returns_store_path=store_p, output_dir=reports_dir)

    # 5. Check Strict Survivor Definition
    survivors: List[Dict[str, Any]] = []

    for item in evaluated_results:
        cid = item["candidate_id"]
        dsr_info = dsr_map.get(cid)
        dsr_verdict = dsr_info.verdict if dsr_info else "fails_dsr"

        factor_info = factor_map.get(cid)
        factor_verdict = factor_info.verdict if factor_info else "factor_repackaging"
        alpha_significant = factor_info.alpha_significant if factor_info else False

        is_redundant = cid in invalidated_ids

        item["dsr_verdict"] = dsr_verdict
        item["factor_verdict"] = factor_verdict
        item["alpha_significant"] = alpha_significant
        item["is_redundant"] = is_redundant

        # Survivor Condition Check
        cond1 = dsr_verdict != "fails_dsr"
        cond2 = factor_verdict != "factor_repackaging" and alpha_significant
        cond3 = not is_redundant

        if cond1 and cond2 and cond3:
            item["survivor"] = True
            survivors.append(item)
        else:
            item["survivor"] = False

    return evaluated_results, survivors


def run_genetic_discovery_loop(
    population_size: int = 300,
    max_generations: int = 40,
    elite_fraction: float = 0.20,
    crossover_prob: float = 0.70,
    max_depth: int = 6,
    check_every: int = 10,
    survivor_target: int = 5,
    seed: int = 42,
    tickers: list = None,  # defaults to UNIVERSE_60 if None (backward compatible)
    log_path: str = "data/trial_log.jsonl",
    status_log_path: str = "data/trial_status_log.jsonl",
    store_path: str = "data/trial_returns.parquet",
    dsr_log_path: str = "data/dsr_results.jsonl",
    db_path: str = "data/alpha_registry.db",
    reports_dir: str = "reports",
    runs_dir: str = "data/agent/genetic_runs",
    cache_dir: str = "data/cache",
):
    print("=" * 100)
    print("ALPHA FORENSICS — AUTONOMOUS GENETIC PROGRAMMING ALPHA DISCOVERY LOOP")
    print("=" * 100)

    log_p = Path(log_path)
    status_p = Path(status_log_path)
    store_p = Path(store_path)
    dsr_p = Path(dsr_log_path)
    db_p = Path(db_path)
    cache_p = Path(cache_dir)
    runs_p = Path(runs_dir)
    runs_p.mkdir(parents=True, exist_ok=True)

    # 1. Load Panel & Holdout
    universe = tickers if tickers is not None else UNIVERSE_60
    if tickers is None or universe == UNIVERSE_60:
        universe_name = "UNIVERSE_60"
    elif universe == UNIVERSE_150:
        universe_name = "UNIVERSE_150"
    else:
        universe_name = "CUSTOM"

    requested_universe_size = len(universe)
    print(f"\n[Step 1] Loading Market Data Panel & Reserving Holdout... ({requested_universe_size} tickers)")
    panel = build_panel(
        tickers=universe,
        start_date="2020-01-01",
        end_date="2023-12-31",
        missing_threshold=0.05,
        cache_dir=cache_p,
    )
    actual_universe_size = len(panel.universe)
    dev_panel, oos_panel = reserve_oos_holdout(panel, oos_fraction=0.15)
    print(f"Development Panel : {len(dev_panel.prices)} dates x {len(dev_panel.universe)} tickers")
    print(f"Holdout Panel     : {len(oos_panel.prices)} dates x {len(oos_panel.universe)} tickers")

    # 2. Stage 1: Population Initialization
    print(f"\n[Step 2] Initializing GP Population (Size = {population_size}, Max Depth = {max_depth})...")
    candidates = initialize_gp_population(population_size=population_size, max_depth=4, seed=seed)

    generation_history: List[Dict[str, Any]] = []
    cumulative_survivors: List[Dict[str, Any]] = []

    print("\n" + "=" * 115)
    print(f"{'Gen':<5} | {'Best IC (|IC|)':<15} | {'Mean IC':<10} | {'Unique Hashes':<14} | {'Unique Behaviors':<16} | {'Passed Screen':<14} | {'Status':<20}")
    print("-" * 115)

    # 3. Stages 2-5: Evolutionary Generation Loop
    for gen in range(1, max_generations + 1):
        # Fitness Evaluation
        pop_individuals = evaluate_population_fitness(candidates, dev_panel, generation=gen)

        fitness_scores = [ind.fitness for ind in pop_individuals]
        best_fit = max(fitness_scores) if fitness_scores else 0.0
        mean_fit = float(np.mean(fitness_scores)) if fitness_scores else 0.0
        unique_hashes = len({ind.canonical_hash for ind in pop_individuals})
        unique_behaviors = len({ind.behavioral_fingerprint for ind in pop_individuals if ind.behavioral_fingerprint not in ("invalid", "low_coverage", "degenerate")})
        passed_count = sum(1 for ind in pop_individuals if ind.passed_screen)

        gen_info = {
            "generation": gen,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "unique_hashes": unique_hashes,
            "unique_behaviors": unique_behaviors,
            "passed_screen": passed_count,
        }
        generation_history.append(gen_info)

        print(
            f"{gen:<5} | {best_fit:<15.6f} | {mean_fit:<10.6f} | {unique_hashes:<14} | {unique_behaviors:<16} | {passed_count:<14} | Evolving..."
        )

        # Stage 6 & 7: Periodic Gauntlet Check
        if gen % check_every == 0 or gen == max_generations:
            print(f"\n---> [Periodic Check at Generation {gen}] Running Full Gauntlet on Top Candidates...")
            top_pop = sorted(pop_individuals, key=lambda ind: ind.fitness, reverse=True)[:25]
            top_candidates = [ind.candidate for ind in top_pop]

            gauntlet_res, new_survivors = run_full_gauntlet_for_candidates(
                candidates=top_candidates,
                panel=panel,
                dev_panel=dev_panel,
                oos_panel=oos_panel,
                log_p=log_p,
                status_p=status_p,
                store_p=store_p,
                dsr_p=dsr_p,
                db_p=db_p,
                reports_dir=reports_dir,
            )

            for s in new_survivors:
                if not any(x["candidate_id"] == s["candidate_id"] for x in cumulative_survivors):
                    cumulative_survivors.append(s)

            print(f"---> Generation {gen} Gauntlet complete. New Survivors: {len(new_survivors)} | Total Cumulative Survivors: {len(cumulative_survivors)}/{survivor_target}")

            if len(cumulative_survivors) >= survivor_target:
                print(f"\n[STOP CONDITION MET] Discovered {len(cumulative_survivors)} genuine non-redundant alpha survivors! Stopping loop early at generation {gen}.")
                break

        # Breed Next Generation
        candidates = breed_next_generation(
            current_population=pop_individuals,
            population_size=population_size,
            elite_fraction=elite_fraction,
            crossover_prob=crossover_prob,
            max_depth=max_depth,
            generation=gen,
            seed=seed,
        )

    # 4. Save Run History & Produce Final Summary
    timestamp_str = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    run_file = runs_p / f"run_{timestamp_str}.json"

    run_output = {
        "run_timestamp": timestamp_str,
        "universe_name": universe_name,
        "requested_universe_size": requested_universe_size,
        "actual_universe_size": actual_universe_size,
        "population_size": population_size,
        "max_generations": max_generations,
        "elite_fraction": elite_fraction,
        "crossover_prob": crossover_prob,
        "max_depth": max_depth,
        "total_generations_run": len(generation_history),
        "total_survivors_found": len(cumulative_survivors),
        "survivors": cumulative_survivors,
        "generation_history": generation_history,
    }

    with run_file.open("w", encoding="utf-8") as f:
        json.dump(run_output, f, indent=2)

    print("\n" + "=" * 100)
    print("AUTONOMOUS GENETIC PROGRAMMING DISCOVERY LOOP COMPLETE")
    print("=" * 100)
    print(f"Total Generations Run : {len(generation_history)}")
    print(f"Genuine Survivors     : {len(cumulative_survivors)} / {survivor_target}")
    print(f"Run Output File       : {run_file}")

    if cumulative_survivors:
        print("\nGENUINE ALPHA SURVIVORS DISCOVERED:")
        for s in cumulative_survivors:
            print(f"  * [{s['candidate_id']}] {s['expression_string']}")
            print(f"    5bps Sharpe={s['sharpe_5bps']:.4f} | 10bps Sharpe={s['sharpe_10bps']:.4f} | DSR={s['dsr_verdict']} | Alpha={s['factor_verdict']}")
    else:
        print("\n[HONEST RESULT] Zero candidates met all 3 strict survivor criteria (Passes DSR, Distinct Alpha Intercept, Non-Redundant).")
        print("This is the honest forensic output of evolutionary search over current market data.")

    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous GP Alpha Discovery Loop")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast reduced smoke test (20 pop, 3 gens)")
    parser.add_argument("--pop-size", type=int, default=300, help="Population size (default 300)")
    parser.add_argument("--max-gens", type=int, default=40, help="Max generations (default 40)")
    parser.add_argument("--check-every", type=int, default=10, help="Run gauntlet every N generations (default 10)")
    parser.add_argument("--survivor-target", type=int, default=5, help="Target survivor count (default 5)")
    parser.add_argument("--universe-150", action="store_true", help="Run against UNIVERSE_150 instead of UNIVERSE_60")

    args = parser.parse_args()

    if args.smoke_test:
        pop_sz = 20
        max_g = 3
        check_ev = 3
    else:
        pop_sz = args.pop_size
        max_g = args.max_gens
        check_ev = args.check_every

    universe = UNIVERSE_150 if args.universe_150 else None  # None → defaults to UNIVERSE_60

    run_genetic_discovery_loop(
        population_size=pop_sz,
        max_generations=max_g,
        check_every=check_ev,
        survivor_target=args.survivor_target,
        tickers=universe,
    )
