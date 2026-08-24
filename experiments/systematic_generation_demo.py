"""
Systematic Alpha Generation Demo (Phase 6).

Demonstrates parameter variant expansion and reproducible compositional random candidate generation,
followed by fast triage screening on the 60-ticker Panel.

PREVIEW DISCLAIMER:
    This is an early preview of the systematic candidate generation funnel. Triage screening is a
    pre-filter only. Real walk-forward validation (Phase 7), cost sweeps (Phase 8), and statistical
    significance testing (Phase 10) will be applied in later pipeline phases.
"""

import sys
from collections import Counter
from pathlib import Path
import pandas as pd

# Add src/ to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.universe import UNIVERSE_60
from data.panel import build_panel
from alpha.strategies.registry import get_all_strategies
from alpha.generator.variant_expansion import expand_variants
from alpha.generator.compositional_generator import generate_random_candidates
from alpha.generator.screening import screen_candidates


def run_demo():
    print("=" * 90)
    print("ALPHA FORENSICS — PHASE 6 SYSTEMATIC ALPHA GENERATION DEMO")
    print("=" * 90)
    print("PREVIEW DISCLAIMER: Triage pre-screening preview only.")
    print("Full walk-forward (Phase 7), cost sweeps (Phase 8), and statistics (Phase 10) apply later.")
    print("-" * 90)

    # 1. Load real 60-ticker Panel
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    print(f"Loading 60-ticker Panel [{start_date} to {end_date}]...")

    try:
        panel = build_panel(
            tickers=UNIVERSE_60,
            start_date=start_date,
            end_date=end_date,
            missing_threshold=0.05,
            cache_dir="data/cache",
        )
        print(f"Loaded Panel successfully! Dates: {len(panel.prices)}, Tickers: {len(panel.universe)}")
    except Exception as e:
        print(f"Warning: Could not fetch live data ({e}). Creating synthetic demo panel...")
        dates = pd.date_range(start_date, periods=250, freq="B")
        import numpy as np
        np.random.seed(42)
        p_df = pd.DataFrame(100.0 + np.random.randn(len(dates), 60).cumsum(axis=0), index=dates, columns=UNIVERSE_60)
        v_df = pd.DataFrame(10000 + np.random.randint(0, 5000, size=(len(dates), 60)), index=dates, columns=UNIVERSE_60)
        panel = build_panel(tickers=UNIVERSE_60, start_date=dates[0], end_date=dates[-1], prices_df=p_df, volume_df=v_df)

    # 2. Mode 1: Parameter Variant Expansion
    print("\n--- MODE 1: Parameter Variant Expansion ---")
    variant_candidates = []
    strategies = get_all_strategies()
    for strat in strategies:
        expanded = expand_variants(strat.build, strat.variant_params, parent_family=strat.name)
        print(f"Expanded strategy '{strat.name}': {len(expanded)} unique candidates")
        variant_candidates.extend(expanded)

    print(f"Total variant expansion candidates: {len(variant_candidates)}")

    # 3. Mode 2: Compositional Random Generation
    print("\n--- MODE 2: Compositional Random Generation ---")
    budget = 200
    max_depth = 4
    seed = 42
    print(f"Generating random candidates: budget={budget}, max_depth={max_depth}, seed={seed}...")
    random_candidates = generate_random_candidates(budget=budget, max_depth=max_depth, seed=seed)
    print(f"Generated {len(random_candidates)} unique compositional candidates.")

    # Combine all candidates and deduplicate across both sets
    all_candidates_raw = variant_candidates + random_candidates
    seen_hashes = set()
    combined_candidates = []
    for c in all_candidates_raw:
        h = c.canonical_hash()
        if h not in seen_hashes:
            seen_hashes.add(h)
            combined_candidates.append(c)

    print(f"\nTotal combined unique candidates before screening: {len(combined_candidates)}")

    # 4. Fast Triage Screening
    print("\n--- SCREENING: Fast Triage Filter ---")
    passed, rejected = screen_candidates(combined_candidates, panel, ic_threshold=0.005, min_coverage=0.8)

    rejection_reasons = Counter([reason for _, reason in rejected])

    # 5. Funnel Summary Display
    print("\n" + "=" * 90)
    print("SYSTEMATIC GENERATION FUNNEL SUMMARY")
    print("=" * 90)
    print(f"Total Raw Candidates Generated : {len(all_candidates_raw)}")
    print(f"Total Unique (Post-Dedup)     : {len(combined_candidates)}")
    print(f"Passed Screening Filter       : {len(passed)} ({len(passed)/len(combined_candidates):.1%})")
    print(f"Rejected                      : {len(rejected)}")
    print("\nRejection Breakdown by Reason:")
    for reason, count in rejection_reasons.items():
        print(f"  - {reason:<22}: {count}")
    print("=" * 90)

    if passed:
        print("\nSample Passed Candidate:")
        sample = passed[0]
        print(f"  Method     : {sample.generation_method}")
        print(f"  Parent     : {sample.parent_family or 'None'}")
        print(f"  Expression : {sample.expression_string}")
        print(f"  Hash       : {sample.canonical_hash()[:16]}...")
    print("=" * 90)


if __name__ == "__main__":
    run_demo()
