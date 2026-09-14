"""
Population Management & Evolutionary Selection Loop for Genetic Programming Alpha Discovery.
"""

import copy
import logging
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple, Any
import numpy as np
import pandas as pd

from core.panel import Panel
from backtesting.metrics import predictive_metrics
from alpha.expressions.tree import Expression
from alpha.expressions.serialize import to_string
from alpha.generator.candidate import GeneratedCandidate
from alpha.generator.compositional_generator import generate_random_candidates
from alpha.generator.genetic.mutation import mutate_expression
from alpha.generator.genetic.crossover import crossover_expressions
from agent.registry_introspection import validate_expression_vocabulary

import hashlib


def compute_behavioral_fingerprint(signal: pd.DataFrame) -> str:
    """
    Computes a behavioral fingerprint for a signal DataFrame using ALL valid dates.

    Two signals with identical cross-sectional rank ordering across ALL dates will
    produce the exact same fingerprint.  Previous code sampled every 5th date
    (iloc[::5]) for speed, but that creates false-negative risk for slow-moving
    signals (e.g. Momentum(126), Momentum(252)) whose ranks barely change
    day-to-day — two genuinely distinct signals can agree on sampled dates while
    diverging in between.  For population_size=300 on a ~1,000-date panel, hashing
    all rows takes < 1 ms per candidate and eliminates the risk entirely.
    """
    valid_signal = signal.dropna(how="all")
    if valid_signal.empty:
        return "empty_signal"

    # Normalize to cross-sectional percent ranks (portfolio weights depend only on rank order)
    ranked = valid_signal.rank(axis=1, pct=True).round(4).fillna(-999.0)

    # Hash ALL dates — no subsampling
    return hashlib.md5(ranked.to_numpy().tobytes()).hexdigest()


@dataclass
class Individual:
    candidate: GeneratedCandidate
    expression_string: str
    canonical_hash: str
    fitness: float  # Absolute IC mean (|IC|) or 0.0 if invalid/degenerate/duplicate
    ic_mean: float
    coverage: float
    passed_screen: bool
    generation: int
    behavioral_fingerprint: str = ""
    is_behavioral_duplicate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression_string": self.expression_string,
            "canonical_hash": self.canonical_hash,
            "fitness": self.fitness,
            "ic_mean": self.ic_mean,
            "coverage": self.coverage,
            "passed_screen": self.passed_screen,
            "generation": self.generation,
            "behavioral_fingerprint": self.behavioral_fingerprint,
            "is_behavioral_duplicate": self.is_behavioral_duplicate,
        }


def evaluate_population_fitness(
    candidates: List[GeneratedCandidate],
    panel: Panel,
    generation: int = 0,
    ic_threshold: float = 0.005,
    min_coverage: float = 0.8,
) -> List[Individual]:
    """
    Cheap Stage 2 fitness evaluation for a population of candidate expressions on panel.

    Fitness Metric:
        Continuous absolute mean Information Coefficient (|IC mean|) if candidate passes
        coverage, non-degeneracy, AND behavioral uniqueness filters; 0.0 otherwise.

    Returns:
        List[Individual]: List of Individual objects populated with fitness metrics.
    """
    total_cells = panel.prices.size
    results: List[Individual] = []
    seen_fingerprints: Set[str] = set()

    for cand in candidates:
        expr = cand.expression
        c_hash = expr.canonical_hash()
        expr_str = cand.expression_string or to_string(expr)

        # 1. Signal Evaluation
        try:
            signal = expr.evaluate(panel)
        except Exception:
            results.append(
                Individual(
                    candidate=cand,
                    expression_string=expr_str,
                    canonical_hash=c_hash,
                    fitness=0.0,
                    ic_mean=0.0,
                    coverage=0.0,
                    passed_screen=False,
                    generation=generation,
                    behavioral_fingerprint="invalid",
                    is_behavioral_duplicate=False,
                )
            )
            continue

        # 2. Coverage Check
        non_nan_count = signal.notna().to_numpy().sum()
        coverage = float(non_nan_count / total_cells) if total_cells > 0 else 0.0
        if coverage < min_coverage:
            results.append(
                Individual(
                    candidate=cand,
                    expression_string=expr_str,
                    canonical_hash=c_hash,
                    fitness=0.0,
                    ic_mean=0.0,
                    coverage=coverage,
                    passed_screen=False,
                    generation=generation,
                    behavioral_fingerprint="low_coverage",
                    is_behavioral_duplicate=False,
                )
            )
            continue

        # 3. Degenerate Signal Check (cross-sectional variance)
        xs_stds = signal.std(axis=1)
        valid_stds = xs_stds.dropna()
        if valid_stds.empty or float(valid_stds.median()) < 1e-6 or (valid_stds < 1e-6).mean() > 0.5:
            results.append(
                Individual(
                    candidate=cand,
                    expression_string=expr_str,
                    canonical_hash=c_hash,
                    fitness=0.0,
                    ic_mean=0.0,
                    coverage=coverage,
                    passed_screen=False,
                    generation=generation,
                    behavioral_fingerprint="degenerate",
                    is_behavioral_duplicate=False,
                )
            )
            continue

        # 4. Behavioral Fingerprint Deduplication Check
        fp = compute_behavioral_fingerprint(signal)
        is_dup = fp in seen_fingerprints
        if not is_dup:
            seen_fingerprints.add(fp)

        # 5. Predictiveness Check (IC mean)
        try:
            pred = predictive_metrics(signal, panel, execution_lag_days=1)
            raw_ic = float(pred.get("ic_mean", 0.0))
            if np.isnan(raw_ic):
                raw_ic = 0.0

            # Demote behavioral duplicates to 0.0 fitness so they do not consume elitism slots
            fit = 0.0 if is_dup else abs(raw_ic)
            passed = (fit >= ic_threshold) and not is_dup
        except Exception:
            raw_ic = 0.0
            fit = 0.0
            passed = False

        results.append(
            Individual(
                candidate=cand,
                expression_string=expr_str,
                canonical_hash=c_hash,
                fitness=fit,
                ic_mean=raw_ic,
                coverage=coverage,
                passed_screen=passed,
                generation=generation,
                behavioral_fingerprint=fp,
                is_behavioral_duplicate=is_dup,
            )
        )

    return results


def select_tournament(
    population: List[Individual],
    k: int = 3,
    rng: Optional[random.Random] = None,
) -> Individual:
    """Selects a parent using tournament selection of size k."""
    if rng is None:
        rng = random.Random()

    sample = rng.sample(population, min(k, len(population)))
    return max(sample, key=lambda ind: ind.fitness)


def initialize_gp_population(
    population_size: int = 300,
    max_depth: int = 4,
    seed: int = 42,
) -> List[GeneratedCandidate]:
    """Generates the initial GP population using random compositional generation."""
    return generate_random_candidates(budget=population_size, max_depth=max_depth, seed=seed)


def breed_next_generation(
    current_population: List[Individual],
    population_size: int = 300,
    elite_fraction: float = 0.20,
    crossover_prob: float = 0.70,
    max_depth: int = 6,
    generation: int = 1,
    seed: int = 42,
    existing_hashes: Optional[Set[str]] = None,
) -> List[GeneratedCandidate]:
    """
    Breeds the next generation using Elitism + Crossover + Mutation.

    Elitism:
      Top (elite_fraction * population_size) individuals are preserved verbatim.

    Breeding:
      Remaining slots are filled via subtree crossover (crossover_prob) or mutation (1 - crossover_prob).
      Enforces canonical_hash() deduplication.
    """
    rng = random.Random(seed + generation * 1000)
    seen_hashes: Set[str] = set(existing_hashes) if existing_hashes else set()

    # Sort population by fitness descending
    sorted_pop = sorted(current_population, key=lambda ind: ind.fitness, reverse=True)

    next_candidates: List[GeneratedCandidate] = []

    # 1. Elitism: Keep top elite candidates
    num_elites = max(1, int(population_size * elite_fraction))
    for ind in sorted_pop[:num_elites]:
        h = ind.canonical_hash
        if h not in seen_hashes:
            seen_hashes.add(h)
            next_candidates.append(ind.candidate)

    # 2. Fill remaining population via Crossover & Mutation
    attempts = 0
    max_attempts = population_size * 20
    now = pd.Timestamp.now()

    while len(next_candidates) < population_size and attempts < max_attempts:
        attempts += 1
        r_val = rng.random()

        if r_val < crossover_prob:
            p1 = select_tournament(sorted_pop, k=3, rng=rng)
            p2 = select_tournament(sorted_pop, k=3, rng=rng)
            c1_expr, c2_expr = crossover_expressions(p1.candidate.expression, p2.candidate.expression, max_depth=max_depth, rng=rng)
            expr_to_add = c1_expr if rng.random() < 0.5 else c2_expr
        else:
            p = select_tournament(sorted_pop, k=3, rng=rng)
            expr_to_add = mutate_expression(p.candidate.expression, max_depth=max_depth, rng=rng)

        c_hash = expr_to_add.canonical_hash()
        if c_hash in seen_hashes:
            continue

        seen_hashes.add(c_hash)
        cand = GeneratedCandidate(
            expression=expr_to_add,
            expression_string=to_string(expr_to_add),
            generation_method="genetic_programming",
            seed=seed,
            parent_family=None,
            param_values={},
            generated_at=now,
        )
        next_candidates.append(cand)

    return next_candidates
