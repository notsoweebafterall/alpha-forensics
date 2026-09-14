# Alpha Forensics

Alpha Forensics is a quantitative research system for discovering, testing, and falsifying alpha hypotheses. It combines evolutionary search with a validation pipeline to generate candidate trading signals, evaluate them out-of-sample, test their robustness, and determine whether apparent performance survives statistical and factor-based scrutiny.

**Core loop:** Generate, Test, Falsify, Repeat.

Every out-of-sample evaluation happens exactly once per candidate ("single-look"), is logged to an append-only ledger, and downstream OOS-based statistics are computed from evaluations recorded in the ledger.

A full writeup of the methodology, experiments, and results is available in `paper/alpha_forensics_paper.pdf`.

## Current Results

- 3 generation mechanisms tested: random compositional search, LLM-guided proposal, and genetic programming.
- Genetic programming: 300 candidates x 40 generations, mutation/crossover/selection over expression trees.
- Two universes tested: 60 tickers and a 169-ticker expanded universe (from a 174-ticker list, with 5 tickers dropped for delisting or missing data).
- No candidate passed the complete survivor criteria (DSR-significant, not factor-repackaged, non-redundant) across the three mechanisms and both universes.
- The strongest candidate was inactive on 84.7% of out-of-sample days, with 48.8% of its total return magnitude concentrated in its five largest single-day moves.
- Expanding the universe from 60 to 169 tickers did not raise the discovery ceiling; peak signal quality was lower on the larger universe. This is consistent with primitive vocabulary being an important constraint on discovery, though a single run per universe does not let us rule out search stochasticity as an alternative explanation.

## How It Works

**Technical backbone:** Genetic Programming, Expression Trees, Candidate Generation, OOS Validation, Statistical Testing, Factor Analysis, Redundancy, Provenance.

Candidates are represented as expression trees (composable primitives and operators, e.g. `Rank(Momentum(20))`), with deterministic structural hashing for identity and deduplication. Three generation mechanisms populate the candidate pool:

- **Genetic programming**, the primary search mechanism. Mutation and crossover operate directly on expression trees; selection is driven by cross-sectional information coefficient; a behavioral-fingerprint gate catches candidates that are structurally different but rank-invariant under monotonic transforms (a failure mode discovered and fixed during development).
- **Random compositional generation**, used as a baseline for comparing search quality.
- **LLM-guided proposal** uses Groq to reason over trial history and propose candidate expressions.

## Validation Gauntlet

| Phase | What it does |
|---|---|
| 7 | Walk-forward OOS validation (single-look, guarded) |
| 8 | Transaction cost sensitivity |
| 9 | Parameter landscape / stability |
| 10 | Statistical forensics: Deflated & Probabilistic Sharpe Ratio |
| 11 | Factor exposure regression (alpha intercept, HAC standard errors) |
| 12 | Market regime classification |
| 13 | Redundancy detection (return-correlation based) |
| 14 | Alpha registry (queryable index over trial data) |
| 15 | Portfolio construction |
| 16 | Reporting |
| 17 | Flagship experiment at real scale |

Phases 1 through 6 (data foundation, backtesting engine, expression system, strategy library, hypothesis spec, generation) are one-time infrastructure. Phases 7 onward, plus the GP loop, run repeatedly as new candidates are discovered.

## Core Design Principles

- **Single-look discipline**: an OOS evaluation, once logged, is never repeated.
- **Append-only, immutable ledger**: invalidated candidates get a status-change event, not a deleted row.
- **Canonical identity plus behavioral deduplication**: structural hashing catches identical trees; a separate fingerprint gate catches behaviorally-identical trees with different structure, protecting the DSR trial count from inflation.
- **Result integrity**: negative and empty results are retained in the research record.
- **Non-silent data degradation**: missing/delisted tickers are dropped with an explicit warning; live-data failures fall back to a clearly-labeled synthetic panel.

## Methodological Findings

Four issues were found and corrected during development, each documented in detail in the paper:

1. **Residual Sharpe bug**: the original factor-exposure test measured leftover alpha as the Sharpe ratio of regression residuals, which is mathematically guaranteed to be approximately zero for any OLS regression with an intercept, regardless of the true signal. Replaced with the regression intercept and its Newey-West t-statistic.
2. **Behavioral duplicate bug**: structurally distinct expressions that are rank-invariant under monotonic transforms (e.g. `Rank(X)` and `SignedPower(Rank(X), 2.0)`) produced identical trades but were logged as separate trials, inflating the DSR trial count. Fixed with a behavioral fingerprint gate.
3. **Walk-forward fold evaluation bug**: the fold-level training diagnostic was scored in-sample rather than on its designated held-out validation window. Confirmed not to affect any survivor determination, since the final OOS evaluation runs through a fully separate code path.
4. **Redundancy clustering transitivity bug**: candidates reaching a redundancy cluster only through transitive chains were logged with a reason string falsely claiming a direct above-threshold correlation to the cluster representative. Fixed to report transitive membership honestly.

## Current Limitations

- Primitive vocabulary is price/volume-derived only; the evidence is consistent with this being an important constraint on discovery, though not formally isolated from other factors.
- No cross-run research memory; each GP run searches independently.
- Neither ticker universe has a documented selection methodology or selection date; survivorship bias in universe construction is a real, unresolved threat to validity, not a checked and cleared item.
- Only one GP run was conducted per universe, with no repeated runs across different random seeds.
- Regime analysis is scoped to whichever regimes occur in a candidate's own OOS window.
- Redundancy testing is scoped to the current candidate universe.
- Portfolio construction uses simple equal-weight / inverse-volatility schemes.
- DSR corrects for multiple-testing selection bias; it does not eliminate all forms of data snooping (e.g. snooping in feature/primitive definition itself).
- The local data cache stores price/volume data under a fixed path not scoped to a specific universe or run, so a later experiment can silently overwrite the data underlying an earlier one.
- The expression-tree system functions as a working DSL in substance, though its vocabulary/parameter grid is currently duplicated by hand across several files rather than centralized in one registry.

## Stack

Quantitative Research, Genetic Programming, Python, Time-Series Analysis, Statistical Validation (DSR/PSR, factor regression), Financial Data (yfinance), Experiment Design and Provenance, LLM Integration (Groq).