# Alpha Forensics

A research-oriented alpha discovery and validation pipeline. This is **not** a
trading bot and **not** a dashboard — it's a forensics system for testing
whether a candidate trading signal has real, statistically defensible
out-of-sample edge, or whether it's noise, an overfit artifact, or a
duplicate of something already tested.

The system's core discipline: every OOS (out-of-sample) evaluation happens
exactly once per candidate ("single-look"), is logged to an append-only,
immutable ledger, and every downstream number — Sharpe ratios, DSR verdicts,
regime robustness, portfolio weights — traces back to one of those real,
guarded evaluations. Nothing here is synthetic, hardcoded, or fabricated as
if it were a real result. If a stage produces no viable candidate, the
system says so plainly rather than dressing up a negative result.

## Current honest state (as of the last flagship run)

Running the full pipeline on 169 real tickers over 2019-2023:

- 36 total candidate trials logged across all phases
- 7 valid, non-redundant, top-level candidates
- 1 explicitly evaluated and **failed** DSR (deflated Sharpe ratio)
- 6 not yet DSR-evaluated
- **0 candidates currently survive rigorous DSR filtering**
- Portfolio construction under rigorous criteria returns an **empty
  portfolio** — this is correct, not a bug. No alpha has been found yet
  that clears the statistical bar this system enforces.
- One pair of candidates (`cross_sectional_momentum` and
  `volatility_adjusted_momentum`) turned out to be redundant
  (correlation 0.964) — effectively the same trade under two names.

This is the expected, useful outcome for a rigorous discovery pipeline in
early stages: it's much easier to find signals that *look* good than
signals that survive real scrutiny. The absence of a surviving candidate
is itself a real finding, not a failure of the tooling.

## Architecture — phase by phase

| Phase | What it does |
|---|---|
| 1 | Data & feature foundation (Panel abstraction over price/volume) |
| 2 | Backtesting engine (position sizing, execution lag, dollar-neutral) |
| 3 | Alpha expression system (composable primitives + structural dedup) |
| 4 | Strategy library (named, parameterized strategy definitions) |
| 5 | Hypothesis specification (economic rationale → testable spec) |
| 6 | Systematic candidate generation (variant expansion, triage screening) |
| 7 | Walk-forward / true OOS validation (guarded single-look) |
| 8 | Transaction cost sensitivity sweeps |
| 9 | Parameter landscape / stability analysis |
| 10 | Statistical forensics — Probabilistic & Deflated Sharpe Ratio |
| 11 | Factor exposure regression & cross-sector/period generalization |
| 12 | Market regime classification & robustness testing |
| 13 | Redundancy detection (correlation-based, not just structural) |
| 14 | Alpha registry — queryable SQLite index over all trial data |
| 15 | Cross-candidate portfolio construction |
| 16 | Consolidated, honest reporting |
| 17 | Flagship experiment orchestrator (full pipeline at real scale) |

Phases 1-6 are one-time infrastructure; phases 7 onward are what actually
runs repeatedly as new candidates are discovered.

## Core design principles

**Single-look discipline.** Once a candidate's OOS data has been evaluated,
it is never re-evaluated. Every phase checks the trial ledger before
touching OOS data and reuses the logged result if the candidate is already
there. This is enforced by an in-process access guard (see
`validation/oos.py`) plus the ledger check pattern used consistently
throughout.

**Append-only, immutable ledger.** `data/trial_log.jsonl` never has a line
edited or deleted. A candidate later found to be invalid (e.g. redundant)
gets a separate status-change event in `data/trial_status_log.jsonl` —
the original measurement stays on record, but downstream tools resolve an
"effective status" from the latest event.

**Canonical candidate identity.** Every candidate_id is built via a single
shared function, `build_candidate_id()` (`src/alpha/strategies/identity.py`)
— never hand-assembled inline. This guarantees the same expression
evaluated under the same parameters always maps to the same identity
across every phase, which is what makes the single-look ledger check
and duplicate/redundancy detection actually reliable.

**Honest reporting over flattering reporting.** Phase 16's report and
Phase 10's DSR verdicts state what the data shows, including negative or
empty results, rather than being written to make the project look further
along than it is.

**Real network fetch with graceful, non-silent degradation.** Live data
comes from yfinance; if it's unavailable, the system falls back to a
clearly-labeled synthetic panel rather than pretending to have real data
— and individually-unavailable tickers (delisted, renamed) are dropped
from the universe rather than aborting the entire panel to synthetic data.

## Running it

```bash
pip install -r requirements.txt
pytest                                    # full test suite
python experiments/walk_forward_validation_demo.py   # Phase 7
python experiments/cost_sensitivity_demo.py           # Phase 8
python experiments/parameter_robustness_demo.py       # Phase 9
python experiments/statistical_forensics_demo.py      # Phase 10
python experiments/factor_exposure_demo.py            # Phase 11
python experiments/regime_analysis_demo.py            # Phase 12
python experiments/redundancy_analysis_demo.py        # Phase 13
python experiments/registry_build_demo.py             # Phase 14
python experiments/portfolio_construction_demo.py     # Phase 15
python experiments/reporting_demo.py                  # Phase 16
```

Or run everything at once, at real scale, in isolated storage that never
touches the paths above:

```bash
python experiments/flagship_experiment_demo.py        # Phase 17
```

This can take a while on a large universe/date range — it's a real,
sequential re-execution of phases 7 through 16, not a quick demo.

## Known limitations (read before trusting a verdict)

- DSR does not protect against regime shifts, data-snooping in feature
  *definition* (as opposed to parameter search), or cost mismodeling.
- Regime robustness (Phase 12) is only ever evaluated across whichever
  regimes occurred within a candidate's specific OOS holdout window — a
  candidate's OOS window may simply never touch some regime types, which
  is reported explicitly rather than silently ignored.
- Redundancy detection (Phase 13) is scoped to top-level, full-OOS-window
  candidates only — sector/period/regime/cost sub-slices are excluded
  from correlation comparison since they're conditional sub-populations,
  not independent alternative strategies.
- Portfolio construction (Phase 15) uses simple equal-weight and
  inverse-volatility schemes only — no mean-variance/max-Sharpe
  optimization yet.
- The alpha registry (Phase 14) is a rebuildable derived index, never a
  source of truth — the JSONL/parquet ledgers are authoritative.

## Alpha discovery

Candidates can originate from established strategy families (Phase 4) or
from explicit economic hypotheses expressed through composable alpha
primitives (Phase 5), which the discovery layer (Phase 6) expands into
testable variants via parameter grids and compositional generation. The
validation pipeline then determines whether any apparent edge survives
true out-of-sample testing, transaction costs, statistical multiple-testing
correction (DSR), factor exposure analysis, regime robustness checks, and
redundancy deduplication against everything already discovered.

The current strategy library ships five baseline families — cross-sectional
momentum, time-series momentum, mean reversion, volatility-adjusted
momentum, low volatility (`src/alpha/strategies/library.py`) — as the seed
set the generation layer expands from. See "Current honest state" above
for the latest numbers — zero survivors so far is an honest intermediate
result, not a failure state: the system is designed to distinguish
signals that appear promising from those that survive increasingly strict
validation.