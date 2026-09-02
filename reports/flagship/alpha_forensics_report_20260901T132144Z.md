# Alpha Forensics Report

**Generated:** 20260901T132144Z UTC

---

## 1. Executive Summary

The project has evaluated 36 total candidate alpha strategies across all phases, of which 7 are currently valid, non-redundant, top-level candidates. Of the 7 candidates with stored returns, 1 explicitly fail DSR and 6 have not been DSR-evaluated. Under rigorous portfolio eligibility criteria (must survive DSR), the current portfolio is empty — no candidate meets all criteria. This is the honest, correct result given present alpha quality, not a reporting gap.

---

## 2. Candidate Eligibility Funnel

| Stage | Filter | Count |
| --- | --- | --- |
| Stage 0 | Total logged candidates (all phases) | 36 |
| Stage 0b | Currently VALID (any scope) | 34 |
| Stage 1 | Valid + non-redundant + top-level only | 7 |
| Stage 2 | + Has stored daily net returns | 7 |
| Stage 3a | + Explicitly fails DSR | 1 |
| Stage 3b | + Not yet DSR-evaluated (verdict NULL) | 6 |
| Stage 3c | + **Survives DSR** (eligible for rigorous portfolio) | **0** |

> Stage 3a and 3b candidates are excluded from the rigorous portfolio. Stage 3b reflects candidates with stored returns that have not yet been explicitly run through Phase 10 DSR evaluation — not that they passed.

---

## 3. Per-Candidate Summary (Top-Level Candidates, by OOS Sharpe)

| Candidate ID | Phase | OOS Sharpe | Status | DSR Verdict | Redundant With |
| --- | --- | --- | --- | --- | --- |
| cross_sectional_momentum__param_lookback_126 | Phase 9 Parameter Landscape | +0.0000 | VALID | NULL (not evaluated) | — |
| volatility_adjusted_momentum__param_lookback_126_window_20 | Phase 9 Parameter Landscape | +0.0000 | VALID | NULL (not evaluated) | — |
| volatility_adjusted_momentum__param_lookback_126_window_60 | Phase 9 Parameter Landscape | +0.0000 | VALID | NULL (not evaluated) | — |
| low_volatility | Phase 7 Walk-Forward Validation | -0.1167 | VALID | NULL (not evaluated) | — |
| volatility_adjusted_momentum | Phase 9 Parameter Landscape | -0.7090 | VALID | fails_dsr | — |
| cross_sectional_momentum__param_lookback_20 | Phase 9 Parameter Landscape | -0.7285 | VALID | NULL (not evaluated) | — |
| volatility_adjusted_momentum__param_lookback_60_window_60 | Phase 9 Parameter Landscape | -0.9952 | INVALIDATED | NULL (not evaluated) | volatility_adjusted_momentum |
| cross_sectional_momentum | Phase 7 Walk-Forward Validation | -1.1907 | INVALIDATED | fails_dsr | volatility_adjusted_momentum |
| volume_momentum_combo | Phase 7 Walk-Forward Validation | -8.5543 | VALID | NULL (not evaluated) | — |

---

## 4. Portfolio Construction Results

### 4a. Rigorous Mode (`require_dsr_survival=True`)

**Eligibility Funnel:**

- Stage 1 (Valid, non-redundant, top-level): 7 candidates
- Stage 2 (+ has stored returns): 7 candidates
- Stage 3 (DSR filter): 0 candidates
- **Final eligible constituents: 0 candidates**


**Result:** Stage 3 filter (dsr_verdict == 'survives_dsr') yielded 0 eligible candidates. Empty portfolio returned.

### 4b. Relaxed Mode (`require_dsr_survival=False`)

**Eligibility Funnel:**

- Stage 1 (Valid, non-redundant, top-level): 7 candidates
- Stage 2 (+ has stored returns): 7 candidates
- Stage 3 (DSR filter): 7 candidates
- **Final eligible constituents: 7 candidates**

| Candidate ID | Equal Weight | Inverse-Vol Weight |
| --- | --- | --- |
| cross_sectional_momentum__param_lookback_126 | 0.1429 | 0.3333 |
| volatility_adjusted_momentum__param_lookback_126_window_20 | 0.1429 | 0.3333 |
| volatility_adjusted_momentum__param_lookback_126_window_60 | 0.1429 | 0.3333 |
| low_volatility | 0.1429 | 0.0000 |
| volatility_adjusted_momentum | 0.1429 | 0.0000 |
| cross_sectional_momentum__param_lookback_20 | 0.1429 | 0.0000 |
| volume_momentum_combo | 0.1429 | 0.0000 |


**Equal-Weight Portfolio OOS Sharpe:** -4.5290

**Inverse-Volatility Portfolio OOS Sharpe:** +0.0000


---

## 5. Consolidated Caveats

**DSR Limitations** (from `src/statistics/dsr.py`):

DSR corrects for selection bias and data mining bias resulting from evaluating multiple
candidate alpha strategies (N trials) before selecting the top-performing strategy. It
computes the probability that the observed Sharpe ratio exceeds the expected maximum
Sharpe ratio under the null hypothesis that all trials are uninformative.

DSR specifically addresses selection bias across multiple trial evaluations. It does
**NOT** protect against:
- Data snooping or lookahead bias embedded directly into feature engineering or universe definition.
- Structural non-stationarity or macroeconomic regime shifts.
- Mismodeling of execution friction, market impact, or transaction costs.
- Overfitting within parameter landscapes when parameter tuning is unconstrained.

**Regime Coverage Limitation** (from `src/regimes/regime_analysis.py`):

Because the OOS holdout window sits chronologically at the end of the historical dataset
by construction, regime robustness testing evaluates candidate performance ONLY across
whichever market regimes actually occurred within that specific tail window. Regimes not
present in the OOS tail window (or with fewer than `min_regime_days` observations) are
explicitly recorded in `skipped_regimes` with explanatory reasons rather than silently
dropped, preventing callers from mistaking partial regime coverage for full-spectrum
evaluation across all 4 market regimes.

**Redundancy Analysis Scope Restriction** (from `src/redundancy/correlation_analysis.py`):

Redundancy analysis is strictly restricted to top-level candidates evaluated over the full
OOS window. Sub-slice trials containing `__sector_`, `__period_`, `__regime_`, or `__cost_`
in their candidate_id are filtered out because comparing a sub-slice to a full-window
candidate mixes different underlying populations and date ranges.

Candidates logged to `trial_log.jsonl` before Phase 13 was deployed do not have a stored
returns series in `trial_returns.parquet`. They are reported with an explicit reason
("no stored returns series") rather than silently dropped or crashed on.

