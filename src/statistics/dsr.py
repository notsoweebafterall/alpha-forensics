"""
Deflated Sharpe Ratio (DSR) Implementation (Bailey & López de Prado, 2014).

What DSR Corrects For:
    DSR corrects for selection bias and data mining bias resulting from evaluating multiple candidate
    alpha strategies (N trials) before selecting the top-performing strategy. When many trials are tested,
    the maximum observed Sharpe ratio among them is inflated due to pure random variation. DSR computes
    the probability that the observed Sharpe ratio exceeds the expected maximum Sharpe ratio under the
    null hypothesis that all trials are uninformative.

Theoretical Assumptions:
    1. Returns are approximately normal after adjusting for skewness (gamma_3) and kurtosis (gamma_4).
    2. Trials are assumed moderately independent (or conservatively counted across all tested candidates).

Required Inputs:
    - observed_sharpe (float): The observed Sharpe ratio of the candidate strategy.
    - sharpe_variance (float): Variance of Sharpe ratios across all N evaluated trials V[SR_hat].
    - n_trials (int): Total number of distinct candidate trials evaluated (N).
    - skew (float): Return skewness of the evaluated candidate (gamma_3). Default 0.0.
    - kurtosis (float): Return kurtosis (non-excess kurtosis gamma_4, where normal distribution = 3.0). Default 3.0.
    - track_record_length (int): Number of return observations T (e.g., number of trading days). Default 252.

What DSR Does NOT Protect Against:
    DSR specifically addresses selection bias across multiple trial evaluations. It does NOT protect against:
    - Data snooping or lookahead bias embedded directly into feature engineering or universe definition (Phase 3).
    - Structural non-stationarity or macroeconomic regime shifts (Phase 11/12).
    - Mismodeling of execution friction, market impact, or transaction costs (Phase 8).
    - Overfitting within parameter landscapes when parameter tuning is unconstrained (Phase 9).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Union
import scipy.stats as stats

# Euler-Mascheroni constant
EULER_MASCHERONI = 0.57721566490153286060


def expected_max_sharpe_under_trials(sharpe_variance: float, n_trials: int) -> float:
    """
    Computes the expected maximum Sharpe ratio SR_0 under N independent standard normal trials.

    Formula (Bailey & López de Prado, 2014):
        SR_0 = sqrt(V[SR_hat]) * [ (1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N*e)) ]
    where gamma is the Euler-Mascheroni constant (~0.57721566) and Phi^{-1} is the probit function.

    Args:
        sharpe_variance: Variance of Sharpe ratios across all N trials (V[SR_hat]).
        n_trials: Total number of candidate trials evaluated (N).

    Returns:
        float: Expected maximum Sharpe ratio under N trials (SR_0). Returns 0.0 if N <= 1 or variance <= 0.
    """
    if n_trials <= 1 or sharpe_variance <= 0.0:
        return 0.0

    sigma_sr = math.sqrt(sharpe_variance)
    gamma = EULER_MASCHERONI
    e = math.e

    # Quantile terms Phi^{-1}(1 - 1/N) and Phi^{-1}(1 - 1/(N*e))
    q1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    q2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * e)))

    sr0 = sigma_sr * ((1.0 - gamma) * q1 + gamma * q2)
    return max(0.0, float(sr0))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_variance: float,
    n_trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    track_record_length: int = 252,
) -> float:
    """
    Computes the Deflated Sharpe Ratio (DSR) as the Probabilistic Sharpe Ratio against benchmark SR_0.

    Formula:
        DSR = PSR(SR_0) = Phi( (SR_hat - SR_0) * sqrt(T - 1) / sqrt(1 - gamma_3*SR_hat + (gamma_4 - 1)/4 * SR_hat^2) )

    Args:
        observed_sharpe: Observed Sharpe ratio of candidate (SR_hat).
        sharpe_variance: Variance of Sharpes across all N trials.
        n_trials: Total distinct trials evaluated (N).
        skew: Return skewness (gamma_3). Default 0.0.
        kurtosis: Return non-excess kurtosis (gamma_4, normal = 3.0). Default 3.0.
        track_record_length: Number of return observations T. Default 252.

    Returns:
        float: Deflated Sharpe Ratio (probability between 0.0 and 1.0).
    """
    if track_record_length <= 1:
        raise ValueError(f"track_record_length T must be > 1, got {track_record_length}")

    # Ensure non-excess kurtosis representation (if excess kurtosis normal=0 was passed, convert to non-excess)
    gamma_4 = kurtosis + 3.0 if kurtosis < 1.0 else kurtosis
    gamma_3 = skew

    sr_0 = expected_max_sharpe_under_trials(sharpe_variance, n_trials)

    # Standard error denominator under Mertens/Opdyke formulation
    denom_sq = 1.0 - gamma_3 * observed_sharpe + ((gamma_4 - 1.0) / 4.0) * (observed_sharpe ** 2)
    if denom_sq <= 0.0:
        denom_sq = 1e-8

    se = math.sqrt(denom_sq / (track_record_length - 1))
    z = (observed_sharpe - sr_0) / se

    return float(stats.norm.cdf(z))


@dataclass
class DSRResult:
    candidate_id: str
    observed_sharpe: float
    n_trials: int
    expected_max_sharpe: float
    dsr: float
    verdict: str  # "survives_dsr" if dsr >= 0.95 else "fails_dsr"
    caveats: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        candidate_id: str,
        observed_sharpe: float,
        sharpe_variance: float,
        n_trials: int,
        skew: float = 0.0,
        kurtosis: float = 3.0,
        track_record_length: int = 252,
        extra_caveats: Optional[List[str]] = None,
    ) -> "DSRResult":
        """Factory method to compute DSR and build a populated DSRResult object."""
        sr_0 = expected_max_sharpe_under_trials(sharpe_variance, n_trials)
        dsr_val = deflated_sharpe_ratio(
            observed_sharpe=observed_sharpe,
            sharpe_variance=sharpe_variance,
            n_trials=n_trials,
            skew=skew,
            kurtosis=kurtosis,
            track_record_length=track_record_length,
        )

        verdict = "survives_dsr" if dsr_val >= 0.95 else "fails_dsr"

        caveats: List[str] = [
            f"Trial count N={n_trials} conservatively includes all historical distinct candidate evaluations.",
            "DSR assumes return series approximate normality adjusted for skewness and kurtosis.",
            "DSR does NOT protect against regime shifts, data snooping in feature definitions, or cost mismodeling.",
        ]
        if extra_caveats:
            caveats.extend(extra_caveats)

        return cls(
            candidate_id=candidate_id,
            observed_sharpe=observed_sharpe,
            n_trials=n_trials,
            expected_max_sharpe=sr_0,
            dsr=dsr_val,
            verdict=verdict,
            caveats=caveats,
        )


@dataclass
class DSRLogRecord:
    candidate_id: str
    dsr_score: float
    verdict: str
    n_trials: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DSRLogRecord":
        return cls(
            candidate_id=str(data["candidate_id"]),
            dsr_score=float(data["dsr_score"]),
            verdict=str(data["verdict"]),
            n_trials=int(data["n_trials"]),
            timestamp=str(data["timestamp"]),
        )


def log_dsr_result(
    dsr_res: DSRResult,
    log_path: Union[str, Path] = "data/dsr_results.jsonl",
) -> None:
    """
    Appends a DSR verdict record to the persistent dsr_results.jsonl log.

    Append-only discipline:
        A candidate's DSR verdict can change over time as total trial count N grows.
        This writes a chronological event log of DSR verdicts over time. Rebuilding
        the registry uses the most recent entry per candidate_id.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = DSRLogRecord(
        candidate_id=dsr_res.candidate_id,
        dsr_score=float(dsr_res.dsr),
        verdict=str(dsr_res.verdict),
        n_trials=int(dsr_res.n_trials),
        timestamp=datetime.now().isoformat(),
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_dict()) + "\n")


def load_dsr_log(
    log_path: Union[str, Path] = "data/dsr_results.jsonl",
) -> List[DSRLogRecord]:
    """Loads all DSRLogRecord entries from the jsonl file."""
    path = Path(log_path)
    if not path.exists():
        return []
    records: List[DSRLogRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(DSRLogRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return records


def load_latest_dsr_verdicts(
    log_path: Union[str, Path] = "data/dsr_results.jsonl",
) -> Dict[str, DSRLogRecord]:
    """Loads a mapping of candidate_id -> most recent DSRLogRecord."""
    records = load_dsr_log(log_path=log_path)
    latest: Dict[str, DSRLogRecord] = {}
    for r in records:
        latest[r.candidate_id] = r
    return latest
