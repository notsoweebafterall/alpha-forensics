from dataclasses import dataclass
import pandas as pd


@dataclass
class ExperimentMetadata:
    experiment_id: str
    source: str              # "strategy" | "hypothesis" | "generated"
    hypothesis_text: str | None
    expression: str
    parent_family: str | None
    generation_method: str
    universe: list[str]
    date_range: tuple[pd.Timestamp, pd.Timestamp]
    config_snapshot: dict
    created_at: pd.Timestamp
