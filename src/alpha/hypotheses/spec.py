"""
Hypothesis specification data structure for structured research hypotheses.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HypothesisSpec:
    hypothesis_id: str
    hypothesis_text: str
    economic_rationale: str
    composition: str
    source_features: List[str]
    parent_family: Optional[str] = None
    author_notes: Optional[str] = None

    def __post_init__(self):
        if not self.hypothesis_id or not isinstance(self.hypothesis_id, str):
            raise ValueError("hypothesis_id must be a non-empty string.")

        if not self.hypothesis_text or not self.hypothesis_text.strip():
            raise ValueError("hypothesis_text cannot be empty.")

        if not self.economic_rationale or not self.economic_rationale.strip():
            raise ValueError("economic_rationale cannot be empty.")

        if not self.composition or not self.composition.strip():
            raise ValueError("composition cannot be empty.")

        if not isinstance(self.source_features, list):
            raise ValueError("source_features must be a list of feature/operator names.")
