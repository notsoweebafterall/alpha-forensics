"""
Data structure representing a systematically generated alpha candidate.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

from alpha.expressions.tree import Expression


@dataclass
class GeneratedCandidate:
    expression: Expression
    expression_string: str
    generation_method: str        # "variant_expansion" | "compositional_random"
    seed: Optional[int]
    parent_family: Optional[str]
    param_values: Dict[str, Any]
    generated_at: pd.Timestamp

    def canonical_hash(self) -> str:
        """Returns the canonical structural hash of the underlying expression."""
        return self.expression.canonical_hash()
