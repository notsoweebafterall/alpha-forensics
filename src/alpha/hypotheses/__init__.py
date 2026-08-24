from .spec import HypothesisSpec
from .translator import translate
from .registry import get_all_hypotheses, get_hypothesis

__all__ = [
    "HypothesisSpec",
    "translate",
    "get_all_hypotheses",
    "get_hypothesis",
]
