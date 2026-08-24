"""
Abstract expression tree representation for alpha signals.
"""

from abc import ABC, abstractmethod
import pandas as pd

from core.panel import Panel
from core.signal import Signal, validate_signal
from alpha.primitives.inputs import PrimitiveInput
from alpha.primitives.operators import Operator
from .dedup import canonical_hash


class Expression(ABC):
    """Abstract Base Class for alpha expressions."""

    @abstractmethod
    def _evaluate_raw(self, panel: Panel) -> pd.DataFrame:
        """Internal recursive evaluation returning raw DataFrame."""
        pass

    def evaluate(self, panel: Panel) -> Signal:
        """
        Evaluates the expression tree on a Panel and validates the root output
        against core.validate_signal.

        Returns:
            Signal: Validated signal DataFrame (date x ticker).
        """
        raw_signal = self._evaluate_raw(panel)
        validate_signal(raw_signal, panel)
        return raw_signal

    @abstractmethod
    def depth(self) -> int:
        """Returns maximum tree depth (Leaf depth = 1)."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Returns human-readable, parseable expression string."""
        pass

    def canonical_hash(self) -> str:
        """Returns canonical structural hash of the expression tree."""
        return canonical_hash(self)

    def __repr__(self) -> str:
        return self.to_string()

    def __str__(self) -> str:
        return self.to_string()


class Leaf(Expression):
    """Leaf node wrapping a primitive input."""

    def __init__(self, primitive: PrimitiveInput):
        self.primitive = primitive

    def _evaluate_raw(self, panel: Panel) -> pd.DataFrame:
        return self.primitive.evaluate(panel)

    def depth(self) -> int:
        return 1

    def to_string(self) -> str:
        return self.primitive.to_string()


class UnaryNode(Expression):
    """Unary node wrapping an operator and a child expression."""

    def __init__(self, operator: Operator, child: Expression):
        self.operator = operator
        self.child = child

    def _evaluate_raw(self, panel: Panel) -> pd.DataFrame:
        child_df = self.child._evaluate_raw(panel)
        return self.operator.apply(child_df)

    def depth(self) -> int:
        return 1 + self.child.depth()

    def to_string(self) -> str:
        from alpha.primitives.operators import Lag, Diff, SignedPower
        if isinstance(self.operator, Lag):
            return f"Lag({self.child.to_string()}, {self.operator.periods})"
        elif isinstance(self.operator, Diff):
            return f"Diff({self.child.to_string()}, {self.operator.periods})"
        elif isinstance(self.operator, SignedPower):
            return f"SignedPower({self.child.to_string()}, {self.operator.power})"
        else:
            return f"{self.operator.to_string()}({self.child.to_string()})"


class BinaryNode(Expression):
    """Binary node wrapping an operator and two child expressions (left, right)."""

    def __init__(self, operator: Operator, left: Expression, right: Expression):
        self.operator = operator
        self.left = left
        self.right = right

    def _evaluate_raw(self, panel: Panel) -> pd.DataFrame:
        left_df = self.left._evaluate_raw(panel)
        right_df = self.right._evaluate_raw(panel)
        return self.operator.apply(left_df, right_df)

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def to_string(self) -> str:
        return f"{self.operator.to_string()}({self.left.to_string()}, {self.right.to_string()})"
