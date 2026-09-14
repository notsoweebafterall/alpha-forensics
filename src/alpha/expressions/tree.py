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


# ---------------------------------------------------------------------------
# Constant bounds — design decision documented here because it affects
# both ConstantNode construction and validate_expression_vocabulary.
#
# Allowed constant range: 0.001 <= |value| <= 1000.0, and value != 0.
#
# Rationale:
#   * Lower bound (0.001): prevents near-zero denominators in Divide() that
#     would make the safe-eps guard in Divide.apply() meaningless — a constant
#     of 1e-8 passes Divide's eps check only because it equals eps, producing
#     NaN for every cell.  We require constants to have genuine financial scale.
#   * Upper bound (1000.0): no financial ratio built from daily returns/vol/rank
#     signals needs a scalar larger than 1000; values beyond this are almost
#     certainly LLM/mutation accidents, not intentional signal design.
#   * Zero explicitly forbidden: Divide(x, Constant(0)) is always NaN.
# ---------------------------------------------------------------------------

CONSTANT_MIN_ABS: float = 0.001
CONSTANT_MAX_ABS: float = 1000.0


def validate_constant_value(value: float) -> list[str]:
    """Returns a list of violation messages for an out-of-bounds constant value."""
    violations: list[str] = []
    if value == 0.0:
        violations.append("Constant value 0.0 is not allowed (always produces NaN in Divide).")
    elif abs(value) < CONSTANT_MIN_ABS:
        violations.append(
            f"Constant value {value} has |value| < {CONSTANT_MIN_ABS} — "
            "near-zero constants act as effective divide-by-zero denominators."
        )
    elif abs(value) > CONSTANT_MAX_ABS:
        violations.append(
            f"Constant value {value} has |value| > {CONSTANT_MAX_ABS} — "
            "unreasonably large scalar, likely a mutation/LLM generation accident."
        )
    return violations


class ConstantNode(Expression):
    """
    Constant scalar node — a leaf-equivalent node that broadcasts a fixed numeric
    value across the full panel date × ticker grid.

    Evaluation contract:
        Returns a DataFrame of shape (n_dates, n_tickers) filled uniformly with
        self.value.  This satisfies validate_signal's column/date-range checks
        while passing cleanly through binary operators like Divide and Multiply.

    Serialization:
        to_string() produces  Constant(value)  — round-trips through parse().

    Bounds:
        Values must satisfy 0.001 <= |value| <= 1000.0 and value != 0.
        See validate_constant_value() and CONSTANT_MIN_ABS / CONSTANT_MAX_ABS.
    """

    def __init__(self, value: float):
        violations = validate_constant_value(value)
        if violations:
            raise ValueError(
                f"ConstantNode received out-of-bounds value {value!r}: {'; '.join(violations)}"
            )
        self.value = value

    def _evaluate_raw(self, panel: Panel) -> pd.DataFrame:
        # Broadcast scalar across full panel grid so validate_signal sees the
        # correct index/columns shape.
        return pd.DataFrame(
            self.value,
            index=panel.prices.index,
            columns=panel.prices.columns,
        )

    def depth(self) -> int:
        return 1

    def to_string(self) -> str:
        # Use repr for the value so integers stay integers (1 → "1") and floats
        # keep their precision (0.5 → "0.5").
        return f"Constant({self.value!r})"
