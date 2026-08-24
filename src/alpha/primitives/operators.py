"""
Unary and binary operator primitives for expression tree evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import pandas as pd


class Operator(ABC):
    """Abstract base class for all tree operators."""

    @abstractmethod
    def apply(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        """Applies operator to input DataFrame(s)."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """String representation of operator name."""
        pass

    @abstractmethod
    def params_dict(self) -> Dict[str, Any]:
        """Operator parameter dictionary."""
        pass

    @property
    def is_commutative(self) -> bool:
        """Whether operator is commutative (for canonical hashing)."""
        return False

    def __repr__(self) -> str:
        return self.to_string()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.params_dict() == other.params_dict()


# -------------------------------------------------------------------
# Unary Operators
# -------------------------------------------------------------------

class Rank(Operator):
    """Cross-sectional percentile rank across tickers."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)

    def to_string(self) -> str:
        return "Rank"

    def params_dict(self) -> Dict[str, Any]:
        return {}


class ZScore(Operator):
    """Cross-sectional z-score across tickers."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        # Avoid division by zero
        std_clean = std.replace(0.0, np.nan)
        z = df.sub(mean, axis=0).div(std_clean, axis=0)
        return z

    def to_string(self) -> str:
        return "ZScore"

    def params_dict(self) -> Dict[str, Any]:
        return {}


class Lag(Operator):
    """Time-series lag operator."""

    def __init__(self, periods: int = 1):
        self.periods = periods

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.shift(self.periods)

    def to_string(self) -> str:
        return f"Lag({self.periods})"

    def params_dict(self) -> Dict[str, Any]:
        return {"periods": self.periods}


class Diff(Operator):
    """Time-series difference operator."""

    def __init__(self, periods: int = 1):
        self.periods = periods

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.diff(self.periods)

    def to_string(self) -> str:
        return f"Diff({self.periods})"

    def params_dict(self) -> Dict[str, Any]:
        return {"periods": self.periods}


class Negate(Operator):
    """Arithmetic negation operator (-x)."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return -df

    def to_string(self) -> str:
        return "Negate"

    def params_dict(self) -> Dict[str, Any]:
        return {}


class Abs(Operator):
    """Absolute value operator (|x|)."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.abs()

    def to_string(self) -> str:
        return "Abs"

    def params_dict(self) -> Dict[str, Any]:
        return {}


class SignedPower(Operator):
    """Signed power operator: sign(x) * |x|^power."""

    def __init__(self, power: float):
        self.power = power

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return np.sign(df) * (df.abs() ** self.power)

    def to_string(self) -> str:
        return f"SignedPower({self.power})"

    def params_dict(self) -> Dict[str, Any]:
        return {"power": self.power}


# -------------------------------------------------------------------
# Binary Operators
# -------------------------------------------------------------------

class Add(Operator):
    """Addition operator (left + right). Commutative."""

    def apply(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return left + right

    def to_string(self) -> str:
        return "Add"

    def params_dict(self) -> Dict[str, Any]:
        return {}

    @property
    def is_commutative(self) -> bool:
        return True


class Subtract(Operator):
    """Subtraction operator (left - right). Non-commutative."""

    def apply(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return left - right

    def to_string(self) -> str:
        return "Subtract"

    def params_dict(self) -> Dict[str, Any]:
        return {}


class Multiply(Operator):
    """Multiplication operator (left * right). Commutative."""

    def apply(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return left * right

    def to_string(self) -> str:
        return "Multiply"

    def params_dict(self) -> Dict[str, Any]:
        return {}

    @property
    def is_commutative(self) -> bool:
        return True


class Divide(Operator):
    """
    Safe division operator (left / right). Non-commutative.

    Replaces denominators with absolute value < eps with NaN to guarantee
    no inf / -inf values are produced.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def apply(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        safe_right = right.mask(right.abs() < self.eps, np.nan)
        out = left / safe_right
        # Replace any residual inf/-inf with NaN
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def to_string(self) -> str:
        return "Divide"

    def params_dict(self) -> Dict[str, Any]:
        return {"eps": self.eps}
