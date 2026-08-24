"""
Leaf-node wrappers around Phase 1 features library primitives.
Each primitive is a thin reference to a function in src/features/.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

from core.panel import Panel
import features.returns as f_returns
import features.rolling as f_rolling
import features.volume as f_volume
import features.risk as f_risk


class PrimitiveInput(ABC):
    """Abstract base class for all input primitives."""

    @abstractmethod
    def evaluate(self, panel: Panel) -> pd.DataFrame:
        """Evaluates primitive on a Panel, returning a DataFrame (date x ticker)."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Returns string representation of primitive."""
        pass

    @abstractmethod
    def params_dict(self) -> Dict[str, Any]:
        """Returns parameter dictionary for hashing and serialization."""
        pass

    def __repr__(self) -> str:
        return self.to_string()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.params_dict() == other.params_dict()


class PriceReturn(PrimitiveInput):
    def __init__(self, lag: int = 1):
        self.lag = lag

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_returns.simple_return(panel, lag=self.lag)

    def to_string(self) -> str:
        return f"PriceReturn({self.lag})"

    def params_dict(self) -> Dict[str, Any]:
        return {"lag": self.lag}


class Momentum(PrimitiveInput):
    def __init__(self, lookback: int):
        self.lookback = lookback

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_returns.momentum(panel, lookback=self.lookback)

    def to_string(self) -> str:
        return f"Momentum({self.lookback})"

    def params_dict(self) -> Dict[str, Any]:
        return {"lookback": self.lookback}


class RollingMean(PrimitiveInput):
    def __init__(self, window: int):
        self.window = window

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_rolling.rolling_mean(panel, window=self.window)

    def to_string(self) -> str:
        return f"RollingMean({self.window})"

    def params_dict(self) -> Dict[str, Any]:
        return {"window": self.window}


class RollingStd(PrimitiveInput):
    def __init__(self, window: int):
        self.window = window

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_rolling.rolling_std(panel, window=self.window)

    def to_string(self) -> str:
        return f"RollingStd({self.window})"

    def params_dict(self) -> Dict[str, Any]:
        return {"window": self.window}


class RollingZScore(PrimitiveInput):
    def __init__(self, window: int):
        self.window = window

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_rolling.rolling_zscore(panel, window=self.window)

    def to_string(self) -> str:
        return f"RollingZScore({self.window})"

    def params_dict(self) -> Dict[str, Any]:
        return {"window": self.window}


class RollingRank(PrimitiveInput):
    def __init__(self, window: int):
        self.window = window

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_rolling.rolling_rank(panel, window=self.window)

    def to_string(self) -> str:
        return f"RollingRank({self.window})"

    def params_dict(self) -> Dict[str, Any]:
        return {"window": self.window}


class VolumeChange(PrimitiveInput):
    def __init__(self, lookback: int = 1):
        self.lookback = lookback

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_volume.volume_change(panel, lookback=self.lookback)

    def to_string(self) -> str:
        return f"VolumeChange({self.lookback})"

    def params_dict(self) -> Dict[str, Any]:
        return {"lookback": self.lookback}


class Turnover(PrimitiveInput):
    def __init__(self, lookback: int):
        self.lookback = lookback

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_volume.turnover(panel, lookback=self.lookback)

    def to_string(self) -> str:
        return f"Turnover({self.lookback})"

    def params_dict(self) -> Dict[str, Any]:
        return {"lookback": self.lookback}


class RealizedVolatility(PrimitiveInput):
    def __init__(self, window: int):
        self.window = window

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_risk.realized_volatility(panel, window=self.window)

    def to_string(self) -> str:
        return f"RealizedVolatility({self.window})"

    def params_dict(self) -> Dict[str, Any]:
        return {"window": self.window}


class Drawdown(PrimitiveInput):
    def __init__(self, window: int):
        self.window = window

    def evaluate(self, panel: Panel) -> pd.DataFrame:
        return f_risk.drawdown(panel, window=self.window)

    def to_string(self) -> str:
        return f"Drawdown({self.window})"

    def params_dict(self) -> Dict[str, Any]:
        return {"window": self.window}
