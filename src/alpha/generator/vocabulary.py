"""
Centralized, auditable vocabulary whitelist for systematic alpha candidate generation.
Imports real classes from alpha.primitives.
"""

from typing import Dict, List, Type, Any

from alpha.primitives.inputs import (
    PrimitiveInput,
    PriceReturn,
    Momentum,
    RollingMean,
    RollingStd,
    RollingZScore,
    RollingRank,
    VolumeChange,
    Turnover,
    RealizedVolatility,
    Drawdown,
)
from alpha.primitives.operators import (
    Operator,
    Rank,
    ZScore,
    Lag,
    Diff,
    Negate,
    Abs,
    SignedPower,
    Add,
    Subtract,
    Multiply,
    Divide,
)

# Whitelisted Leaf Primitive Classes and their parameter grids
LEAF_PRIMITIVES: Dict[Type[PrimitiveInput], Dict[str, List[Any]]] = {
    PriceReturn: {"lag": [1, 5, 10]},
    Momentum: {"lookback": [20, 60, 126, 252]},
    RollingMean: {"window": [10, 20, 60]},
    RollingStd: {"window": [10, 20, 60]},
    RollingZScore: {"window": [10, 20, 60]},
    RollingRank: {"window": [10, 20, 60]},
    VolumeChange: {"lookback": [1, 5, 20]},
    Turnover: {"lookback": [5, 20, 60]},
    RealizedVolatility: {"window": [20, 60]},
    Drawdown: {"window": [20, 60]},
}

# Whitelisted Unary Operator Classes and their parameter grids
UNARY_OPS: Dict[Type[Operator], Dict[str, List[Any]]] = {
    Rank: {},
    ZScore: {},
    Negate: {},
    Abs: {},
    Lag: {"periods": [1, 5]},
    Diff: {"periods": [1, 5]},
    SignedPower: {"power": [0.5, 2.0]},
}

# Whitelisted Binary Operator Classes
BINARY_OPS: List[Type[Operator]] = [Add, Subtract, Multiply, Divide]
