"""
Catalog of 5 economically-motivated alpha strategy definitions.
All strategies are constructed strictly using Phase 3 Expression trees.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Any

from alpha.expressions.tree import Expression, Leaf, UnaryNode, BinaryNode
import alpha.primitives.inputs as in_prim
import alpha.primitives.operators as op_prim


@dataclass
class StrategyDefinition:
    name: str
    family: str
    hypothesis_text: str
    build: Callable[[Dict[str, Any]], Expression]
    default_params: Dict[str, Any]
    variant_params: Dict[str, List[Any]]


def _build_cross_sectional_momentum(params: Dict[str, Any]) -> Expression:
    lookback = params.get("lookback", 60)
    return UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(lookback=lookback)))


def _build_short_term_reversal(params: Dict[str, Any]) -> Expression:
    lookback = params.get("lookback", 5)
    return UnaryNode(
        op_prim.Rank(),
        UnaryNode(op_prim.Negate(), Leaf(in_prim.Momentum(lookback=lookback))),
    )


def _build_volatility_adjusted_momentum(params: Dict[str, Any]) -> Expression:
    lookback = params.get("lookback", 60)
    window = params.get("window", 20)
    return UnaryNode(
        op_prim.Rank(),
        BinaryNode(
            op_prim.Divide(),
            Leaf(in_prim.Momentum(lookback=lookback)),
            Leaf(in_prim.RealizedVolatility(window=window)),
        ),
    )


def _build_volume_confirmed_momentum(params: Dict[str, Any]) -> Expression:
    lookback = params.get("lookback", 60)
    return BinaryNode(
        op_prim.Multiply(),
        UnaryNode(op_prim.Rank(), Leaf(in_prim.Momentum(lookback=lookback))),
        UnaryNode(op_prim.Rank(), Leaf(in_prim.VolumeChange(lookback=lookback))),
    )


def _build_low_volatility(params: Dict[str, Any]) -> Expression:
    window = params.get("window", 60)
    return UnaryNode(
        op_prim.Rank(),
        UnaryNode(op_prim.Negate(), Leaf(in_prim.RealizedVolatility(window=window))),
    )


# Catalog dictionary of the 5 canonical strategy definitions
STRATEGY_CATALOG: Dict[str, StrategyDefinition] = {
    "cross_sectional_momentum": StrategyDefinition(
        name="cross_sectional_momentum",
        family="momentum",
        hypothesis_text="Recent relative winners continue outperforming over the next short horizon.",
        build=_build_cross_sectional_momentum,
        default_params={"lookback": 60},
        variant_params={"lookback": [20, 60, 126]},
    ),
    "short_term_reversal": StrategyDefinition(
        name="short_term_reversal",
        family="reversal",
        hypothesis_text="Short-horizon extreme price moves partially revert due to overreaction.",
        build=_build_short_term_reversal,
        default_params={"lookback": 5},
        variant_params={"lookback": [5, 10, 20]},
    ),
    "volatility_adjusted_momentum": StrategyDefinition(
        name="volatility_adjusted_momentum",
        family="momentum",
        hypothesis_text="Momentum risk-adjusted by recent volatility is a cleaner signal than raw momentum.",
        build=_build_volatility_adjusted_momentum,
        default_params={"lookback": 60, "window": 20},
        variant_params={"lookback": [60, 126], "window": [20, 60]},
    ),
    "volume_confirmed_momentum": StrategyDefinition(
        name="volume_confirmed_momentum",
        family="momentum",
        hypothesis_text="Momentum is more reliable when accompanied by rising volume and institutional participation.",
        build=_build_volume_confirmed_momentum,
        default_params={"lookback": 60},
        variant_params={"lookback": [20, 60, 126]},
    ),
    "low_volatility": StrategyDefinition(
        name="low_volatility",
        family="volatility",
        hypothesis_text="Lower-volatility assets earn better risk-adjusted returns than the cross-section average (the low-vol anomaly).",
        build=_build_low_volatility,
        default_params={"window": 60},
        variant_params={"window": [20, 60, 126]},
    ),
}
