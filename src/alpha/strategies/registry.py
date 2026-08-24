"""
In-memory strategy registry catalog.
"""

from typing import List
from .library import StrategyDefinition, STRATEGY_CATALOG


def get_all_strategies() -> List[StrategyDefinition]:
    """
    Returns a list of all registered strategy definitions in the catalog.
    """
    return list(STRATEGY_CATALOG.values())


def get_strategy(name: str) -> StrategyDefinition:
    """
    Retrieves a strategy definition by name.

    Args:
        name: Name of the strategy.

    Returns:
        StrategyDefinition: Target strategy definition.

    Raises:
        KeyError: If strategy name is not found in the registry catalog.
    """
    if name not in STRATEGY_CATALOG:
        available = sorted(list(STRATEGY_CATALOG.keys()))
        raise KeyError(
            f"Strategy '{name}' not found in registry catalog. Available strategies: {available}"
        )
    return STRATEGY_CATALOG[name]
