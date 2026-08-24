"""
Factors module for Alpha Forensics.

Provides self-constructed factor modeling (MKT, MOM, VOL, SECTOR),
factor exposure regression with HAC standard errors, and cross-sector / cross-period
generalization testing.
"""

from .sector_map import SECTOR_MAP
from .factor_construction import (
    FactorPanel,
    construct_market_factor,
    construct_momentum_factor,
    construct_volatility_factor,
    construct_sector_returns,
    build_factor_panel,
)
from .exposure import (
    FactorExposureResult,
    run_factor_regression,
)
from .generalization import (
    GeneralizationResult,
    evaluate_generalization,
)

__all__ = [
    "SECTOR_MAP",
    "FactorPanel",
    "construct_market_factor",
    "construct_momentum_factor",
    "construct_volatility_factor",
    "construct_sector_returns",
    "build_factor_panel",
    "FactorExposureResult",
    "run_factor_regression",
    "GeneralizationResult",
    "evaluate_generalization",
]
