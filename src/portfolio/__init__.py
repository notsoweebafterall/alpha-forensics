"""
Portfolio Construction Package — cross-candidate capital allocation & meta-portfolio weighting.
"""

from .portfolio_construction import (
    PortfolioResult,
    build_portfolio,
)

__all__ = [
    "PortfolioResult",
    "build_portfolio",
]
