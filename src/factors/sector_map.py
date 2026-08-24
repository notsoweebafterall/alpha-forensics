"""
Hardcoded GICS Sector mapping for all 60 tickers in UNIVERSE_60.

Provides static ticker-to-sector mapping used for sector return factor construction
and cross-sector sub-universe generalization testing.
"""

from typing import Dict
from data.universe import UNIVERSE_60

SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "AMD": "Technology",
    "CSCO": "Technology",
    "INTC": "Technology",
    "QCOM": "Technology",
    "IBM": "Technology",
    "ADBE": "Technology",
    # Communication Services
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "T": "Communication Services",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "MS": "Financials",
    "GS": "Financials",
    "C": "Financials",
    "BLK": "Financials",
    "SCHW": "Financials",
    "AXP": "Financials",
    "CB": "Financials",
    # Healthcare
    "LLY": "Healthcare",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "PFE": "Healthcare",
    "TMO": "Healthcare",
    "ABT": "Healthcare",
    "AMGN": "Healthcare",
    "DHR": "Healthcare",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "PEP": "Consumer Staples",
    "KO": "Consumer Staples",
    "PM": "Consumer Staples",
    # Industrials
    "GE": "Industrials",
    "CAT": "Industrials",
    "HON": "Industrials",
    "BA": "Industrials",
    "UNP": "Industrials",
    "LMT": "Industrials",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    # Utilities
    "NEE": "Utilities",
    # Real Estate
    "PLD": "Real Estate",
}

# Startup verification: Ensure every ticker in UNIVERSE_60 has a mapping entry
_missing = set(UNIVERSE_60) - set(SECTOR_MAP.keys())
if _missing:
    raise ValueError(f"Missing sector mappings for UNIVERSE_60 tickers: {sorted(_missing)}")
