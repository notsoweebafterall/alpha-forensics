from dataclasses import dataclass
import pandas as pd


@dataclass
class Panel:
    prices: pd.DataFrame   # index=date, columns=ticker, adjusted close
    volume: pd.DataFrame   # index=date, columns=ticker
    returns: pd.DataFrame  # index=date, columns=ticker, computed from prices
    universe: list[str]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
