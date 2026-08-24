import numpy as np
import pandas as pd
from typing import TypeAlias

from .panel import Panel

Signal: TypeAlias = pd.DataFrame


def validate_signal(signal: pd.DataFrame, panel: Panel) -> None:
    """
    Validates that a signal DataFrame conforms to panel constraints.

    Checks:
    - Signal contains no inf or -inf values.
    - Signal column tickers are a subset of panel.universe.
    - Signal index dates lie within [panel.start_date, panel.end_date].

    NaN values are explicitly allowed to represent 'no position'.

    Raises:
        ValueError: If any validation check fails.
    """
    if not isinstance(signal, pd.DataFrame):
        raise ValueError("Signal must be a pandas DataFrame.")

    # Check inf values
    values = signal.to_numpy()
    if np.isinf(values).any():
        raise ValueError("Signal contains infinite (inf or -inf) values.")

    # Check out-of-universe tickers
    invalid_tickers = set(signal.columns) - set(panel.universe)
    if invalid_tickers:
        raise ValueError(
            f"Signal contains out-of-universe tickers: {sorted(list(invalid_tickers))}"
        )

    # Check out-of-range dates
    if not signal.empty:
        signal_start = signal.index.min()
        signal_end = signal.index.max()
        if signal_start < panel.start_date or signal_end > panel.end_date:
            raise ValueError(
                f"Signal dates [{signal_start}, {signal_end}] fall outside "
                f"panel date range [{panel.start_date}, {panel.end_date}]."
            )
