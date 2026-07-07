"""
Momentum Indicators.

Responsibilities:
- RSI
- MACD
- ROC
- Williams %R
- CCI
- Stochastic Oscillator

Adds momentum features to the incoming dataframe.
"""

from __future__ import annotations

import pandas as pd

from core.exceptions import IndicatorError
from core.logger import get_logger

logger = get_logger(__name__)


class MomentumIndicators:
    """Momentum indicator engine."""

    def calculate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            raise IndicatorError("Empty dataframe received.")

        required = {"high", "low", "close"}
        missing = required.difference(dataframe.columns)
        if missing:
            raise IndicatorError(f"Missing required columns: {sorted(missing)}")

        df = dataframe.copy()

        # RSI(14)
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD (12,26,9)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # ROC(12)
        df["roc_12"] = df["close"].pct_change(12) * 100

        # Williams %R(14)
        hh = df["high"].rolling(14, min_periods=14).max()
        ll = df["low"].rolling(14, min_periods=14).min()
        df["williams_r_14"] = -100 * ((hh - df["close"]) / (hh - ll))

        # CCI(20)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(20, min_periods=20).mean()
        mad = tp.rolling(20, min_periods=20).apply(
            lambda x: (x - x.mean()).abs().mean(),
            raw=False,
        )
        df["cci_20"] = (tp - sma_tp) / (0.015 * mad)

        # Stochastic (14,3)
        low14 = df["low"].rolling(14, min_periods=14).min()
        high14 = df["high"].rolling(14, min_periods=14).max()
        df["stoch_k"] = 100 * (df["close"] - low14) / (high14 - low14)
        df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=3).mean()

        logger.info("Momentum indicators calculated.")

        return df
