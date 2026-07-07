"""
Market data provider.

Responsibilities:
- Download OHLCV market data
- Normalize columns
- Return MarketData records
- No indicator calculations
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf

from core.schemas import MarketData
from core.exceptions import DataError
from core.logger import get_logger

logger = get_logger(__name__)


class MarketDataProvider:
    """Fetch and normalize market data."""

    REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    def fetch(
        self,
        symbol: str,
        interval: str = "1d",
        period: str = "1y",
    ) -> pd.DataFrame:
        """Return normalized OHLCV dataframe."""
        try:
            df = yf.download(
                tickers=symbol,
                interval=interval,
                period=period,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise DataError(f"Failed to download data for {symbol}") from exc

        if df.empty:
            raise DataError(f"No data returned for {symbol}")

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataError(f"Missing columns: {missing}")

        df = df.reset_index()
        df = df.rename(
            columns={
                "Date": "timestamp",
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        df["symbol"] = symbol
        df["timeframe"] = interval

        return df[
            [
                "timestamp",
                "symbol",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].copy()

    def to_schema(self, dataframe: pd.DataFrame) -> list[MarketData]:
        """Convert dataframe into MarketData schema objects."""
        records: list[MarketData] = []

        for row in dataframe.to_dict("records"):
            records.append(
                MarketData(
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    timestamp=pd.to_datetime(row["timestamp"]).to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )

        return records
