"""
SELL Strategy Engine

Institutional Version

Responsibilities
----------------

Trend Analysis
Momentum Analysis
Volume Analysis
Breakdown Analysis
Failed Breakout Analysis
Ichimoku Analysis
Market Filter
Sector Filter
Fundamental Filter
News Filter

Produces

SELL

or

NO_TRADE

No scoring.

No probability.

No position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.constants import SELL
from core.constants import NO_TRADE

from core.logger import get_logger
from core.exceptions import StrategyError

logger = get_logger(__name__)


# ==========================================================
# RESULT
# ==========================================================


@dataclass(slots=True)
class SellDecision:

    action: str

    passed: bool

    confidence: float

    reasons: list[str] = field(default_factory=list)

    technical_checks: dict[str, bool] = field(default_factory=dict)


# ==========================================================
# ENGINE
# ==========================================================


class SellStrategyEngine:
    """
    Complete SELL Strategy
    """

    REQUIRED_COLUMNS = {
        "close",
        "ema_20",
        "ema_50",
        "ema_200",
        "sma_20",
        "sma_50",
        "sma_200",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "adx_14",
        "supertrend",
        "cloud_trend",
        "vwap",
        "volume",
        "volume_sma_20",
        "obv",
        "cmf_20",
        "mfi_14",
        "roc_12",
        "cci_20",
        "stochastic_k",
        "stochastic_d",
        "williams_r_14",
        "market_regime",
        "volatility_state",
        "breadth",
        "atr_14",
        "bb_middle",
        "kc_middle",
        "dc_middle",
        "pivot",
        "gap_up",
        "gap_down",
        "gap_day",
        "is_breakdown",
        "is_pullback",
    }

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        fundamentals: dict[str, Any],
        news_score: float,
        market_score: float,
        sector_score: float,
    ) -> SellDecision:

        if dataframe.empty:
            raise StrategyError("Empty dataframe.")

        missing = self.REQUIRED_COLUMNS.difference(dataframe.columns)

        if missing:

            raise StrategyError(f"Missing columns: {sorted(missing)}")

        row = dataframe.iloc[-1]

        checks = {}

        reasons = []
        # ==========================================================
        # TREND ENGINE
        # ==========================================================

        # --------------------------------------------------
        # EMA TREND
        # --------------------------------------------------

        checks["ema_alignment"] = row["ema_20"] < row["ema_50"] < row["ema_200"]

        if checks["ema_alignment"]:

            reasons.append("Bearish EMA alignment.")

        # --------------------------------------------------
        # SMA TREND
        # --------------------------------------------------

        checks["sma_alignment"] = row["sma_20"] < row["sma_50"] < row["sma_200"]

        if checks["sma_alignment"]:

            reasons.append("Bearish SMA alignment.")

        # --------------------------------------------------
        # PRICE BELOW EMA20
        # --------------------------------------------------

        checks["price_below_ema20"] = row["close"] < row["ema_20"]

        if checks["price_below_ema20"]:

            reasons.append("Price below EMA20.")

        # --------------------------------------------------
        # PRICE BELOW VWAP
        # --------------------------------------------------

        checks["price_below_vwap"] = row["close"] < row["vwap"]

        if checks["price_below_vwap"]:

            reasons.append("Price below VWAP.")

        # --------------------------------------------------
        # SUPERTREND
        # --------------------------------------------------

        checks["supertrend"] = not bool(row["supertrend"])

        if checks["supertrend"]:

            reasons.append("Supertrend is bearish.")

        # --------------------------------------------------
        # ICHIMOKU CLOUD
        # --------------------------------------------------

        checks["ichimoku"] = row["cloud_trend"] == "BEAR"

        if checks["ichimoku"]:

            reasons.append("Price below Ichimoku Cloud.")

        # --------------------------------------------------
        # LONG TERM MARKET TREND
        # --------------------------------------------------

        checks["market_trend"] = row["market_regime"] == "BEAR"

        if checks["market_trend"]:

            reasons.append("Bearish market regime.")
        # ==========================================================
        # MOMENTUM ENGINE
        # ==========================================================

        # --------------------------------------------------
        # RSI
        # --------------------------------------------------

        rsi = float(row["rsi_14"])

        checks["rsi"] = 30 <= rsi <= 45

        if checks["rsi"]:

            reasons.append("Weak RSI momentum.")

        # --------------------------------------------------
        # MACD
        # --------------------------------------------------

        checks["macd_cross"] = row["macd"] < row["macd_signal"]

        if checks["macd_cross"]:

            reasons.append("Bearish MACD crossover.")

        # --------------------------------------------------
        # MACD HISTOGRAM
        # --------------------------------------------------

        checks["macd_histogram"] = row["macd_histogram"] < 0

        if checks["macd_histogram"]:

            reasons.append("Negative MACD histogram.")

        # --------------------------------------------------
        # ADX
        # --------------------------------------------------

        checks["adx"] = row["adx_14"] >= 25

        if checks["adx"]:

            reasons.append("Strong bearish trend.")

        # --------------------------------------------------
        # ROC
        # --------------------------------------------------

        checks["roc"] = row["roc_12"] < 0

        if checks["roc"]:

            reasons.append("Negative Rate of Change.")

        # --------------------------------------------------
        # CCI
        # --------------------------------------------------

        checks["cci"] = row["cci_20"] < -100

        if checks["cci"]:

            reasons.append("CCI indicates bearish momentum.")

        # --------------------------------------------------
        # STOCHASTIC
        # --------------------------------------------------

        checks["stochastic"] = row["stochastic_k"] < row["stochastic_d"]

        if checks["stochastic"]:

            reasons.append("Bearish stochastic crossover.")

        # --------------------------------------------------
        # WILLIAMS %R
        # --------------------------------------------------

        checks["williams_r"] = row["williams_r_14"] < -50

        if checks["williams_r"]:

            reasons.append("Williams %R confirms weakness.")
        # ==========================================================
        # VOLUME ENGINE
        # ==========================================================

        # --------------------------------------------------
        # VOLUME ABOVE AVERAGE
        # --------------------------------------------------

        checks["volume"] = row["volume"] > row["volume_sma_20"]

        if checks["volume"]:

            reasons.append("Selling volume above 20-day average.")

        # --------------------------------------------------
        # VOLUME SPIKE
        # --------------------------------------------------

        volume_ratio = row["volume"] / max(
            row["volume_sma_20"],
            1,
        )

        checks["volume_spike"] = volume_ratio >= 1.50

        if checks["volume_spike"]:

            reasons.append("Heavy institutional selling volume detected.")

        # --------------------------------------------------
        # OBV
        # --------------------------------------------------

        checks["obv"] = row["obv"] < 0

        if checks["obv"]:

            reasons.append("On Balance Volume is negative.")

        # --------------------------------------------------
        # CMF
        # --------------------------------------------------

        checks["cmf"] = row["cmf_20"] < 0

        if checks["cmf"]:

            reasons.append("Negative Chaikin Money Flow.")

        # --------------------------------------------------
        # MONEY FLOW INDEX
        # --------------------------------------------------

        checks["mfi"] = 20 <= row["mfi_14"] <= 50

        if checks["mfi"]:

            reasons.append("Money Flow Index indicates distribution.")

        # --------------------------------------------------
        # VWAP CONFIRMATION
        # --------------------------------------------------

        checks["vwap_confirmation"] = row["close"] < row["vwap"]

        if checks["vwap_confirmation"]:

            reasons.append("VWAP confirms bearish pressure.")

        # --------------------------------------------------
        # DISTRIBUTION
        # --------------------------------------------------

        checks["distribution"] = checks["obv"] and checks["cmf"]

        if checks["distribution"]:

            reasons.append("Volume profile indicates institutional distribution.")
        # ==========================================================
        # BREAKDOWN ENGINE
        # ==========================================================

        # --------------------------------------------------
        # BREAKDOWN
        # --------------------------------------------------

        checks["breakdown"] = bool(row["is_breakdown"])

        if checks["breakdown"]:

            reasons.append("20-period breakdown detected.")

        # --------------------------------------------------
        # FAILED BREAKOUT
        # --------------------------------------------------

        checks["failed_breakout"] = bool(
            row.get(
                "failed_breakout",
                False,
            )
        )

        if checks["failed_breakout"]:

            reasons.append("Failed breakout confirmed.")

        # --------------------------------------------------
        # GAP FILTER
        # --------------------------------------------------

        checks["gap_filter"] = not bool(row["gap_up"])

        if checks["gap_filter"]:

            reasons.append("No bullish opening gap.")

        # --------------------------------------------------
        # PIVOT
        # --------------------------------------------------

        checks["pivot"] = row["close"] < row["pivot"]

        if checks["pivot"]:

            reasons.append("Price below pivot.")

        # --------------------------------------------------
        # BOLLINGER
        # --------------------------------------------------

        checks["bollinger"] = row["close"] < row["bb_middle"]

        if checks["bollinger"]:

            reasons.append("Below Bollinger middle band.")

        # --------------------------------------------------
        # KELTNER
        # --------------------------------------------------

        checks["keltner"] = row["close"] < row["kc_middle"]

        if checks["keltner"]:

            reasons.append("Below Keltner middle channel.")

        # --------------------------------------------------
        # DONCHIAN
        # --------------------------------------------------

        checks["donchian"] = row["close"] < row["dc_middle"]

        if checks["donchian"]:

            reasons.append("Below Donchian midpoint.")

        # --------------------------------------------------
        # ATR FILTER
        # --------------------------------------------------

        checks["atr_filter"] = row["atr_14"] > 0 and row["atr_14"] < (
            row["close"] * 0.05
        )

        if checks["atr_filter"]:

            reasons.append("ATR within acceptable range.")

        # --------------------------------------------------
        # BREAKDOWN CONFIRMATION
        # --------------------------------------------------

        checks["confirmed_breakdown"] = (
            checks["breakdown"] and checks["volume"] and checks["volume_spike"]
        )

        if checks["confirmed_breakdown"]:

            reasons.append("High conviction breakdown confirmed by volume.")
        # ==========================================================
        # MARKET FILTER
        # ==========================================================

        # --------------------------------------------------
        # MARKET REGIME
        # --------------------------------------------------

        checks["bear_market"] = row["market_regime"] == "BEAR"

        if checks["bear_market"]:

            reasons.append("Bear market confirmed.")

        # --------------------------------------------------
        # VOLATILITY
        # --------------------------------------------------

        checks["volatility"] = row["volatility_state"] != "LOW"

        if checks["volatility"]:

            reasons.append("Volatility supports downside movement.")

        # --------------------------------------------------
        # GAP DAY FILTER
        # --------------------------------------------------

        checks["gap_day"] = not bool(
            row.get(
                "gap_day",
                False,
            )
        )

        if checks["gap_day"]:

            reasons.append("No abnormal market gap.")

        # --------------------------------------------------
        # MARKET BREADTH
        # --------------------------------------------------

        checks["breadth"] = (
            row.get(
                "breadth",
                "NEUTRAL",
            )
            == "WEAK"
        )

        if checks["breadth"]:

            reasons.append("Weak market breadth.")

        # ==========================================================
        # SECTOR FILTER
        # ==========================================================

        checks["sector"] = sector_score >= 70

        if checks["sector"]:

            reasons.append("Sector showing relative weakness.")

        # ==========================================================
        # FUNDAMENTAL FILTER
        # ==========================================================

        revenue = fundamentals.get(
            "revenue_growth",
            0,
        )

        earnings = fundamentals.get(
            "earnings_growth",
            0,
        )

        roe = fundamentals.get(
            "roe",
            100,
        )

        debt = fundamentals.get(
            "debt_to_equity",
            0,
        )

        cash = fundamentals.get(
            "operating_cashflow",
            1,
        )

        checks["fundamental"] = (
            revenue < 0 and earnings < 0 and roe < 10 and debt > 2 and cash < 0
        )

        if checks["fundamental"]:

            reasons.append("Weak company fundamentals.")

        # ==========================================================
        # NEWS FILTER
        # ==========================================================

        checks["news"] = news_score >= 60

        if checks["news"]:

            reasons.append("Negative news sentiment supports bearish trade.")

        # ==========================================================
        # MARKET SCORE FILTER
        # ==========================================================

        checks["market_score"] = market_score >= 60

        if checks["market_score"]:

            reasons.append("Overall market score favors bearish trades.")
        # ==========================================================
        # DECISION ENGINE
        # ==========================================================

        # --------------------------------------------------
        # MANDATORY CONDITIONS
        # --------------------------------------------------

        mandatory_checks = {
            "ema_alignment": checks["ema_alignment"],
            "market_trend": checks["market_trend"],
            "volume": checks["volume"],
            "macd_cross": checks["macd_cross"],
            "fundamental": checks["fundamental"],
            "news": checks["news"],
        }

        mandatory_passed = all(mandatory_checks.values())

        # --------------------------------------------------
        # OPTIONAL CONDITIONS
        # --------------------------------------------------

        optional_checks = {
            key: value for key, value in checks.items() if key not in mandatory_checks
        }

        optional_passed = sum(optional_checks.values())

        optional_total = len(optional_checks)

        optional_ratio = optional_passed / max(
            optional_total,
            1,
        )

        # --------------------------------------------------
        # FINAL QUALIFICATION
        # --------------------------------------------------

        qualified = mandatory_passed and optional_ratio >= 0.65

        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        confidence = round(
            (sum(checks.values()) / len(checks)) * 100,
            2,
        )

        # --------------------------------------------------
        # REASONS
        # --------------------------------------------------

        if mandatory_passed:

            reasons.append("Mandatory SELL filters passed.")

        else:

            failed = [key for key, value in mandatory_checks.items() if not value]

            reasons.append("Mandatory filters failed: " + ", ".join(failed))

        reasons.append(f"Optional confirmation: " f"{optional_passed}/{optional_total}")

        reasons.append(f"Confidence: {confidence:.2f}%")

        # --------------------------------------------------
        # ACTION
        # --------------------------------------------------

        action = SELL if qualified else NO_TRADE
        # ==========================================================
        # DIAGNOSTICS
        # ==========================================================

        passed_checks = sum(bool(value) for value in checks.values())

        failed_checks = len(checks) - passed_checks

        reasons.append(f"Checks Passed : {passed_checks}")

        reasons.append(f"Checks Failed : {failed_checks}")

        logger.info(
            "SELL Strategy | Action=%s | Confidence=%.2f | Passed=%d/%d",
            action,
            confidence,
            passed_checks,
            len(checks),
        )

        # ==========================================================
        # RETURN
        # ==========================================================

        return SellDecision(
            action=action,
            passed=qualified,
            confidence=confidence,
            reasons=reasons,
            technical_checks=checks,
        )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    @staticmethod
    def to_dict(
        decision: SellDecision,
    ) -> dict[str, Any]:

        return {
            "action": decision.action,
            "passed": decision.passed,
            "confidence": decision.confidence,
            "reasons": decision.reasons,
            "technical_checks": decision.technical_checks,
        }

    # ==========================================================
    # DEBUG REPORT
    # ==========================================================

    @staticmethod
    def debug_report(
        decision: SellDecision,
    ) -> str:

        report = []

        report.append("========== SELL STRATEGY ==========")

        report.append(f"ACTION : {decision.action}")

        report.append(f"PASSED : {decision.passed}")

        report.append(f"CONFIDENCE : {decision.confidence:.2f}%")

        report.append("")

        report.append("Reasons:")

        for reason in decision.reasons:

            report.append(f" • {reason}")

        report.append("")

        report.append("Checks:")

        for key, value in sorted(decision.technical_checks.items()):

            status = "PASS" if value else "FAIL"

            report.append(f"{key:<30} {status}")

        return "\n".join(report)


# ==========================================================
# END OF FILE
# ==========================================================
