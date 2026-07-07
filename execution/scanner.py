"""
Institutional Market Scanner

Production Version

Responsibilities
----------------
• Download market data
• Build features
• Execute every engine
• Rank opportunities
• Produce final trade list

Pipeline

Market Data
    │
    ▼
Features
    │
    ▼
Indicators
    │
    ▼
BUY Strategy
SELL Strategy
    │
    ▼
BUY Score
SELL Score
    │
    ▼
BUY Probability
SELL Probability
    │
    ▼
Decision Engine
    │
    ▼
Validation
    │
    ▼
Risk
    │
    ▼
Position Sizing
    │
    ▼
Portfolio Rules
    │
    ▼
Final Ranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.logger import get_logger

from data.market_data import MarketData

from features.feature_engineering import FeatureEngineeringEngine

from strategy.buy_strategy import BuyStrategyEngine
from strategy.sell_strategy import SellStrategyEngine

from strategy.buy_scoring import BuyScoringEngine
from strategy.sell_scoring import SellScoringEngine

from strategy.buy_probability import BuyProbabilityEngine
from strategy.sell_probability import SellProbabilityEngine

from decision.decision_engine import DecisionEngine
from decision.validation_engine import ValidationEngine

from risk.risk_manager import RiskManager
from risk.position_sizing import PositionSizingEngine
from risk.portfolio_rules import PortfolioRulesEngine

logger = get_logger(__name__)


# ==========================================================
# RESULT
# ==========================================================


@dataclass(slots=True)
class ScanResult:

    symbol: str

    action: str

    score: float

    probability: float

    confidence: float

    ranking: float

    position_size: int

    portfolio_allowed: bool

    diagnostics: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# SCANNER
# ==========================================================


class MarketScanner:
    """
    Master Scanner
    """

    def __init__(self):

        self.market = MarketDataEngine()

        self.features = FeatureEngineeringEngine()

        self.buy_strategy = BuyStrategyEngine()

        self.sell_strategy = SellStrategyEngine()

        self.buy_score = BuyScoringEngine()

        self.sell_score = SellScoringEngine()

        self.buy_probability = BuyProbabilityEngine()

        self.sell_probability = SellProbabilityEngine()

        self.decision = DecisionEngine()

        self.validation = ValidationEngine()

        self.risk = RiskManager()

        self.position = PositionSizingEngine()

        self.portfolio = PortfolioRulesEngine()

        logger.info("Market Scanner initialized.")

    # ==========================================================
    # SCAN SINGLE SYMBOL
    # ==========================================================

    def scan_symbol(
        self,
        symbol: str,
        portfolio: dict[str, Any],
        broker_status: dict[str, Any],
        market_state: dict[str, Any],
    ) -> ScanResult:

        logger.info(
            "Scanning %s",
            symbol,
        )

        diagnostics = {}

        try:

            # --------------------------------------------------
            # DOWNLOAD DATA
            # --------------------------------------------------

            dataframe = self.market.get_history(
                symbol=symbol,
            )

            if dataframe.empty:

                raise ValueError("No market data received.")

            diagnostics["candles"] = len(dataframe)

            diagnostics["symbol"] = symbol

            # --------------------------------------------------
            # FEATURE ENGINEERING
            # --------------------------------------------------

            dataframe = self.features.generate(dataframe)

            latest = dataframe.iloc[-1]

            diagnostics["latest_close"] = round(
                float(latest["close"]),
                2,
            )

            # --------------------------------------------------
            # STRATEGIES
            # --------------------------------------------------

            buy_decision = self.buy_strategy.evaluate(dataframe)

            sell_decision = self.sell_strategy.evaluate(dataframe)

            diagnostics["buy_signal"] = buy_decision.action

            diagnostics["sell_signal"] = sell_decision.action

            # --------------------------------------------------
            # SCORING
            # --------------------------------------------------

            buy_score = self.buy_score.calculate(
                dataframe,
                buy_decision,
            )

            sell_score = self.sell_score.calculate(
                dataframe,
                sell_decision,
            )

            diagnostics["buy_score"] = round(
                buy_score.overall,
                2,
            )

            diagnostics["sell_score"] = round(
                sell_score.overall,
                2,
            )
            # ==========================================================
            # PROBABILITY ENGINES
            # ==========================================================

            buy_probability = self.buy_probability.calculate(
                dataframe=dataframe,
                decision=buy_decision,
                score=buy_score,
            )

            sell_probability = self.sell_probability.calculate(
                dataframe=dataframe,
                decision=sell_decision,
                score=sell_score,
            )

            diagnostics["buy_probability"] = round(
                buy_probability.win_probability,
                2,
            )

            diagnostics["sell_probability"] = round(
                sell_probability.success_probability,
                2,
            )

            # ==========================================================
            # DECISION ENGINE
            # ==========================================================

            final_decision = self.decision.evaluate(
                buy_decision=buy_decision,
                sell_decision=sell_decision,
                buy_score=buy_score,
                sell_score=sell_score,
                buy_probability=buy_probability,
                sell_probability=sell_probability,
            )

            diagnostics["decision"] = final_decision.action

            diagnostics["ranking"] = round(
                final_decision.ranking,
                2,
            )

            diagnostics["confidence"] = round(
                final_decision.confidence,
                2,
            )

            # ==========================================================
            # VALIDATION ENGINE
            # ==========================================================

            validation = self.validation.validate(
                decision=final_decision,
                dataframe=dataframe,
                portfolio=portfolio,
                broker_status=broker_status,
                market_state=market_state,
            )

            diagnostics["validation_passed"] = validation.passed

            diagnostics["validation_action"] = validation.action

            diagnostics["validation_warnings"] = len(validation.warnings)

            if not validation.passed:

                logger.info(
                    "%s rejected by Validation Engine.",
                    symbol,
                )
            # ==========================================================
            # RISK MANAGER
            # ==========================================================

            risk_result = self.risk.evaluate(
                validation=validation,
                decision=final_decision,
                dataframe=dataframe,
                portfolio=portfolio,
                market=market_state,
            )

            diagnostics["risk_safe"] = risk_result.safe

            diagnostics["risk_grade"] = risk_result.risk_grade

            diagnostics["total_risk"] = round(
                risk_result.total_risk,
                2,
            )

            # ==========================================================
            # POSITION SIZING
            # ==========================================================

            position_result = self.position.calculate(
                decision=final_decision,
                validation=validation,
                risk=risk_result,
                dataframe=dataframe,
                portfolio=portfolio,
            )

            diagnostics["quantity"] = position_result.quantity

            diagnostics["position_value"] = round(
                position_result.position_value,
                2,
            )

            diagnostics["allocation"] = round(
                position_result.allocation_percent,
                4,
            )

            # ==========================================================
            # PORTFOLIO RULES
            # ==========================================================

            portfolio_result = self.portfolio.evaluate(
                decision=final_decision,
                validation=validation,
                risk=risk_result,
                sizing=position_result,
                portfolio=portfolio,
            )

            diagnostics["portfolio_allowed"] = portfolio_result.allowed

            diagnostics["portfolio_score"] = round(
                portfolio_result.portfolio_score,
                2,
            )

            # ==========================================================
            # FINAL SCORE
            # ==========================================================

            if final_decision.action == "BUY":

                final_score = final_decision.buy_score

                probability = final_decision.buy_probability

            elif final_decision.action == "SELL":

                final_score = final_decision.sell_score

                probability = final_decision.sell_probability

            else:

                final_score = 0.0

                probability = 0.0

            # ==========================================================
            # BUILD RESULT
            # ==========================================================

            return ScanResult(
                symbol=symbol,
                action=final_decision.action,
                score=round(
                    final_score,
                    2,
                ),
                probability=round(
                    probability,
                    2,
                ),
                confidence=round(
                    final_decision.confidence,
                    2,
                ),
                ranking=round(
                    final_decision.ranking,
                    2,
                ),
                position_size=(position_result.quantity),
                portfolio_allowed=(portfolio_result.allowed),
                diagnostics=diagnostics,
            )
        # ==========================================================
        # EXCEPTION HANDLING
        # ==========================================================

        except Exception as exc:

            logger.exception(
                "Scanner failed for %s",
                symbol,
            )

            diagnostics["error"] = str(exc)

            return ScanResult(
                symbol=symbol,
                action="ERROR",
                score=0.0,
                probability=0.0,
                confidence=0.0,
                ranking=0.0,
                position_size=0,
                portfolio_allowed=False,
                diagnostics=diagnostics,
            )

    # ==========================================================
    # SCAN MULTIPLE SYMBOLS
    # ==========================================================

    def scan_symbols(
        self,
        symbols: list[str],
        portfolio: dict[str, Any],
        broker_status: dict[str, Any],
        market_state: dict[str, Any],
    ) -> list[ScanResult]:

        logger.info(
            "Starting scan of %d symbols.",
            len(symbols),
        )

        results: list[ScanResult] = []

        total = len(symbols)

        # ==========================================================
        # MAIN LOOP
        # ==========================================================

        for index, symbol in enumerate(
            symbols,
            start=1,
        ):

            logger.info(
                "[%d/%d] %s",
                index,
                total,
                symbol,
            )

            result = self.scan_symbol(
                symbol=symbol,
                portfolio=portfolio,
                broker_status=broker_status,
                market_state=market_state,
            )

            results.append(result)

        # ==========================================================
        # SCAN SUMMARY
        # ==========================================================

        logger.info(
            "Completed scanning %d symbols.",
            len(results),
        )

        diagnostics = {
            "total_symbols": total,
            "successful_scans": sum(result.action != "ERROR" for result in results),
            "failed_scans": sum(result.action == "ERROR" for result in results),
        }

        logger.info(
            "Scan Summary: %s",
            diagnostics,
        )
        # ==========================================================
        # FILTER RESULTS
        # ==========================================================

        valid_results = [result for result in results if (result.action != "ERROR")]

        diagnostics["valid_results"] = len(valid_results)

        diagnostics["invalid_results"] = len(results) - len(valid_results)

        # ==========================================================
        # REMOVE REJECTED TRADES
        # ==========================================================

        executable_results = [
            result
            for result in valid_results
            if (
                result.portfolio_allowed
                and result.action
                in (
                    "BUY",
                    "SELL",
                )
            )
        ]

        diagnostics["executable_results"] = len(executable_results)

        # ==========================================================
        # RANK RESULTS
        # ==========================================================

        executable_results.sort(
            key=lambda result: (
                result.ranking,
                result.confidence,
                result.score,
                result.probability,
            ),
            reverse=True,
        )

        # ==========================================================
        # ASSIGN RANKS
        # ==========================================================

        for rank, result in enumerate(
            executable_results,
            start=1,
        ):

            result.diagnostics["rank"] = rank

            result.diagnostics["scanner_score"] = round(
                (
                    result.ranking * 0.40
                    + result.confidence * 0.30
                    + result.score * 0.20
                    + result.probability * 0.10
                ),
                2,
            )

        # ==========================================================
        # BEST CANDIDATE
        # ==========================================================

        if executable_results:

            best = executable_results[0]

            logger.info(
                "Top Candidate: %s | %s | Rank %.2f",
                best.symbol,
                best.action,
                best.ranking,
            )

        else:

            logger.info("No executable opportunities found.")

        # ==========================================================
        # STORE SCAN STATISTICS
        # ==========================================================

        diagnostics["buy_count"] = sum(
            result.action == "BUY" for result in executable_results
        )

        diagnostics["sell_count"] = sum(
            result.action == "SELL" for result in executable_results
        )

        diagnostics["average_ranking"] = round(
            sum(result.ranking for result in executable_results)
            / max(
                len(executable_results),
                1,
            ),
            2,
        )

        diagnostics["average_probability"] = round(
            sum(result.probability for result in executable_results)
            / max(
                len(executable_results),
                1,
            ),
            2,
        )
        # ==========================================================
        # CONFIGURATION
        # ==========================================================

        max_trade_candidates = int(
            market_state.get(
                "max_trade_candidates",
                20,
            )
        )

        max_watchlist = int(
            market_state.get(
                "max_watchlist",
                50,
            )
        )

        diagnostics["max_trade_candidates"] = max_trade_candidates

        diagnostics["max_watchlist"] = max_watchlist

        # ==========================================================
        # TOP TRADE CANDIDATES
        # ==========================================================

        trade_candidates = executable_results[:max_trade_candidates]

        diagnostics["trade_candidates"] = len(trade_candidates)

        # ==========================================================
        # WATCHLIST
        # ==========================================================

        watchlist = [
            result
            for result in valid_results
            if (
                result.action
                in (
                    "BUY",
                    "SELL",
                )
            )
        ]

        watchlist.sort(
            key=lambda result: (
                result.ranking,
                result.confidence,
            ),
            reverse=True,
        )

        watchlist = watchlist[:max_watchlist]

        diagnostics["watchlist_size"] = len(watchlist)

        # ==========================================================
        # BUY / SELL BREAKDOWN
        # ==========================================================

        buy_candidates = [
            result for result in trade_candidates if result.action == "BUY"
        ]

        sell_candidates = [
            result for result in trade_candidates if result.action == "SELL"
        ]

        diagnostics["buy_candidates"] = len(buy_candidates)

        diagnostics["sell_candidates"] = len(sell_candidates)

        # ==========================================================
        # SCANNER ANALYTICS
        # ==========================================================

        diagnostics["highest_probability"] = round(
            max(
                (result.probability for result in trade_candidates),
                default=0.0,
            ),
            2,
        )

        diagnostics["highest_confidence"] = round(
            max(
                (result.confidence for result in trade_candidates),
                default=0.0,
            ),
            2,
        )

        diagnostics["highest_score"] = round(
            max(
                (result.score for result in trade_candidates),
                default=0.0,
            ),
            2,
        )

        # ==========================================================
        # SCAN COMPLETENESS
        # ==========================================================

        diagnostics["scan_completed"] = True

        diagnostics["scan_timestamp"] = pd.Timestamp.utcnow().isoformat()

        logger.info("Scanner analytics completed.")
        # ==========================================================
        # CONFIDENCE DISTRIBUTION
        # ==========================================================

        confidence_distribution = {
            "very_high": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for result in trade_candidates:

            if result.confidence >= 90:

                confidence_distribution["very_high"] += 1

            elif result.confidence >= 80:

                confidence_distribution["high"] += 1

            elif result.confidence >= 65:

                confidence_distribution["medium"] += 1

            else:

                confidence_distribution["low"] += 1

        diagnostics["confidence_distribution"] = confidence_distribution

        # ==========================================================
        # PROBABILITY DISTRIBUTION
        # ==========================================================

        probability_distribution = {
            "90_plus": 0,
            "80_plus": 0,
            "70_plus": 0,
            "below_70": 0,
        }

        for result in trade_candidates:

            if result.probability >= 90:

                probability_distribution["90_plus"] += 1

            elif result.probability >= 80:

                probability_distribution["80_plus"] += 1

            elif result.probability >= 70:

                probability_distribution["70_plus"] += 1

            else:

                probability_distribution["below_70"] += 1

        diagnostics["probability_distribution"] = probability_distribution

        # ==========================================================
        # MARKET BREADTH
        # ==========================================================

        total_buy = len(buy_candidates)

        total_sell = len(sell_candidates)

        total_signals = max(
            total_buy + total_sell,
            1,
        )

        buy_breadth = (total_buy / total_signals) * 100

        sell_breadth = (total_sell / total_signals) * 100

        diagnostics["buy_breadth"] = round(
            buy_breadth,
            2,
        )

        diagnostics["sell_breadth"] = round(
            sell_breadth,
            2,
        )

        # ==========================================================
        # SCANNER QUALITY
        # ==========================================================

        scanner_quality = (
            diagnostics["average_ranking"] * 0.35
            + diagnostics["average_probability"] * 0.35
            + buy_breadth * 0.15
            + (100.0 - sell_breadth) * 0.15
        )

        scanner_quality = round(
            min(
                scanner_quality,
                100.0,
            ),
            2,
        )

        diagnostics["scanner_quality"] = scanner_quality

        # ==========================================================
        # RECOMMENDATION SUMMARY
        # ==========================================================

        recommendations = []

        if scanner_quality >= 90:

            recommendations.append("Excellent market opportunities detected.")

        elif scanner_quality >= 80:

            recommendations.append("Strong trading session.")

        elif scanner_quality >= 70:

            recommendations.append("Moderate-quality opportunities.")

        else:

            recommendations.append("Exercise caution.")

        if total_buy > total_sell:

            recommendations.append("BUY opportunities dominate.")

        elif total_sell > total_buy:

            recommendations.append("SELL opportunities dominate.")

        else:

            recommendations.append("Balanced market.")

        diagnostics["recommendations"] = recommendations

        logger.info("Scanner quality evaluation completed.")
        # ==========================================================
        # FINAL SCANNER VALIDATION
        # ==========================================================

        validation_errors: list[str] = []

        if diagnostics["total_symbols"] <= 0:

            validation_errors.append("No symbols supplied.")

        if diagnostics["successful_scans"] < 0:

            validation_errors.append("Invalid successful scan count.")

        if diagnostics["failed_scans"] < 0:

            validation_errors.append("Invalid failed scan count.")

        if diagnostics["buy_count"] < 0:

            validation_errors.append("Invalid BUY count.")

        if diagnostics["sell_count"] < 0:

            validation_errors.append("Invalid SELL count.")

        diagnostics["validation_errors"] = validation_errors

        # ==========================================================
        # FAIL SAFE
        # ==========================================================

        if validation_errors:

            logger.error(
                "Scanner validation failed: %s",
                validation_errors,
            )

            diagnostics["scan_completed"] = False

            diagnostics["fail_safe"] = True

            trade_candidates = []

            watchlist = []

            buy_candidates = []

            sell_candidates = []

            recommendations = ["Scanner entered fail-safe mode."]

        else:

            diagnostics["fail_safe"] = False

        # ==========================================================
        # FINAL SUMMARY
        # ==========================================================

        diagnostics["summary"] = {
            "symbols_scanned": diagnostics["total_symbols"],
            "successful": diagnostics["successful_scans"],
            "failed": diagnostics["failed_scans"],
            "trade_candidates": len(trade_candidates),
            "watchlist": len(watchlist),
            "buy_candidates": len(buy_candidates),
            "sell_candidates": len(sell_candidates),
            "scanner_quality": scanner_quality,
        }

        # ==========================================================
        # FINAL REPORT
        # ==========================================================

        logger.info(
            "Scanner completed with %d trade candidates.",
            len(trade_candidates),
        )

        logger.info(
            "Scanner quality %.2f",
            scanner_quality,
        )

        logger.info(
            "Watchlist size %d",
            len(watchlist),
        )

        # ==========================================================
        # RETURN RESULTS
        # ==========================================================

        return trade_candidates

    # ==========================================================
    # EXPORT TO DATAFRAME
    # ==========================================================

    @staticmethod
    def export_dataframe(
        results: list[ScanResult],
    ) -> pd.DataFrame:

        rows = []

        for result in results:

            rows.append(
                {
                    "Symbol": result.symbol,
                    "Action": result.action,
                    "Score": round(
                        result.score,
                        2,
                    ),
                    "Probability": round(
                        result.probability,
                        2,
                    ),
                    "Confidence": round(
                        result.confidence,
                        2,
                    ),
                    "Ranking": round(
                        result.ranking,
                        2,
                    ),
                    "Position": result.position_size,
                    "Portfolio": result.portfolio_allowed,
                }
            )

        return pd.DataFrame(rows)

    # ==========================================================
    # EXPORT CSV
    # ==========================================================

    @staticmethod
    def export_csv(
        results: list[ScanResult],
        filename: str,
    ) -> None:

        dataframe = MarketScanner.export_dataframe(results)

        dataframe.to_csv(
            filename,
            index=False,
        )

        logger.info(
            "Scanner CSV exported: %s",
            filename,
        )

    # ==========================================================
    # EXPORT JSON
    # ==========================================================

    @staticmethod
    def export_json(
        results: list[ScanResult],
        filename: str,
    ) -> None:

        dataframe = MarketScanner.export_dataframe(results)

        dataframe.to_json(
            filename,
            orient="records",
            indent=4,
        )

        logger.info(
            "Scanner JSON exported: %s",
            filename,
        )

    # ==========================================================
    # TOP RESULTS
    # ==========================================================

    @staticmethod
    def top_results(
        results: list[ScanResult],
        limit: int = 10,
    ) -> list[ScanResult]:

        ordered = sorted(
            results,
            key=lambda item: (
                item.ranking,
                item.confidence,
                item.score,
            ),
            reverse=True,
        )

        return ordered[:limit]

    # ==========================================================
    # SCANNER SUMMARY
    # ==========================================================

    @staticmethod
    def summary(
        results: list[ScanResult],
    ) -> str:

        total = len(results)

        buy_count = sum(result.action == "BUY" for result in results)

        sell_count = sum(result.action == "SELL" for result in results)

        average_probability = round(
            sum(result.probability for result in results)
            / max(
                total,
                1,
            ),
            2,
        )

        average_ranking = round(
            sum(result.ranking for result in results)
            / max(
                total,
                1,
            ),
            2,
        )

        return (
            f"Scanned={total}"
            f" | BUY={buy_count}"
            f" | SELL={sell_count}"
            f" | Avg Rank={average_ranking:.2f}"
            f" | Avg Prob={average_probability:.2f}%"
        )

    # ==========================================================
    # MARKET OVERVIEW
    # ==========================================================

    @staticmethod
    def market_overview(
        results: list[ScanResult],
    ) -> dict[str, Any]:

        overview = {
            "total_symbols": len(results),
            "buy": sum(result.action == "BUY" for result in results),
            "sell": sum(result.action == "SELL" for result in results),
            "hold": sum(result.action == "HOLD" for result in results),
            "errors": sum(result.action == "ERROR" for result in results),
            "portfolio_approved": sum(result.portfolio_allowed for result in results),
        }

        return overview

    # ==========================================================
    # RANKING REPORT
    # ==========================================================

    @staticmethod
    def ranking_report(
        results: list[ScanResult],
        limit: int = 20,
    ) -> str:

        ordered = sorted(
            results,
            key=lambda item: (
                item.ranking,
                item.confidence,
            ),
            reverse=True,
        )[:limit]

        lines = []

        lines.append("=" * 90)

        lines.append("TOP MARKET OPPORTUNITIES")

        lines.append("=" * 90)

        lines.append(
            f"{'Rank':<6}"
            f"{'Symbol':<15}"
            f"{'Action':<10}"
            f"{'Score':>8}"
            f"{'Prob':>8}"
            f"{'Conf':>8}"
            f"{'Qty':>8}"
        )

        lines.append("-" * 90)

        for rank, result in enumerate(
            ordered,
            start=1,
        ):

            lines.append(
                f"{rank:<6}"
                f"{result.symbol:<15}"
                f"{result.action:<10}"
                f"{result.score:>8.2f}"
                f"{result.probability:>8.2f}"
                f"{result.confidence:>8.2f}"
                f"{result.position_size:>8}"
            )

        lines.append("=" * 90)

        return "\n".join(lines)

    # ==========================================================
    # DEBUG REPORT
    # ==========================================================

    @staticmethod
    def debug_report(
        results: list[ScanResult],
    ) -> str:

        report: list[str] = []

        report.append("=" * 120)
        report.append("MARKET SCANNER REPORT")
        report.append("=" * 120)
        report.append("")

        report.append(MarketScanner.summary(results))

        report.append("")
        report.append("")

        report.append(
            f"{'Rank':<6}"
            f"{'Symbol':<15}"
            f"{'Action':<10}"
            f"{'Score':>8}"
            f"{'Prob':>8}"
            f"{'Conf':>8}"
            f"{'Rank':>8}"
            f"{'Qty':>8}"
        )

        report.append("-" * 120)

        ordered = sorted(
            results,
            key=lambda item: (
                item.ranking,
                item.confidence,
                item.score,
            ),
            reverse=True,
        )

        for index, result in enumerate(
            ordered,
            start=1,
        ):

            report.append(
                f"{index:<6}"
                f"{result.symbol:<15}"
                f"{result.action:<10}"
                f"{result.score:>8.2f}"
                f"{result.probability:>8.2f}"
                f"{result.confidence:>8.2f}"
                f"{result.ranking:>8.2f}"
                f"{result.position_size:>8}"
            )

        report.append("")
        report.append("=" * 120)
        report.append("TOP 10 DIAGNOSTICS")
        report.append("=" * 120)

        for result in ordered[:10]:

            report.append("")
            report.append(f"[{result.symbol}]")

            report.append("-" * 80)

            for key, value in sorted(result.diagnostics.items()):

                report.append(f"{key:<35} : {value}")

        report.append("")
        report.append("=" * 120)
        report.append("END OF REPORT")
        report.append("=" * 120)

        return "\n".join(report)


# ==========================================================
# END OF FILE
# ==========================================================
