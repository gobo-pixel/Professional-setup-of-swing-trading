"""
WIRED ORCHESTRATOR (PRODUCTION FLOW)

Single controlled execution pipeline:
NO module is allowed to bypass this flow.

Modes:
- LIVE
- BACKTEST
- PAPER
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import time

from core.logger import get_logger

from data.data_engine import DataEngine
from features.feature_engineering import FeatureEngineeringEngine
from market.market_regime import MarketRegimeEngine
from strategy.buy_strategy import BuyStrategyEngine
from strategy.sell_strategy import SellStrategyEngine
from decision.decision_engine import DecisionEngine
from risk.risk_manager import RiskManager
from execution.scanner import Scanner
from execution.broker import BrokerEngine
from portfolio.portfolio import PortfolioEngine
from analytics.analytics import AnalyticsEngine

logger = get_logger(__name__)


# ==========================================================
# ORCHESTRATOR STATE
# ==========================================================


@dataclass
class OrchestratorContext:

    mode: str  # LIVE / BACKTEST / PAPER

    cycle_id: int = 0

    timestamp: float = 0.0

    last_data: Any = None

    last_features: Any = None

    last_signals: Any = None

    last_decision: Any = None

    last_risk: Any = None

    last_execution: Any = None


# ==========================================================
# WIRED ORCHESTRATOR
# ==========================================================


class WiredOrchestrator:

    def __init__(self, mode: str = "BACKTEST"):

        self.mode = mode

        self.data_engine = DataEngine()

        self.feature_engine = FeatureEngineeringEngine()

        self.market = MarketRegimeEngine()

        self.buy_strategy = BuyStrategyEngine()

        self.sell_strategy = SellStrategyEngine()

        self.decision_engine = DecisionEngine()

        self.risk_engine = RiskManager()

        self.scanner = Scanner()

        self.broker = BrokerEngine()

        self.portfolio = PortfolioEngine()

        self.analytics = AnalyticsEngine()

        self.context = OrchestratorContext(mode=mode)

        logger.info(f"WIRED ORCHESTRATOR initialized in {mode} mode")

    # ==========================================================
    # MAIN EXECUTION LOOP
    # ==========================================================

    def run_cycle(
        self,
        symbols: list[str],
        portfolio_state: Any,
    ) -> OrchestratorContext:

        self.context.cycle_id += 1

        self.context.timestamp = time.time()

        # ==========================================================
        # STEP 1: DATA FETCH
        # ==========================================================

        data = self.data_engine.get_market_data(symbols)

        self.context.last_data = data

        # ==========================================================
        # STEP 2: FEATURE GENERATION
        # ==========================================================

        features = self.feature_engine.build_features(data)

        self.context.last_features = features

        # ==========================================================
        # STEP 3: MARKET REGIME DETECTION
        # ==========================================================

        regime = self.market.detect(features)

        self.context.last_risk = {"regime": regime}

        # ==========================================================
        # STEP 4: STRATEGY SIGNAL GENERATION
        # ==========================================================

        buy_signals = self.buy_strategy.generate(
            features=features,
            regime=regime,
        )

        sell_signals = self.sell_strategy.generate(
            features=features,
            regime=regime,
        )

        signals = {
            "buy": buy_signals,
            "sell": sell_signals,
        }

        self.context.last_signals = signals

        # ==========================================================
        # STEP 5: DECISION ENGINE
        # ==========================================================

        decision = self.decision_engine.evaluate(
            signals=signals,
            portfolio=portfolio_state,
            regime=regime,
        )

        self.context.last_decision = decision
        # ==========================================================
        # STEP 6: RISK ENGINE (FINAL GATE BEFORE EXECUTION)
        # ==========================================================

        risk_check = self.risk_engine.evaluate_order(
            validation=self.decision_engine.validate(decision),
            decision=decision,
            portfolio=portfolio_state,
            market={"regime": regime, "event_day": False, "vix": 20},
        )

        self.context.last_risk = risk_check

        if not risk_check["approved"]:

            logger.warning(
                "Risk engine blocked execution cycle %d",
                self.context.cycle_id,
            )

            self.context.last_execution = {
                "status": "BLOCKED",
                "reason": risk_check,
            }

            return self.context

        # ==========================================================
        # STEP 7: SCANNER (ORDER PREPARATION LAYER)
        # ==========================================================

        scan_candidates = self.scanner.prepare_orders(
            decision=decision,
            signals=signals,
            portfolio=portfolio_state,
        )

        # ==========================================================
        # STEP 8: EXECUTION VIA BROKER
        # ==========================================================

        executed_orders = []

        for order in scan_candidates:

            result = self.broker.execute_order(
                order=order,
                mode=self.mode,
            )

            executed_orders.append(result)

        self.context.last_execution = {
            "status": "EXECUTED",
            "orders": executed_orders,
        }

        # ==========================================================
        # STEP 9: PORTFOLIO UPDATE
        # ==========================================================

        updated_portfolio = self.portfolio.update_from_broker(
            executed_orders=executed_orders,
            portfolio_state=portfolio_state,
        )

        return self.context
        # ==========================================================
        # STEP 10: PORTFOLIO RECONCILIATION (SOURCE OF TRUTH UPDATE)
        # ==========================================================

        self.portfolio.reconcile(
            broker_updates=executed_orders,
            portfolio_state=updated_portfolio,
        )

        # ==========================================================
        # STEP 11: EQUITY + PNL UPDATE
        # ==========================================================

        portfolio_snapshot = self.portfolio.snapshot()

        equity = portfolio_snapshot.get("equity", 0.0)

        pnl = portfolio_snapshot.get("total_pnl", 0.0)

        self.context.last_portfolio = {
            "equity": equity,
            "pnl": pnl,
            "exposure": portfolio_snapshot.get("exposure", 0.0),
        }

        # ==========================================================
        # STEP 12: ANALYTICS FEED (POST-CYCLE INTELLIGENCE)
        # ==========================================================

        analytics_input = {
            "cycle_id": self.context.cycle_id,
            "equity": equity,
            "pnl": pnl,
            "orders": executed_orders,
            "signals": signals,
            "decision": decision,
            "risk": risk_check,
            "regime": regime,
        }

        analytics_state = self.analytics.update(analytics_input)

        self.context.last_analytics = analytics_state

        # ==========================================================
        # STEP 13: CYCLE COMPLETION LOG
        # ==========================================================

        logger.info(
            "Cycle %d completed | Equity=%.2f | PnL=%.2f",
            self.context.cycle_id,
            equity,
            pnl,
        )

        return self.context

    # ==========================================================
    # LIVE RUN LOOP
    # ==========================================================

    def run_live(
        self,
        symbols: list[str],
        portfolio_state: Any,
        sleep_seconds: int = 60,
    ) -> None:

        logger.info("LIVE MODE STARTED")

        while True:

            try:

                self.run_cycle(
                    symbols=symbols,
                    portfolio_state=portfolio_state,
                )

                time.sleep(sleep_seconds)

            except Exception as e:

                logger.exception(
                    "LIVE LOOP ERROR: %s",
                    str(e),
                )

                self.emergency_stop()

                break

    # ==========================================================
    # BACKTEST RUN LOOP
    # ==========================================================

    def run_backtest(
        self,
        historical_data: dict[str, Any],
        portfolio_state: Any,
    ) -> list[OrchestratorContext]:

        logger.info("BACKTEST MODE STARTED")

        results = []

        for step in range(len(next(iter(historical_data.values())))):

            sliced_data = {k: v.iloc[: step + 1] for k, v in historical_data.items()}

            self.context.timestamp = time.time()

            ctx = self.run_cycle(
                symbols=list(historical_data.keys()),
                portfolio_state=portfolio_state,
            )

            results.append(ctx)

        return results

    # ==========================================================
    # PAPER MODE (SIMULATION WITH LIVE FEED)
    # ==========================================================

    def run_paper(
        self,
        symbols: list[str],
        portfolio_state: Any,
        sleep_seconds: int = 10,
    ) -> None:

        logger.info("PAPER TRADING MODE STARTED")

        while True:

            self.run_cycle(
                symbols=symbols,
                portfolio_state=portfolio_state,
            )

            time.sleep(sleep_seconds)

    # ==========================================================
    # EMERGENCY STOP
    # ==========================================================

    def emergency_stop(self) -> None:

        logger.critical("EMERGENCY STOP ACTIVATED")

        self.context.last_execution = {
            "status": "STOPPED",
            "reason": "EMERGENCY_TRIGGER",
        }

    # ==========================================================
    # KILL SWITCH (GLOBAL SAFETY OVERRIDE)
    # ==========================================================

    def kill_switch(self, portfolio_state: Any) -> bool:

        equity = getattr(portfolio_state, "equity", 0.0)
        pnl = getattr(portfolio_state, "total_pnl", 0.0)
        exposure = getattr(portfolio_state, "exposure", 0.0)

        if pnl < -0.05 * equity:
            logger.critical("KILL SWITCH: DAILY LOSS LIMIT BREACHED")
            return True

        if exposure > 0.95:
            logger.critical("KILL SWITCH: EXCESS EXPOSURE")
            return True

        return False

    # ==========================================================
    # CIRCUIT BREAKER (VOLATILITY PROTECTION)
    # ==========================================================

    def circuit_breaker(self, risk_check: dict[str, Any]) -> bool:

        if risk_check.get("volatility", 0.0) > 0.03:
            logger.warning("CIRCUIT BREAKER: HIGH VOLATILITY")
            return True

        if risk_check.get("liquidity_risk", False):
            logger.warning("CIRCUIT BREAKER: LIQUIDITY RISK")
            return True

        return False

    # ==========================================================
    # CYCLE VALIDATION (DATA INTEGRITY CHECK)
    # ==========================================================

    def validate_cycle(self, context: OrchestratorContext) -> bool:

        if context.last_data is None:
            logger.error("VALIDATION FAILED: NO DATA")
            return False

        if context.last_features is None:
            logger.error("VALIDATION FAILED: NO FEATURES")
            return False

        if context.last_signals is None:
            logger.error("VALIDATION FAILED: NO SIGNALS")
            return False

        return True

    # ==========================================================
    # PRE-RUN GUARD (BEFORE EVERY CYCLE)
    # ==========================================================

    def pre_cycle_guard(self, portfolio_state: Any, risk_check: dict[str, Any]) -> bool:

        if self.kill_switch(portfolio_state):
            self.emergency_stop()
            return False

        if self.circuit_breaker(risk_check):
            self.emergency_stop()
            return False

        return True

    # ==========================================================
    # EXECUTION TRACE LOGGER (FULL AUDIT TRAIL)
    # ==========================================================

    def trace_cycle(self) -> dict[str, Any]:

        trace = {
            "cycle_id": self.context.cycle_id,
            "timestamp": self.context.timestamp,
            "data_loaded": self.context.last_data is not None,
            "features_generated": self.context.last_features is not None,
            "signals_generated": self.context.last_signals is not None,
            "decision": self.context.last_decision,
            "risk": self.context.last_risk,
            "execution": self.context.last_execution,
            "portfolio": self.context.last_portfolio,
        }

        logger.debug("CYCLE TRACE: %s", trace)

        return trace

    # ==========================================================
    # DECISION EXPLAINABILITY LOGGER
    # ==========================================================

    def explain_decision(self) -> dict[str, Any]:

        decision = self.context.last_decision

        explanation = {
            "approved": getattr(decision, "approved", None),
            "reason": getattr(decision, "reason", None),
            "confidence": getattr(decision, "confidence", None),
            "signals_used": self.context.last_signals,
            "market_regime": self.context.last_risk.get("regime", None),
        }

        logger.info("DECISION EXPLANATION: %s", explanation)

        return explanation

    # ==========================================================
    # TRADE LINEAGE TRACKING
    # ==========================================================

    def trace_trades(self) -> list[dict[str, Any]]:

        execution = self.context.last_execution or {}

        orders = execution.get("orders", [])

        lineage = []

        for order in orders:

            lineage.append(
                {
                    "cycle_id": self.context.cycle_id,
                    "order": order,
                    "signals": self.context.last_signals,
                    "decision": self.context.last_decision,
                    "risk": self.context.last_risk,
                }
            )

        self.context.last_execution["lineage"] = lineage

        return lineage

    # ==========================================================
    # DEBUG SNAPSHOT (SYSTEM STATE DUMP)
    # ==========================================================

    def debug_snapshot(self) -> dict[str, Any]:

        snapshot = {
            "cycle_id": self.context.cycle_id,
            "mode": self.mode,
            "context": self.context,
            "portfolio": self.context.last_portfolio,
            "analytics": self.context.last_analytics,
        }

        logger.debug("DEBUG SNAPSHOT GENERATED")

        return snapshot

    # ==========================================================
    # CLEAN SHUTDOWN (SAFE TERMINATION)
    # ==========================================================

    def shutdown(self) -> None:

        logger.info("SHUTDOWN INITIATED")

        try:

            final_snapshot = self.debug_snapshot()

            self.storage.save(final_snapshot)

        except Exception as e:

            logger.exception("SHUTDOWN SNAPSHOT FAILED: %s", str(e))

        self.context.last_execution = {
            "status": "SHUTDOWN",
            "reason": "MANUAL_OR_SYSTEM_EXIT",
        }

        logger.info("SYSTEM SHUTDOWN COMPLETE")

    # ==========================================================
    # FINAL CYCLE WRAPPER (HARD ENTRYPOINT)
    # ==========================================================

    def execute_cycle(
        self,
        symbols: list[str],
        portfolio_state: Any,
    ) -> OrchestratorContext:

        if not self.validate_cycle(self.context):

            logger.error("CYCLE ABORTED: VALIDATION FAILED")

            return self.context

        risk_check = self.context.last_risk or {}

        if not self.pre_cycle_guard(portfolio_state, risk_check):

            logger.warning("CYCLE BLOCKED BY SAFETY LAYER")

            return self.context

        return self.run_cycle(symbols, portfolio_state)

    # ==========================================================
    # SYSTEM HEALTH STATUS
    # ==========================================================

    def system_health(self) -> dict[str, Any]:

        return {
            "mode": self.mode,
            "cycle_id": self.context.cycle_id,
            "last_execution_status": getattr(
                self.context.last_execution, "status", None
            ),
            "has_data": self.context.last_data is not None,
            "has_signals": self.context.last_signals is not None,
            "has_decision": self.context.last_decision is not None,
            "has_portfolio": self.context.last_portfolio is not None,
            "has_analytics": self.context.last_analytics is not None,
        }


# ==========================================================
# FINAL ENTRYPOINT (PRODUCTION RUN HOOK)
# ==========================================================


def main():

    orchestrator = WiredOrchestrator(mode="BACKTEST")

    symbols = ["AAPL", "MSFT", "TSLA"]

    portfolio_state = {"equity": 100000}

    try:

        orchestrator.run_backtest(
            historical_data={},  # injected externally
            portfolio_state=portfolio_state,
        )

    except KeyboardInterrupt:

        orchestrator.shutdown()


# ==========================================================
# END OF ORCHESTRATOR
# ==========================================================
