"""
WIRED ORCHESTRATOR (PRODUCTION FLOW)

Single controlled execution pipeline synchronized with the frozen contract.
NO module is allowed to bypass this flow.

Modes:
- LIVE
- BACKTEST
- PAPER
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

# Core System Utilities
from core.logger import get_logger

# Fixed Package Internal Mapping Layer
from data.data_engine import DataEngine
from features.feature_engineering import FeatureEngineeringEngine
from market.market_regime import MarketRegimeEngine
from strategy.buy_strategy import BuyStrategyEngine
from strategy.sell_strategy import SellStrategyEngine
from decision.decision_engine import DecisionEngine
from risk.risk_manager import RiskManager
from execution.scanner import (
    MarketScanner,
)  # Fixed: Imported correct class matching folder
from execution.broker import BrokerEngine
from portfolio.portfolio import PortfolioEngine
from analytics.analytics import AnalyticsEngine

logger = get_logger(__name__)


# ==========================================================
# ORCHESTRATOR STATE CONTEXT
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
    last_portfolio: Dict[str, Any] = None
    last_analytics: Any = None


# ==========================================================
# WIRED ORCHESTRATOR CORE
# ==========================================================


class WiredOrchestrator:
    def __init__(self, mode: str = "BACKTEST"):
        self.mode = mode

        # Core Engines Initialization
        self.data_engine = DataEngine()
        self.feature_engine = FeatureEngineeringEngine()
        self.market = MarketRegimeEngine()
        self.buy_strategy = BuyStrategyEngine()
        self.sell_strategy = SellStrategyEngine()
        self.decision_engine = DecisionEngine()
        self.risk_engine = RiskManager()
        self.scanner = MarketScanner()  # Fixed reference to standard interface
        self.broker = BrokerEngine()
        self.portfolio = PortfolioEngine()
        self.analytics = AnalyticsEngine()

        self.context = OrchestratorContext(mode=mode)
        logger.info(f"WIRED ORCHESTRATOR successfully initialized in {mode} mode")

    # ==========================================================
    # MAIN SEQUENTIAL PIPELINE LOOP
    # ==========================================================

    def run_cycle(
        self, symbols: list[str], portfolio_state: Any
    ) -> OrchestratorContext:
        self.context.cycle_id += 1
        self.context.timestamp = time.time()

        logger.info(
            f"--- Triggering Execution Matrix Sequence for Cycle: {self.context.cycle_id} ---"
        )

        # STEP 1: DATA FETCH
        data = self.data_engine.get_market_data(symbols)
        self.context.last_data = data
        if not data or (isinstance(data, dict) and len(data) == 0):
            logger.error(
                "Cycle processing suspended: Empty data structure returned from engine context."
            )
            return self.context

        # STEP 2: FEATURE GENERATION
        features = self.feature_engine.build_features(data)
        self.context.last_features = features

        # STEP 3: MARKET REGIME DETECTION
        regime = self.market.detect(features)
        self.context.last_risk = {"regime": regime}

        # STEP 4: STRATEGY SIGNAL GENERATION (Synchronized Parameters)
        buy_signals = self.buy_strategy.generate(features=features, regime=regime)
        sell_signals = self.sell_strategy.generate(features=features, regime=regime)

        signals = {"buy": buy_signals, "sell": sell_signals}
        self.context.last_signals = signals

        # STEP 5: DECISION ENGINE INTEGRATION
        decision = self.decision_engine.evaluate(
            signals=signals,
            portfolio=portfolio_state,
            regime=regime,
        )
        self.context.last_decision = decision

        # STEP 6: RISK ENGINE VALUATION GATEWAY
        risk_check = self.risk_engine.evaluate_order(
            validation=self.decision_engine.validate(decision),
            decision=decision,
            portfolio=portfolio_state,
            market={"regime": regime, "event_day": False, "vix": 20},
        )
        self.context.last_risk = risk_check

        if not risk_check.get("approved", False):
            logger.warning(
                f"Risk controls blocked execution pipeline commands at cycle iteration: {self.context.cycle_id}"
            )
            self.context.last_execution = {"status": "BLOCKED", "reason": risk_check}
            return self.context

        # STEP 7: MASTER OPPORTUNITIES SCANNER AND SIZING LAYER
        # Hand off execution metadata securely into our compiled MarketScanner matrix
        broker_status = {"status": "ONLINE", "mode": self.mode}
        market_state = {
            "regime": regime,
            "max_trade_candidates": 20,
            "max_watchlist": 50,
        }

        scan_candidates = self.scanner.scan_symbols(
            symbols=symbols,
            portfolio=(
                portfolio_state
                if isinstance(portfolio_state, dict)
                else getattr(portfolio_state, "__dict__", {})
            ),
            broker_status=broker_status,
            market_state=market_state,
        )

        # STEP 8: ORDER SUBMISSION TO BROKER ENGINE
        executed_orders = []
        for order in scan_candidates:
            if order.action in ["BUY", "SELL"] and order.portfolio_allowed:
                result = self.broker.execute_order(order=order, mode=self.mode)
                executed_orders.append(result)

        self.context.last_execution = {"status": "EXECUTED", "orders": executed_orders}

        # STEP 9: PORTFOLIO STATE UPDATE TRACKING
        updated_portfolio = self.portfolio.update_from_broker(
            executed_orders=executed_orders, portfolio_state=portfolio_state
        )

        # STEP 10: PORTFOLIO RECONCILIATION
        self.portfolio.reconcile(
            broker_updates=executed_orders, portfolio_state=updated_portfolio
        )

        # STEP 11: FINANCIAL MATRIX SNAPSHOTS
        portfolio_snapshot = self.portfolio.snapshot()
        equity = portfolio_snapshot.get("equity", 100000.0)
        pnl = portfolio_snapshot.get("total_pnl", 0.0)

        self.context.last_portfolio = {
            "equity": equity,
            "pnl": pnl,
            "exposure": portfolio_snapshot.get("exposure", 0.0),
        }

        # STEP 12: DEEP POST-CYCLE POST-MORTEM ANALYTICS
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

        # STEP 13: LOG SUCCESS SNAPSHOT
        logger.info(
            f"System execution pass complete. Status -> Cycle: {self.context.cycle_id} | Equity: {equity:.2f} | Realized PnL: {pnl:.2f}"
        )
        return self.context

    # ==========================================================
    # AUTOMATED LOOP RUN TIME MANAGERS
    # ==========================================================

    def run_live(
        self, symbols: list[str], portfolio_state: Any, sleep_seconds: int = 60
    ) -> None:
        logger.info("CRITICAL NODE OVERRIDE: LIVE MODE TRANSACTION PROCESSING DEPLOYED")
        while True:
            try:
                if not self.execute_cycle_safely(symbols, portfolio_state):
                    break
                time.sleep(sleep_seconds)
            except Exception as e:
                logger.exception(f"LIVE TERMINATION EXCEPTION TRACE DETECTED: {str(e)}")
                self.emergency_stop()
                break

    def run_backtest(
        self, historical_data: dict[str, Any], portfolio_state: Any
    ) -> list[OrchestratorContext]:
        logger.info("HISTORICAL PERFORMANCE SIMULATION ENGINE TRIGGERED")
        results = []
        if not historical_data:
            logger.warning(
                "No historical feed records passed to simulator. Exiting run loop execution context."
            )
            return results

        total_steps = len(next(iter(historical_data.values())))
        for step in range(total_steps):
            sliced_data = {k: v.iloc[: step + 1] for k, v in historical_data.items()}
            self.context.timestamp = time.time()
            ctx = self.run_cycle(
                symbols=list(historical_data.keys()), portfolio_state=portfolio_state
            )
            results.append(ctx)
        return results

    def run_paper(
        self, symbols: list[str], portfolio_state: Any, sleep_seconds: int = 10
    ) -> None:
        logger.info("PAPER RUN STATE MACHINE INITIATED")
        while True:
            self.run_cycle(symbols=symbols, portfolio_state=portfolio_state)
            time.sleep(sleep_seconds)

    def execute_cycle_safely(self, symbols: list[str], portfolio_state: Any) -> bool:
        if not self.validate_cycle(self.context):
            logger.error(
                "Cycle processing abandoned: Engine configuration checking parameters failed."
            )
            return False

        risk_check = self.context.last_risk or {}
        if not self.pre_cycle_guard(portfolio_state, risk_check):
            logger.warning("Pipeline processing terminated at security firewall.")
            return False

        self.run_cycle(symbols, portfolio_state)
        return True

    # ==========================================================
    # SYSTEM FIREWALL PROTECTION CIRCUITS
    # ==========================================================

    def emergency_stop(self) -> None:
        logger.critical(
            "EMERGENCY PROCESS SAFETY OVERRIDE ACTIVE -> ISOLATING TRANSACTIONS"
        )
        self.context.last_execution = {
            "status": "STOPPED",
            "reason": "EMERGENCY_TRIGGER",
        }

    def kill_switch(self, portfolio_state: Any) -> bool:
        equity = (
            getattr(portfolio_state, "equity", 100000.0)
            if not isinstance(portfolio_state, dict)
            else portfolio_state.get("equity", 100000.0)
        )
        pnl = (
            getattr(portfolio_state, "total_pnl", 0.0)
            if not isinstance(portfolio_state, dict)
            else portfolio_state.get("total_pnl", 0.0)
        )
        exposure = (
            getattr(portfolio_state, "exposure", 0.0)
            if not isinstance(portfolio_state, dict)
            else portfolio_state.get("exposure", 0.0)
        )

        if pnl < -0.05 * equity:
            logger.critical(
                "CRITICAL ACCOUNT LOSS MARGIN INFRINGEMENT DETECTED -> SHUTTING DOWN PIPELINE"
            )
            return True
        if exposure > 0.95:
            logger.critical(
                "PORTFOLIO ASSET OVER-EXPOSURE SAFETY TRIGGERED -> DISABLING PIPELINE ORDERS"
            )
            return True
        return False

    def circuit_breaker(self, risk_check: dict[str, Any]) -> bool:
        if risk_check.get("volatility", 0.0) > 0.03:
            logger.warning(
                "CIRCUIT BREAKER ENGAGED: MAXIMUM ASSET SYSTEM VOLATILITY BREACHED"
            )
            return True
        if risk_check.get("liquidity_risk", False):
            logger.warning("CIRCUIT BREAKER ENGAGED: MARKET DEPTH FAILURE DETECTED")
            return True
        return False

    def validate_cycle(self, context: OrchestratorContext) -> bool:
        if context.cycle_id > 0 and context.last_data is None:
            logger.error(
                "DATA INTEGRITY VERIFICATION FAILURE: Empty reference arrays detected."
            )
            return False
        return True

    def pre_cycle_guard(self, portfolio_state: Any, risk_check: dict[str, Any]) -> bool:
        if self.kill_switch(portfolio_state):
            self.emergency_stop()
            return False
        if self.circuit_breaker(risk_check):
            self.emergency_stop()
            return False
        return True

    def debug_snapshot(self) -> dict[str, Any]:
        return {
            "cycle_id": self.context.cycle_id,
            "mode": self.mode,
            "context": self.context,
            "portfolio": self.context.last_portfolio,
            "analytics": self.context.last_analytics,
        }

    def shutdown(self) -> None:
        logger.info("SYSTEM DISCONNECT AND GRACEFUL SHUTDOWN INITIATED")
        self.context.last_execution = {
            "status": "SHUTDOWN",
            "reason": "MANUAL_OR_SYSTEM_EXIT",
        }
        logger.info("PLATFORM DOWNSTREAM SERVICES DE-ACTIVATED CLEANLY")


def main():
    orchestrator = WiredOrchestrator(mode="BACKTEST")
    symbols = ["AAPL", "MSFT", "TSLA"]
    portfolio_state = {"equity": 100000, "total_pnl": 0.0, "exposure": 0.0}
    try:
        orchestrator.run_cycle(symbols=symbols, portfolio_state=portfolio_state)
    except KeyboardInterrupt:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
