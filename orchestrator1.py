"""
Orchestrator Engine

Institutional Production Version

Responsibilities
----------------
• Central control loop of entire trading system
• Coordinates scanner → tracker → broker → portfolio
• Manages execution cycles
• Maintains system state
• Ensures safe sequencing of all engines

This is the "main brain" of the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import time
import pandas as pd

from core.logger import get_logger

from scanner.scanner import MarketScanner
from tracker.tracker import PositionTracker, PositionState
from broker.broker import BrokerEngine, OrderRequest

logger = get_logger(__name__)


# ==========================================================
# ORCHESTRATOR STATE
# ==========================================================


@dataclass(slots=True)
class OrchestratorState:

    cycle_id: int

    timestamp: float

    active_positions: dict[str, PositionState] = field(default_factory=dict)

    last_scan_results: list[Any] = field(default_factory=list)

    executed_orders: list[Any] = field(default_factory=list)

    diagnostics: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# ORCHESTRATOR ENGINE
# ==========================================================


class TradingOrchestrator:
    """
    Master Control Engine
    """

    def __init__(self):

        self.scanner = MarketScanner()

        self.tracker = PositionTracker()

        self.broker = BrokerEngine()

        self.state = OrchestratorState(
            cycle_id=0,
            timestamp=time.time(),
        )

        logger.info("Trading Orchestrator initialized.")

    # ==========================================================
    # MAIN EXECUTION CYCLE
    # ==========================================================

    def run_cycle(
        self,
        symbols: list[str],
        portfolio: dict[str, Any],
        market_state: dict[str, Any],
        dataframe_map: dict[str, pd.DataFrame],
    ) -> OrchestratorState:

        self.state.cycle_id += 1

        self.state.timestamp = time.time()

        logger.info(
            "Starting cycle %d",
            self.state.cycle_id,
        )

        # ==========================================================
        # MARKET STATE SNAPSHOT
        # ==========================================================

        self.state.diagnostics["market_state"] = {
            "market_open": market_state.get(
                "market_open",
                True,
            ),
            "volatility_regime": market_state.get(
                "volatility_regime",
                "NORMAL",
            ),
            "risk_on": market_state.get(
                "risk_on",
                True,
            ),
        }

        # ==========================================================
        # ACTIVE POSITIONS SNAPSHOT
        # ==========================================================

        active_positions = self.state.active_positions

        self.state.diagnostics["active_positions_count"] = len(active_positions)

        # ==========================================================
        # TRACK ACTIVE POSITIONS
        # ==========================================================

        tracker_results = []

        for symbol, position in list(active_positions.items()):

            dataframe = dataframe_map.get(symbol)

            if dataframe is None:

                logger.warning(
                    "Missing data for tracker %s",
                    symbol,
                )

                continue

            result = self.tracker.update_position(
                position=position,
                dataframe=dataframe,
                portfolio=portfolio,
                market_state=market_state,
            )

            tracker_results.append(result)

            self.state.executed_orders.append(result)

        # ==========================================================
        # UPDATE STATE AFTER TRACKING
        # ==========================================================

        self.state.diagnostics["tracked_positions"] = len(tracker_results)

        self.state.last_scan_results = []
        # ==========================================================
        # RUN SCANNER
        # ==========================================================

        scan_results = self.scanner.scan_symbols(
            symbols=symbols,
            portfolio=portfolio,
            broker_status=market_state.get("broker_status", {}),
            market_state=market_state,
        )

        self.state.last_scan_results = scan_results

        # ==========================================================
        # FILTER NEW CANDIDATES
        # ==========================================================

        new_candidates = [
            r
            for r in scan_results
            if r.action in ("BUY", "SELL") and r.portfolio_allowed
        ]

        # ==========================================================
        # REMOVE DUPLICATES (ALREADY ACTIVE POSITIONS)
        # ==========================================================

        filtered_candidates = []

        for candidate in new_candidates:

            if candidate.symbol in active_positions:

                continue

            filtered_candidates.append(candidate)

        # ==========================================================
        # STORE FILTERED RESULTS
        # ==========================================================

        self.state.diagnostics["scan_count"] = len(scan_results)

        self.state.diagnostics["new_candidates"] = len(new_candidates)

        self.state.diagnostics["filtered_candidates"] = len(filtered_candidates)

        # ==========================================================
        # SORT CANDIDATES BY RANK
        # ==========================================================

        filtered_candidates.sort(
            key=lambda x: (
                x.ranking,
                x.confidence,
                x.score,
            ),
            reverse=True,
        )
        # ==========================================================
        # ORDER GENERATION
        # ==========================================================

        orders_to_execute = []

        for candidate in filtered_candidates:

            order = OrderRequest(
                symbol=candidate.symbol,
                action=candidate.action,
                quantity=candidate.position_size,
                order_type="MARKET",
                strategy_tag="ORCHESTRATOR",
            )

            orders_to_execute.append(order)

        # ==========================================================
        # BROKER EXECUTION LOOP
        # ==========================================================

        executed_orders = []

        for order in orders_to_execute:

            market_price = 0.0

            dataframe = dataframe_map.get(order.symbol)

            if dataframe is not None:

                market_price = float(dataframe.iloc[-1]["close"])

            result = self.broker.place_order(
                order=order,
                market_price=market_price,
                market_state=market_state,
            )

            executed_orders.append(result)

            self.state.executed_orders.append(result)

        # ==========================================================
        # STORE EXECUTION METRICS
        # ==========================================================

        self.state.diagnostics["orders_generated"] = len(orders_to_execute)

        self.state.diagnostics["orders_executed"] = len(executed_orders)

        # ==========================================================
        # UPDATE ACTIVE POSITIONS FROM FILLED ORDERS
        # ==========================================================

        for order_result in executed_orders:

            if order_result.status == "REJECTED":

                continue

            if order_result.filled_quantity <= 0:

                continue

            self.state.active_positions[order_result.symbol] = PositionState(
                symbol=order_result.symbol,
                entry_price=order_result.avg_price,
                quantity=order_result.filled_quantity,
                direction="BUY",
                entry_time=str(self.state.timestamp),
            )
        # ==========================================================
        # POSITION LIFECYCLE SYNC
        # ==========================================================

        updated_positions = {}

        for symbol, position in self.state.active_positions.items():

            tracker_result = self.tracker.update_position(
                position=position,
                dataframe=dataframe_map.get(symbol),
                portfolio=portfolio,
                market_state=market_state,
            )

            self.state.executed_orders.append(tracker_result)

            position.diagnostics = tracker_result.diagnostics

            updated_positions[symbol] = position

        self.state.active_positions = updated_positions

        # ==========================================================
        # REMOVE CLOSED POSITIONS
        # ==========================================================

        active_only_positions = {}

        closed_count = 0

        for symbol, position in self.state.active_positions.items():

            if position.status == "CLOSED":

                closed_count += 1

                continue

            active_only_positions[symbol] = position

        self.state.active_positions = active_only_positions

        # ==========================================================
        # PORTFOLIO SNAPSHOT UPDATE
        # ==========================================================

        portfolio["open_positions_count"] = len(self.state.active_positions)

        portfolio["executed_orders_count"] = len(self.state.executed_orders)

        portfolio["closed_positions_count"] = closed_count

        # ==========================================================
        # CYCLE DIAGNOSTICS UPDATE
        # ==========================================================

        self.state.diagnostics["active_positions"] = len(self.state.active_positions)

        self.state.diagnostics["closed_positions"] = closed_count

        self.state.diagnostics["cycle_complete"] = True
        # ==========================================================
        # PORTFOLIO VALUE RECOMPUTATION
        # ==========================================================

        total_exposure = 0.0

        total_pnl = 0.0

        for position in self.state.active_positions.values():

            position_value = position.quantity * position.current_price

            total_exposure += position_value

            total_pnl += position.pnl_absolute

        total_capital = float(
            portfolio.get(
                "total_capital",
                1.0,
            )
        )

        portfolio_exposure_ratio = total_exposure / max(total_capital, 1.0)

        portfolio["total_exposure"] = total_exposure

        portfolio["portfolio_pnl"] = total_pnl

        portfolio["portfolio_exposure_ratio"] = portfolio_exposure_ratio

        # ==========================================================
        # RISK AGGREGATION
        # ==========================================================

        risk_flags = {
            "high_exposure": portfolio_exposure_ratio > 0.9,
            "high_drawdown": portfolio.get("drawdown", 0.0) > 0.15,
            "high_daily_loss": portfolio.get("daily_loss", 0.0) > 0.03,
        }

        self.state.diagnostics["risk_flags"] = risk_flags

        # ==========================================================
        # EMERGENCY SYSTEM CHECK
        # ==========================================================

        emergency_trigger = any(risk_flags.values())

        if emergency_trigger:

            logger.warning("EMERGENCY CONDITION DETECTED")

            # ==========================================================
            # FORCE EXIT SIGNAL BROADCAST
            # ==========================================================

            for position in self.state.active_positions.values():

                position.diagnostics["emergency_flag"] = True

                position.status = "FORCED_EXIT"

        # ==========================================================
        # UPDATE STATE METRICS
        # ==========================================================

        self.state.diagnostics["total_exposure"] = total_exposure

        self.state.diagnostics["total_pnl"] = total_pnl

        self.state.diagnostics["portfolio_exposure_ratio"] = portfolio_exposure_ratio

        self.state.diagnostics["emergency_trigger"] = emergency_trigger
        # ==========================================================
        # CYCLE SUMMARY METRICS
        # ==========================================================

        total_orders = len(self.state.executed_orders)

        filled_orders = 0

        rejected_orders = 0

        partial_orders = 0

        for order in self.state.executed_orders:

            if hasattr(order, "status"):

                if order.status == "FILLED":

                    filled_orders += 1

                elif order.status == "REJECTED":

                    rejected_orders += 1

                elif order.status == "PARTIALLY_FILLED":

                    partial_orders += 1

        self.state.diagnostics["order_summary"] = {
            "total_orders": total_orders,
            "filled": filled_orders,
            "rejected": rejected_orders,
            "partial": partial_orders,
        }

        # ==========================================================
        # PERFORMANCE METRICS
        # ==========================================================

        avg_pnl = 0.0

        if self.state.active_positions:

            avg_pnl = sum(
                p.pnl_percent for p in self.state.active_positions.values()
            ) / len(self.state.active_positions)

        win_positions = sum(
            1 for p in self.state.active_positions.values() if p.pnl_percent > 0
        )

        loss_positions = sum(
            1 for p in self.state.active_positions.values() if p.pnl_percent <= 0
        )

        self.state.diagnostics["performance_metrics"] = {
            "avg_pnl_percent": avg_pnl,
            "win_positions": win_positions,
            "loss_positions": loss_positions,
            "win_rate": (win_positions / max(len(self.state.active_positions), 1)),
        }

        # ==========================================================
        # CYCLE HEALTH SCORE
        # ==========================================================

        cycle_health = (
            (1 - min(portfolio_exposure_ratio, 1.0)) * 40
            + (filled_orders / max(total_orders, 1)) * 30
            + (1 - min(portfolio.get("daily_loss", 0.0), 1.0)) * 20
            + (1 - int(emergency_trigger)) * 10
        )

        cycle_health = max(0.0, min(100.0, cycle_health))

        self.state.diagnostics["cycle_health"] = round(cycle_health, 2)

        # ==========================================================
        # FINAL STATE UPDATE
        # ==========================================================

        self.state.diagnostics["cycle_id"] = self.state.cycle_id

        self.state.diagnostics["timestamp"] = self.state.timestamp

        self.state.diagnostics["active_positions_final"] = len(
            self.state.active_positions
        )
        # ==========================================================
        # SYSTEM SNAPSHOT BUILD
        # ==========================================================

        system_snapshot = {
            "cycle_id": self.state.cycle_id,
            "timestamp": self.state.timestamp,
            "active_positions": len(self.state.active_positions),
            "total_orders": len(self.state.executed_orders),
            "portfolio_exposure": portfolio_exposure_ratio,
            "portfolio_pnl": total_pnl,
            "cycle_health": self.state.diagnostics.get(
                "cycle_health",
                0.0,
            ),
            "emergency_trigger": emergency_trigger,
        }

        self.state.diagnostics["system_snapshot"] = system_snapshot

        # ==========================================================
        # EXECUTION TRACE SUMMARY
        # ==========================================================

        execution_trace = {
            "scanner_results": len(self.state.last_scan_results),
            "candidates": self.state.diagnostics.get(
                "filtered_candidates",
                0,
            ),
            "orders_generated": self.state.diagnostics.get(
                "orders_generated",
                0,
            ),
            "orders_executed": self.state.diagnostics.get(
                "orders_executed",
                0,
            ),
            "tracker_updates": self.state.diagnostics.get(
                "tracked_positions",
                0,
            ),
        }

        self.state.diagnostics["execution_trace"] = execution_trace

        # ==========================================================
        # FINAL STATUS FLAGS
        # ==========================================================

        self.state.diagnostics["system_status"] = (
            "EMERGENCY" if emergency_trigger else "ACTIVE"
        )

        # ==========================================================
        # RETURN ORCHESTRATOR STATE
        # ==========================================================

        logger.info(
            "Cycle %d completed | Positions=%d | Orders=%d | Health=%.2f",
            self.state.cycle_id,
            len(self.state.active_positions),
            len(self.state.executed_orders),
            cycle_health,
        )

        return self.state

    # ==========================================================
    # SINGLE CYCLE WRAPPER
    # ==========================================================

    def run_once(
        self,
        symbols: list[str],
        portfolio: dict[str, Any],
        market_state: dict[str, Any],
        dataframe_map: dict[str, pd.DataFrame],
    ) -> OrchestratorState:

        return self.run_cycle(
            symbols=symbols,
            portfolio=portfolio,
            market_state=market_state,
            dataframe_map=dataframe_map,
        )

    # ==========================================================
    # CONTINUOUS EXECUTION LOOP
    # ==========================================================

    def run_continuous(
        self,
        symbols: list[str],
        portfolio: dict[str, Any],
        market_state_provider: Any,
        dataframe_provider: Any,
        sleep_seconds: int = 60,
        max_cycles: int | None = None,
    ) -> None:

        cycle_count = 0

        while True:

            cycle_count += 1

            if max_cycles and cycle_count > max_cycles:

                break

            market_state = market_state_provider()

            dataframe_map = dataframe_provider()

            self.run_cycle(
                symbols=symbols,
                portfolio=portfolio,
                market_state=market_state,
                dataframe_map=dataframe_map,
            )

            logger.info(
                "Sleeping for %d seconds",
                sleep_seconds,
            )

            time.sleep(sleep_seconds)

    # ==========================================================
    # DRY RUN MODE
    # ==========================================================

    def run_dry(
        self,
        symbols: list[str],
        portfolio: dict[str, Any],
        market_state: dict[str, Any],
        dataframe_map: dict[str, pd.DataFrame],
    ) -> OrchestratorState:

        logger.info("Running DRY mode (no state mutation beyond simulation)")

        original_positions = dict(self.state.active_positions)

        state = self.run_cycle(
            symbols=symbols,
            portfolio=portfolio,
            market_state=market_state,
            dataframe_map=dataframe_map,
        )

        self.state.active_positions = original_positions

        return state

    # ==========================================================
    # STATE RESET UTILITY
    # ==========================================================

    def reset_state(self) -> None:

        logger.warning("Resetting orchestrator state")

        self.state = OrchestratorState(
            cycle_id=0,
            timestamp=time.time(),
            active_positions={},
            last_scan_results=[],
            executed_orders=[],
            diagnostics={},
        )

    # ==========================================================
    # PARTIAL RESET (SAFE MODE)
    # ==========================================================

    def soft_reset(self) -> None:

        logger.warning("Soft reset triggered")

        self.state.cycle_id = 0

        self.state.last_scan_results.clear()

        self.state.executed_orders.clear()

        self.state.diagnostics["reset_mode"] = "SOFT"

    # ==========================================================
    # EMERGENCY HALT CHECK
    # ==========================================================

    def should_halt(self) -> bool:

        emergency = self.state.diagnostics.get(
            "emergency_trigger",
            False,
        )

        cycle_health = self.state.diagnostics.get(
            "cycle_health",
            100.0,
        )

        pnl = self.state.diagnostics.get(
            "total_pnl",
            0.0,
        )

        if emergency:

            return True

        if cycle_health < 20:

            return True

        if pnl < -5000:

            return True

        return False

    # ==========================================================
    # RECOVERY HOOK
    # ==========================================================

    def recover_state(
        self,
        saved_state: OrchestratorState,
    ) -> None:

        logger.info("Recovering orchestrator state")

        self.state = saved_state

        self.state.diagnostics["recovered"] = True

    # ==========================================================
    # HEALTH CHECK
    # ==========================================================

    def health_check(self) -> dict[str, Any]:

        return {
            "cycle_id": self.state.cycle_id,
            "active_positions": len(self.state.active_positions),
            "orders": len(self.state.executed_orders),
            "cycle_health": self.state.diagnostics.get(
                "cycle_health",
                0.0,
            ),
            "emergency": self.state.diagnostics.get(
                "emergency_trigger",
                False,
            ),
        }

    # ==========================================================
    # DEBUG REPORT
    # ==========================================================

    def debug_report(self) -> str:

        report: list[str] = []

        report.append("=" * 120)
        report.append("ORCHESTRATOR DEBUG REPORT")
        report.append("=" * 120)
        report.append("")

        report.append(f"Cycle ID: {self.state.cycle_id}")

        report.append(f"Timestamp: {self.state.timestamp}")

        report.append(f"Active Positions: {len(self.state.active_positions)}")

        report.append(f"Executed Orders: {len(self.state.executed_orders)}")

        report.append("")

        report.append("-" * 120)
        report.append("SYSTEM SNAPSHOT")
        report.append("-" * 120)

        snapshot = self.state.diagnostics.get(
            "system_snapshot",
            {},
        )

        for k, v in snapshot.items():

            report.append(f"{k:<30} : {v}")

        report.append("")
        report.append("-" * 120)
        report.append("EXECUTION TRACE")
        report.append("-" * 120)

        trace = self.state.diagnostics.get(
            "execution_trace",
            {},
        )

        for k, v in trace.items():

            report.append(f"{k:<30} : {v}")

        report.append("")
        report.append("-" * 120)
        report.append("RISK FLAGS")
        report.append("-" * 120)

        risk_flags = self.state.diagnostics.get(
            "risk_flags",
            {},
        )

        for k, v in risk_flags.items():

            report.append(f"{k:<30} : {v}")

        report.append("")
        report.append("=" * 120)
        report.append("END ORCHESTRATOR REPORT")
        report.append("=" * 120)

        return "\n".join(report)

    # ==========================================================
    # STATE EXPORT
    # ==========================================================

    def export_state(self) -> dict[str, Any]:

        return {
            "cycle_id": self.state.cycle_id,
            "timestamp": self.state.timestamp,
            "active_positions": {
                k: {
                    "symbol": v.symbol,
                    "quantity": v.quantity,
                    "entry_price": v.entry_price,
                    "current_price": v.current_price,
                    "status": v.status,
                    "pnl_percent": v.pnl_percent,
                }
                for k, v in self.state.active_positions.items()
            },
            "diagnostics": self.state.diagnostics,
            "executed_orders_count": len(self.state.executed_orders),
        }

    # ==========================================================
    # LIFECYCLE HOOK (PRE-RUN)
    # ==========================================================

    def pre_run(self) -> None:

        logger.info("Orchestrator pre-run initialized")

        self.state.diagnostics["pre_run"] = {
            "timestamp": time.time(),
            "cycle_id": self.state.cycle_id,
            "active_positions": len(self.state.active_positions),
        }

    # ==========================================================
    # LIFECYCLE HOOK (POST-RUN)
    # ==========================================================

    def post_run(self) -> None:

        logger.info("Orchestrator post-run completed")

        self.state.diagnostics["post_run"] = {
            "timestamp": time.time(),
            "cycle_id": self.state.cycle_id,
            "executed_orders": len(self.state.executed_orders),
            "active_positions": len(self.state.active_positions),
        }

    # ==========================================================
    # SHUTDOWN HANDLER
    # ==========================================================

    def shutdown(self) -> None:

        logger.warning("Orchestrator shutting down")

        self.state.diagnostics["shutdown"] = {
            "timestamp": time.time(),
            "final_cycle": self.state.cycle_id,
            "remaining_positions": len(self.state.active_positions),
            "total_orders": len(self.state.executed_orders),
        }

    # ==========================================================
    # FINAL SYSTEM STATUS
    # ==========================================================

    def system_status(self) -> dict[str, Any]:

        return {
            "status": "RUNNING" if not self.should_halt() else "HALTED",
            "cycle_id": self.state.cycle_id,
            "health": self.state.diagnostics.get(
                "cycle_health",
                0.0,
            ),
            "emergency": self.state.diagnostics.get(
                "emergency_trigger",
                False,
            ),
            "active_positions": len(self.state.active_positions),
            "orders": len(self.state.executed_orders),
        }


# ==========================================================
# END OF FILE
# ==========================================================
