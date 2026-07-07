"""
Position Tracker Engine

Institutional Production Version

Responsibilities
----------------
• Track all open positions in real time
• Update price, PnL, highs/lows
• Maintain trade state
• Feed exit strategy engine
• Trigger risk updates
• Generate portfolio-level position snapshot

This engine does NOT:
• Open trades
• Close trades directly
• Modify strategy logic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.logger import get_logger

from risk.exit_strategy import ExitStrategyEngine
from risk.risk_manager import RiskManager
from risk.position_sizing import PositionSizingEngine

logger = get_logger(__name__)


# ==========================================================
# POSITION STATE
# ==========================================================


@dataclass(slots=True)
class PositionState:

    symbol: str

    entry_price: float

    quantity: int

    direction: str  # BUY / SELL

    entry_time: str

    current_price: float = 0.0

    pnl_percent: float = 0.0

    pnl_absolute: float = 0.0

    highest_price: float = 0.0

    lowest_price: float = 0.0

    holding_days: int = 0

    status: str = "ACTIVE"

    diagnostics: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# TRACKER RESULT
# ==========================================================


@dataclass(slots=True)
class TrackerResult:

    symbol: str

    action: str

    pnl_percent: float

    pnl_absolute: float

    status: str

    exit_signal: str

    diagnostics: dict[str, Any] = field(default_factory=dict)


# ==========================================================
# TRACKER ENGINE
# ==========================================================


class PositionTracker:
    """
    Live Position Tracking Engine
    """

    def __init__(self):

        self.exit_engine = ExitStrategyEngine()

        self.risk_engine = RiskManager()

        self.sizing_engine = PositionSizingEngine()

        logger.info("Position Tracker initialized.")

    def update_position(
        self,
        position: PositionState,
        dataframe: pd.DataFrame,
        portfolio: dict[str, Any],
        market_state: dict[str, Any],
    ) -> TrackerResult:

        logger.info(
            "Tracking %s",
            position.symbol,
        )

        diagnostics: dict[str, Any] = {}

        latest = dataframe.iloc[-1]
        # ==========================================================
        # PRICE UPDATE
        # ==========================================================

        current_price = float(
            latest.get(
                "close",
                position.entry_price,
            )
        )

        position.current_price = current_price

        diagnostics["current_price"] = round(
            current_price,
            2,
        )

        # ==========================================================
        # HIGH / LOW TRACKING
        # ==========================================================

        if position.highest_price == 0.0:

            position.highest_price = current_price

        if position.lowest_price == 0.0:

            position.lowest_price = current_price

        position.highest_price = max(
            position.highest_price,
            current_price,
        )

        position.lowest_price = min(
            position.lowest_price,
            current_price,
        )

        diagnostics["highest_price"] = round(
            position.highest_price,
            2,
        )

        diagnostics["lowest_price"] = round(
            position.lowest_price,
            2,
        )

        # ==========================================================
        # PnL CALCULATION
        # ==========================================================

        price_diff = current_price - position.entry_price

        if position.direction == "SELL":

            price_diff *= -1

        position.pnl_absolute = price_diff * position.quantity

        position.pnl_percent = (
            price_diff
            / max(
                position.entry_price,
                0.01,
            )
        ) * 100

        diagnostics["pnl_absolute"] = round(
            position.pnl_absolute,
            2,
        )

        diagnostics["pnl_percent"] = round(
            position.pnl_percent,
            2,
        )

        # ==========================================================
        # HOLDING TIME UPDATE
        # ==========================================================

        position.holding_days = int(position.holding_days) + 1

        diagnostics["holding_days"] = position.holding_days

        # ==========================================================
        # STATUS UPDATE
        # ==========================================================

        if position.pnl_percent <= -risk_engine.MAX_DAILY_LOSS * 100:

            position.status = "RISK_STOP"

        diagnostics["status"] = position.status
        # ==========================================================
        # EXIT STRATEGY EVALUATION
        # ==========================================================

        exit_decision = self.exit_engine.evaluate(
            decision={"action": position.direction},
            risk=self.risk_engine,
            sizing=self.sizing_engine,
            dataframe=dataframe,
            position={
                "entry_price": position.entry_price,
                "holding_days": position.holding_days,
                "highest_price": position.highest_price,
                "lowest_price": position.lowest_price,
            },
        )

        diagnostics["exit_action"] = exit_decision.action

        diagnostics["exit_percent"] = exit_decision.exit_percent

        diagnostics["exit_confidence"] = exit_decision.confidence

        # ==========================================================
        # STOP LOSS CHECK
        # ==========================================================

        stop_triggered = False

        if position.direction == "BUY":

            if current_price <= exit_decision.stop_loss:

                stop_triggered = True

        else:

            if current_price >= exit_decision.stop_loss:

                stop_triggered = True

        diagnostics["stop_triggered"] = stop_triggered

        # ==========================================================
        # TRAILING STOP CHECK
        # ==========================================================

        trailing_triggered = False

        if position.direction == "BUY":

            if current_price <= exit_decision.trailing_stop:

                trailing_triggered = True

        else:

            if current_price >= exit_decision.trailing_stop:

                trailing_triggered = True

        diagnostics["trailing_triggered"] = trailing_triggered

        # ==========================================================
        # TAKE PROFIT CHECK
        # ==========================================================

        target_hit = False

        if position.direction == "BUY":

            if current_price >= exit_decision.take_profit:

                target_hit = True

        else:

            if current_price <= exit_decision.take_profit:

                target_hit = True

        diagnostics["target_hit"] = target_hit

        # ==========================================================
        # EXIT SIGNAL AGGREGATION
        # ==========================================================

        exit_signal = "HOLD"

        if stop_triggered:

            exit_signal = "STOP_EXIT"

        elif trailing_triggered:

            exit_signal = "TRAIL_EXIT"

        elif target_hit:

            exit_signal = "TARGET_EXIT"

        elif exit_decision.action == "PARTIAL_EXIT":

            exit_signal = "PARTIAL_EXIT"

        diagnostics["exit_signal"] = exit_signal
        # ==========================================================
        # POSITION STATE TRANSITION
        # ==========================================================

        exit_reason = None

        if exit_signal == "STOP_EXIT":

            position.status = "CLOSED"

            exit_reason = "Stop-loss triggered"

        elif exit_signal == "TRAIL_EXIT":

            position.status = "CLOSED"

            exit_reason = "Trailing stop triggered"

        elif exit_signal == "TARGET_EXIT":

            position.status = "CLOSED"

            exit_reason = "Target hit"

        elif exit_signal == "PARTIAL_EXIT":

            position.status = "PARTIAL"

            exit_reason = "Partial exit triggered"

        diagnostics["exit_reason"] = exit_reason

        diagnostics["position_status"] = position.status

        # ==========================================================
        # FINAL PnL LOCK (IF CLOSED)
        # ==========================================================

        if position.status == "CLOSED":

            position.pnl_absolute = float(
                position.quantity * (current_price - position.entry_price)
            )

            if position.direction == "SELL":

                position.pnl_absolute *= -1

            position.pnl_percent = (
                position.pnl_absolute
                / max(
                    position.entry_price * position.quantity,
                    1.0,
                )
            ) * 100

        # ==========================================================
        # PARTIAL POSITION ADJUSTMENT
        # ==========================================================

        if position.status == "PARTIAL":

            reduced_qty = int(position.quantity * 0.5)

            position.quantity = max(
                reduced_qty,
                1,
            )

            diagnostics["reduced_quantity"] = position.quantity

        # ==========================================================
        # HIGHEST / LOWEST LOCK UPDATE
        # ==========================================================

        if position.status != "CLOSED":

            position.highest_price = max(
                position.highest_price,
                current_price,
            )

            position.lowest_price = min(
                position.lowest_price,
                current_price,
            )

        # ==========================================================
        # FINAL STATUS FLAGS
        # ==========================================================

        is_closed = position.status == "CLOSED"

        is_active = position.status == "ACTIVE"

        is_partial = position.status == "PARTIAL"

        diagnostics["is_closed"] = is_closed

        diagnostics["is_active"] = is_active

        diagnostics["is_partial"] = is_partial
        # ==========================================================
        # RISK RE-EVALUATION HOOK
        # ==========================================================

        updated_risk = self.risk_engine.evaluate(
            validation=None,
            decision={"action": position.direction},
            dataframe=dataframe,
            portfolio=portfolio,
            market=market_state,
        )

        diagnostics["updated_risk"] = updated_risk.total_risk

        diagnostics["risk_grade"] = updated_risk.risk_grade

        # ==========================================================
        # PORTFOLIO EXPOSURE UPDATE
        # ==========================================================

        portfolio_exposure = float(
            portfolio.get(
                "open_exposure",
                0.0,
            )
        )

        position_exposure = position.quantity * current_price

        total_capital = float(
            portfolio.get(
                "total_capital",
                1.0,
            )
        )

        updated_exposure = portfolio_exposure + (
            position_exposure
            / max(
                total_capital,
                1.0,
            )
        )

        diagnostics["portfolio_exposure"] = round(
            portfolio_exposure,
            4,
        )

        diagnostics["updated_exposure"] = round(
            updated_exposure,
            4,
        )

        # ==========================================================
        # POSITION VALUE UPDATE
        # ==========================================================

        position_value = position.quantity * current_price

        diagnostics["position_value"] = round(
            position_value,
            2,
        )

        # ==========================================================
        # UNREALIZED PnL SNAPSHOT
        # ==========================================================

        unrealized_pnl = position_value - (position.entry_price * position.quantity)

        if position.direction == "SELL":

            unrealized_pnl *= -1

        unrealized_pnl_percent = (
            unrealized_pnl
            / max(
                position.entry_price * position.quantity,
                1.0,
            )
        ) * 100

        diagnostics["unrealized_pnl"] = round(
            unrealized_pnl,
            2,
        )

        diagnostics["unrealized_pnl_percent"] = round(
            unrealized_pnl_percent,
            2,
        )

        # ==========================================================
        # TRADE LIFECYCLE SNAPSHOT
        # ==========================================================

        lifecycle = {
            "symbol": position.symbol,
            "status": position.status,
            "direction": position.direction,
            "holding_days": position.holding_days,
            "entry_price": position.entry_price,
            "current_price": current_price,
            "highest_price": position.highest_price,
            "lowest_price": position.lowest_price,
            "unrealized_pnl": round(
                unrealized_pnl,
                2,
            ),
            "unrealized_pnl_percent": round(
                unrealized_pnl_percent,
                2,
            ),
            "updated_risk": updated_risk.total_risk,
            "updated_exposure": round(
                updated_exposure,
                4,
            ),
        }

        diagnostics["lifecycle"] = lifecycle
        # ==========================================================
        # EXIT ENGINE FINAL EVALUATION
        # ==========================================================

        exit_decision = self.exit_engine.evaluate(
            decision={"action": position.direction},
            risk=updated_risk,
            sizing=self.sizing_engine,
            dataframe=dataframe,
            position={
                "entry_price": position.entry_price,
                "holding_days": position.holding_days,
                "highest_price": position.highest_price,
                "lowest_price": position.lowest_price,
                "emergency_exit": False,
            },
        )

        diagnostics["exit_action"] = exit_decision.action

        diagnostics["exit_confidence"] = exit_decision.confidence

        # ==========================================================
        # EXIT OVERRIDE LOGIC
        # ==========================================================

        override_exit = False

        override_reason = None

        if updated_risk.total_risk >= 90:

            override_exit = True

            override_reason = "Extreme risk detected"

        if updated_risk.risk_grade == "F":

            override_exit = True

            override_reason = "Risk grade failure"

        if portfolio.get("emergency_stop", False):

            override_exit = True

            override_reason = "Portfolio emergency stop"

        # ==========================================================
        # FORCE LIQUIDATION
        # ==========================================================

        force_exit = False

        if override_exit:

            force_exit = True

            position.status = "FORCED_EXIT"

            exit_decision.action = "FULL_EXIT"

            exit_decision.exit_percent = 100.0

            diagnostics["force_exit"] = True

            diagnostics["force_reason"] = override_reason

            logger.warning(
                "Force exit triggered: %s",
                override_reason,
            )

        else:

            diagnostics["force_exit"] = False

        # ==========================================================
        # STOP CONDITION RECHECK
        # ==========================================================

        stop_triggered = (
            position.status != "CLOSED"
            and position.status != "FORCED_EXIT"
            and position.direction == "BUY"
            and current_price <= exit_decision.stop_loss
        ) or (
            position.status != "CLOSED"
            and position.status != "FORCED_EXIT"
            and position.direction == "SELL"
            and current_price >= exit_decision.stop_loss
        )

        diagnostics["stop_triggered"] = stop_triggered

        # ==========================================================
        # FINAL EXIT OVERRIDE PRIORITY
        # ==========================================================

        if force_exit or stop_triggered:

            position.status = "CLOSED"

            exit_decision.action = "FULL_EXIT"

            exit_decision.exit_percent = 100.0

            diagnostics["final_exit_override"] = True

        else:

            diagnostics["final_exit_override"] = False
        # ==========================================================
        # FINAL EXIT REASON CONSOLIDATION
        # ==========================================================

        exit_reason = exit_decision.reasons[0] if exit_decision.reasons else None

        if force_exit:

            exit_reason = diagnostics.get(
                "force_reason",
                "Force exit triggered",
            )

        elif stop_triggered:

            exit_reason = "Stop-loss triggered"

        elif position.status == "CLOSED":

            exit_reason = "Position closed"

        diagnostics["exit_reason"] = exit_reason

        # ==========================================================
        # STATUS NORMALIZATION
        # ==========================================================

        if position.status == "FORCED_EXIT":

            normalized_status = "CLOSED"

        else:

            normalized_status = position.status

        diagnostics["normalized_status"] = normalized_status

        # ==========================================================
        # FINAL PnL LOCK (RE-CALIBRATION SAFETY)
        # ==========================================================

        final_position_value = position.quantity * current_price

        if position.direction == "SELL":

            final_position_value = position.quantity * position.entry_price - (
                position.quantity * (current_price - position.entry_price)
            )

        realized_pnl = final_position_value - (position.entry_price * position.quantity)

        realized_pnl_percent = (
            realized_pnl
            / max(
                position.entry_price * position.quantity,
                1.0,
            )
        ) * 100

        diagnostics["realized_pnl"] = round(
            realized_pnl,
            2,
        )

        diagnostics["realized_pnl_percent"] = round(
            realized_pnl_percent,
            2,
        )

        # ==========================================================
        # FINAL POSITION STATE SNAPSHOT
        # ==========================================================

        final_state = {
            "symbol": position.symbol,
            "status": normalized_status,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "current_price": current_price,
            "quantity": position.quantity,
            "holding_days": position.holding_days,
            "highest_price": position.highest_price,
            "lowest_price": position.lowest_price,
            "realized_pnl": round(realized_pnl, 2),
            "realized_pnl_percent": round(realized_pnl_percent, 2),
        }

        diagnostics["final_state"] = final_state

        # ==========================================================
        # TRACKER RESULT BUILD (START)
        # ==========================================================

        tracker_result = TrackerResult(
            symbol=position.symbol,
            action=exit_decision.action,
            pnl_percent=round(
                realized_pnl_percent,
                2,
            ),
            pnl_absolute=round(
                realized_pnl,
                2,
            ),
            status=normalized_status,
            exit_signal=diagnostics.get(
                "exit_action",
                "HOLD",
            ),
            diagnostics=diagnostics,
        )

        logger.info(
            "TrackerResult created for %s",
            position.symbol,
        )

        return tracker_result
        # ==========================================================
        # PORTFOLIO AGGREGATION HOOK
        # ==========================================================

        portfolio_positions = portfolio.get(
            "positions",
            {},
        )

        portfolio_positions[position.symbol] = {
            "quantity": position.quantity,
            "entry_price": position.entry_price,
            "current_price": current_price,
            "direction": position.direction,
            "pnl": realized_pnl,
            "pnl_percent": realized_pnl_percent,
            "status": normalized_status,
        }

        portfolio["positions"] = portfolio_positions

        diagnostics["portfolio_position_count"] = len(portfolio_positions)

        # ==========================================================
        # TOTAL PORTFOLIO EXPOSURE RE-CALCULATION
        # ==========================================================

        total_exposure = 0.0

        total_pnl = 0.0

        for sym, pos in portfolio_positions.items():

            position_value = pos["quantity"] * pos["current_price"]

            total_exposure += position_value

            total_pnl += pos["pnl"]

        total_capital = float(
            portfolio.get(
                "total_capital",
                1.0,
            )
        )

        portfolio_exposure_ratio = total_exposure / max(
            total_capital,
            1.0,
        )

        diagnostics["total_exposure"] = round(
            total_exposure,
            2,
        )

        diagnostics["portfolio_exposure_ratio"] = round(
            portfolio_exposure_ratio,
            4,
        )

        diagnostics["portfolio_pnl"] = round(
            total_pnl,
            2,
        )

        # ==========================================================
        # RISK AGGREGATION VIEW
        # ==========================================================

        portfolio_risk_view = {
            "open_positions": len(portfolio_positions),
            "exposure_ratio": round(portfolio_exposure_ratio, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_position_pnl": round(
                total_pnl / max(len(portfolio_positions), 1),
                2,
            ),
            "risk_flag": updated_risk.total_risk > 70,
        }

        diagnostics["portfolio_risk_view"] = portfolio_risk_view

        # ==========================================================
        # SYSTEM HEALTH METRICS
        # ==========================================================

        system_health = {
            "tracker_status": "ACTIVE",
            "risk_status": updated_risk.risk_grade,
            "positions_monitored": len(portfolio_positions),
            "forced_exits": int(
                sum(1 for p in portfolio_positions.values() if p["status"] == "CLOSED")
            ),
        }

        diagnostics["system_health"] = system_health

        # ==========================================================
        # FINAL TRACKER SNAPSHOT UPDATE
        # ==========================================================

        diagnostics["tracker_snapshot"] = {
            "symbol": position.symbol,
            "status": normalized_status,
            "pnl_percent": round(realized_pnl_percent, 2),
            "portfolio_exposure": round(portfolio_exposure_ratio, 4),
            "system_health": system_health,
        }

    # ==========================================================
    # BATCH TRACKER UTILITY
    # ==========================================================

    def update_portfolio(
        self,
        positions: list[PositionState],
        dataframe_map: dict[str, pd.DataFrame],
        portfolio: dict[str, Any],
        market_state: dict[str, Any],
    ) -> list[TrackerResult]:

        results: list[TrackerResult] = []

        logger.info(
            "Updating %d positions",
            len(positions),
        )

        for position in positions:

            dataframe = dataframe_map.get(position.symbol)

            if dataframe is None:

                logger.warning(
                    "Missing data for %s",
                    position.symbol,
                )

                continue

            result = self.update_position(
                position=position,
                dataframe=dataframe,
                portfolio=portfolio,
                market_state=market_state,
            )

            results.append(result)

        return results

    # ==========================================================
    # TRACKER SUMMARY
    # ==========================================================

    @staticmethod
    def summary(
        results: list[TrackerResult],
    ) -> str:

        total = len(results)

        closed = sum(r.status == "CLOSED" for r in results)

        active = sum(r.status == "ACTIVE" for r in results)

        partial = sum(r.status == "PARTIAL" for r in results)

        avg_pnl = round(
            sum(r.pnl_percent for r in results) / max(total, 1),
            2,
        )

        return (
            f"Tracked={total}"
            f" | Active={active}"
            f" | Closed={closed}"
            f" | Partial={partial}"
            f" | AvgPnL={avg_pnl:.2f}%"
        )

    # ==========================================================
    # TOP MOVERS
    # ==========================================================

    @staticmethod
    def top_movers(
        results: list[TrackerResult],
        limit: int = 10,
    ) -> list[TrackerResult]:

        return sorted(
            results,
            key=lambda r: r.pnl_percent,
            reverse=True,
        )[:limit]

    # ==========================================================
    # DEBUG REPORT
    # ==========================================================

    @staticmethod
    def debug_report(
        results: list[TrackerResult],
    ) -> str:

        report: list[str] = []

        report.append("=" * 120)
        report.append("TRACKER SYSTEM REPORT")
        report.append("=" * 120)
        report.append("")

        report.append(Tracker.summary(results))

        report.append("")
        report.append("-" * 120)

        for r in results:

            report.append(
                f"{r.symbol:<15}"
                f"{r.action:<10}"
                f"{r.status:<12}"
                f"{r.pnl_percent:>10.2f}%"
                f"{r.pnl_absolute:>15.2f}"
            )

        report.append("")
        report.append("=" * 120)
        report.append("END TRACKER REPORT")
        report.append("=" * 120)

        return "\n".join(report)


# ==========================================================
# END OF FILE
# ==========================================================
