import json
import time
from core.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:

    def generate(self, analytics: dict, portfolio: dict, risk: dict) -> dict:

        report = {
            "timestamp": time.time(),
            "analytics": analytics,
            "portfolio": portfolio,
            "risk": risk,
            "summary": {
                "equity": portfolio.get("equity", 0),
                "pnl": portfolio.get("pnl", 0),
                "risk_level": risk.get("grade", "UNKNOWN"),
            },
        }

        logger.info("Report generated")

        return report

    def save_json(self, report: dict, path: str = None):

        path = path or f"storage/reports/report_{int(time.time())}.json"

        with open(path, "w") as f:

            json.dump(report, f, indent=2)

        logger.info(f"Report saved at {path}")
