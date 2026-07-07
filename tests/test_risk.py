from risk.risk_manager import RiskManager


def test_risk_block():
    r = RiskManager()

    decision = {"action": "BUY"}
    portfolio = {"equity": 1000}

    # Keyword arguments use karke parameters ko explicitly pass kiya
    result = r.evaluate(decision=decision, portfolio=portfolio, market="HIGH_VOL")

    assert "approved" in result
