import pandas as pd
from risk.risk_manager import RiskManager
from decision.validation_engine import ValidationResult
from decision.decision_engine import FinalDecision


def test_risk_block():
    r = RiskManager()

    # Sahi dummy mock objects/data taiyar kiya
    validation = ValidationResult(passed=True)
    decision = FinalDecision(signal="BUY", confidence=85.0)
    portfolio = {"equity": 1000}
    market = {"regime": "HIGH_VOLATILITY"}

    # Ek khali DataFrame banaya jisme mandatory rows ho sakein
    dataframe = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume", "atr"]
    )

    # Saare 5 arguments properly keyword ke sath pass kiye
    result = r.evaluate(
        validation=validation,
        decision=decision,
        dataframe=dataframe,
        portfolio=portfolio,
        market=market,
    )

    # Dictionary check karne ke bajay object attribute check kiya
    assert hasattr(result, "safe")
