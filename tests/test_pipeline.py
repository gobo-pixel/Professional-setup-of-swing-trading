from orchestrator import WiredOrchestrator


def test_imports():
    assert WiredOrchestrator is not None


def test_init():
    orch = WiredOrchestrator(mode="BACKTEST")
    assert orch.mode == "BACKTEST"
