import json
import os


class TradeStore:

    def __init__(self):

        self.path = "storage/trades"

        os.makedirs(self.path, exist_ok=True)

    def save_trade(self, trade: dict):

        file = f"{self.path}/{trade['id']}.json"

        with open(file, "w") as f:

            json.dump(trade, f, indent=2)
