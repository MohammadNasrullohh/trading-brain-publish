from __future__ import annotations

import json
import unittest
from pathlib import Path

from trading_brain import TradingBrain


ROOT = Path(__file__).resolve().parents[1]


def load_example(name: str) -> dict:
    path = ROOT / "examples" / name
    return json.loads(path.read_text(encoding="utf-8"))


class TradingBrainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.brain = TradingBrain()

    def test_bullish_example_returns_long(self) -> None:
        result = self.brain.analyze_payload(load_example("bullish_btc.json"))
        self.assertEqual(result["summary"]["verdict"], "LONG")
        self.assertGreater(result["scores"]["long"], result["scores"]["short"])
        self.assertEqual(result["summary"]["market_type"], "crypto")
        self.assertGreaterEqual(result["meta"]["neuron_count"], 120)
        self.assertIn("snr", str(result["plan"]["setup_type"]).lower())

    def test_bearish_example_returns_short(self) -> None:
        result = self.brain.analyze_payload(load_example("bearish_eth.json"))
        self.assertEqual(result["summary"]["verdict"], "SHORT")
        self.assertGreater(result["scores"]["short"], result["scores"]["long"])
        self.assertIn("snr", str(result["plan"]["setup_type"]).lower())

    def test_range_example_returns_no_trade(self) -> None:
        result = self.brain.analyze_payload(load_example("no_trade_range.json"))
        self.assertEqual(result["summary"]["verdict"], "NO TRADE")

    def test_forex_example_returns_long(self) -> None:
        result = self.brain.analyze_payload(load_example("forex_eurusd_bullish.json"))
        self.assertEqual(result["summary"]["verdict"], "LONG")
        self.assertEqual(result["summary"]["market_type"], "forex")
        self.assertGreaterEqual(result["meta"]["neuron_count"], 120)
        self.assertIn("snr", str(result["plan"]["setup_type"]).lower())

    def test_drawdown_pause_returns_no_trade(self) -> None:
        result = self.brain.analyze_payload(load_example("drawdown_pause.json"))
        self.assertEqual(result["summary"]["verdict"], "NO TRADE")
        self.assertGreaterEqual(result["risk"]["current_drawdown_percent"], 5.0)
        self.assertIn("loss streak tinggi, brain masuk mode recovery", result["blockers"])


if __name__ == "__main__":
    unittest.main()
