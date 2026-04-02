from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_brain import adaptive_growth, signal_memory


def _payload(symbol: str, price: float, market_type: str = "crypto") -> dict:
    return {
        "symbol": symbol,
        "timeframe": "15m",
        "style": "intraday",
        "market_type": market_type,
        "price": price,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "atr": max(price * 0.002, 0.001),
    }


def _result(symbol: str, market_type: str, verdict: str, entry: float, stop: float, tp1: float, tp2: float) -> dict:
    return {
        "summary": {
            "symbol": symbol,
            "timeframe": "15m",
            "market_type": market_type,
            "verdict": verdict,
            "confidence": 0.82,
        },
        "brain_output": {
            "summary": {
                "symbol": symbol,
                "timeframe": "15m",
                "market_type": market_type,
                "verdict": verdict,
                "confidence": 0.82,
            },
            "plan": {
                "entry_zone": [entry, entry],
                "stop_loss": stop,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "risk_reward": 2.1,
            },
        },
    }


class AdaptiveGrowthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.signal_dir_patch = patch.object(signal_memory, "DATA_DIR", self.data_dir)
        self.signal_file_patch = patch.object(signal_memory, "DATA_PATH", self.data_dir / "signal_memory.json")
        self.growth_dir_patch = patch.object(adaptive_growth, "DATA_DIR", self.data_dir)
        self.growth_file_patch = patch.object(adaptive_growth, "DATA_PATH", self.data_dir / "adaptive_growth.json")
        self.signal_dir_patch.start()
        self.signal_file_patch.start()
        self.growth_dir_patch.start()
        self.growth_file_patch.start()

    def tearDown(self) -> None:
        self.growth_file_patch.stop()
        self.growth_dir_patch.stop()
        self.signal_file_patch.stop()
        self.signal_dir_patch.stop()
        self.temp_dir.cleanup()

    def test_growth_profile_compounds_when_pair_history_is_in_sync(self) -> None:
        for price in (68000.0, 68250.0, 68510.0, 68740.0):
            payload = _payload("BTCUSDT", price, "crypto")
            result = _result("BTCUSDT", "crypto", "LONG", price, price - 140.0, price + 310.0, price + 460.0)
            signal_memory.register_signal(payload, result)
            signal_memory.reconcile_market_memory(
                "BTCUSDT",
                "15m",
                "intraday",
                {
                    **payload,
                    "high": price + 320.0,
                    "low": price - 60.0,
                    "price": price + 300.0,
                    "close": price + 300.0,
                },
            )

        profile = adaptive_growth.derive_growth_profile("BTCUSDT", "15m", "intraday", "crypto")
        stored = adaptive_growth.load_growth_profile("BTCUSDT", "15m", "intraday", "crypto")

        self.assertEqual(profile["growth_cycle"], "compound")
        self.assertEqual(profile["adaptation_mode"], "compounding")
        self.assertGreaterEqual(profile["maturity"], 50)
        self.assertGreater(profile["pair_weight"], 0.9)
        self.assertGreater(profile["score_shift"], 0.0)
        self.assertIn("Compounding Drive", profile["focus_titles"])
        self.assertIsNotNone(stored)
        self.assertGreaterEqual(stored["cycles"], 1)

    def test_growth_profile_enters_protect_mode_after_losses(self) -> None:
        for price in (1.1010, 1.1030, 1.1050):
            payload = _payload("EURUSD", price, "forex")
            result = _result("EURUSD", "forex", "LONG", price, price - 0.0010, price + 0.0018, price + 0.0028)
            signal_memory.register_signal(payload, result)
            signal_memory.reconcile_market_memory(
                "EURUSD",
                "15m",
                "intraday",
                {
                    **payload,
                    "high": price + 0.0002,
                    "low": price - 0.0012,
                    "price": price - 0.0011,
                    "close": price - 0.0011,
                },
            )

        profile = adaptive_growth.derive_growth_profile("EURUSD", "15m", "intraday", "forex")

        self.assertEqual(profile["growth_cycle"], "protect")
        self.assertEqual(profile["adaptation_mode"], "capital_protection")
        self.assertGreaterEqual(profile["protection_bias"], 0.2)
        self.assertGreaterEqual(profile["rr_floor"], 1.7)
        self.assertLess(profile["score_shift"], 0.0)
        self.assertIn("Survival Instinct", profile["focus_titles"])


if __name__ == "__main__":
    unittest.main()
