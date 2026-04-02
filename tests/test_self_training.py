from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_brain import self_training, signal_memory


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


def _result(
    symbol: str,
    market_type: str,
    verdict: str,
    entry: float,
    stop: float,
    tp1: float,
    tp2: float,
    *,
    confidence: float = 0.82,
    risk_reward: float = 2.1,
) -> dict:
    return {
        "summary": {
            "symbol": symbol,
            "timeframe": "15m",
            "market_type": market_type,
            "verdict": verdict,
            "confidence": confidence,
        },
        "brain_output": {
            "summary": {
                "symbol": symbol,
                "timeframe": "15m",
                "market_type": market_type,
                "verdict": verdict,
                "confidence": confidence,
            },
            "plan": {
                "entry_zone": [entry, entry],
                "stop_loss": stop,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "risk_reward": risk_reward,
            },
        },
    }


class SelfTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.signal_dir_patch = patch.object(signal_memory, "DATA_DIR", self.data_dir)
        self.signal_file_patch = patch.object(signal_memory, "DATA_PATH", self.data_dir / "signal_memory.json")
        self.training_dir_patch = patch.object(self_training, "DATA_DIR", self.data_dir)
        self.training_file_patch = patch.object(self_training, "DATA_PATH", self.data_dir / "self_training.json")
        self.signal_dir_patch.start()
        self.signal_file_patch.start()
        self.training_dir_patch.start()
        self.training_file_patch.start()

    def tearDown(self) -> None:
        self.training_file_patch.stop()
        self.training_dir_patch.stop()
        self.signal_file_patch.stop()
        self.signal_dir_patch.stop()
        self.temp_dir.cleanup()

    def test_trainer_learns_profitable_direction_from_closed_wins(self) -> None:
        for price in (68000.0, 68250.0, 68500.0, 68720.0):
            payload = _payload("BTCUSDT", price, "crypto")
            result = _result("BTCUSDT", "crypto", "LONG", price, price - 140.0, price + 300.0, price + 450.0)
            signal_memory.register_signal(payload, result)
            signal_memory.reconcile_market_memory(
                "BTCUSDT",
                "15m",
                "intraday",
                {
                    **payload,
                    "high": price + 320.0,
                    "low": price - 40.0,
                    "price": price + 310.0,
                    "close": price + 310.0,
                },
            )

        profile = self_training.derive_training_profile("BTCUSDT", "15m", "intraday", "crypto")
        stored = self_training.load_training_profile("BTCUSDT", "15m", "intraday", "crypto")

        self.assertIn(profile["trainer_state"], {"improving", "compounding"})
        self.assertEqual(profile["preferred_direction"], "long")
        self.assertGreater(profile["long_edge"], 0.0)
        self.assertGreaterEqual(profile["training_days"], 1)
        self.assertIsNotNone(stored)

    def test_trainer_raises_discipline_after_low_rr_losses(self) -> None:
        for price in (1.1010, 1.1030, 1.1050):
            payload = _payload("EURUSD", price, "forex")
            result = _result(
                "EURUSD",
                "forex",
                "LONG",
                price,
                price - 0.0010,
                price + 0.0011,
                price + 0.0012,
                confidence=0.84,
                risk_reward=1.1,
            )
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

        profile = self_training.derive_training_profile("EURUSD", "15m", "intraday", "forex")

        self.assertEqual(profile["trainer_state"], "defensive")
        self.assertGreaterEqual(profile["low_rr_losses"], 2)
        self.assertGreater(profile["rr_floor_delta"], 0.1)
        self.assertLess(profile["confidence_shift"], 0.0)
        self.assertIn("RR", " ".join(profile["lesson_notes"]).upper())

    def test_training_dashboard_leaderboard_exposes_updated_fields(self) -> None:
        for price in (2.10, 2.14, 2.18):
            payload = _payload("XRPUSDT", price, "crypto")
            result = _result("XRPUSDT", "crypto", "SHORT", price, price + 0.04, price - 0.08, price - 0.12)
            signal_memory.register_signal(payload, result)
            signal_memory.reconcile_market_memory(
                "XRPUSDT",
                "15m",
                "intraday",
                {
                    **payload,
                    "high": price + 0.01,
                    "low": price - 0.09,
                    "price": price - 0.08,
                    "close": price - 0.08,
                },
            )

        dashboard = self_training.get_training_dashboard("XRPUSDT", "15m", "intraday", "crypto")

        self.assertTrue(dashboard["leaderboard"])
        top = dashboard["leaderboard"][0]
        self.assertEqual(top["symbol"], "XRPUSDT")
        self.assertIn("updated_at", top)
        self.assertTrue(top["updated_at"])
        self.assertIn("source_scope", top)


if __name__ == "__main__":
    unittest.main()
