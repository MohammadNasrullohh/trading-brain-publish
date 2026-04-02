from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_brain import signal_memory


def _payload(symbol: str, price: float, timeframe: str = "15m", style: str = "intraday") -> dict:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "style": style,
        "market_type": "forex" if symbol.endswith("USD") else "crypto",
        "price": price,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "atr": max(price * 0.002, 0.001),
    }


def _result(symbol: str, timeframe: str, verdict: str, confidence: float, entry: float, stop: float, tp1: float, tp2: float) -> dict:
    return {
        "summary": {
            "symbol": symbol,
            "timeframe": timeframe,
            "market_type": "forex" if symbol.endswith("USD") else "crypto",
            "verdict": verdict,
            "confidence": confidence,
        },
        "brain_output": {
            "summary": {
                "symbol": symbol,
                "timeframe": timeframe,
                "market_type": "forex" if symbol.endswith("USD") else "crypto",
                "verdict": verdict,
                "confidence": confidence,
            },
            "plan": {
                "entry_zone": [entry, entry],
                "stop_loss": stop,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "risk_reward": 2.0,
            },
        },
    }


class SignalMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.path_patch = patch.object(signal_memory, "DATA_DIR", self.data_dir)
        self.file_patch = patch.object(signal_memory, "DATA_PATH", self.data_dir / "signal_memory.json")
        self.path_patch.start()
        self.file_patch.start()

    def tearDown(self) -> None:
        self.file_patch.stop()
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_register_signal_and_close_with_win_updates_history(self) -> None:
        payload = _payload("EURUSD", 1.0800)
        result = _result("EURUSD", "15m", "LONG", 0.82, 1.0800, 1.0788, 1.0824, 1.0840)

        context = signal_memory.register_signal(payload, result)
        self.assertEqual(context["open_signals"], 1)

        signal_memory.reconcile_market_memory(
            "EURUSD",
            "15m",
            "intraday",
            {
                **payload,
                "high": 1.0826,
                "low": 1.0794,
                "price": 1.0825,
                "close": 1.0825,
            },
        )
        history = signal_memory.get_signal_history("EURUSD", "15m", "intraday", "forex")

        self.assertEqual(history["summary"]["wins"], 1)
        self.assertEqual(history["summary"]["losses"], 0)
        self.assertEqual(history["summary"]["win_rate"], 100.0)
        self.assertTrue(any(item["status"] == "win" for item in history["recent"]))

    def test_learning_context_enters_cooldown_after_three_losses(self) -> None:
        for index in range(3):
            entry = 1.1000 + (index * 0.0010)
            payload = _payload("GBPUSD", entry)
            result = _result("GBPUSD", "15m", "LONG", 0.8, entry, entry - 0.0010, entry + 0.0016, entry + 0.0024)
            signal_memory.register_signal(payload, result)
            signal_memory.reconcile_market_memory(
                "GBPUSD",
                "15m",
                "intraday",
                {
                    **payload,
                    "high": entry + 0.0002,
                    "low": entry - 0.0012,
                    "price": entry - 0.0011,
                    "close": entry - 0.0011,
                },
            )

        context = signal_memory.learning_context("GBPUSD", "15m", "intraday", "forex")

        self.assertEqual(context["state"], "cooldown")
        self.assertEqual(context["loss_streak"], 3)
        self.assertFalse(context["prime_penalty"])
        self.assertEqual(context["wins"], 0)
        self.assertEqual(context["losses"], 3)

    def test_learning_context_can_fallback_to_market_memory_when_pair_history_is_thin(self) -> None:
        payload = _payload("ETHUSDT", 3250.0)
        result = _result("ETHUSDT", "15m", "LONG", 0.81, 3250.0, 3210.0, 3310.0, 3360.0)
        signal_memory.register_signal(payload, result)
        signal_memory.reconcile_market_memory(
            "ETHUSDT",
            "15m",
            "intraday",
            {
                **payload,
                "high": 3314.0,
                "low": 3232.0,
                "price": 3312.0,
                "close": 3312.0,
            },
        )

        for symbol, entry, status in [
            ("BTCUSDT", 68000.0, "win"),
            ("SOLUSDT", 180.0, "win"),
            ("BNBUSDT", 610.0, "loss"),
        ]:
            payload = _payload(symbol, entry)
            result = _result(symbol, "15m", "LONG", 0.79, entry, entry * 0.992, entry * 1.012, entry * 1.02)
            signal_memory.register_signal(payload, result)
            market_state = {
                **payload,
                "price": entry,
                "close": entry,
            }
            if status == "win":
                market_state["high"] = entry * 1.013
                market_state["low"] = entry * 0.998
            else:
                market_state["high"] = entry * 1.002
                market_state["low"] = entry * 0.989
                market_state["price"] = entry * 0.9895
                market_state["close"] = entry * 0.9895
            signal_memory.reconcile_market_memory(symbol, "15m", "intraday", market_state)

        context = signal_memory.learning_context("ADAUSDT", "15m", "intraday", "crypto")

        self.assertEqual(context["memory_scope"], "market")
        self.assertTrue(context["market_fallback_active"])
        self.assertEqual(context["market_scored_total"], 4)
        self.assertEqual(context["pair_scored_total"], 0)
        self.assertEqual(context["state"], "in_sync")

    def test_memory_dashboard_exposes_per_signal_curve_for_same_day_closes(self) -> None:
        for index, status in enumerate(["win", "loss", "win"], start=1):
            entry = 1.2000 + (index * 0.0010)
            payload = _payload("EURUSD", entry)
            result = _result("EURUSD", "15m", "LONG", 0.78, entry, entry - 0.0010, entry + 0.0018, entry + 0.0025)
            signal_memory.register_signal(payload, result)
            market_state = {
                **payload,
                "high": entry + (0.0021 if status == "win" else 0.0002),
                "low": entry - (0.0012 if status == "loss" else 0.0004),
                "price": entry + (0.0019 if status == "win" else -0.0011),
                "close": entry + (0.0019 if status == "win" else -0.0011),
            }
            signal_memory.reconcile_market_memory("EURUSD", "15m", "intraday", market_state)

        dashboard = signal_memory.build_memory_dashboard("EURUSD", "15m", "intraday", "forex")
        curve = dashboard["performance_curve_signal"]

        self.assertEqual(len(curve), 3)
        self.assertEqual(curve[-1]["net_r"], 3.0)
        self.assertEqual(curve[1]["status"], "loss")

    def test_memory_dashboard_selected_pair_detail_includes_open_signal_and_score(self) -> None:
        payload = _payload("BTCUSDT", 68250.0)
        payload["_signal_score"] = 108.4
        payload["_signal_rank"] = 1
        result = _result("BTCUSDT", "15m", "LONG", 0.84, 68240.0, 68090.0, 68620.0, 68840.0)

        signal_memory.register_signal(payload, result, source="best_setups")
        dashboard = signal_memory.build_memory_dashboard("BTCUSDT", "15m", "intraday", "crypto")
        detail = dashboard["selected_pair_detail"]

        self.assertIsNotNone(detail)
        self.assertEqual(detail["symbol"], "BTCUSDT")
        self.assertEqual(detail["top_score"], 108.4)
        self.assertEqual(detail["active_signal"]["entry_mid"], 68240.0)
        self.assertEqual(detail["active_signal"]["score"], 108.4)
        self.assertEqual(detail["active_signal"]["score_rank"], 1)
        self.assertEqual(detail["open_signals"], 1)

    def test_memory_dashboard_open_breakdown_lists_all_open_pairs(self) -> None:
        for symbol, score in [("BTCUSDT", 104.0), ("XRPUSDT", 112.0), ("EURUSD", 101.5)]:
            payload = _payload(symbol, 68000.0 if symbol == "BTCUSDT" else 2.2 if symbol == "XRPUSDT" else 1.082)
            payload["_signal_score"] = score
            result = _result(symbol, "15m", "LONG", 0.81, payload["price"], payload["price"] * 0.995, payload["price"] * 1.01, payload["price"] * 1.02)
            signal_memory.register_signal(payload, result, source="best_setups")

        dashboard = signal_memory.build_memory_dashboard("BTCUSDT", "15m", "intraday", "crypto")
        open_symbols = [item["symbol"] for item in dashboard["open_breakdown"]]

        self.assertIn("BTCUSDT", open_symbols)
        self.assertIn("XRPUSDT", open_symbols)
        self.assertIn("EURUSD", open_symbols)

    def test_legacy_json_store_is_migrated_into_shared_database(self) -> None:
        legacy_store = {
            "version": 1,
            "signals": [
                {
                    "id": "sig-1",
                    "symbol": "BTCUSDT",
                    "timeframe": "15m",
                    "style": "intraday",
                    "market_type": "crypto",
                    "status": "win",
                }
            ],
            "events": [{"id": "evt-1", "signal_id": "sig-1"}],
        }
        signal_memory.DATA_PATH.write_text(json.dumps(legacy_store), encoding="utf-8")

        migrated = signal_memory._read_store()
        db_path = self.data_dir / "brain_state.db"

        self.assertTrue(db_path.exists())
        self.assertEqual(migrated["signals"][0]["id"], "sig-1")

        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT payload_json FROM module_documents WHERE module = ?",
                ("signal_memory",),
            ).fetchone()
        self.assertIsNotNone(row)

        signal_memory.DATA_PATH.unlink()
        persisted = signal_memory._read_store()
        self.assertEqual(persisted["signals"][0]["id"], "sig-1")


if __name__ == "__main__":
    unittest.main()
