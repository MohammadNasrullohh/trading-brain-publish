from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_brain import adaptive_growth
from trading_brain import claw_research
from trading_brain.web_bridge import analyze_for_web, list_example_files, load_scene_data


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEARNING = {
    "symbol": "TEST",
    "timeframe": "15m",
    "style": "intraday",
    "market_type": "crypto",
    "state": "warming",
    "score_bias": 0.0,
    "confidence_bias": 0.0,
    "win_rate": None,
    "wins": 0,
    "losses": 0,
    "scored_total": 0,
    "sample_size": 0,
    "pair_scored_total": 0,
    "market_scored_total": 0,
    "memory_scope": "warming",
    "market_fallback_active": False,
    "loss_streak": 0,
    "open_signals": 0,
    "prime_penalty": False,
    "note": "No history yet.",
    "recent_history": [],
}


def load_example(name: str) -> dict:
    path = ROOT / "examples" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _node_id_by_title(scene: dict, title: str) -> str:
    for node in scene["nodes"]:
        if node["title"] == title:
            return node["id"]
    raise AssertionError(f"Node title not found: {title}")


class WebBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.path_patch = patch.object(adaptive_growth, "DATA_DIR", self.data_dir)
        self.file_patch = patch.object(adaptive_growth, "DATA_PATH", self.data_dir / "adaptive_growth.json")
        self.claw_path_patch = patch.object(claw_research, "DATA_DIR", self.data_dir)
        self.claw_file_patch = patch.object(claw_research, "DATA_PATH", self.data_dir / "claw_research_memory.json")
        self.web_learning_patch = patch("trading_brain.web_bridge.signal_learning_context", return_value=dict(DEFAULT_LEARNING))
        self.adaptive_learning_patch = patch("trading_brain.adaptive_growth.signal_learning_context", return_value=dict(DEFAULT_LEARNING))
        self.path_patch.start()
        self.file_patch.start()
        self.claw_path_patch.start()
        self.claw_file_patch.start()
        self.web_learning_patch.start()
        self.adaptive_learning_patch.start()

    def tearDown(self) -> None:
        self.adaptive_learning_patch.stop()
        self.web_learning_patch.stop()
        self.claw_file_patch.stop()
        self.claw_path_patch.stop()
        self.file_patch.stop()
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_web_bridge_returns_long_visual_state_for_bullish_case(self) -> None:
        result = analyze_for_web(load_example("bullish_btc.json"), mode="super")
        visual = result["visual_state"]
        scene = load_scene_data()
        verdict_node_id = _node_id_by_title(scene, "Verdict Gate")
        long_setup_id = _node_id_by_title(scene, "Long Setup")

        self.assertEqual(visual["mode"], "LONG")
        self.assertEqual(visual["mode_index"], 0)
        self.assertEqual(visual["verdict_node_id"], verdict_node_id)
        self.assertEqual(visual["pair_profile"]["key"], "BTC")
        self.assertEqual(result["adaptive_profile"]["growth_cycle"], "bootstrap")
        self.assertEqual(result["claw_research"]["focus"], "snr")
        self.assertIn("BTCUSDT", result["result"]["strategic_brief"]["claw_summary"])
        self.assertGreaterEqual(visual["node_weights"][long_setup_id], 0.8)
        self.assertGreaterEqual(visual["node_weights"][verdict_node_id], 0.8)

    def test_web_bridge_returns_defensive_visual_state_for_drawdown_case(self) -> None:
        result = analyze_for_web(load_example("drawdown_pause.json"), mode="super")
        visual = result["visual_state"]
        scene = load_scene_data()
        drawdown_guard_id = _node_id_by_title(scene, "Drawdown Guard")
        no_trade_id = _node_id_by_title(scene, "No Trade Filter")

        self.assertEqual(visual["mode"], "NO_TRADE")
        self.assertEqual(visual["mode_index"], 3)
        self.assertGreaterEqual(visual["node_weights"][drawdown_guard_id], 0.8)
        self.assertGreaterEqual(visual["node_weights"][no_trade_id], 0.8)

    def test_examples_listing_contains_known_payloads(self) -> None:
        examples = list_example_files()
        self.assertIn("bullish_btc.json", examples)
        self.assertIn("drawdown_pause.json", examples)

    def test_web_bridge_returns_wti_pair_profile_for_commodity_payload(self) -> None:
        payload = {
            "symbol": "WTI",
            "timeframe": "15m",
            "style": "intraday",
            "market_type": "commodity",
            "session": "us",
            "price": 93.55,
            "open": 93.08,
            "high": 94.22,
            "low": 92.72,
            "close": 93.55,
            "atr": 0.96,
            "levels": {
                "support": [93.02, 92.54],
                "resistance": [94.08, 94.66],
                "previous_high": 94.3,
                "previous_low": 92.48,
                "session_high": 94.22,
                "session_low": 92.72,
            },
            "indicators": {
                "ema_fast": 93.42,
                "ema_slow": 93.18,
                "rsi": 52,
                "macd_histogram": 0.22,
                "volume_trend": "steady",
                "vwap": 93.36,
                "adx": 20,
                "stochastic": 54,
                "delta_volume": 0.18,
                "bollinger_position": 0.57,
                "cci": 34,
            },
            "risk": {
                "max_risk_percent": 0.35,
                "leverage": 5,
                "current_drawdown_percent": 1.1,
                "max_daily_loss_percent": 4,
                "loss_streak": 0,
            },
            "sentiment": {
                "score": 0.04,
                "headline_risk": False,
                "correlation_bias": 0.08,
                "macro_risk": False,
            },
            "context": {
                "regime_hint": "ranging",
                "session_quality_hint": "normal",
                "market_type_hint": "commodity",
            },
            "microstructure": {
                "spread": 0.05,
                "fee_bps": 0,
                "slippage_bps": 3,
                "weekend": False,
                "liquidity_score": 0.72,
                "orderbook_imbalance": 0.05,
            },
        }
        result = analyze_for_web(payload, mode="super")
        visual = result["visual_state"]
        self.assertEqual(visual["pair_profile"]["key"], "WTI")
        self.assertEqual(result["adaptive_profile"]["adaptation_mode"], "calibrating")

    def test_web_bridge_returns_forex_pair_profile_for_major_pair(self) -> None:
        result = analyze_for_web(load_example("forex_eurusd_bullish.json"), mode="super")
        visual = result["visual_state"]
        self.assertEqual(visual["pair_profile"]["key"], "EURUSD")
        self.assertIn("Session", visual["pair_profile"]["template_name"])
        self.assertIn("Execution", visual["pair_profile"]["tags"])

    def test_web_bridge_builds_mtf_summary_and_applies_guidance(self) -> None:
        payload = load_example("bullish_btc.json")
        payload["_mtf_snapshots"] = {
            "4h": {"context": {"bias_hint": "bullish", "structure_hint": "bullish", "regime_hint": "trend"}, "price": 68200},
            "1h": {"context": {"bias_hint": "bullish", "structure_hint": "bullish", "regime_hint": "trend"}, "price": 68240},
            "15m": {"context": {"bias_hint": "bullish", "structure_hint": "bullish", "regime_hint": "expansion"}, "price": 68280},
            "5m": {"context": {"bias_hint": "bearish", "structure_hint": "bearish", "regime_hint": "pullback"}, "price": 68210},
        }

        result = analyze_for_web(payload, mode="super")

        self.assertEqual(result["mtf_summary"]["consensus_bias"], "bullish")
        self.assertEqual(result["mtf_summary"]["higher_timeframe_bias"], "bullish")
        self.assertEqual(result["result"]["mtf_summary"]["consensus_bias"], "bullish")
        self.assertGreater(result["result"]["brain_output"]["summary"]["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
