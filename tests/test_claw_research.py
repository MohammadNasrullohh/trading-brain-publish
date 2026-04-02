from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_brain import claw_research


def _payload() -> dict:
    return {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "style": "intraday",
        "market_type": "crypto",
        "execution_profile": "balanced",
        "price": 66420,
        "sentiment": {
            "score": 0.18,
            "headline_risk": False,
            "macro_risk": False,
        },
        "live_context": {
            "listing_profile": "mature",
            "fresh_listing_candidate": False,
        },
    }


def _analysis() -> dict:
    return {
        "summary": {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "market_type": "crypto",
            "verdict": "LONG",
            "confidence": 0.84,
        },
        "brain_output": {
            "summary": {
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "market_type": "crypto",
                "verdict": "LONG",
                "confidence": 0.84,
            },
            "plan": {
                "setup_type": "breakout_retest",
                "entry_zone": [66380, 66420],
                "stop_loss": 66240,
                "take_profit_1": 66720,
                "take_profit_2": 66980,
                "risk_reward": 2.3,
            },
            "reasons": ["Momentum clean", "Breakout follow-through"],
            "warnings": [],
            "blockers": [],
            "training": {
                "trainer_state": "improving",
            },
        },
        "strategic_brief": {
            "primary_thesis": "BTC memimpin momentum saat orderflow tetap bersih.",
            "edge_summary": "Breakout masih sehat.",
        },
    }


DEFAULT_LEARNING = {
    "state": "in_sync",
    "score_bias": 2.5,
    "confidence_bias": 0.03,
    "win_rate": 62.0,
    "wins": 5,
    "losses": 3,
    "loss_streak": 0,
    "recent_history": [],
}


class ClawResearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.path_patch = patch.object(claw_research, "DATA_DIR", self.data_dir)
        self.file_patch = patch.object(claw_research, "DATA_PATH", self.data_dir / "claw_research_memory.json")
        self.path_patch.start()
        self.file_patch.start()

    def tearDown(self) -> None:
        self.file_patch.stop()
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_build_claw_research_tracks_session_and_positive_bias_for_clean_long(self) -> None:
        research = claw_research.build_claw_research(
            _payload(),
            _analysis(),
            DEFAULT_LEARNING,
            news_snapshot={
                "summary": {
                    "score": 0.26,
                    "headline_risk": False,
                    "macro_risk": False,
                    "mood": "tailwind",
                },
                "articles": [{"title": "Bitcoin breakout holds volume support"}],
            },
        )

        self.assertEqual(research["focus"], "breakout")
        self.assertEqual(research["risk_posture"], "offense")
        self.assertGreater(research["score_delta"], 0.0)
        self.assertEqual(research["session"]["turn_count"], 1)

    def test_merge_and_bias_attach_research_metadata(self) -> None:
        research = claw_research.build_claw_research(
            _payload(),
            _analysis(),
            DEFAULT_LEARNING,
            news_snapshot={"summary": {"score": 0.0, "headline_risk": False, "macro_risk": False, "mood": "neutral"}, "articles": []},
        )
        merged = claw_research.merge_claw_research(_analysis(), research)
        biased = claw_research.apply_claw_research_bias(
            {"symbol": "BTCUSDT", "score": 72.0, "prime_setup": True},
            research,
        )

        self.assertIn("claw_summary", merged["strategic_brief"])
        self.assertEqual(merged["summary"]["research_focus"], research["focus"])
        self.assertEqual(biased["claw_focus"], research["focus"])
        self.assertGreaterEqual(biased["score"], 72.0)


if __name__ == "__main__":
    unittest.main()
