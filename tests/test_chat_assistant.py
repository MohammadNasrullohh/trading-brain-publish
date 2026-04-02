from __future__ import annotations

import unittest

from trading_brain.chat_assistant import answer_chat


def _context() -> dict:
    return {
        "payload": {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
        },
        "result": {
            "summary": {
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "verdict": "LONG",
                "confidence": 0.82,
            },
            "brain_output": {
                "plan": {
                    "entry_zone": [68420.0, 68420.0],
                    "stop_loss": 68280.0,
                    "take_profit_1": 68780.0,
                    "take_profit_2": 69040.0,
                    "risk_reward": 2.4,
                },
                "reasons": ["Zona support masih bertahan."],
                "warnings": ["Butuh jaga headline risk."],
                "blockers": [],
            },
            "strategic_brief": {
                "primary_thesis": "Long di zona support selama struktur tetap bertahan.",
            },
        },
        "memory_dashboard": {
            "global_summary": {
                "closed_signals": 26,
                "tracked_pairs": 9,
                "win_rate": 58.4,
            },
            "pair_summary": {
                "state": "in_sync",
                "sample_size": 18,
            },
            "open_breakdown": [
                {
                    "symbol": "BTCUSDT",
                    "open_signals": 2,
                    "active_signal": {
                        "verdict": "LONG",
                    },
                },
                {
                    "symbol": "XAUUSD",
                    "open_signals": 1,
                    "active_signal": {
                        "verdict": "SHORT",
                    },
                },
            ],
        },
        "training_dashboard": {
            "current": {
                "trainer_state": "improving",
                "sample_size": 18,
                "training_days": 7,
            },
            "overview": {
                "active_contexts": 6,
            },
        },
        "recommendations": {
            "items": [
                {
                    "symbol": "ETHUSDT",
                    "verdict": "LONG",
                    "score": 103.4,
                    "confidence": 84,
                    "risk_reward": 2.1,
                    "reason": "ETH sedang retest support dan lebih rapi dari pair lain.",
                }
            ]
        },
        "news": {
            "headline_mood": "risk-on",
            "headline_risk": "normal",
            "articles": [
                {
                    "title": "Crypto majors rebound on fresh flows",
                    "source": "Desk Feed",
                }
            ],
        },
    }


class ChatAssistantTest(unittest.TestCase):
    def test_current_signal_answer_mentions_levels(self) -> None:
        payload = answer_chat("Apa call sekarang?", _context())
        self.assertIn("BUY / LONG", payload["reply"])
        self.assertIn("Entry", payload["reply"])
        self.assertIn("SL", payload["reply"])

    def test_best_setup_answer_uses_recommendation_context(self) -> None:
        payload = answer_chat("Best setup mana?", _context())
        self.assertIn("ETHUSDT", payload["reply"])
        self.assertTrue(any("Score" in chip for chip in payload["chips"]))

    def test_readiness_answer_reports_phase(self) -> None:
        payload = answer_chat("Readiness sistem berapa?", _context())
        self.assertIn("%", payload["reply"])
        self.assertTrue(any(chip in {"Building", "Ready", "Live Ready", "Learning", "Defensive", "Warming"} for chip in payload["chips"]))


if __name__ == "__main__":
    unittest.main()
