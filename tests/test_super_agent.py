from __future__ import annotations

import json
import unittest
from pathlib import Path

from trading_brain import SuperTradingAgent


ROOT = Path(__file__).resolve().parents[1]


def load_example(name: str) -> dict:
    path = ROOT / "examples" / name
    return json.loads(path.read_text(encoding="utf-8"))


class SuperTradingAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = SuperTradingAgent()

    def test_super_agent_engages_on_clean_bullish_case(self) -> None:
        result = self.agent.analyze_payload(load_example("bullish_btc.json"))
        self.assertEqual(result["summary"]["agent_state"], "ENGAGE")
        self.assertEqual(result["summary"]["verdict"], "LONG")
        self.assertEqual(result["action_plan"]["primary_action"], "LONG")
        self.assertGreaterEqual(result["summary"]["readiness_score"], 70)
        self.assertIn(result["summary"]["mission_posture"], {"PRESS", "ATTACK"})
        self.assertTrue(result["summary"]["dominant_playbook"])
        self.assertIn("mission_control", result)
        self.assertIn("desk_consensus", result)
        self.assertIn("scenario_map", result)
        self.assertGreaterEqual(result["meta"]["base_neuron_count"], 120)

    def test_super_agent_recovers_on_drawdown_case(self) -> None:
        result = self.agent.analyze_payload(load_example("drawdown_pause.json"))
        self.assertEqual(result["summary"]["agent_state"], "RECOVER")
        self.assertEqual(result["summary"]["verdict"], "NO TRADE")
        self.assertEqual(result["action_plan"]["execution_mode"], "risk_shutdown")
        self.assertEqual(result["summary"]["mission_posture"], "LOCKDOWN")
        self.assertEqual(
            result["mission_control"]["capital_allocation"]["capital_mode"],
            "flat",
        )
        self.assertEqual(result["desk_consensus"]["risk_desk"]["stance"], "blocked")
        self.assertTrue(result["risk_protocol"]["hard_stop_conditions"])


if __name__ == "__main__":
    unittest.main()
