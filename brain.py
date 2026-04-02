from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_brain import SuperTradingAgent, TradingBrain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading brain CLI")
    parser.add_argument("input", help="Path ke file JSON market input")
    parser.add_argument(
        "--mode",
        choices=["brain", "super"],
        default="brain",
        help="Pilih mode analisa standar atau super-agent",
    )
    parser.add_argument("--compact", action="store_true", help="Output JSON tanpa indent")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    if args.mode == "super":
        agent = SuperTradingAgent()
        result = agent.analyze_payload(payload)
    else:
        brain = TradingBrain()
        result = brain.analyze_payload(payload)
    if args.compact:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
