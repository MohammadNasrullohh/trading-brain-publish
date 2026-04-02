from __future__ import annotations

from typing import Any

from .models import BrainState, MarketInput
from .neurons import ALL_NEURONS


class TradingBrain:
    def __init__(self) -> None:
        self.neurons = ALL_NEURONS

    def analyze(self, market_input: MarketInput) -> dict[str, Any]:
        state = BrainState(market=market_input)
        state.features["neuron_count"] = len(self.neurons)
        for neuron in self.neurons:
            neuron.run(state)
        return state.output

    def analyze_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        market_input = MarketInput.from_dict(payload)
        return self.analyze(market_input)
