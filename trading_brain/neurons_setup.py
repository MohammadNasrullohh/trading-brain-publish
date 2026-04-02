from __future__ import annotations

from .models import BrainState
from .neuron_base import Neuron, price_anchor
from .utils import distance_pct


class LongSetupNeuron(Neuron):
    name = "neuron11_long_setup"

    def run(self, state: BrainState) -> None:
        market = state.market
        price = price_anchor(state)
        support = state.features.get("nearest_support")
        structure = state.features.get("structure")
        bias = state.features.get("higher_timeframe_bias")
        components = 0

        if bias == "bullish":
            components += 1
        if structure == "bullish":
            components += 1
        if support is not None:
            support_distance = distance_pct(price, support)
            if support_distance is not None and support_distance <= 1.2:
                components += 1
                state.features["setup_type_long"] = "snr support zone"
        if market.indicators.ema_fast is not None and market.indicators.ema_slow is not None and price:
            if price >= market.indicators.ema_fast >= market.indicators.ema_slow:
                components += 1

        if components >= 3:
            self.boost(state, "long", 2.0, "setup long punya bias dan lokasi entry yang masuk akal")
        elif components == 2:
            self.boost(state, "long", 1.0, "setup long mulai terbentuk")


class ShortSetupNeuron(Neuron):
    name = "neuron12_short_setup"

    def run(self, state: BrainState) -> None:
        market = state.market
        price = price_anchor(state)
        resistance = state.features.get("nearest_resistance")
        structure = state.features.get("structure")
        bias = state.features.get("higher_timeframe_bias")
        components = 0

        if bias == "bearish":
            components += 1
        if structure == "bearish":
            components += 1
        if resistance is not None:
            resistance_distance = distance_pct(price, resistance)
            if resistance_distance is not None and resistance_distance <= 1.2:
                components += 1
                state.features["setup_type_short"] = "snr resistance zone"
        if market.indicators.ema_fast is not None and market.indicators.ema_slow is not None and price:
            if price <= market.indicators.ema_fast <= market.indicators.ema_slow:
                components += 1

        if components >= 3:
            self.boost(state, "short", 2.0, "setup short punya bias dan lokasi entry yang masuk akal")
        elif components == 2:
            self.boost(state, "short", 1.0, "setup short mulai terbentuk")


class BreakoutRetestNeuron(Neuron):
    name = "neuron13_breakout_retest"

    def run(self, state: BrainState) -> None:
        price = price_anchor(state)
        volume_trend = (state.market.indicators.volume_trend or "").strip().lower()
        broken_resistance = state.features.get("broken_resistance")
        broken_support = state.features.get("broken_support")

        if broken_resistance is not None:
            distance_from_break = distance_pct(price, broken_resistance)
            if distance_from_break is not None and distance_from_break <= 0.4 and volume_trend in {"rising", "strong", "expanding"}:
                state.features["setup_type_long"] = "snr support zone"
                self.boost(state, "long", 1.0, "resistance pecah berubah jadi support zone yang layak diretest")

        if broken_support is not None:
            distance_from_break = distance_pct(price, broken_support)
            if distance_from_break is not None and distance_from_break <= 0.4 and volume_trend in {"rising", "strong", "expanding"}:
                state.features["setup_type_short"] = "snr resistance zone"
                self.boost(state, "short", 1.0, "support pecah berubah jadi resistance zone yang layak diretest")


class PullbackReversalNeuron(Neuron):
    name = "neuron14_pullback_reversal"

    def run(self, state: BrainState) -> None:
        price = price_anchor(state)
        support = state.features.get("nearest_support")
        resistance = state.features.get("nearest_resistance")
        bias = state.features.get("higher_timeframe_bias")
        rsi = state.market.indicators.rsi

        if bias == "bullish" and support is not None:
            support_distance = distance_pct(price, support)
            if support_distance is not None and support_distance <= 0.7:
                state.features["setup_type_long"] = "snr support zone"
                self.boost(state, "long", 0.5, "harga dekat area pullback bullish")

        if bias == "bearish" and resistance is not None:
            resistance_distance = distance_pct(price, resistance)
            if resistance_distance is not None and resistance_distance <= 0.7:
                state.features["setup_type_short"] = "snr resistance zone"
                self.boost(state, "short", 0.5, "harga dekat area pullback bearish")

        if bias == "netral" and rsi is not None:
            if support is not None and distance_pct(price, support) is not None and distance_pct(price, support) <= 0.25 and rsi <= 35:
                state.add_warning("ada potensi reversal dari support, tetapi konfirmasi belum kuat")
            if resistance is not None and distance_pct(price, resistance) is not None and distance_pct(price, resistance) <= 0.25 and rsi >= 65:
                state.add_warning("ada potensi reversal dari resistance, tetapi konfirmasi belum kuat")


class ConfluenceScoreNeuron(Neuron):
    name = "neuron15_confluence_score"

    def run(self, state: BrainState) -> None:
        state.features["confluence_long"] = round(state.long_score, 2)
        state.features["confluence_short"] = round(state.short_score, 2)
