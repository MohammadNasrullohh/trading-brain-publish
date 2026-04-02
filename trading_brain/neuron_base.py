from __future__ import annotations

from .models import BrainState, TradePlan
from .utils import distance_pct, midpoint, round_price


def price_anchor(state: BrainState) -> float:
    market = state.market
    return market.price or market.close or market.open or 0.0


def buffer_size(state: BrainState) -> float:
    price = price_anchor(state)
    atr = state.market.atr or price * 0.004
    return max(atr * 0.35, price * 0.0015)


def build_long_candidate(state: BrainState) -> TradePlan:
    price = price_anchor(state)
    support = state.features.get("nearest_support")
    support_distance = distance_pct(price, support)
    anchor = support if support is not None and support_distance is not None and support_distance <= 1.5 else price
    buffer = buffer_size(state)
    entry_low = min(anchor, price)
    entry_high = max(anchor, price)
    entry_mid = midpoint(entry_low, entry_high)
    stop_loss = (support - buffer) if support is not None else (entry_mid - (buffer * 1.5))
    if stop_loss >= entry_mid:
        stop_loss = entry_mid - max(buffer, price * 0.002)

    plan = TradePlan(
        direction="long",
        setup_type=str(state.features.get("setup_type_long", "snr support zone")),
        entry_zone=[round_price(entry_low), round_price(entry_high)],
        stop_loss=round_price(stop_loss),
        valid=True,
    )
    state.candidate_long = plan
    state.features["entry_mid_long"] = entry_mid
    return plan


def build_short_candidate(state: BrainState) -> TradePlan:
    price = price_anchor(state)
    resistance = state.features.get("nearest_resistance")
    resistance_distance = distance_pct(price, resistance)
    anchor = resistance if resistance is not None and resistance_distance is not None and resistance_distance <= 1.5 else price
    buffer = buffer_size(state)
    entry_low = min(anchor, price)
    entry_high = max(anchor, price)
    entry_mid = midpoint(entry_low, entry_high)
    stop_loss = (resistance + buffer) if resistance is not None else (entry_mid + (buffer * 1.5))
    if stop_loss <= entry_mid:
        stop_loss = entry_mid + max(buffer, price * 0.002)

    plan = TradePlan(
        direction="short",
        setup_type=str(state.features.get("setup_type_short", "snr resistance zone")),
        entry_zone=[round_price(entry_low), round_price(entry_high)],
        stop_loss=round_price(stop_loss),
        valid=True,
    )
    state.candidate_short = plan
    state.features["entry_mid_short"] = entry_mid
    return plan


class Neuron:
    name = "neuron"

    def run(self, state: BrainState) -> None:
        raise NotImplementedError

    def boost(self, state: BrainState, direction: str, score: float, reason: str | None = None) -> None:
        if direction == "long":
            state.long_score += score
            if reason:
                state.add_reason("long", reason)
        if direction == "short":
            state.short_score += score
            if reason:
                state.add_reason("short", reason)
