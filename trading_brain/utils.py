from __future__ import annotations

from typing import Iterable


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def round_price(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    magnitude = abs(numeric)
    if magnitude >= 100:
        digits = 2
    elif magnitude >= 1:
        digits = 5
    elif magnitude >= 0.01:
        digits = 6
    else:
        digits = 8
    return round(numeric, digits)


def midpoint(first: float, second: float) -> float:
    return (first + second) / 2.0


def distance_pct(price: float | None, level: float | None) -> float | None:
    if price in (None, 0) or level is None:
        return None
    return abs(price - level) / price * 100.0


def nearest_below(values: Iterable[float], reference: float) -> float | None:
    candidates = [value for value in values if value <= reference]
    return max(candidates) if candidates else None


def nearest_above(values: Iterable[float], reference: float) -> float | None:
    candidates = [value for value in values if value >= reference]
    return min(candidates) if candidates else None


def farthest_below(values: Iterable[float], reference: float) -> float | None:
    candidates = [value for value in values if value < reference]
    return max(candidates) if candidates else None


def farthest_above(values: Iterable[float], reference: float) -> float | None:
    candidates = [value for value in values if value > reference]
    return min(candidates) if candidates else None


def risk_reward(entry: float | None, stop: float | None, target: float | None, direction: str) -> float | None:
    if entry is None or stop is None or target is None:
        return None
    if direction == "long":
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    if risk <= 0:
        return None
    return reward / risk


def normalize_label(value: str | None, default: str = "unknown") -> str:
    if not value:
        return default
    return str(value).strip().lower()
