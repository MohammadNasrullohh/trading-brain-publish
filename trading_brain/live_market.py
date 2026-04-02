from __future__ import annotations

import json
import math
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .utils import round_price


BINANCE_HTTP_BASE = "https://api.binance.com"
FRANKFURTER_HTTP_BASE = "https://api.frankfurter.dev"
STOOQ_HTTP_BASE = "https://stooq.com"
HTTP_TIMEOUT_SECONDS = 5
STREAM_TIMEOUT_SECONDS = 70

OANDA_REST_BASES = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

OANDA_STREAM_BASES = {
    "practice": "https://stream-fxpractice.oanda.com",
    "live": "https://stream-fxtrade.oanda.com",
}

BINANCE_INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
    "1mo": "1M",
}

OANDA_GRANULARITIES = {
    "5s": "S5",
    "10s": "S10",
    "15s": "S15",
    "30s": "S30",
    "1m": "M1",
    "2m": "M2",
    "4m": "M4",
    "5m": "M5",
    "10m": "M10",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "2h": "H2",
    "4h": "H4",
    "6h": "H6",
    "8h": "H8",
    "12h": "H12",
    "1d": "D",
    "1w": "W",
}

CONTEXT_CANDLE_LIMIT = 160

SESSION_WINDOWS = {
    "1m": 90,
    "3m": 80,
    "5m": 72,
    "15m": 48,
    "30m": 40,
    "1h": 30,
    "2h": 24,
    "4h": 18,
    "6h": 16,
    "8h": 14,
    "12h": 12,
    "1d": 10,
    "1w": 8,
}

STOOQ_SYMBOLS = {
    "XAUUSD": {
        "stooq_symbol": "xauusd",
        "market_type": "forex",
        "label": "Spot Gold",
    },
    "WTI": {
        "stooq_symbol": "cl.f",
        "market_type": "commodity",
        "label": "WTI Crude Oil",
    },
    "USOIL": {
        "stooq_symbol": "cl.f",
        "market_type": "commodity",
        "label": "WTI Crude Oil",
    },
    "CL.F": {
        "stooq_symbol": "cl.f",
        "market_type": "commodity",
        "label": "WTI Crude Oil",
    },
}

CRYPTO_DISCOVERY_STABLE_BASES = {
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "TUSD",
    "USDP",
    "DAI",
    "EUR",
    "AEUR",
}

CRYPTO_DISCOVERY_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

CRYPTO_DISCOVERY_CORE_BASES = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "DOGE",
    "ADA",
    "AVAX",
    "LINK",
    "TRX",
    "SUI",
    "TON",
]

FOREX_DISCOVERY_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCHF",
    "USDCAD",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "EURGBP",
    "AUDJPY",
    "EURCHF",
    "GBPCHF",
    "XAUUSD",
]

COMMODITY_DISCOVERY_SYMBOLS = [
    "WTI",
    "XAUUSD",
]

CRYPTO_DISCOVERY_ALIASES = {
    "leaders": "leaders",
    "leader": "leaders",
    "core": "leaders",
    "market": "leaders",
    "auto": "leaders",
    "all": "all_liquid",
    "all_liquid": "all_liquid",
    "broad": "all_liquid",
    "full": "all_liquid",
    "fresh": "fresh",
    "new": "fresh",
    "new_listing": "fresh",
    "new_listings": "fresh",
}


def _http_get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "TradingBrain/2.0",
            **(headers or {}),
        },
    )
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timeframe_minutes(timeframe: str | None) -> int | None:
    raw = str(timeframe or "").strip().lower()
    if not raw:
        return None
    if raw.endswith("m"):
        return int(raw[:-1] or 0) or None
    if raw.endswith("h"):
        return (int(raw[:-1] or 0) * 60) or None
    if raw.endswith("d"):
        return (int(raw[:-1] or 1) * 1440) or None
    if raw.endswith("w"):
        return (int(raw[:-1] or 1) * 10080) or None
    if raw in {"1mo", "1mth", "1month"}:
        return 43200
    return None


def _normalized_crypto_discovery_mode(value: str | None) -> str:
    raw = str(value or "leaders").strip().lower()
    return CRYPTO_DISCOVERY_ALIASES.get(raw, "leaders")


def normalize_symbol(symbol: str | None) -> str:
    raw = str(symbol or "").strip().upper()
    if ":" in raw:
        raw = raw.split(":")[-1]
    if raw.endswith("=X"):
        raw = raw[:-2]
    return raw.replace("/", "")


def infer_market_type(symbol: str | None, market_type: str | None = None) -> str:
    hint = str(market_type or "").strip().lower()
    if hint in {"crypto", "forex", "commodity"}:
        return hint

    normalized = normalize_symbol(symbol)
    if normalized in STOOQ_SYMBOLS:
        return STOOQ_SYMBOLS[normalized]["market_type"]
    if len(normalized) == 6 and normalized.isalpha():
        return "forex"
    if any(normalized.endswith(suffix) for suffix in ("USDT", "USDC", "BUSD", "FDUSD", "BTC", "ETH")):
        return "crypto"
    return "unknown"


def discover_market_symbols(
    market_type: str | None,
    *,
    base_symbol: str | None = None,
    timeframe: str | None = None,
    limit: int = 8,
    discovery_mode: str | None = None,
) -> list[str]:
    resolved_market = infer_market_type(base_symbol, market_type)
    normalized_base = normalize_symbol(base_symbol)
    max_items = max(1, min(int(limit or 8), 60))

    if resolved_market == "crypto":
        normalized_mode = _normalized_crypto_discovery_mode(discovery_mode)
        payload = _http_get_json(f"{BINANCE_HTTP_BASE}/api/v3/ticker/24hr")
        ranked: list[tuple[float, str, str]] = []
        min_quote_volume = (
            25_000_000
            if normalized_mode == "leaders"
            else 6_000_000
            if normalized_mode == "all_liquid"
            else 1_800_000
        )
        for item in payload if isinstance(payload, list) else []:
            symbol = normalize_symbol(item.get("symbol"))
            if not symbol.endswith("USDT"):
                continue
            if not symbol.isascii() or not symbol.isalnum():
                continue
            base_asset = symbol[:-4]
            if not base_asset or len(base_asset) > 12 or base_asset in CRYPTO_DISCOVERY_STABLE_BASES:
                continue
            if any(base_asset.endswith(suffix) for suffix in CRYPTO_DISCOVERY_LEVERAGED_SUFFIXES):
                continue

            quote_volume = _safe_float(item.get("quoteVolume"))
            change_percent = abs(_safe_float(item.get("priceChangePercent")))
            last_price = _safe_float(item.get("lastPrice"))
            trade_count = _safe_float(item.get("count"))
            if quote_volume <= min_quote_volume or last_price <= 0:
                continue

            score = (math.log10(max(quote_volume, 1.0)) * 7.5) + min(change_percent, 14.0)
            score += min(math.log10(max(trade_count, 1.0)) * 1.4, 6.0)
            if normalized_base and symbol == normalized_base:
                score += 4.0
            category = "core" if base_asset in CRYPTO_DISCOVERY_CORE_BASES else "satellite"
            if normalized_mode == "leaders" and category == "core":
                score += 2.2
            if normalized_mode == "all_liquid" and category == "satellite":
                score += 1.0
            ranked.append((score, symbol, category))

        ranked.sort(key=lambda item: item[0], reverse=True)
        discovered: list[str] = []
        if normalized_mode == "fresh":
            candidate_pool = [symbol for _, symbol, _ in ranked[: max(max_items * 5, 24)]]
            fresh_ranked: list[tuple[float, str]] = []
            scan_timeframe = normalize_timeframe(timeframe)
            for index, symbol in enumerate(candidate_pool):
                try:
                    candles = _binance_candles(symbol, scan_timeframe, CONTEXT_CANDLE_LIMIT)
                except Exception:  # noqa: BLE001
                    continue
                window_size = len(candles)
                if not candles or window_size >= CONTEXT_CANDLE_LIMIT:
                    continue
                completeness = window_size / CONTEXT_CANDLE_LIMIT
                momentum_bonus = max(0.0, 1.0 - completeness) * 26.0
                freshness_score = momentum_bonus + max(0.0, 12.0 - (index * 0.3))
                fresh_ranked.append((freshness_score, symbol))
            fresh_ranked.sort(key=lambda item: item[0], reverse=True)
            discovered = [symbol for _, symbol in fresh_ranked]
            if not discovered:
                discovered = [symbol for _, symbol, _ in ranked]
        elif normalized_mode == "leaders":
            core_first = [symbol for _, symbol, category in ranked if category == "core"]
            satellite = [symbol for _, symbol, category in ranked if category != "core"]
            discovered = core_first + satellite
        else:
            discovered = [symbol for _, symbol, _ in ranked]
        if normalized_base:
            discovered.insert(0, normalized_base)
        return list(dict.fromkeys(discovered))[:max_items]

    if resolved_market == "forex":
        discovered = [normalized_base] if normalized_base else []
        discovered.extend(FOREX_DISCOVERY_SYMBOLS)
        return list(dict.fromkeys(symbol for symbol in discovered if symbol))[:max_items]

    if resolved_market == "commodity":
        discovered = [normalized_base] if normalized_base else []
        discovered.extend(COMMODITY_DISCOVERY_SYMBOLS)
        return list(dict.fromkeys(symbol for symbol in discovered if symbol))[:max_items]

    fallback = [normalized_base] if normalized_base else []
    fallback.extend(["BTCUSDT", "XAUUSD", "WTI"])
    return list(dict.fromkeys(symbol for symbol in fallback if symbol))[:max_items]


def normalize_timeframe(timeframe: str | None) -> str:
    raw = str(timeframe or "15m").strip().lower()
    return BINANCE_INTERVALS.get(raw, "15m")


def normalize_oanda_granularity(timeframe: str | None) -> str:
    raw = str(timeframe or "15m").strip().lower()
    return OANDA_GRANULARITIES.get(raw, "M15")


def parse_forex_pair(symbol: str | None) -> tuple[str, str] | None:
    normalized = normalize_symbol(symbol)
    if len(normalized) == 6 and normalized.isalpha():
        return normalized[:3], normalized[3:]
    return None


def oanda_available() -> bool:
    return bool(os.getenv("OANDA_API_TOKEN") and os.getenv("OANDA_ACCOUNT_ID"))


def _oanda_environment() -> str:
    env_name = str(os.getenv("OANDA_ENV", "practice")).strip().lower()
    return env_name if env_name in OANDA_REST_BASES else "practice"


def _oanda_headers() -> dict[str, str]:
    token = os.getenv("OANDA_API_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept-Datetime-Format": "RFC3339",
    }


def _oanda_account_id() -> str:
    account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
    if not account_id:
        raise RuntimeError("OANDA_ACCOUNT_ID is not configured.")
    return account_id


def _oanda_instrument(symbol: str | None) -> str:
    pair = parse_forex_pair(symbol)
    if not pair:
        raise ValueError(f"Unsupported forex symbol: {symbol}")
    base, quote = pair
    return f"{base}_{quote}"


def _stooq_symbol_meta(symbol: str | None) -> dict[str, str] | None:
    normalized = normalize_symbol(symbol)
    return STOOQ_SYMBOLS.get(normalized)


def _oanda_rest_base() -> str:
    return OANDA_REST_BASES[_oanda_environment()]


def _oanda_stream_base() -> str:
    return OANDA_STREAM_BASES[_oanda_environment()]


def _oanda_mid_price(price_payload: dict[str, Any]) -> float:
    if price_payload.get("closeoutBid") and price_payload.get("closeoutAsk"):
        bid = float(price_payload["closeoutBid"])
        ask = float(price_payload["closeoutAsk"])
        return (bid + ask) / 2

    bid_bucket = (price_payload.get("bids") or [{}])[0]
    ask_bucket = (price_payload.get("asks") or [{}])[0]
    bid = float(bid_bucket.get("price", 0.0))
    ask = float(ask_bucket.get("price", 0.0))
    if bid and ask:
        return (bid + ask) / 2
    return bid or ask


def _oanda_snapshot_from_price_and_candle(
    config: dict[str, Any],
    price_payload: dict[str, Any],
    candle_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    price = _oanda_mid_price(price_payload)
    snapshot: dict[str, Any] = {
        **config,
        "price": price,
        "close": price,
        "event_time": price_payload.get("time"),
        "change_percent_24h": None,
        "bid": float(price_payload.get("closeoutBid", 0.0) or 0.0),
        "ask": float(price_payload.get("closeoutAsk", 0.0) or 0.0),
    }

    if candle_payload:
        mid = candle_payload.get("mid") or {}
        snapshot.update(
            {
                "open": float(mid.get("o", price)),
                "high": float(mid.get("h", price)),
                "low": float(mid.get("l", price)),
                "close": float(mid.get("c", price)),
                "candle_volume": int(candle_payload.get("volume", 0) or 0),
                "event_time": candle_payload.get("time") or snapshot["event_time"],
            }
        )
    else:
        snapshot.update(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
        )

    return snapshot


def _stooq_csv_row(symbol: str) -> dict[str, Any]:
    query = urlencode({"s": symbol, "f": "sd2t2ohlcv", "h": "", "e": "csv"})
    request = Request(
        f"{STOOQ_HTTP_BASE}/q/l/?{query}",
        headers={
            "Accept": "text/csv,text/plain",
            "User-Agent": "TradingBrain/2.0",
        },
    )
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8", errors="ignore")

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"No Stooq quote data for {symbol}")

    headers = [value.strip() for value in lines[0].split(",")]
    values = [value.strip() for value in lines[1].split(",")]
    row = dict(zip(headers, values))

    close = row.get("Close")
    if not close or close == "N/D":
        raise RuntimeError(f"Stooq quote is unavailable for {symbol}")
    return row


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None

    multiplier = 2 / (period + 1)
    ema_value = sum(values[:period]) / period
    for value in values[period:]:
        ema_value = (value - ema_value) * multiplier + ema_value
    return ema_value


def _rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for previous, current in zip(values, values[1:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for index in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[index]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[index]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(candles: list[dict[str, Any]], period: int = 14) -> float | None:
    if len(candles) <= period:
        return None

    true_ranges: list[float] = []
    for previous, current in zip(candles, candles[1:]):
        high = _safe_float(current["high"])
        low = _safe_float(current["low"])
        previous_close = _safe_float(previous["close"])
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))

    if len(true_ranges) < period:
        return None

    atr_value = sum(true_ranges[:period]) / period
    for current_range in true_ranges[period:]:
        atr_value = ((atr_value * (period - 1)) + current_range) / period
    return atr_value


def _macd_histogram(values: list[float]) -> float | None:
    if len(values) < 35:
        return None

    macd_line: list[float] = []
    for index in range(len(values)):
        window = values[: index + 1]
        ema_fast = _ema(window, 12)
        ema_slow = _ema(window, 26)
        if ema_fast is None or ema_slow is None:
            continue
        macd_line.append(ema_fast - ema_slow)

    signal = _ema(macd_line, 9)
    if signal is None or not macd_line:
        return None
    return macd_line[-1] - signal


def _stochastic(candles: list[dict[str, Any]], period: int = 14) -> float | None:
    if len(candles) < period:
        return None

    window = candles[-period:]
    low_value = min(_safe_float(candle["low"]) for candle in window)
    high_value = max(_safe_float(candle["high"]) for candle in window)
    close = _safe_float(window[-1]["close"])
    if high_value == low_value:
        return 50.0
    return ((close - low_value) / (high_value - low_value)) * 100


def _bollinger_position(values: list[float], period: int = 20) -> float | None:
    if len(values) < period:
        return None

    window = values[-period:]
    mean = sum(window) / period
    variance = sum((value - mean) ** 2 for value in window) / period
    deviation = math.sqrt(variance)
    upper = mean + (2 * deviation)
    lower = mean - (2 * deviation)
    if upper == lower:
        return 0.5
    return (window[-1] - lower) / (upper - lower)


def _cci(candles: list[dict[str, Any]], period: int = 20) -> float | None:
    if len(candles) < period:
        return None

    typical_prices = [
        (_safe_float(candle["high"]) + _safe_float(candle["low"]) + _safe_float(candle["close"])) / 3
        for candle in candles[-period:]
    ]
    sma = sum(typical_prices) / period
    mean_deviation = sum(abs(value - sma) for value in typical_prices) / period
    if mean_deviation == 0:
        return 0.0
    return (typical_prices[-1] - sma) / (0.015 * mean_deviation)


def _adx(candles: list[dict[str, Any]], period: int = 14) -> float | None:
    if len(candles) <= period + 1:
        return None

    true_ranges: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []

    for previous, current in zip(candles, candles[1:]):
        current_high = _safe_float(current["high"])
        current_low = _safe_float(current["low"])
        previous_high = _safe_float(previous["high"])
        previous_low = _safe_float(previous["low"])
        previous_close = _safe_float(previous["close"])

        up_move = current_high - previous_high
        down_move = previous_low - current_low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        true_ranges.append(max(current_high - current_low, abs(current_high - previous_close), abs(current_low - previous_close)))

    if len(true_ranges) < period:
        return None

    atr_value = sum(true_ranges[:period])
    plus_value = sum(plus_dm[:period])
    minus_value = sum(minus_dm[:period])
    dx_values: list[float] = []

    for index in range(period, len(true_ranges)):
        atr_value = atr_value - (atr_value / period) + true_ranges[index]
        plus_value = plus_value - (plus_value / period) + plus_dm[index]
        minus_value = minus_value - (minus_value / period) + minus_dm[index]
        if atr_value == 0:
            continue
        plus_di = (plus_value / atr_value) * 100
        minus_di = (minus_value / atr_value) * 100
        denominator = plus_di + minus_di
        if denominator == 0:
            continue
        dx_values.append((abs(plus_di - minus_di) / denominator) * 100)

    if not dx_values:
        return None
    if len(dx_values) <= period:
        return sum(dx_values) / len(dx_values)

    adx_value = sum(dx_values[:period]) / period
    for value in dx_values[period:]:
        adx_value = ((adx_value * (period - 1)) + value) / period
    return adx_value


def _vwap(candles: list[dict[str, Any]], period: int = 30) -> float | None:
    if not candles:
        return None

    window = candles[-period:]
    weighted_price = 0.0
    total_volume = 0.0
    for candle in window:
        high = _safe_float(candle["high"])
        low = _safe_float(candle["low"])
        close = _safe_float(candle["close"])
        volume = _safe_float(candle.get("volume"))
        typical_price = (high + low + close) / 3
        weighted_price += typical_price * volume
        total_volume += volume
    if total_volume == 0:
        return None
    return weighted_price / total_volume


def _volume_trend(candles: list[dict[str, Any]]) -> str | None:
    if len(candles) < 12:
        return None

    recent = candles[-5:]
    baseline = candles[-20:-5] or candles[:-5]
    if not baseline:
        return None

    recent_average = sum(_safe_float(candle.get("volume")) for candle in recent) / len(recent)
    baseline_average = sum(_safe_float(candle.get("volume")) for candle in baseline) / len(baseline)
    if baseline_average <= 0:
        return None

    if recent_average >= baseline_average * 1.15:
        return "rising"
    if recent_average <= baseline_average * 0.88:
        return "falling"
    return "stable"


def _level_candidates(candles: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    if len(candles) < 5:
        return values

    for index in range(2, len(candles) - 2):
        current = _safe_float(candles[index][field_name])
        neighbors = [
            _safe_float(candles[index - 2][field_name]),
            _safe_float(candles[index - 1][field_name]),
            _safe_float(candles[index + 1][field_name]),
            _safe_float(candles[index + 2][field_name]),
        ]
        if field_name == "high" and current >= max(neighbors):
            values.append(current)
        if field_name == "low" and current <= min(neighbors):
            values.append(current)
    return values


def _dedupe_levels(values: list[float], price: float, limit: int = 3) -> list[float]:
    unique: list[float] = []
    minimum_gap = max(price * 0.0012, 0.0003)
    for value in values:
        if any(abs(value - existing) < minimum_gap for existing in unique):
            continue
        unique.append(value)
        if len(unique) >= limit:
            break
    return unique


def _session_window(timeframe: str | None, total: int) -> int:
    base = SESSION_WINDOWS.get(str(timeframe or "15m").strip().lower(), 24)
    return max(6, min(base, total))


def _derived_bias(price: float, ema_fast: float | None, ema_slow: float | None) -> str | None:
    if ema_fast is None or ema_slow is None:
        return None
    if price >= ema_fast >= ema_slow:
        return "bullish"
    if price <= ema_fast <= ema_slow:
        return "bearish"
    return "netral"


def _derived_structure(candle: dict[str, Any]) -> str:
    high = _safe_float(candle["high"])
    low = _safe_float(candle["low"])
    close = _safe_float(candle["close"])
    if high <= low:
        return "netral"
    close_position = (close - low) / (high - low)
    if close_position >= 0.65:
        return "bullish"
    if close_position <= 0.35:
        return "bearish"
    return "netral"


def _derived_regime(price: float, atr_value: float | None, adx_value: float | None, ema_fast: float | None, ema_slow: float | None) -> str:
    if price <= 0:
        return "ranging"

    atr_pct = (atr_value / price * 100) if atr_value else None
    ema_gap_pct = (abs(ema_fast - ema_slow) / price * 100) if ema_fast is not None and ema_slow is not None else None

    if adx_value is not None and adx_value >= 24 and ema_gap_pct is not None and ema_gap_pct >= 0.18:
        return "trending"
    if atr_pct is not None and atr_pct >= 1.8:
        return "expansion"
    if atr_pct is not None and atr_pct <= 0.35:
        return "compression"
    return "ranging"


def _derive_live_context(candles: list[dict[str, Any]], price: float, timeframe: str) -> dict[str, Any]:
    closes = [_safe_float(candle["close"]) for candle in candles]
    highs = [_safe_float(candle["high"]) for candle in candles]
    lows = [_safe_float(candle["low"]) for candle in candles]

    ema_fast = _ema(closes, 9)
    ema_slow = _ema(closes, 21)
    rsi_value = _rsi(closes, 14)
    macd_histogram = _macd_histogram(closes)
    stochastic = _stochastic(candles, 14)
    atr_value = _atr(candles, 14)
    adx_value = _adx(candles, 14)
    vwap_value = _vwap(candles, 30)
    bollinger_position = _bollinger_position(closes, 20)
    cci_value = _cci(candles, 20)
    volume_trend = _volume_trend(candles)

    support_candidates = sorted(value for value in _level_candidates(candles, "low") if value < price)
    resistance_candidates = sorted(value for value in _level_candidates(candles, "high") if value > price)

    if not support_candidates:
        support_candidates = sorted(value for value in lows[:-1] if value < price)
    if not resistance_candidates:
        resistance_candidates = sorted(value for value in highs[:-1] if value > price)

    support = _dedupe_levels(list(reversed(support_candidates)), price)
    resistance = _dedupe_levels(resistance_candidates, price)

    if support:
        support = sorted(round_price(value) for value in support)
    if resistance:
        resistance = sorted(round_price(value) for value in resistance)

    session_size = _session_window(timeframe, len(candles))
    session_slice = candles[-session_size:]
    prior_slice = candles[-(session_size * 2):-session_size] or candles[:-1]

    latest_candle = candles[-1]
    structure_hint = _derived_structure(latest_candle)
    bias_hint = _derived_bias(price, ema_fast, ema_slow)
    regime_hint = _derived_regime(price, atr_value, adx_value, ema_fast, ema_slow)
    window_size = len(candles)
    completeness = window_size / CONTEXT_CANDLE_LIMIT if CONTEXT_CANDLE_LIMIT else 1.0
    if completeness <= 0.35:
        listing_profile = "new_listing"
    elif completeness <= 0.65:
        listing_profile = "fresh_listing"
    elif completeness <= 0.85:
        listing_profile = "young_market"
    else:
        listing_profile = "established"
    timeframe_minutes = _timeframe_minutes(timeframe)
    history_age_hours = round((window_size * timeframe_minutes) / 60, 2) if timeframe_minutes else None

    latest_volume = _safe_float(latest_candle.get("volume"))
    baseline_volume = sum(_safe_float(candle.get("volume")) for candle in session_slice[:-1]) / max(len(session_slice) - 1, 1)
    if baseline_volume > 0 and latest_volume >= baseline_volume * 1.12:
        session_quality = "high"
    elif baseline_volume > 0 and latest_volume <= baseline_volume * 0.82:
        session_quality = "thin"
    else:
        session_quality = "normal"

    return {
        "atr": round_price(atr_value),
        "levels": {
            "support": support,
            "resistance": resistance,
            "demand": support[:2],
            "supply": resistance[:2],
            "previous_high": round_price(max(_safe_float(candle["high"]) for candle in prior_slice)) if prior_slice else None,
            "previous_low": round_price(min(_safe_float(candle["low"]) for candle in prior_slice)) if prior_slice else None,
            "session_high": round_price(max(_safe_float(candle["high"]) for candle in session_slice)) if session_slice else None,
            "session_low": round_price(min(_safe_float(candle["low"]) for candle in session_slice)) if session_slice else None,
        },
        "indicators": {
            "ema_fast": round_price(ema_fast),
            "ema_slow": round_price(ema_slow),
            "rsi": round(rsi_value, 2) if rsi_value is not None else None,
            "macd_histogram": round_price(macd_histogram),
            "volume_trend": volume_trend,
            "vwap": round_price(vwap_value),
            "adx": round(adx_value, 2) if adx_value is not None else None,
            "stochastic": round(stochastic, 2) if stochastic is not None else None,
            "open_interest_delta": None,
            "funding_rate": None,
            "delta_volume": None,
            "bollinger_position": round(bollinger_position, 4) if bollinger_position is not None else None,
            "cci": round(cci_value, 2) if cci_value is not None else None,
        },
        "context": {
            "regime_hint": regime_hint,
            "structure_hint": structure_hint,
            "bias_hint": bias_hint,
            "session_quality_hint": session_quality,
        },
        "live_context": {
            "source": "derived_candles",
            "window_size": window_size,
            "history_completeness": round(completeness, 3),
            "listing_profile": listing_profile,
            "fresh_listing_candidate": listing_profile in {"new_listing", "fresh_listing", "young_market"},
            "history_age_hours": history_age_hours,
            "refreshed_at": latest_candle.get("time"),
        },
    }


def _binance_candles(symbol: str, timeframe: str, limit: int = CONTEXT_CANDLE_LIMIT) -> list[dict[str, Any]]:
    kline_query = urlencode({"symbol": symbol, "interval": timeframe, "limit": limit})
    raw_candles = _http_get_json(f"{BINANCE_HTTP_BASE}/api/v3/klines?{kline_query}")
    return [
        {
            "time": int(candle[6]),
            "open": _safe_float(candle[1]),
            "high": _safe_float(candle[2]),
            "low": _safe_float(candle[3]),
            "close": _safe_float(candle[4]),
            "volume": _safe_float(candle[5]),
        }
        for candle in raw_candles
    ]


def _oanda_candles(instrument: str, timeframe: str, count: int = CONTEXT_CANDLE_LIMIT) -> list[dict[str, Any]]:
    candle_query = urlencode(
        {
            "price": "M",
            "granularity": normalize_oanda_granularity(timeframe),
            "count": count,
        }
    )
    candle_payload = _http_get_json(
        f"{_oanda_rest_base()}/v3/instruments/{instrument}/candles?{candle_query}",
        headers=_oanda_headers(),
    )
    candles = candle_payload.get("candles") or []
    return [
        {
            "time": candle.get("time"),
            "open": _safe_float((candle.get("mid") or {}).get("o")),
            "high": _safe_float((candle.get("mid") or {}).get("h")),
            "low": _safe_float((candle.get("mid") or {}).get("l")),
            "close": _safe_float((candle.get("mid") or {}).get("c")),
            "volume": _safe_float(candle.get("volume")),
        }
        for candle in candles
    ]


def _parse_oanda_stream_payload(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    payload_type = payload.get("type")
    if payload_type == "HEARTBEAT":
        return {
            **config,
            "supported": True,
            "heartbeat": True,
            "event_time": payload.get("time"),
        }

    if payload_type != "PRICE":
        return None

    price = _oanda_mid_price(payload)
    return {
        **config,
        "supported": True,
        "price": price,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "event_time": payload.get("time"),
        "change_percent_24h": None,
        "bid": float(payload.get("closeoutBid", 0.0) or 0.0),
        "ask": float(payload.get("closeoutAsk", 0.0) or 0.0),
    }


def iter_oanda_live_stream(symbol: str | None, timeframe: str | None, market_type: str | None = "forex"):
    config = build_live_config(symbol, timeframe, market_type)
    if config.get("provider") != "oanda":
        raise RuntimeError("OANDA stream is not configured for this request.")

    account_id = _oanda_account_id()
    stream_query = urlencode({
        "instruments": config["instrument"],
        "snapshot": "false",
    })
    stream_url = f"{_oanda_stream_base()}/v3/accounts/{account_id}/pricing/stream?{stream_query}"
    request = Request(
        stream_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "TradingBrain/2.0",
            **_oanda_headers(),
        },
    )

    with urlopen(request, timeout=STREAM_TIMEOUT_SECONDS) as response:
        for raw_line in response:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            payload = json.loads(line)
            normalized = _parse_oanda_stream_payload(config, payload)
            if normalized:
                yield normalized


def build_live_config(symbol: str | None, timeframe: str | None, market_type: str | None = None) -> dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    resolved_market = infer_market_type(normalized_symbol, market_type)
    normalized_timeframe = normalize_timeframe(timeframe)
    stooq_meta = _stooq_symbol_meta(normalized_symbol)

    if resolved_market == "crypto" and normalized_symbol:
        stream_symbol = normalized_symbol.lower()
        websocket_url = (
            "wss://stream.binance.com:9443/stream?streams="
            f"{stream_symbol}@miniTicker/{stream_symbol}@kline_{normalized_timeframe}"
        )
        return {
            "supported": True,
            "market_type": "crypto",
            "symbol": normalized_symbol,
            "timeframe": normalized_timeframe,
            "provider": "binance",
            "transport": "websocket",
            "realtime": True,
            "poll_ms": 2500,
            "context_refresh_ms": 15000,
            "websocket": {
                "url": websocket_url,
                "streams": [
                    f"{stream_symbol}@miniTicker",
                    f"{stream_symbol}@kline_{normalized_timeframe}",
                ],
            },
            "source_note": "Live tick and kline stream from Binance Spot public market data.",
        }

    pair = parse_forex_pair(normalized_symbol)
    if resolved_market == "forex" and pair:
        base, quote = pair
        if oanda_available():
            instrument = _oanda_instrument(normalized_symbol)
            return {
                "supported": True,
                "market_type": "forex",
                "symbol": normalized_symbol,
                "timeframe": normalized_timeframe,
                "provider": "oanda",
                "transport": "sse",
                "realtime": True,
                "poll_ms": 12000,
                "context_refresh_ms": 12000,
                "pair": {
                    "base": base,
                    "quote": quote,
                },
                "instrument": instrument,
                "stream_url": (
                    f"/api/live/stream?symbol={normalized_symbol}"
                    f"&timeframe={normalized_timeframe}&market_type=forex"
                ),
                "source_note": "Live FX broker feed via OANDA pricing stream.",
            }
        if stooq_meta:
            return {
                "supported": True,
                "market_type": stooq_meta["market_type"],
                "symbol": normalized_symbol,
                "timeframe": normalized_timeframe,
                "provider": "stooq",
                "transport": "poll",
                "realtime": False,
                "poll_ms": 20000,
                "context_refresh_ms": 20000,
                "stooq_symbol": stooq_meta["stooq_symbol"],
                "source_note": f"Reference quote feed for {stooq_meta['label']} via Stooq public market data.",
            }
        return {
            "supported": True,
            "market_type": "forex",
            "symbol": normalized_symbol,
            "timeframe": normalized_timeframe,
            "provider": "frankfurter",
            "transport": "poll",
            "realtime": False,
            "poll_ms": 15000,
            "context_refresh_ms": 15000,
            "pair": {
                "base": base,
                "quote": quote,
            },
            "source_note": "Reference FX rates from Frankfurter. Useful for live monitoring, but not a tick-by-tick feed.",
        }

    if resolved_market == "commodity" and stooq_meta:
        return {
            "supported": True,
            "market_type": "commodity",
            "symbol": normalized_symbol,
            "timeframe": normalized_timeframe,
            "provider": "stooq",
            "transport": "poll",
            "realtime": False,
            "poll_ms": 20000,
            "context_refresh_ms": 20000,
            "stooq_symbol": stooq_meta["stooq_symbol"],
            "source_note": f"Reference quote feed for {stooq_meta['label']} via Stooq public market data.",
        }

    return {
        "supported": False,
        "market_type": resolved_market,
        "symbol": normalized_symbol,
        "timeframe": normalized_timeframe,
        "provider": "none",
        "transport": "none",
        "realtime": False,
        "poll_ms": 0,
        "source_note": "No live feed provider configured for this market.",
    }


def _fetch_binance_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    symbol = config["symbol"]
    timeframe = config["timeframe"]

    ticker_query = urlencode({"symbol": symbol})
    ticker = _http_get_json(f"{BINANCE_HTTP_BASE}/api/v3/ticker/24hr?{ticker_query}")
    candles = _binance_candles(symbol, timeframe)
    candle = candles[-1]
    context = _derive_live_context(candles, _safe_float(ticker["lastPrice"]), timeframe)

    return {
        **config,
        "price": float(ticker["lastPrice"]),
        "change_percent_24h": float(ticker["priceChangePercent"]),
        "open_24h": float(ticker["openPrice"]),
        "high_24h": float(ticker["highPrice"]),
        "low_24h": float(ticker["lowPrice"]),
        "volume_24h": float(ticker["volume"]),
        "quote_volume_24h": float(ticker["quoteVolume"]),
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "candle_volume": candle["volume"],
        "event_time": int(ticker["closeTime"]),
        **context,
    }


def _fetch_frankfurter_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    base = config["pair"]["base"]
    quote = config["pair"]["quote"]
    rate = _http_get_json(f"{FRANKFURTER_HTTP_BASE}/v2/rate/{base}/{quote}")
    price = float(rate["rate"])
    return {
        **config,
        "price": price,
        "change_percent_24h": None,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "event_time": rate["date"],
        "reference_date": rate["date"],
    }


def _fetch_stooq_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    row = _stooq_csv_row(config["stooq_symbol"])
    price = _safe_float(row.get("Close"))
    open_price = _safe_float(row.get("Open"), price)
    high_price = _safe_float(row.get("High"), price)
    low_price = _safe_float(row.get("Low"), price)
    range_value = max(high_price - low_price, price * 0.0025)
    atr_value = round_price(range_value)

    support_candidates = sorted({
        round_price(low_price),
        round_price(price - (range_value * 0.35)),
        round_price(price - (range_value * 0.65)),
    })
    resistance_candidates = sorted({
        round_price(high_price),
        round_price(price + (range_value * 0.35)),
        round_price(price + (range_value * 0.65)),
    })
    support = [value for value in support_candidates if value < price]
    resistance = [value for value in resistance_candidates if value > price]

    if not support:
        support = [round_price(price - range_value * 0.4)]
    if not resistance:
        resistance = [round_price(price + range_value * 0.4)]

    structure_hint = _derived_structure(
        {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": price,
        }
    )
    bias_hint = "bullish" if price >= open_price else "bearish"
    regime_hint = "expansion" if (range_value / price) * 100 >= 1.0 else "ranging"
    event_time = " ".join(
        part for part in (row.get("Date"), row.get("Time")) if part and part != "N/D"
    ) or row.get("Date") or row.get("Time")

    return {
        **config,
        "price": price,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": price,
        "change_percent_24h": None,
        "event_time": event_time,
        "atr": atr_value,
        "levels": {
            "support": support,
            "resistance": resistance,
            "demand": support[:2],
            "supply": resistance[:2],
            "previous_high": round_price(high_price),
            "previous_low": round_price(low_price),
            "session_high": round_price(high_price),
            "session_low": round_price(low_price),
        },
        "indicators": {
            "ema_fast": round_price((price + open_price) / 2),
            "ema_slow": round_price((price + open_price + low_price) / 3),
            "rsi": 56.0 if price >= open_price else 44.0,
            "macd_histogram": round_price(price - open_price),
            "volume_trend": "steady",
            "vwap": round_price((high_price + low_price + price) / 3),
            "adx": round(22.0 if regime_hint == "expansion" else 18.0, 2),
            "stochastic": round(68.0 if structure_hint == "bullish" else 32.0 if structure_hint == "bearish" else 50.0, 2),
            "open_interest_delta": None,
            "funding_rate": None,
            "delta_volume": None,
            "bollinger_position": 0.72 if structure_hint == "bullish" else 0.28 if structure_hint == "bearish" else 0.5,
            "cci": 96.0 if structure_hint == "bullish" else -96.0 if structure_hint == "bearish" else 0.0,
        },
        "context": {
            "regime_hint": regime_hint,
            "structure_hint": structure_hint,
            "bias_hint": bias_hint,
            "session_quality_hint": "normal",
            "market_type_hint": config["market_type"],
        },
        "live_context": {
            "source": "stooq_quote_derived",
            "window_size": 1,
            "refreshed_at": event_time,
        },
    }


def _fetch_oanda_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    account_id = _oanda_account_id()
    price_query = urlencode({"instruments": config["instrument"]})
    price_payload = _http_get_json(
        f"{_oanda_rest_base()}/v3/accounts/{account_id}/pricing?{price_query}",
        headers=_oanda_headers(),
    )
    prices = price_payload.get("prices") or []
    if not prices:
        raise RuntimeError(f"No OANDA pricing data for {config['instrument']}")

    candles = _oanda_candles(config["instrument"], config["timeframe"])
    latest_candle = candles[-1] if candles else None
    snapshot = _oanda_snapshot_from_price_and_candle(
        config,
        prices[0],
        {
            "time": latest_candle.get("time"),
            "volume": latest_candle.get("volume"),
            "mid": {
                "o": latest_candle.get("open"),
                "h": latest_candle.get("high"),
                "l": latest_candle.get("low"),
                "c": latest_candle.get("close"),
            },
        } if latest_candle else None,
    )
    if candles:
        snapshot.update(_derive_live_context(candles, snapshot["price"], config["timeframe"]))
    return snapshot


def fetch_live_snapshot(symbol: str | None, timeframe: str | None, market_type: str | None = None) -> dict[str, Any]:
    config = build_live_config(symbol, timeframe, market_type)
    if not config["supported"]:
        return config

    if config["market_type"] == "crypto":
        return _fetch_binance_snapshot(config)
    if config.get("provider") == "stooq":
        return _fetch_stooq_snapshot(config)
    if config.get("provider") == "oanda":
        return _fetch_oanda_snapshot(config)
    if config["market_type"] == "forex":
        return _fetch_frankfurter_snapshot(config)
    return config
