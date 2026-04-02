from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .claw_research import build_claw_research, merge_claw_research
from .engine import TradingBrain
from .self_training import get_training_dashboard
from .signal_memory import build_memory_dashboard
from .signal_memory import learning_context as signal_learning_context
from .signal_memory import register_signal
from .super_agent import SuperTradingAgent


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "visuals" / "neuron_scene_data.json"
EXAMPLES_DIR = ROOT / "examples"

MODE_INDEX = {
    "LONG": 0,
    "SHORT": 1,
    "WAIT": 2,
    "NO_TRADE": 3,
}

BASE_GROUP_WEIGHTS: dict[str, dict[str, float]] = {
    "LONG": {
        "core": 0.50,
        "market": 0.72,
        "signal": 0.82,
        "quality": 0.36,
        "adaptive": 0.78,
        "cortex": 0.82,
        "pro": 0.70,
        "setup": 0.92,
        "plan_micro": 0.86,
        "risk_micro": 0.20,
        "risk": 0.46,
        "output": 0.94,
    },
    "SHORT": {
        "core": 0.50,
        "market": 0.72,
        "signal": 0.82,
        "quality": 0.36,
        "adaptive": 0.78,
        "cortex": 0.82,
        "pro": 0.70,
        "setup": 0.92,
        "plan_micro": 0.86,
        "risk_micro": 0.20,
        "risk": 0.46,
        "output": 0.94,
    },
    "WAIT": {
        "core": 0.40,
        "market": 0.68,
        "signal": 0.45,
        "quality": 0.80,
        "adaptive": 0.74,
        "cortex": 0.76,
        "pro": 0.72,
        "setup": 0.42,
        "plan_micro": 0.26,
        "risk_micro": 0.58,
        "risk": 0.78,
        "output": 0.66,
    },
    "NO_TRADE": {
        "core": 0.34,
        "market": 0.46,
        "signal": 0.18,
        "quality": 0.80,
        "adaptive": 0.88,
        "cortex": 0.72,
        "pro": 0.56,
        "setup": 0.14,
        "plan_micro": 0.08,
        "risk_micro": 0.90,
        "risk": 0.96,
        "output": 0.84,
    },
}

STOPWORDS = {
    "yang",
    "dan",
    "atau",
    "untuk",
    "dari",
    "saat",
    "lebih",
    "akan",
    "pada",
    "dengan",
    "sebagai",
    "agar",
    "karena",
    "sudah",
    "belum",
    "bila",
    "jika",
    "tetap",
    "mode",
    "brain",
    "trading",
    "market",
    "layer",
    "utama",
    "micro",
}

PAIR_TEMPLATES: dict[str, dict[str, Any]] = {
    "BTC": {
        "key": "BTC",
        "label": "BTC Momentum Desk",
        "character": "24/7 momentum, crowding, dan liquidity sweep.",
        "template_name": "BTC Liquidity Engine",
        "tempo": "Fast continuation / breakout",
        "bias_lens": "Microstructure + orderflow",
        "description": "Template ini menekankan momentum, funding, orderbook pressure, breakout quality, dan conviction risk.",
        "palette": ["#f7931a", "#66d9ef", "#ffd166"],
        "group_biases": {"market": 0.08, "pro": 0.14, "setup": 0.08, "risk": 0.04},
        "stage_biases": {"market": 0.04, "pro": 0.08, "setup": 0.04},
        "node_titles": ["Crypto Microstruct", "Orderbook Pressure", "Breakout Quality", "Trend Continuation", "Liquidity Context"],
        "keywords": ["crypto", "funding", "open interest", "orderbook", "liquidity", "breakout", "momentum", "trend"],
        "focus_candidates": ["Crypto Microstruct", "Orderbook Pressure", "Breakout Quality"],
        "tags": ["Momentum", "Liquidity", "Crowding", "Breakout"],
    },
    "XAUUSD": {
        "key": "XAUUSD",
        "label": "Gold Macro Desk",
        "character": "Macro-sensitive, reaction-heavy, dan sering mean-revert sebelum expansion.",
        "template_name": "Gold Event Reprice",
        "tempo": "Session reaction / macro repricing",
        "bias_lens": "Event sentiment + execution location",
        "description": "Template Gold mendorong pembacaan event sentiment, session quality, reference memory, dan entry yang efisien.",
        "palette": ["#ffd166", "#f4f1de", "#8db9ff"],
        "group_biases": {"market": 0.1, "pro": 0.08, "setup": 0.04, "risk": 0.12},
        "stage_biases": {"market": 0.04, "pro": 0.04, "risk": 0.08},
        "node_titles": ["Event Sentiment", "Mean Reversion", "Execution Location", "Reference Memory", "Session Context"],
        "keywords": ["macro", "event", "sentiment", "session", "mean reversion", "reference", "execution", "liquidity"],
        "focus_candidates": ["Event Sentiment", "Execution Location", "Mean Reversion"],
        "tags": ["Macro", "Session", "Mean Reversion", "Execution"],
    },
    "FOREX": {
        "key": "FOREX",
        "label": "FX Session Desk",
        "character": "Session-driven, spread-sensitive, dan menghukum entry yang terlambat.",
        "template_name": "FX Session Matrix",
        "tempo": "London / New York rotation",
        "bias_lens": "Session flow + execution location",
        "description": "Template FX menekankan session context, trend strength, spread discipline, mean reversion pressure, dan entry yang efisien.",
        "palette": ["#8db9ff", "#7ce0c3", "#d7e3ff"],
        "group_biases": {"market": 0.09, "pro": 0.08, "setup": 0.07, "risk": 0.11},
        "stage_biases": {"market": 0.05, "pro": 0.05, "risk": 0.06},
        "node_titles": ["Session Context", "Trend Strength", "Execution Location", "Mean Reversion", "Spread Slippage Guard"],
        "keywords": ["session", "trend", "execution", "mean reversion", "spread", "forex", "macro", "liquidity"],
        "focus_candidates": ["Session Context", "Execution Location", "Trend Strength"],
        "tags": ["Majors", "Session", "Execution", "Discipline"],
    },
    "WTI": {
        "key": "WTI",
        "label": "WTI Flow Desk",
        "character": "Headline-sensitive, volatile, dan rawan trap saat energi bereaksi ke berita.",
        "template_name": "WTI Volatility Grid",
        "tempo": "Expansion / trap / reclaim",
        "bias_lens": "Volatility + event risk",
        "description": "Template WTI menonjolkan volatility engine, trap detection, execution location, dan headline risk energi.",
        "palette": ["#ff9f68", "#ffd166", "#8db9ff"],
        "group_biases": {"market": 0.12, "pro": 0.08, "setup": 0.1, "risk": 0.1},
        "stage_biases": {"market": 0.06, "setup": 0.06, "risk": 0.04},
        "node_titles": ["Volatility Engine", "Trap Detection", "Execution Location", "Event Sentiment", "Liquidity Context"],
        "keywords": ["volatility", "trap", "execution", "event", "liquidity", "range", "breakout", "risk"],
        "focus_candidates": ["Volatility Engine", "Trap Detection", "Execution Location"],
        "tags": ["Volatility", "Trap", "Energy", "Execution"],
    },
}

FOREX_PAIR_VARIANTS: dict[str, dict[str, Any]] = {
    "EURUSD": {
        "label": "EURUSD Session Desk",
        "character": "Cenderung paling bersih untuk struktur, retest, dan follow-through saat overlap session.",
        "palette": ["#8db9ff", "#4dd4a0", "#dbe8ff"],
        "tags": ["Major", "Session", "Clean Structure", "Execution"],
    },
    "GBPUSD": {
        "label": "GBPUSD Drive Desk",
        "character": "Lebih impulsif, suka spike cepat, dan butuh timing entry yang disiplin.",
        "palette": ["#66d9ef", "#8df7c7", "#ffd166"],
        "tempo": "London drive / pullback",
        "tags": ["Major", "Momentum", "London", "Execution"],
    },
    "USDJPY": {
        "label": "USDJPY Velocity Desk",
        "character": "Cepat saat session pindah dan sering lanjut setelah break level penting.",
        "palette": ["#9ec2ff", "#ffd166", "#f4f1de"],
        "tempo": "Tokyo to New York momentum",
        "tags": ["Major", "Velocity", "Session", "Breakout"],
    },
    "AUDUSD": {
        "label": "AUDUSD Trend Desk",
        "character": "Lebih halus, enak dibaca saat trend berjalan dan risk sentiment jelas.",
        "palette": ["#7ce0c3", "#8db9ff", "#f4f1de"],
        "tags": ["Major", "Trend", "Risk Mood", "Session"],
    },
    "NZDUSD": {
        "label": "NZDUSD Flow Desk",
        "character": "Geraknya lebih ringan, tapi clean saat market risk-on atau risk-off jelas.",
        "palette": ["#9fe6d2", "#8db9ff", "#ffd6a5"],
        "tags": ["Major", "Flow", "Risk Mood", "Execution"],
    },
    "USDCHF": {
        "label": "USDCHF Defense Desk",
        "character": "Sering bergerak rapat dan lebih cocok dibaca dengan disiplin level dan timing.",
        "palette": ["#d6e4ff", "#8db9ff", "#7ce0c3"],
        "tags": ["Major", "Defense", "Range", "Execution"],
    },
    "USDCAD": {
        "label": "USDCAD Oil-Link Desk",
        "character": "Sering sensitif ke dolar dan oil flow, jadi bagus untuk konteks rotasi makro.",
        "palette": ["#8db9ff", "#ffbe76", "#e9f3ff"],
        "tags": ["Major", "Macro", "Oil Link", "Session"],
    },
    "EURJPY": {
        "label": "EURJPY Momentum Desk",
        "character": "Lebih agresif dari EURUSD dan sering memberi continuation yang bersih saat trend hidup.",
        "palette": ["#9cb7ff", "#ffd166", "#7ce0c3"],
        "tempo": "Momentum continuation",
        "tags": ["Cross", "Momentum", "Session", "Trend"],
    },
    "GBPJPY": {
        "label": "GBPJPY Expansion Desk",
        "character": "Volatil, ekspansif, dan butuh neuron risk yang lebih aktif sebelum entry.",
        "palette": ["#ff9f68", "#ffd166", "#8db9ff"],
        "tempo": "Expansion / retrace / expansion",
        "tags": ["Cross", "Expansion", "Volatility", "Execution"],
    },
    "EURGBP": {
        "label": "EURGBP Balance Desk",
        "character": "Sering lebih tenang dan range-driven, cocok untuk baca squeeze dan reversion.",
        "palette": ["#d7e3ff", "#8db9ff", "#7ce0c3"],
        "tempo": "Range rotation / squeeze",
        "tags": ["Cross", "Range", "Mean Reversion", "Control"],
    },
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def _keyword_fragments(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _normalize(text))
    return {word for word in words if len(word) >= 3 and word not in STOPWORDS}


def _mode_name(base_result: dict[str, Any]) -> str:
    verdict = base_result.get("summary", {}).get("verdict", "WAIT")
    if verdict == "NO TRADE":
        return "NO_TRADE"
    return verdict if verdict in MODE_INDEX else "WAIT"


def _base_result(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("brain_output", result)


def _active_plan(base_result: dict[str, Any]) -> dict[str, Any]:
    return base_result.get("plan") or base_result.get("conditional_plan") or {}


def _adaptive_profile(base_result: dict[str, Any]) -> dict[str, Any]:
    adaptive = base_result.get("adaptive")
    return adaptive if isinstance(adaptive, dict) else {}


def _normalized_symbol(base_result: dict[str, Any]) -> str:
    raw = str(base_result.get("summary", {}).get("symbol", "") or "").upper().replace("/", "")
    return raw.replace(":", "")


def _pair_template(base_result: dict[str, Any]) -> dict[str, Any] | None:
    symbol = _normalized_symbol(base_result)
    if symbol.startswith("BTC"):
        return PAIR_TEMPLATES["BTC"]
    if symbol == "XAUUSD":
        return PAIR_TEMPLATES["XAUUSD"]
    if symbol in {"WTI", "USOIL", "CL.F"}:
        return PAIR_TEMPLATES["WTI"]
    if len(symbol) == 6 and symbol.isalpha():
        template = copy.deepcopy(PAIR_TEMPLATES["FOREX"])
        variant = FOREX_PAIR_VARIANTS.get(symbol, {})
        template.update(variant)
        template["key"] = symbol
        if "label" not in variant:
            template["label"] = f"{symbol} FX Desk"
        if symbol.endswith("JPY") and "tempo" not in variant:
            template["tempo"] = "Session drive / momentum"
        if symbol.startswith("USD") and "bias_lens" not in variant:
            template["bias_lens"] = "Dollar flow + execution location"
        return template
    return None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _execution_profile(payload: dict[str, Any] | None) -> str:
    raw = str(
        (payload or {}).get("execution_profile")
        or ((payload or {}).get("context", {}) or {}).get("execution_profile")
        or "balanced"
    ).strip().lower()
    if raw in {"precision", "near_zero_float", "zero_float", "sniper", "a_plus"}:
        return "precision"
    return "balanced"


def _execution_profile_meta(profile: str) -> dict[str, str]:
    if profile == "precision":
        return {
            "execution_profile": "precision",
            "execution_profile_label": "Precision / Near-Zero Float",
            "floating_policy": "near_zero_float",
        }
    return {
        "execution_profile": "balanced",
        "execution_profile_label": "Balanced",
        "floating_policy": "balanced",
    }


def _entry_midpoint(entry_zone: list[Any] | None) -> float | None:
    values = [float(value) for value in (entry_zone or []) if _safe_float(value) is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return (min(values) + max(values)) / 2.0


def _entry_width(entry_zone: list[Any] | None) -> float | None:
    values = [float(value) for value in (entry_zone or []) if _safe_float(value) is not None]
    if len(values) < 2:
        return 0.0 if values else None
    return abs(max(values) - min(values))


def _precision_thresholds(payload: dict[str, Any], base_result: dict[str, Any]) -> dict[str, float]:
    summary = base_result.get("summary", {})
    style = str(payload.get("style") or "intraday").strip().lower()
    is_scalp = style in {"scalp", "scalping"}
    market_type = str(summary.get("market_type") or payload.get("market_type") or "").strip().lower()

    max_leverage = 5.0
    if market_type == "crypto":
        max_leverage = 4.0 if is_scalp else 5.0
    elif market_type == "forex":
        max_leverage = 6.0 if is_scalp else 8.0

    return {
        "min_confidence": 0.8 if is_scalp else 0.76,
        "min_rr": 1.0 if is_scalp else 1.25,
        "max_rr": 3.2 if is_scalp else 4.4,
        "max_warnings": 1.0 if is_scalp else 2.0,
        "max_friction_bps": 12.0 if is_scalp else 18.0,
        "max_account_heat": 0.42 if is_scalp else 0.5,
        "max_drawdown_percent": 2.0 if is_scalp else 2.8,
        "max_effective_risk": 0.34 if is_scalp else 0.5,
        "max_leverage": max_leverage,
        "max_entry_atr": 0.18 if is_scalp else 0.38,
        "max_stop_atr": 0.62 if is_scalp else 1.15,
        "max_tp1_atr": 1.5 if is_scalp else 3.4,
        "max_zone_atr": 0.18 if is_scalp else 0.42,
        "max_entry_percent": 0.12 if is_scalp else 0.3,
        "max_stop_percent": 0.34 if is_scalp else 0.8,
        "max_tp1_percent": 0.7 if is_scalp else 1.95,
        "max_zone_percent": 0.14 if is_scalp else 0.34,
    }


def _append_unique(items: list[str], message: str) -> None:
    if message and message not in items:
        items.append(message)


MTF_TIMEFRAMES = ("4h", "1h", "15m", "5m")
MTF_WEIGHTS = {
    "4h": 0.34,
    "1h": 0.28,
    "15m": 0.22,
    "5m": 0.16,
}


def _normalized_bias_hint(value: Any) -> str:
    raw = _normalize(str(value or ""))
    if any(token in raw for token in ("bull", "long", "up")):
        return "bullish"
    if any(token in raw for token in ("bear", "short", "down")):
        return "bearish"
    return "neutral"


def _snapshot_bias(snapshot: dict[str, Any]) -> str:
    context = snapshot.get("context", {}) if isinstance(snapshot.get("context"), dict) else {}
    bias = _normalized_bias_hint(context.get("bias_hint"))
    if bias != "neutral":
        return bias

    price = _safe_float(snapshot.get("price"), _safe_float(snapshot.get("close")))
    indicators = snapshot.get("indicators", {}) if isinstance(snapshot.get("indicators"), dict) else {}
    ema_fast = _safe_float(indicators.get("ema_fast"))
    ema_slow = _safe_float(indicators.get("ema_slow"))
    if price is not None and ema_fast is not None and ema_slow is not None:
        if price >= ema_fast >= ema_slow:
            return "bullish"
        if price <= ema_fast <= ema_slow:
            return "bearish"
    return "neutral"


def _build_mtf_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    raw_snapshots = payload.get("_mtf_snapshots")
    if not isinstance(raw_snapshots, dict):
        return None

    frames: list[dict[str, Any]] = []
    weighted_score = 0.0
    weighted_total = 0.0
    bull_weight = 0.0
    bear_weight = 0.0

    for timeframe in MTF_TIMEFRAMES:
        snapshot = raw_snapshots.get(timeframe)
        if not isinstance(snapshot, dict) or not snapshot:
            continue
        bias = _snapshot_bias(snapshot)
        structure = _normalize(
            str((snapshot.get("context") or {}).get("structure_hint", "neutral"))
        ) if isinstance(snapshot.get("context"), dict) else "neutral"
        regime = _normalize(
            str((snapshot.get("context") or {}).get("regime_hint", "ranging"))
        ) if isinstance(snapshot.get("context"), dict) else "ranging"
        weight = float(MTF_WEIGHTS.get(timeframe, 0.2))
        score = 1.0 if bias == "bullish" else -1.0 if bias == "bearish" else 0.0
        weighted_score += score * weight
        weighted_total += weight
        if bias == "bullish":
            bull_weight += weight
        elif bias == "bearish":
            bear_weight += weight
        frames.append(
            {
                "timeframe": timeframe,
                "bias": bias,
                "structure": structure or "neutral",
                "regime": regime or "ranging",
                "price": _safe_float(snapshot.get("price"), _safe_float(snapshot.get("close"))),
            }
        )

    if not frames or weighted_total <= 0:
        return None

    consensus_score = weighted_score / weighted_total
    if consensus_score >= 0.22:
        consensus_bias = "bullish"
    elif consensus_score <= -0.22:
        consensus_bias = "bearish"
    else:
        consensus_bias = "mixed"

    active_timeframe = str(payload.get("timeframe") or "15m").strip().lower()
    active_frame = next((frame for frame in frames if frame["timeframe"] == active_timeframe), None)
    higher_frames = [frame for frame in frames if frame["timeframe"] in {"4h", "1h"}]
    execution_frames = [frame for frame in frames if frame["timeframe"] in {"15m", "5m"}]
    higher_biases = {frame["bias"] for frame in higher_frames if frame["bias"] != "neutral"}
    execution_biases = {frame["bias"] for frame in execution_frames if frame["bias"] != "neutral"}
    higher_bias = higher_frames[0]["bias"] if len(higher_biases) == 1 else "mixed" if higher_biases else "neutral"
    execution_bias = execution_frames[0]["bias"] if len(execution_biases) == 1 else "mixed" if execution_biases else "neutral"
    alignment_percent = round(max(bull_weight, bear_weight) / weighted_total * 100) if weighted_total else 0
    consensus_label = "Mixed"
    if consensus_bias == "bullish":
        consensus_label = "Bullish"
    elif consensus_bias == "bearish":
        consensus_label = "Bearish"

    note_parts = [f"MTF {consensus_label}"]
    if higher_bias not in {"neutral", "mixed"}:
        note_parts.append(f"HTF {higher_bias}")
    if execution_bias not in {"neutral", "mixed"}:
        note_parts.append(f"Entry {execution_bias}")
    if active_frame and active_frame["bias"] not in {"neutral", "mixed"}:
        note_parts.append(f"Active {active_frame['bias']}")

    return {
        "frames": frames,
        "active_timeframe": active_timeframe,
        "active_bias": active_frame["bias"] if active_frame else "neutral",
        "higher_timeframe_bias": higher_bias,
        "execution_bias": execution_bias,
        "consensus_bias": consensus_bias,
        "consensus_score": round(consensus_score, 3),
        "alignment_percent": alignment_percent,
        "conflicted": bool(bull_weight and bear_weight),
        "note": " • ".join(note_parts),
    }


def _apply_mtf_guidance(result: dict[str, Any], mtf_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not mtf_summary:
        return result

    base = _base_result(result)
    base_summary = base.setdefault("summary", {})
    top_summary = result.setdefault("summary", {}) if isinstance(result.get("summary"), dict) else {}
    verdict = str(base_summary.get("verdict") or top_summary.get("verdict") or "WAIT").strip().upper()
    confidence = _safe_float(base_summary.get("confidence"), _safe_float(top_summary.get("confidence"), 0.0)) or 0.0
    warnings = list(base.get("warnings", []))
    reasons = list(base.get("reasons", []))

    consensus_bias = str(mtf_summary.get("consensus_bias") or "mixed").strip().lower()
    higher_bias = str(mtf_summary.get("higher_timeframe_bias") or "neutral").strip().lower()
    active_bias = str(mtf_summary.get("active_bias") or "neutral").strip().lower()
    alignment_percent = int(mtf_summary.get("alignment_percent") or 0)
    note = str(mtf_summary.get("note") or "").strip()

    confidence_delta = 0.0
    if verdict == "LONG":
        if consensus_bias == "bullish" and higher_bias == "bullish":
            confidence_delta += 0.04
            _append_unique(reasons, "4H/1H/15M/5M selaras bullish.")
        elif consensus_bias == "bearish" or higher_bias == "bearish":
            confidence_delta -= 0.06
            _append_unique(warnings, "Multi-timeframe belum sinkron untuk bias long.")
        elif active_bias == "bullish":
            confidence_delta += 0.02
    elif verdict == "SHORT":
        if consensus_bias == "bearish" and higher_bias == "bearish":
            confidence_delta += 0.04
            _append_unique(reasons, "4H/1H/15M/5M selaras bearish.")
        elif consensus_bias == "bullish" or higher_bias == "bullish":
            confidence_delta -= 0.06
            _append_unique(warnings, "Multi-timeframe belum sinkron untuk bias short.")
        elif active_bias == "bearish":
            confidence_delta += 0.02
    elif note:
        _append_unique(reasons, note)

    if verdict in {"LONG", "SHORT"} and alignment_percent < 58:
        confidence_delta -= 0.03
        _append_unique(warnings, "Alignment 4H/1H/15M/5M masih tipis.")

    if confidence_delta:
        adjusted = round(_clamp(confidence + confidence_delta, 0.01, 0.99), 4)
        base_summary["confidence"] = adjusted
        if top_summary:
            top_summary["confidence"] = adjusted
        mtf_summary["confidence_delta"] = round(confidence_delta, 3)

    base["warnings"] = warnings
    base["reasons"] = reasons
    base["mtf_summary"] = mtf_summary
    if top_summary is not None:
        top_summary["mtf_consensus"] = mtf_summary.get("consensus_bias")
        top_summary["mtf_alignment_percent"] = mtf_summary.get("alignment_percent")
    result["mtf_summary"] = mtf_summary
    return result


def _precision_guard(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    base = _base_result(result)
    summary = base.get("summary", {})
    risk = base.get("risk", {})
    context = base.get("context", {})
    plan = _active_plan(base)
    verdict = str(summary.get("verdict") or "WAIT").upper()
    thresholds = _precision_thresholds(payload, base)
    blockers = list(base.get("blockers", []))
    warnings = list(base.get("warnings", []))
    sentiment = payload.get("sentiment", {}) if isinstance(payload.get("sentiment"), dict) else {}

    if verdict not in {"LONG", "SHORT"}:
        return {
            "profile": "precision",
            "status": "watch",
            "floating_risk": "standby",
            "notes": ["Precision mode menunggu setup directional yang cukup rapat, tapi filter sekarang lebih adaptif."],
            "fallback_verdict": verdict,
            "prime_setup": False,
            "thresholds": thresholds,
            "metrics": {},
        }

    confidence = _safe_float(summary.get("confidence"), 0.0) or 0.0
    rr = _safe_float(plan.get("risk_reward"))
    atr = _safe_float(payload.get("atr"), 0.0) or 0.0
    price = _safe_float(payload.get("price"), _safe_float(payload.get("close"), 0.0)) or 0.0
    entry_mid = _entry_midpoint(plan.get("entry_zone"))
    entry_width = _entry_width(plan.get("entry_zone"))
    stop_loss = _safe_float(plan.get("stop_loss"))
    take_profit_1 = _safe_float(plan.get("take_profit_1"))
    entry_drift = abs(price - entry_mid) if entry_mid is not None and price else None
    stop_distance = abs(entry_mid - stop_loss) if entry_mid is not None and stop_loss is not None else None
    tp1_distance = abs(take_profit_1 - entry_mid) if entry_mid is not None and take_profit_1 is not None else None

    entry_drift_atr = (entry_drift / atr) if atr and entry_drift is not None else None
    stop_distance_atr = (stop_distance / atr) if atr and stop_distance is not None else None
    tp1_distance_atr = (tp1_distance / atr) if atr and tp1_distance is not None else None
    zone_width_atr = (entry_width / atr) if atr and entry_width is not None else None

    entry_drift_pct = ((entry_drift / price) * 100.0) if price and entry_drift is not None else None
    stop_distance_pct = ((stop_distance / entry_mid) * 100.0) if entry_mid and stop_distance is not None else None
    tp1_distance_pct = ((tp1_distance / entry_mid) * 100.0) if entry_mid and tp1_distance is not None else None
    zone_width_pct = ((entry_width / entry_mid) * 100.0) if entry_mid and entry_width is not None else None

    effective_risk = _safe_float(risk.get("effective_risk_percent"), _safe_float(risk.get("max_risk_percent"), 0.0)) or 0.0
    leverage = _safe_float(risk.get("leverage"), 0.0) or 0.0
    friction = _safe_float(risk.get("friction_bps"), 0.0) or 0.0
    account_heat = _safe_float(risk.get("account_heat"), 0.0) or 0.0
    drawdown = _safe_float(risk.get("current_drawdown_percent"), 0.0) or 0.0

    hard_failures: list[str] = []
    soft_failures: list[str] = []

    if not plan.get("entry_zone") or stop_loss is None or take_profit_1 is None:
        hard_failures.append("plan belum lengkap untuk precision execution")
    if confidence < thresholds["min_confidence"]:
        soft_failures.append(f"confidence {round(confidence * 100)}% di bawah standar precision")
    if blockers:
        hard_failures.append("masih ada blocker aktif di brain")
    if len(warnings) > thresholds["max_warnings"]:
        soft_failures.append("warning terlalu banyak untuk mode precision")
    if sentiment.get("headline_risk"):
        hard_failures.append("headline risk masih aktif")
    if sentiment.get("macro_risk"):
        hard_failures.append("macro risk masih aktif")
    if rr is None or rr < thresholds["min_rr"]:
        soft_failures.append("reward-to-risk belum cukup rapat")
    elif rr > thresholds["max_rr"]:
        soft_failures.append("target terlalu jauh untuk near-zero float bias")

    if leverage > thresholds["max_leverage"]:
        hard_failures.append("leverage terlalu agresif untuk precision mode")
    if friction > thresholds["max_friction_bps"]:
        hard_failures.append("friction terlalu tinggi untuk entry cepat")
    if account_heat > thresholds["max_account_heat"]:
        hard_failures.append("account heat masih terlalu tinggi")
    if drawdown > thresholds["max_drawdown_percent"]:
        hard_failures.append("drawdown akun belum cukup ringan")
    if effective_risk > thresholds["max_effective_risk"]:
        soft_failures.append("risk per trade masih terlalu besar")

    session_quality = str(context.get("session_quality") or "").strip().lower()
    if session_quality == "low":
        soft_failures.append("session quality masih rendah")

    regime = str(context.get("regime") or "").strip().lower()
    if regime == "choppy":
        hard_failures.append("regime masih choppy untuk precision entry")

    if entry_drift_atr is not None and entry_drift_atr > thresholds["max_entry_atr"]:
        soft_failures.append("harga sudah terlalu jauh dari entry aktif")
    elif entry_drift_pct is not None and entry_drift_pct > thresholds["max_entry_percent"]:
        soft_failures.append("harga menjauh dari entry lebih cepat dari toleransi precision")

    if stop_distance_atr is not None and stop_distance_atr > thresholds["max_stop_atr"]:
        soft_failures.append("stop loss masih terlalu lebar")
    elif stop_distance_pct is not None and stop_distance_pct > thresholds["max_stop_percent"]:
        soft_failures.append("stop teknikal masih terlalu jauh")

    if tp1_distance_atr is not None and tp1_distance_atr > thresholds["max_tp1_atr"]:
        soft_failures.append("TP1 terlalu jauh sehingga rawan floating lama")
    elif tp1_distance_pct is not None and tp1_distance_pct > thresholds["max_tp1_percent"]:
        soft_failures.append("target pertama terlalu jauh untuk execution cepat")

    if zone_width_atr is not None and zone_width_atr > thresholds["max_zone_atr"]:
        soft_failures.append("zona entry terlalu lebar")
    elif zone_width_pct is not None and zone_width_pct > thresholds["max_zone_percent"]:
        soft_failures.append("lebar zona entry tidak efisien")

    notes = hard_failures + soft_failures
    blocked = bool(notes)
    fallback_verdict = verdict
    if blocked:
        fallback_verdict = "NO TRADE" if hard_failures or len(notes) >= 4 else "WAIT"

    return {
        "profile": "precision",
        "status": "blocked" if blocked else "pass",
        "floating_risk": "elevated" if blocked else "low",
        "notes": notes or ["Precision mode pass: entry rapat, invalidation efisien, dan floating risk rendah."],
        "fallback_verdict": fallback_verdict,
        "prime_setup": not blocked,
        "thresholds": thresholds,
        "metrics": {
            "confidence": round(confidence, 3),
            "risk_reward": rr,
            "entry_drift_atr": None if entry_drift_atr is None else round(entry_drift_atr, 3),
            "stop_distance_atr": None if stop_distance_atr is None else round(stop_distance_atr, 3),
            "tp1_distance_atr": None if tp1_distance_atr is None else round(tp1_distance_atr, 3),
            "zone_width_atr": None if zone_width_atr is None else round(zone_width_atr, 3),
            "entry_drift_percent": None if entry_drift_pct is None else round(entry_drift_pct, 3),
            "stop_distance_percent": None if stop_distance_pct is None else round(stop_distance_pct, 3),
            "tp1_distance_percent": None if tp1_distance_pct is None else round(tp1_distance_pct, 3),
            "zone_width_percent": None if zone_width_pct is None else round(zone_width_pct, 3),
            "friction_bps": round(friction, 2),
            "effective_risk_percent": round(effective_risk, 3),
            "account_heat": round(account_heat, 3),
            "drawdown_percent": round(drawdown, 3),
        },
    }


def _apply_execution_profile(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    profile = _execution_profile(payload)
    adjusted = copy.deepcopy(result)
    meta = _execution_profile_meta(profile)
    base = _base_result(adjusted)
    base_summary = base.get("summary", {})
    plan = _active_plan(base)
    for key, value in meta.items():
        base_summary[key] = value
        adjusted.get("summary", {}).setdefault(key, value)

    if profile != "precision":
        base["profile_guard"] = {
            "profile": "balanced",
            "status": "balanced",
            "floating_risk": "normal",
            "notes": ["Balanced mode membiarkan brain menilai setup dengan filter standar."],
        }
        adjusted["profile_guard"] = base["profile_guard"]
        return adjusted

    guard = _precision_guard(adjusted, payload)
    adjusted["profile_guard"] = guard
    base["profile_guard"] = guard

    if guard["status"] == "pass":
        _append_unique(base.setdefault("reasons", []), guard["notes"][0])
        if "summary" in adjusted:
            adjusted["summary"]["precision_status"] = "pass"
        return adjusted

    new_verdict = guard["fallback_verdict"]
    profile_note = f"precision mode: {guard['notes'][0]}"
    _append_unique(base.setdefault("warnings", []), profile_note)
    if new_verdict == "NO TRADE":
        _append_unique(base.setdefault("blockers", []), profile_note)

    base_summary["verdict"] = new_verdict
    base_summary["confidence"] = min(_safe_float(base_summary.get("confidence"), 0.0) or 0.0, 0.64 if new_verdict == "WAIT" else 0.52)
    if new_verdict == "WAIT":
        base["conditional_plan"] = plan
    else:
        base["conditional_plan"] = {}
    base["plan"] = {}

    if "summary" in adjusted:
        adjusted["summary"]["verdict"] = new_verdict
        adjusted["summary"]["confidence"] = base_summary["confidence"]
        adjusted["summary"]["precision_status"] = "blocked"
        adjusted["summary"]["setup_grade"] = "FILTERED"
        if "agent_state" in adjusted["summary"]:
            adjusted["summary"]["agent_state"] = "STALK" if new_verdict == "WAIT" else "DEFEND"
        if "mission_posture" in adjusted["summary"]:
            adjusted["summary"]["mission_posture"] = "AMBUSH" if new_verdict == "WAIT" else "SHIELD"
        if "dominant_playbook" in adjusted["summary"]:
            adjusted["summary"]["dominant_playbook"] = "precision_hold_fire" if new_verdict == "WAIT" else "precision_capital_preservation"

    action_plan = adjusted.get("action_plan")
    if isinstance(action_plan, dict):
        action_plan["primary_action"] = new_verdict
        action_plan["execution_mode"] = "conditional_execution" if new_verdict == "WAIT" else "capital_preservation"
        action_plan["entry_triggers"] = ["Precision mode menunggu setup lebih rapat sebelum mengizinkan entry baru."]
        action_plan["execution_steps"] = (
            [
                "Jangan paksa entry. Biarkan market kembali ke level rapat dan invalidation efisien.",
                "Aktifkan ulang evaluasi hanya bila price mendekat ke entry dengan noise yang lebih kecil.",
            ]
            if new_verdict == "WAIT"
            else [
                "Tetap flat. Setup ini tidak lolos filter near-zero float.",
                "Scan ulang level live sampai jarak entry, stop, dan TP1 kembali efisien.",
            ]
        )
        action_plan["skip_conditions"] = [profile_note, *action_plan.get("skip_conditions", [])][:4]
        action_plan["take_profit_logic"] = ["TP dinonaktifkan sementara sampai setup precision kembali valid."]
        action_plan["if_then_map"] = [
            {
                "if": "harga kembali dekat ke entry dengan blocker rendah",
                "then": "re-score setup sebelum entry",
            },
            {
                "if": "floating risk tetap tinggi",
                "then": "tetap no-trade",
            },
        ]

    mission_control = adjusted.get("mission_control")
    if isinstance(mission_control, dict):
        mission_control["mission_posture"] = "AMBUSH" if new_verdict == "WAIT" else "SHIELD"
        mission_control["operating_doctrine"] = (
            "Precision mode aktif: tunggu jarak entry yang rapat dan invalidation yang efisien."
            if new_verdict == "WAIT"
            else "Precision mode memveto setup ini untuk menjaga floating tetap serendah mungkin."
        )
        mission_control["constraint_stack"] = [profile_note, *mission_control.get("constraint_stack", [])][:4]
        capital_allocation = mission_control.get("capital_allocation")
        if isinstance(capital_allocation, dict):
            capital_allocation["capital_mode"] = "probe_risk" if new_verdict == "WAIT" else "flat"
            capital_allocation["sizing_style"] = "pilot_position" if new_verdict == "WAIT" else "no_new_positions"

    risk_protocol = adjusted.get("risk_protocol")
    if isinstance(risk_protocol, dict):
        if new_verdict == "WAIT":
            risk_protocol["effective_risk_percent"] = min(_safe_float(risk_protocol.get("effective_risk_percent"), 0.25) or 0.25, 0.25)
        else:
            risk_protocol["effective_risk_percent"] = 0.0
        risk_protocol["hard_stop_conditions"] = [profile_note, *risk_protocol.get("hard_stop_conditions", [])][:5]

    strategic_brief = adjusted.get("strategic_brief")
    if isinstance(strategic_brief, dict):
        strategic_brief["counter_thesis"] = f"Precision filter aktif: {'; '.join(guard['notes'][:2])}."
        strategic_brief["edge_summary"] = (
            "Edge ditahan dulu karena execution profile ini mengutamakan entry rapat dan floating serendah mungkin."
        )

    return adjusted


@lru_cache(maxsize=1)
def load_scene_data() -> dict[str, Any]:
    return json.loads(SCENE_PATH.read_text(encoding="utf-8"))


def list_example_files() -> list[str]:
    return sorted(path.name for path in EXAMPLES_DIR.glob("*.json"))


def load_example_payload(name: str) -> dict[str, Any]:
    safe_name = Path(name).name
    path = EXAMPLES_DIR / safe_name
    if not path.exists():
        raise FileNotFoundError(f"Example not found: {safe_name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _first_matching_node(scene: dict[str, Any], *titles: str) -> dict[str, Any] | None:
    wanted = {_normalize(title) for title in titles if title}
    for node in scene["nodes"]:
        if _normalize(node["title"]) in wanted:
            return node
    return None


def _focus_node(scene: dict[str, Any], base_result: dict[str, Any], pair_template: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = base_result.get("summary", {})
    context = base_result.get("context", {})
    risk = base_result.get("risk", {})
    adaptive = _adaptive_profile(base_result)
    plan = base_result.get("plan") or base_result.get("conditional_plan") or {}
    blockers = [text.lower() for text in base_result.get("blockers", [])]
    verdict = summary.get("verdict", "WAIT")
    setup_type = _normalize(plan.get("setup_type", ""))
    regime = context.get("regime", "")
    adaptive_cycle = _normalize(adaptive.get("growth_cycle", ""))

    if any("loss streak" in item or "recovery" in item for item in blockers):
        node = _first_matching_node(scene, "Recovery Mode")
        if node:
            return node
    if any("drawdown" in item or "loss harian" in item for item in blockers):
        node = _first_matching_node(scene, "Drawdown Guard", "Account Heat")
        if node:
            return node
    if any("no trade" in item for item in blockers):
        node = _first_matching_node(scene, "No Trade Filter")
        if node:
            return node

    if adaptive_cycle:
        node = _first_matching_node(scene, *(adaptive.get("focus_titles") or []))
        if node:
            return node
        if adaptive_cycle == "protect":
            node = _first_matching_node(scene, "Survival Instinct", "Anti Overfit Filter", "Confidence Memory")
            if node:
                return node
        if adaptive_cycle == "compound":
            node = _first_matching_node(scene, "Compounding Drive", "Pair Memory Resonance", "Opportunity Compression")
            if node:
                return node
        if adaptive_cycle in {"bootstrap", "explore"}:
            node = _first_matching_node(scene, "Adaptive Bootstrap", "Exploration Drive", "Market Memory Bridge")
            if node:
                return node

    if "snr support zone" in setup_type:
        node = _first_matching_node(scene, "Long Setup", "Key Levels", "Pullback Reversal")
        if node:
            return node
    if "snr resistance zone" in setup_type:
        node = _first_matching_node(scene, "Short Setup", "Key Levels", "Pullback Reversal")
        if node:
            return node

    if pair_template:
        node = _first_matching_node(scene, *(pair_template.get("focus_candidates") or []))
        if node:
            return node

    if verdict == "LONG":
        node = _first_matching_node(scene, "Long Setup", "Trend Continuation", "Breakout Retest")
        if node:
            return node
    if verdict == "SHORT":
        node = _first_matching_node(scene, "Short Setup", "Trap Detection")
        if node:
            return node
    if verdict == "WAIT":
        if regime == "ranging":
            node = _first_matching_node(scene, "Range Fade", "Execution Location")
            if node:
                return node
        node = _first_matching_node(scene, "Confluence Score", "Execution Location", "Liquidity Context")
        if node:
            return node

    if risk.get("current_drawdown_percent", 0) >= 4.0:
        node = _first_matching_node(scene, "Drawdown Guard")
        if node:
            return node

    return _first_matching_node(scene, "Verdict Gate") or scene["nodes"][-1]


def _build_keywords(base_result: dict[str, Any]) -> set[str]:
    summary = base_result.get("summary", {})
    context = base_result.get("context", {})
    adaptive = _adaptive_profile(base_result)
    plan = base_result.get("plan") or base_result.get("conditional_plan") or {}
    risk = base_result.get("risk", {})

    texts: list[str] = []
    texts.extend(base_result.get("reasons", []))
    texts.extend(base_result.get("warnings", []))
    texts.extend(base_result.get("blockers", []))
    texts.extend(base_result.get("notes", []))
    texts.extend(
        [
            summary.get("verdict", ""),
            summary.get("bias", ""),
            summary.get("market_type", ""),
            context.get("regime", ""),
            context.get("structure", ""),
            context.get("session_quality", ""),
            plan.get("setup_type", ""),
            adaptive.get("growth_cycle", ""),
            adaptive.get("adaptation_mode", ""),
            adaptive.get("memory_scope", ""),
            adaptive.get("note", ""),
        ]
    )

    if risk.get("current_drawdown_percent", 0) >= 4.0:
        texts.append("drawdown recovery account heat")
    if risk.get("loss_streak", 0) >= 2:
        texts.append("loss streak recovery")
    if risk.get("leverage", 0) >= 8:
        texts.append("leverage guard")

    verdict = summary.get("verdict", "")
    if verdict == "LONG":
        texts.append("bull long support snr zone retest target asymmetry")
    elif verdict == "SHORT":
        texts.append("bear short resistance snr zone retest target asymmetry")
    elif verdict == "WAIT":
        texts.append("quality support resistance execution confluence range")
    else:
        texts.append("risk no trade recovery drawdown account heat leverage")

    keywords: set[str] = set()
    for text in texts:
        keywords.update(_keyword_fragments(str(text)))
    return keywords


def _boost_from_keywords(text: str, keywords: set[str]) -> float:
    matches = sum(1 for keyword in keywords if keyword in text)
    return min(0.28, matches * 0.045)


def build_visual_state(result: dict[str, Any]) -> dict[str, Any]:
    scene = load_scene_data()
    base = _base_result(result)
    summary = base.get("summary", {})
    context = base.get("context", {})
    risk = base.get("risk", {})
    adaptive = _adaptive_profile(base)
    plan = base.get("plan") or base.get("conditional_plan") or {}
    blockers = base.get("blockers", [])
    warnings = base.get("warnings", [])
    mode_name = _mode_name(base)
    mode_index = MODE_INDEX[mode_name]
    keywords = _build_keywords(base)
    pair_template = _pair_template(base)
    pair_title_tokens = {_normalize(title) for title in (pair_template or {}).get("node_titles", [])}
    pair_keywords = {_normalize(keyword) for keyword in (pair_template or {}).get("keywords", [])}
    focus_node = _focus_node(scene, base, pair_template)

    market_type = _normalize(summary.get("market_type", ""))
    regime = _normalize(context.get("regime", ""))
    setup_type = _normalize(plan.get("setup_type", ""))
    structure = _normalize(context.get("structure", ""))
    has_plan = bool(plan)
    adaptive_cycle = _normalize(adaptive.get("growth_cycle", ""))
    adaptive_stage_biases = adaptive.get("stage_biases", {}) if isinstance(adaptive.get("stage_biases"), dict) else {}

    node_weights: dict[str, float] = {}

    for node in scene["nodes"]:
        text = _normalize(f"{node['title']} {node.get('description', '')} {node.get('group', '')} {node.get('stage', '')}")
        weight = BASE_GROUP_WEIGHTS[mode_name].get(node["group"], 0.22)
        weight += _boost_from_keywords(text, keywords)

        if pair_template:
            weight += pair_template.get("group_biases", {}).get(node["group"], 0.0)
            weight += pair_template.get("stage_biases", {}).get(node["stage"], 0.0)
            if _normalize(node["title"]) in pair_title_tokens:
                weight += 0.24
            if any(keyword in text for keyword in pair_keywords):
                weight += 0.12

        weight += float(adaptive_stage_biases.get(node["stage"], 0.0))
        weight += float(adaptive_stage_biases.get(node["group"], 0.0))

        if market_type and market_type in text:
            weight += 0.18
        if "crypto" in market_type and "crypto" in text:
            weight += 0.08
        if "forex" in market_type and "forex" in text:
            weight += 0.08

        if regime == "trending" and any(term in text for term in ("trend", "breakout", "momentum")):
            weight += 0.12
        if regime in {"ranging", "range", "choppy"} and any(term in text for term in ("range", "mean reversion", "trap", "execution")):
            weight += 0.14

        if "bullish" in structure and any(term in text for term in ("bull", "long", "support", "breakout")):
            weight += 0.12
        if "bearish" in structure and any(term in text for term in ("bear", "short", "resistance", "trap")):
            weight += 0.12

        if setup_type and setup_type in text:
            weight += 0.24
        if has_plan and any(term in text for term in ("target", "stop loss", "asymmetry", "risk budget")):
            weight += 0.08

        if blockers and any(term in text for term in ("recovery", "drawdown", "account heat", "no trade", "leverage", "risk")):
            weight += 0.16
        if warnings and any(term in text for term in ("liquidity", "spread", "wick", "trap", "quality")):
            weight += 0.08

        if adaptive_cycle == "compound" and any(term in text for term in ("memory", "compound", "continuation", "opportunity", "confidence")):
            weight += 0.14
        if adaptive_cycle == "protect" and any(term in text for term in ("survival", "overfit", "confidence", "risk", "recovery")):
            weight += 0.18
        if adaptive_cycle in {"bootstrap", "explore"} and any(term in text for term in ("adaptive", "exploration", "market memory", "mutation")):
            weight += 0.16

        if risk.get("current_drawdown_percent", 0) >= 4.0 and any(term in text for term in ("drawdown", "recovery", "account heat", "no trade")):
            weight += 0.18
        if risk.get("loss_streak", 0) >= 2 and any(term in text for term in ("recovery", "loss streak", "conviction", "no trade")):
            weight += 0.14
        if risk.get("leverage", 0) >= 8.0 and any(term in text for term in ("leverage", "risk", "no trade")):
            weight += 0.14

        if node["id"] == focus_node["id"]:
            weight += 0.30
        if node["group"] == "output":
            weight += 0.06

        node_weights[node["id"]] = round(_clamp(weight, 0.05, 1.0), 3)

    verdict_gate = _first_matching_node(scene, "Verdict Gate") or scene["nodes"][-1]
    return {
        "mode": mode_name,
        "mode_index": mode_index,
        "locked_node_id": focus_node["id"],
        "verdict_node_id": verdict_gate["id"],
        "focus_stage": focus_node["stage"],
        "focus_point": {
            "x": focus_node["x"],
            "y": focus_node["y"],
            "z": focus_node["z"],
        },
        "node_weights": node_weights,
        "pair_profile": pair_template,
        "adaptive_profile": adaptive,
    }


def analyze_for_web(
    payload: dict[str, Any],
    mode: str = "super",
    *,
    track_signal: bool = False,
    signal_source: str = "web",
    news_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_news_snapshot = news_snapshot
    if resolved_news_snapshot is None and isinstance(payload.get("_news_snapshot"), dict):
        resolved_news_snapshot = payload.get("_news_snapshot")
    mtf_summary = _build_mtf_summary(payload)
    if mode == "super":
        result = SuperTradingAgent().analyze_payload(payload)
    else:
        result = TradingBrain().analyze_payload(payload)
    result = _apply_execution_profile(result, payload)
    result = _apply_mtf_guidance(result, mtf_summary)
    summary = _base_result(result).get("summary", {})
    symbol = str(summary.get("symbol") or payload.get("symbol") or "").strip().upper()
    timeframe = str(summary.get("timeframe") or payload.get("timeframe") or "15m").strip().lower()
    style = str(payload.get("style") or summary.get("style") or "intraday").strip().lower()
    market_type = str(summary.get("market_type") or payload.get("market_type") or "").strip().lower()
    context = (
        register_signal(payload, result, source=signal_source)
        if track_signal
        else signal_learning_context(symbol, timeframe, style, market_type)
    )
    research = build_claw_research(
        payload,
        result,
        context,
        news_snapshot=resolved_news_snapshot,
        source=signal_source if track_signal else "analysis",
    )
    result = merge_claw_research(result, research)
    memory_dashboard = build_memory_dashboard(symbol, timeframe, style, market_type)
    training_dashboard = get_training_dashboard(symbol, timeframe, style, market_type)

    return {
        "mode": mode,
        "result": result,
        "visual_state": build_visual_state(result),
        "learning_context": {key: value for key, value in context.items() if key != "recent_history"},
        "signal_history": context.get("recent_history", []),
        "memory_dashboard": memory_dashboard,
        "training_dashboard": training_dashboard,
        "adaptive_profile": _adaptive_profile(_base_result(result)),
        "claw_research": research,
        "mtf_summary": mtf_summary,
    }
