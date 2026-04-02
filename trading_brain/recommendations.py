from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .claw_research import apply_claw_research_bias, build_claw_research
from .live_market import discover_market_symbols, fetch_live_snapshot, infer_market_type
from .live_news import fetch_live_news
from .signal_memory import learning_context as signal_learning_context, register_signal
from .web_bridge import analyze_for_web


DEFAULT_SYMBOLS = ["ETHUSDT", "SOLUSDT", "XAUUSD", "WTI", "EURUSD", "GBPUSD", "USDJPY"]

MACRO_MIX_SYMBOLS = [
    "XAUUSD",
    "WTI",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "EURJPY",
    "GBPJPY",
    "EURGBP",
]

BALANCED_CROSS_MARKET_SYMBOLS = [
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "XAUUSD",
    "WTI",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
]

CRYPTO_RESEARCH_SYMBOLS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "BTCUSDT"]

SYMBOL_DEFAULTS: dict[str, dict[str, Any]] = {
    "BTCUSDT": {
        "market_type": "crypto",
        "session": "us",
        "risk": {
            "max_risk_percent": 0.5,
            "leverage": 3,
            "current_drawdown_percent": 1.2,
            "max_daily_loss_percent": 5.0,
            "loss_streak": 0,
        },
        "context": {
            "regime_hint": "normal",
            "session_quality_hint": "normal",
            "market_type_hint": "crypto",
        },
        "microstructure": {
            "spread": 6.0,
            "fee_bps": 4,
            "slippage_bps": 3,
            "weekend": False,
            "liquidity_score": 0.82,
            "orderbook_imbalance": 0.0,
        },
        "sentiment": {
            "score": 0.0,
            "headline_risk": False,
            "macro_risk": False,
            "correlation_bias": 0.3,
        },
    },
    "XAUUSD": {
        "market_type": "forex",
        "session": "overlap",
        "risk": {
            "max_risk_percent": 0.4,
            "leverage": 8,
            "current_drawdown_percent": 1.0,
            "max_daily_loss_percent": 4.0,
            "loss_streak": 0,
        },
        "context": {
            "regime_hint": "normal",
            "session_quality_hint": "high",
            "market_type_hint": "forex",
        },
        "microstructure": {
            "spread": 0.35,
            "fee_bps": 0,
            "slippage_bps": 2,
            "weekend": False,
            "liquidity_score": 0.76,
            "orderbook_imbalance": 0.0,
        },
        "sentiment": {
            "score": 0.0,
            "headline_risk": False,
            "macro_risk": False,
            "correlation_bias": 0.12,
        },
    },
    "WTI": {
        "market_type": "commodity",
        "session": "us",
        "risk": {
            "max_risk_percent": 0.35,
            "leverage": 5,
            "current_drawdown_percent": 1.1,
            "max_daily_loss_percent": 4.0,
            "loss_streak": 0,
        },
        "context": {
            "regime_hint": "normal",
            "session_quality_hint": "normal",
            "market_type_hint": "commodity",
        },
        "microstructure": {
            "spread": 0.05,
            "fee_bps": 0,
            "slippage_bps": 3,
            "weekend": False,
            "liquidity_score": 0.72,
            "orderbook_imbalance": 0.0,
        },
        "sentiment": {
            "score": 0.0,
            "headline_risk": False,
            "macro_risk": False,
            "correlation_bias": 0.08,
        },
    },
}

MARKET_SYMBOL_FALLBACK = {
    "crypto": "BTCUSDT",
    "forex": "XAUUSD",
    "commodity": "WTI",
}


def _normalize_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper()


def _verdict_label(verdict: str | None) -> str:
    normalized = str(verdict or "").strip().upper()
    if normalized == "LONG":
        return "BUY / LONG"
    if normalized == "SHORT":
        return "SELL / SHORT"
    if normalized == "NO TRADE":
        return "NO TRADE / DEFENSE"
    if normalized == "WAIT":
        return "WAIT / STANDBY"
    return normalized or "-"


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def _number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _normalized_execution_profile(base_payload: dict[str, Any] | None) -> str:
    payload = base_payload or {}
    raw = str(
        payload.get("execution_profile")
        or (payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}).get("execution_profile")
        or "balanced"
    ).strip().lower()
    return "precision" if raw in {"precision", "near_zero_float", "zero_float", "sniper", "a_plus"} else "balanced"


def _normalized_research_mode(base_payload: dict[str, Any] | None, discovery_mode: str | None = None) -> str:
    payload = base_payload or {}
    raw = str(
        payload.get("research_mode")
        or (payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}).get("research_mode")
        or ""
    ).strip().lower()
    if raw in {"deep", "deep_fresh", "fresh", "listing", "new_listing"}:
        return "deep_fresh"
    if str(discovery_mode or "").strip().lower() == "fresh":
        return "deep_fresh"
    return "standard"


def _healthy_rr_floor(style: str | None, execution_profile: str | None) -> float:
    normalized_style = str(style or "intraday").strip().lower()
    normalized_profile = str(execution_profile or "balanced").strip().lower()
    is_scalp = normalized_style in {"scalp", "scalping"}
    if normalized_profile == "precision":
        return 1.15 if is_scalp else 1.4
    return 1.0 if is_scalp else 1.2


def _extract_plan(result: dict[str, Any]) -> dict[str, Any]:
    base = result.get("brain_output", result)
    return base.get("plan") or base.get("conditional_plan") or {}


def _unique_symbols(symbols: list[str]) -> list[str]:
    unique: list[str] = []
    for symbol in symbols:
        normalized = _normalize_symbol(symbol)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _macro_mix_symbols(active_symbol: str, timeframe: str, discover_limit: int) -> list[str]:
    forex_seed = active_symbol if infer_market_type(active_symbol, "auto") == "forex" else "EURUSD"
    commodity_seed = active_symbol if infer_market_type(active_symbol, "auto") == "commodity" else "WTI"
    symbols = [active_symbol]
    symbols.extend(discover_market_symbols("forex", base_symbol=forex_seed, timeframe=timeframe, limit=max(discover_limit, 10)))
    symbols.extend(discover_market_symbols("commodity", base_symbol=commodity_seed, timeframe=timeframe, limit=max(4, min(discover_limit, 8))))
    symbols.extend(MACRO_MIX_SYMBOLS)
    return _unique_symbols(symbols)


def _cross_market_symbols(active_symbol: str, active_market: str, timeframe: str, discover_limit: int, discovery_mode: str) -> list[str]:
    symbols = [active_symbol]
    if active_market == "crypto":
        symbols.extend(
            discover_market_symbols(
                "crypto",
                base_symbol=active_symbol,
                timeframe=timeframe,
                limit=max(discover_limit, 12),
                discovery_mode=discovery_mode,
            )
        )
        symbols.extend(CRYPTO_RESEARCH_SYMBOLS)
        symbols.extend(_macro_mix_symbols(active_symbol, timeframe, max(10, discover_limit)))
    else:
        symbols.extend(_macro_mix_symbols(active_symbol, timeframe, max(10, discover_limit)))
        symbols.extend(
            discover_market_symbols(
                "crypto",
                base_symbol=None,
                timeframe=timeframe,
                limit=max(8, min(discover_limit, 14)),
                discovery_mode=discovery_mode or "all_liquid",
            )
        )
        symbols.extend(CRYPTO_RESEARCH_SYMBOLS)
    symbols.extend(BALANCED_CROSS_MARKET_SYMBOLS)
    return _unique_symbols(symbols)


def _base_template(symbol: str, timeframe: str, style: str, base_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload or {})
    normalized_symbol = _normalize_symbol(symbol)
    payload_symbol = _normalize_symbol(payload.get("symbol"))
    same_symbol = payload_symbol == normalized_symbol
    payload_market_hint = payload.get("market_type") if same_symbol else None
    inferred_market = infer_market_type(symbol, payload_market_hint)
    fallback_symbol = MARKET_SYMBOL_FALLBACK.get(inferred_market)
    default = copy.deepcopy(SYMBOL_DEFAULTS.get(symbol) or SYMBOL_DEFAULTS.get(fallback_symbol, {}))
    explicit_market = default.get("market_type") or inferred_market
    market_type = explicit_market if explicit_market != "unknown" else infer_market_type(symbol, payload_market_hint)

    risk = {
        **default.get("risk", {}),
        **copy.deepcopy(payload.get("risk", {}) if isinstance(payload.get("risk"), dict) else {}),
    }
    context = {
        **default.get("context", {}),
        **copy.deepcopy(payload.get("context", {}) if same_symbol and isinstance(payload.get("context"), dict) else {}),
        "market_type_hint": market_type,
    }
    microstructure = {
        **default.get("microstructure", {}),
        **copy.deepcopy(payload.get("microstructure", {}) if same_symbol and isinstance(payload.get("microstructure"), dict) else {}),
    }
    sentiment = {
        **default.get("sentiment", {}),
        **copy.deepcopy(payload.get("sentiment", {}) if same_symbol and isinstance(payload.get("sentiment"), dict) else {}),
    }

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "style": style,
        "market_type": market_type,
        "execution_profile": payload.get("execution_profile") or "balanced",
        "research_mode": payload.get("research_mode") or "standard",
        "session": (payload.get("session") if same_symbol else None) or default.get("session") or ("us" if market_type == "crypto" else "overlap" if market_type == "forex" else "us"),
        "risk": risk,
        "context": context,
        "microstructure": microstructure,
        "sentiment": sentiment,
        "levels": copy.deepcopy(payload.get("levels", {}) if same_symbol and isinstance(payload.get("levels"), dict) else {}),
        "indicators": copy.deepcopy(payload.get("indicators", {}) if same_symbol and isinstance(payload.get("indicators"), dict) else {}),
    }


def _apply_snapshot(payload: dict[str, Any], snapshot: dict[str, Any]) -> None:
    payload["price"] = snapshot.get("price", payload.get("price"))
    payload["open"] = snapshot.get("open", payload.get("open"))
    payload["high"] = snapshot.get("high", payload.get("high"))
    payload["low"] = snapshot.get("low", payload.get("low"))
    payload["close"] = snapshot.get("close", payload.get("close", payload.get("price")))
    payload["atr"] = snapshot.get("atr", payload.get("atr"))
    payload["levels"] = {
        **payload.get("levels", {}),
        **copy.deepcopy(snapshot.get("levels", {})),
    }
    payload["indicators"] = {
        **payload.get("indicators", {}),
        **copy.deepcopy(snapshot.get("indicators", {})),
    }
    payload["context"] = {
        **payload.get("context", {}),
        **copy.deepcopy(snapshot.get("context", {})),
        "market_type_hint": payload.get("market_type"),
    }
    payload["live_context"] = copy.deepcopy(snapshot.get("live_context", {}))
    if snapshot.get("provider"):
        payload["context"]["live_provider"] = snapshot.get("provider")


def _apply_news(payload: dict[str, Any], news_snapshot: dict[str, Any]) -> None:
    summary = news_snapshot.get("summary", {}) if isinstance(news_snapshot, dict) else {}
    payload["sentiment"] = {
        **payload.get("sentiment", {}),
        "score": summary.get("score", payload.get("sentiment", {}).get("score")),
        "headline_risk": summary.get("headline_risk", payload.get("sentiment", {}).get("headline_risk")),
        "macro_risk": summary.get("macro_risk", payload.get("sentiment", {}).get("macro_risk")),
    }


def _recommendation_score(analysis: dict[str, Any], news_summary: dict[str, Any], style: str | None = None) -> float:
    result = analysis.get("result", analysis)
    summary = result.get("summary", {})
    base = result.get("brain_output", result)
    adaptive = base.get("adaptive", {}) if isinstance(base.get("adaptive"), dict) else {}
    training = base.get("training", {}) if isinstance(base.get("training"), dict) else {}
    plan = _extract_plan(result)
    verdict = str(summary.get("verdict") or base.get("summary", {}).get("verdict") or "WAIT").upper()
    confidence = float(summary.get("confidence") or base.get("summary", {}).get("confidence") or 0.0)
    blockers = base.get("blockers", [])
    warnings = base.get("warnings", [])
    rr = plan.get("risk_reward")
    setup_type = str(plan.get("setup_type") or "").strip().lower()
    profile_guard = result.get("profile_guard", {}) if isinstance(result, dict) else {}
    profile_status = str(profile_guard.get("status") or "").strip().lower()
    float_risk = str(profile_guard.get("floating_risk") or "").strip().lower()
    execution_profile = str(summary.get("execution_profile") or base.get("summary", {}).get("execution_profile") or "balanced").strip().lower()
    rr_floor = _healthy_rr_floor(style, execution_profile)

    score = confidence * 100.0
    if verdict in {"LONG", "SHORT"}:
        score += 12.0
    elif verdict == "WAIT":
        score -= 10.0
    else:
        score -= 28.0

    if rr is not None:
        rr_value = float(rr)
        score += min(rr_value, 4.0) * 6.0
        if verdict in {"LONG", "SHORT"} and rr_value < rr_floor:
            score -= 12.0 if execution_profile == "precision" else 8.0

    if verdict in {"LONG", "SHORT"}:
        if "snr" in setup_type or "support zone" in setup_type or "resistance zone" in setup_type:
            score += 6.0
        else:
            score -= 8.0

    score -= min(len(blockers) * 8.0, 24.0)
    score -= min(len(warnings) * 2.0, 8.0)

    if news_summary.get("headline_risk"):
        score -= 10.0
    if news_summary.get("macro_risk"):
        score -= 6.0
    if execution_profile == "precision":
        if profile_status == "pass":
            score += 6.0
        elif profile_status in {"balanced", "watch", "", "standby"}:
            score -= 4.0
        else:
            score -= 12.0
    if float_risk == "low":
        score += 4.0
    elif float_risk in {"elevated", "high"}:
        score -= 12.0

    growth_cycle = str(adaptive.get("growth_cycle") or "").strip().lower()
    maturity = float(adaptive.get("maturity") or 0.0)
    if growth_cycle == "compound":
        score += 5.0 + min(maturity, 80.0) * 0.03
    elif growth_cycle == "stabilize":
        score += 2.0 + min(maturity, 70.0) * 0.02
    elif growth_cycle == "protect":
        score -= 10.0
    elif growth_cycle == "bootstrap":
        score -= 2.0

    score += float(adaptive.get("score_shift") or 0.0) * 8.0

    return round(_clamp(score), 2)


def _recommendation_item(
    symbol: str,
    timeframe: str,
    style: str,
    analysis: dict[str, Any],
    snapshot: dict[str, Any],
    news_snapshot: dict[str, Any],
) -> dict[str, Any]:
    result = analysis["result"]
    summary = result.get("summary", {})
    base = result.get("brain_output", result)
    adaptive = base.get("adaptive", {}) if isinstance(base.get("adaptive"), dict) else {}
    training = base.get("training", {}) if isinstance(base.get("training"), dict) else {}
    plan = _extract_plan(result)
    news_summary = news_snapshot.get("summary", {})
    confidence = float(summary.get("confidence") or base.get("summary", {}).get("confidence") or 0.0)
    verdict = summary.get("verdict") or base.get("summary", {}).get("verdict") or "WAIT"
    score = _recommendation_score(analysis, news_summary, style)
    reasons = base.get("reasons", [])
    blockers = base.get("blockers", [])
    edge_summary = result.get("strategic_brief", {}).get("edge_summary")
    primary_thesis = result.get("strategic_brief", {}).get("primary_thesis")
    mood = news_summary.get("mood", "neutral")
    profile_guard = result.get("profile_guard", {}) if isinstance(result, dict) else {}
    profile_status = str(profile_guard.get("status") or "balanced").strip().lower()
    float_risk = str(profile_guard.get("floating_risk") or "normal").strip().lower()
    execution_profile = str(summary.get("execution_profile") or base.get("summary", {}).get("execution_profile") or "balanced").strip().lower()
    profile_note = (profile_guard.get("notes") or [None])[0]
    rr_value = _number(plan.get("risk_reward"))
    setup_type = str(plan.get("setup_type") or "").strip()
    rr_floor = _healthy_rr_floor(style, execution_profile)
    rr_healthy = rr_value is not None and rr_value >= rr_floor
    live_context = snapshot.get("live_context", {}) if isinstance(snapshot.get("live_context"), dict) else {}
    listing_profile = live_context.get("listing_profile")
    fresh_listing = bool(live_context.get("fresh_listing_candidate"))
    history_age_hours = live_context.get("history_age_hours")

    if verdict == "LONG":
        kind = "buy"
    elif verdict == "SHORT":
        kind = "sell"
    else:
        kind = "defense"

    prime_setup = (
        verdict in {"LONG", "SHORT"}
        and not blockers
        and confidence >= (0.76 if execution_profile == "precision" else 0.7)
        and rr_healthy
        and (profile_status not in {"blocked"} if execution_profile == "precision" else True)
    )

    research_note = _research_note(symbol, listing_profile, history_age_hours, news_snapshot)
    survival = _survival_guard(
        price=snapshot.get("price") or snapshot.get("close"),
        entry_zone=plan.get("entry_zone"),
        stop_loss=plan.get("stop_loss"),
        risk_reward=plan.get("risk_reward"),
        fresh_listing=fresh_listing,
        headline_risk=bool(news_summary.get("headline_risk")),
        macro_risk=bool(news_summary.get("macro_risk")),
    )
    prime_setup = prime_setup and not survival["lock_prime"]

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "style": style,
        "market_type": summary.get("market_type") or base.get("summary", {}).get("market_type"),
        "verdict": verdict,
        "display_verdict": _verdict_label(verdict),
        "kind": kind,
        "confidence": round(confidence * 100.0, 1),
        "score": score,
        "prime_setup": prime_setup,
        "profile_status": profile_status,
        "float_risk": float_risk,
        "profile_note": profile_note,
        "execution_profile": execution_profile,
        "headline_mood": mood,
        "headline_risk": bool(news_summary.get("headline_risk")),
        "macro_risk": bool(news_summary.get("macro_risk")),
        "price": snapshot.get("price") or snapshot.get("close"),
        "entry_zone": plan.get("entry_zone"),
        "stop_loss": plan.get("stop_loss"),
        "take_profit_1": plan.get("take_profit_1"),
        "take_profit_2": plan.get("take_profit_2"),
        "risk_reward": plan.get("risk_reward"),
        "rr_floor": rr_floor,
        "rr_healthy": rr_healthy,
        "adaptive_mode": adaptive.get("adaptation_mode"),
        "growth_cycle": adaptive.get("growth_cycle"),
        "adaptive_maturity": adaptive.get("maturity"),
        "memory_scope": adaptive.get("memory_scope"),
        "adaptive_note": adaptive.get("note"),
        "trainer_state": training.get("trainer_state"),
        "trainer_note": training.get("note"),
        "trainer_preferred_direction": training.get("preferred_direction"),
        "trainer_training_days": training.get("training_days"),
        "setup_type": setup_type or "snr zone",
        "technique_family": "SnR Zones",
        "playbook": summary.get("dominant_playbook") or summary.get("setup_grade"),
        "reason": edge_summary or primary_thesis or (reasons[0] if reasons else "Live opportunity scan"),
        "blockers": blockers[:2],
        "research_note": research_note,
        "survival_note": survival["note"],
        "survival_bias": survival["score_bias"],
        "survival_lock": survival["lock_prime"],
        "live_provider": snapshot.get("provider"),
        "news_provider": news_snapshot.get("provider"),
        "articles": news_snapshot.get("articles", [])[:2],
        "listing_profile": listing_profile,
        "fresh_listing": fresh_listing,
        "history_age_hours": history_age_hours,
    }


def _research_note(symbol: str, listing_profile: str | None, history_age_hours: Any, news_snapshot: dict[str, Any]) -> str | None:
    articles = news_snapshot.get("articles", []) if isinstance(news_snapshot, dict) else []
    titles = " ".join(str(article.get("title") or "") for article in articles).lower()
    matched_tags = []
    for keyword in ("listing", "launch", "unlock", "ecosystem", "funding", "roadmap", "volume", "partnership", "airdrop"):
        if keyword in titles and keyword not in matched_tags:
            matched_tags.append(keyword)

    if listing_profile in {"new_listing", "fresh_listing", "young_market"}:
        age_note = f"riwayat {history_age_hours} jam" if history_age_hours is not None else "riwayat masih pendek"
        tag_note = f"fokus headline: {', '.join(matched_tags[:3])}" if matched_tags else "cek launch, unlock, funding, dan volume"
        return f"{symbol} masih {listing_profile.replace('_', ' ')} dengan {age_note}; {tag_note}."

    if matched_tags:
        return f"Riset headline paling relevan saat ini: {', '.join(matched_tags[:3])}."
    return None


def _survival_guard(
    *,
    price: Any,
    entry_zone: Any,
    stop_loss: Any,
    risk_reward: Any,
    fresh_listing: bool,
    headline_risk: bool,
    macro_risk: bool,
) -> dict[str, Any]:
    price_value = _number(price)
    stop_value = _number(stop_loss)
    rr_value = _number(risk_reward)
    entry_values = entry_zone if isinstance(entry_zone, list) else [entry_zone]
    entry_numbers = [_number(value) for value in entry_values]
    entry_numbers = [value for value in entry_numbers if value is not None]
    entry_mid = sum(entry_numbers[:2]) / min(2, len(entry_numbers)) if entry_numbers else price_value

    score_bias = 0.0
    notes: list[str] = []
    lock_prime = False

    if price_value is not None and entry_mid is not None and stop_value is not None and price_value > 0:
        stop_risk_percent = abs(entry_mid - stop_value) / price_value * 100.0
        if stop_risk_percent >= 2.2:
            score_bias -= 14.0
            lock_prime = True
            notes.append("Stop terlalu lebar; survival mode menahan setup.")
        elif stop_risk_percent >= 1.2:
            score_bias -= 6.0
            notes.append("Risk masih agak lebar, jadi score dipotong.")

    if fresh_listing:
        if rr_value is None or rr_value < 2.6:
            score_bias -= 10.0
            lock_prime = True
            notes.append("Fresh coin butuh asymmetry besar; RR sekarang belum cukup.")
        if headline_risk or macro_risk:
            score_bias -= 6.0
            notes.append("Fresh coin dengan headline risk langsung ditahan.")

    return {
        "score_bias": round(score_bias, 2),
        "lock_prime": lock_prime,
        "note": " ".join(notes[:2]) if notes else None,
    }


def _apply_learning_bias(item: dict[str, Any], learning: dict[str, Any]) -> dict[str, Any]:
    adjusted = copy.deepcopy(item)
    score_bias = float(learning.get("score_bias") or 0.0)
    confidence_bias = float(learning.get("confidence_bias") or 0.0) * 100.0
    survival_bias = float(adjusted.get("survival_bias") or 0.0)
    adjusted["learning_state"] = learning.get("state") or "warming"
    adjusted["learning_note"] = learning.get("note")
    adjusted["recent_win_rate"] = learning.get("win_rate")
    adjusted["recent_sample_size"] = learning.get("scored_total") or learning.get("sample_size") or 0
    adjusted["pair_scored_total"] = learning.get("pair_scored_total") or 0
    adjusted["market_scored_total"] = learning.get("market_scored_total") or 0
    adjusted["memory_scope"] = learning.get("memory_scope") or "warming"
    adjusted["market_fallback_active"] = bool(learning.get("market_fallback_active"))
    adjusted["recent_loss_streak"] = learning.get("loss_streak") or 0
    adjusted["open_memory_signals"] = learning.get("open_signals") or 0
    adjusted["learning_score_bias"] = round(score_bias, 2)
    adjusted["confidence_effective"] = round(_clamp(float(adjusted.get("confidence") or 0.0) + confidence_bias), 1)
    adjusted["memory_recent"] = copy.deepcopy((learning.get("recent_history") or [])[:3])
    adjusted["score"] = round(_clamp(float(adjusted.get("score") or 0.0) + score_bias + survival_bias), 2)
    if learning.get("prime_penalty") or adjusted.get("survival_lock"):
        adjusted["prime_setup"] = False
    return adjusted


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float]:
    item = candidate.get("item", {})
    return (
        float(item.get("score") or 0.0),
        float(item.get("confidence_effective") or item.get("confidence") or 0.0),
    )


def _pick_tracking_candidate(
    primary_candidates: list[dict[str, Any]],
    fallback_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for bucket in (primary_candidates, fallback_candidates):
        for candidate in bucket:
            verdict = str(candidate.get("item", {}).get("verdict") or "").strip().upper()
            if verdict in {"LONG", "SHORT"}:
                return candidate
    for bucket in (primary_candidates, fallback_candidates):
        if bucket:
            return bucket[0]
    return None


def _pick_tracking_candidates(
    primary_candidates: list[dict[str, Any]],
    fallback_candidates: list[dict[str, Any]],
    *,
    min_score: float = 100.0,
    limit: int = 3,
) -> list[dict[str, Any]]:
    directional: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for bucket in (primary_candidates, fallback_candidates):
        for candidate in sorted(bucket, key=_candidate_sort_key, reverse=True):
            item = candidate.get("item", {})
            verdict = str(item.get("verdict") or "").strip().upper()
            if verdict not in {"LONG", "SHORT"}:
                continue
            symbol = str(item.get("symbol") or "").strip().upper()
            timeframe = str(item.get("timeframe") or "").strip().lower()
            style = str(item.get("style") or "").strip().lower()
            key = "|".join([symbol, timeframe, style, verdict])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            directional.append(candidate)

    directional.sort(key=_candidate_sort_key, reverse=True)
    best_score = float(directional[0].get("item", {}).get("score") or 0.0) if directional else 0.0
    cohort_floor = max(best_score - 2.5, min_score - 4.0)
    top_scored = [
        candidate
        for candidate in directional
        if (
            float(candidate.get("item", {}).get("score") or 0.0) >= min_score
            or float(candidate.get("item", {}).get("score") or 0.0) >= cohort_floor
        )
        and (
            candidate.get("item", {}).get("prime_setup")
            or float(candidate.get("item", {}).get("confidence") or 0.0) >= 72.0
        )
    ]
    if len(top_scored) < min(limit, len(directional)):
        for candidate in directional:
            if candidate in top_scored:
                continue
            if candidate.get("item", {}).get("prime_setup") or float(candidate.get("item", {}).get("confidence") or 0.0) >= 76.0:
                top_scored.append(candidate)
            if len(top_scored) >= limit:
                break
    if top_scored:
        return top_scored[:limit]

    fallback = _pick_tracking_candidate(primary_candidates, fallback_candidates)
    return [fallback] if fallback else []


def _scan_symbol_candidate(
    symbol: str,
    timeframe: str,
    style: str,
    mode: str,
    base_payload: dict[str, Any] | None,
    research_mode: str,
) -> dict[str, Any]:
    payload = _base_template(symbol, timeframe, style, base_payload)
    payload["research_mode"] = research_mode
    payload.setdefault("context", {})
    if isinstance(payload.get("context"), dict):
        payload["context"]["research_mode"] = research_mode
    snapshot = fetch_live_snapshot(symbol, timeframe, payload["market_type"])
    _apply_snapshot(payload, snapshot)
    news_snapshot = fetch_live_news(symbol, payload["market_type"], limit=4, force=False)
    _apply_news(payload, news_snapshot)
    payload["_news_snapshot"] = copy.deepcopy(news_snapshot)
    analysis = analyze_for_web(payload, mode=mode)
    learning = signal_learning_context(symbol, timeframe, style, payload.get("market_type"))
    research = analysis.get("claw_research", {}) if isinstance(analysis, dict) else {}
    if not research:
        research = build_claw_research(
            payload,
            analysis.get("result", analysis),
            learning,
            news_snapshot=news_snapshot,
            source="recommendation_scan",
        )
    base_item = _recommendation_item(symbol, timeframe, style, analysis, snapshot, news_snapshot)
    base_item = apply_claw_research_bias(base_item, research)
    item = _apply_learning_bias(base_item, learning)
    return {
        "item": item,
        "base_item": base_item,
        "payload": payload,
        "result": analysis["result"],
        "error": None,
    }


def scan_recommendations(
    *,
    symbols: list[str] | None = None,
    timeframe: str = "15m",
    style: str = "intraday",
    mode: str = "super",
    base_payload: dict[str, Any] | None = None,
    scope: str = "hybrid",
    discover_limit: int = 8,
    discovery_mode: str | None = None,
    research_mode: str | None = None,
    prime_only: bool | None = None,
    track_best_setup: bool = False,
) -> dict[str, Any]:
    manual_symbols: list[str] = []
    for symbol in symbols or []:
        normalized = _normalize_symbol(symbol)
        if normalized and normalized not in manual_symbols:
            manual_symbols.append(normalized)

    active_symbol = _normalize_symbol((base_payload or {}).get("symbol"))
    active_market = infer_market_type(active_symbol, (base_payload or {}).get("market_type"))
    execution_profile = _normalized_execution_profile(base_payload)
    normalized_scope = str(scope or "hybrid").strip().lower()
    normalized_discovery_mode = str(discovery_mode or "leaders").strip().lower()
    normalized_research_mode = str(research_mode or _normalized_research_mode(base_payload, normalized_discovery_mode)).strip().lower()
    only_prime = execution_profile == "precision" if prime_only is None else bool(prime_only)
    discovered_symbols: list[str] = []
    if normalized_scope in {"macro", "macro_mix", "fx_macro"}:
        discovered_symbols = _macro_mix_symbols(active_symbol, timeframe, discover_limit)
    elif normalized_scope in {"cross", "cross_market", "global"}:
        discovered_symbols = _cross_market_symbols(active_symbol, active_market, timeframe, discover_limit, normalized_discovery_mode)
    elif normalized_scope in {"market", "hybrid"}:
        discovered_symbols = discover_market_symbols(
            active_market,
            base_symbol=active_symbol,
            timeframe=timeframe,
            limit=discover_limit,
            discovery_mode=normalized_discovery_mode,
        )

    if normalized_scope in {"market", "macro", "macro_mix", "fx_macro", "cross", "cross_market", "global"}:
        normalized_symbols = discovered_symbols
    elif normalized_scope == "manual":
        normalized_symbols = manual_symbols
    else:
        normalized_symbols = []
        for symbol in [*manual_symbols, *discovered_symbols]:
            if symbol and symbol not in normalized_symbols:
                normalized_symbols.append(symbol)

    if not normalized_symbols:
        normalized_symbols = manual_symbols or discovered_symbols or DEFAULT_SYMBOLS[:]

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    max_workers = max(1, min(8, len(normalized_symbols)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_scan_symbol_candidate, symbol, timeframe, style, mode, base_payload, normalized_research_mode): symbol
            for symbol in normalized_symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                payload = future.result()
                if payload.get("item"):
                    candidates.append(payload)
                elif payload.get("error"):
                    skipped.append({"symbol": symbol, "error": str(payload["error"])})
            except Exception as exc:  # noqa: BLE001
                skipped.append({"symbol": symbol, "error": str(exc)})

    filtered_candidates = [candidate for candidate in candidates if only_prime and not candidate["item"].get("prime_setup")]
    fallback_items = []
    tracked_best = None
    tracked_memory: list[dict[str, Any]] = []
    if only_prime:
        fallback_candidates = sorted(filtered_candidates, key=_candidate_sort_key, reverse=True)[:3]
        candidates = [candidate for candidate in candidates if candidate["item"].get("prime_setup")]
    else:
        fallback_candidates = []

    candidates.sort(key=_candidate_sort_key, reverse=True)

    if track_best_setup:
        tracked_candidates = _pick_tracking_candidates(candidates, fallback_candidates)
        for rank, tracked_candidate in enumerate(tracked_candidates, start=1):
            tracked_item_current = tracked_candidate.get("item", {})
            tracked_payload = copy.deepcopy(tracked_candidate["payload"])
            tracked_result = copy.deepcopy(tracked_candidate["result"])
            tracked_payload["_signal_score"] = tracked_item_current.get("score")
            tracked_payload["_signal_rank"] = rank
            tracked_result.setdefault("summary", {})
            tracked_result["summary"]["recommendation_score"] = tracked_item_current.get("score")
            tracked_learning = register_signal(tracked_payload, tracked_result, source="best_setups")
            tracked_item = _apply_learning_bias(tracked_candidate["base_item"], tracked_learning)
            tracked_item["tracked_in_signal_memory"] = True
            tracked_item["signal_source"] = "best_setups"
            tracked_item["memory_track_rank"] = rank
            tracked_candidate["item"] = tracked_item
            tracked_memory.append(copy.deepcopy(tracked_item))
        candidates.sort(key=_candidate_sort_key, reverse=True)
        fallback_candidates.sort(key=_candidate_sort_key, reverse=True)
        tracked_best = copy.deepcopy(tracked_memory[0]) if tracked_memory else None

    items = [candidate["item"] for candidate in candidates]
    fallback_items = [candidate["item"] for candidate in fallback_candidates]
    best = items[0] if items else None
    return {
        "symbols": normalized_symbols,
        "manual_symbols": manual_symbols,
        "discovered_symbols": discovered_symbols,
        "scope": normalized_scope,
        "scope_label": "FX + Gold + Oil" if normalized_scope in {"macro", "macro_mix", "fx_macro"} else "Cross Market" if normalized_scope in {"cross", "cross_market", "global"} else normalized_scope,
        "discovery_mode": normalized_discovery_mode,
        "execution_profile": execution_profile,
        "research_mode": normalized_research_mode,
        "prime_only": only_prime,
        "timeframe": timeframe,
        "style": style,
        "mode": mode,
        "best": best,
        "tracked_best": tracked_best,
        "tracked_memory": tracked_memory,
        "items": items,
        "skipped": skipped,
        "filtered_out": [
            {
                "symbol": candidate["item"]["symbol"],
                "verdict": candidate["item"]["verdict"],
                "profile_status": candidate["item"].get("profile_status"),
                "reason": candidate["item"].get("profile_note") or candidate["item"].get("reason"),
            }
            for candidate in filtered_candidates[:10]
        ],
        "fallback_items": fallback_items,
    }
