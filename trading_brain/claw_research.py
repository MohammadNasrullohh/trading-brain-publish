from __future__ import annotations

import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .storage_db import DB_FILENAME, default_data_dir, read_document, write_document

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = default_data_dir(ROOT)
DATA_PATH = DATA_DIR / "claw_research_memory.json"
STORE_LOCK = threading.RLock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _db_path() -> Path:
    return DATA_DIR / DB_FILENAME


def _legacy_paths() -> tuple[Path, ...]:
    paths = [DATA_PATH]
    fallback = ROOT / "logs" / "claw_research_memory.json"
    if DATA_DIR == default_data_dir(ROOT) and fallback != DATA_PATH:
        paths.append(fallback)
    return tuple(dict.fromkeys(paths))


def _legacy_db_paths() -> tuple[Path, ...]:
    paths = []
    fallback = ROOT / "logs" / DB_FILENAME
    if DATA_DIR == default_data_dir(ROOT) and fallback != _db_path():
        paths.append(fallback)
    return tuple(dict.fromkeys(paths))


def _default_store() -> dict[str, Any]:
    return {
        "version": 1,
        "sessions": {},
    }


def _read_store() -> dict[str, Any]:
    _ensure_data_dir()
    return read_document(
        db_path=_db_path(),
        module="claw_research",
        default_factory=_default_store,
        legacy_paths=_legacy_paths(),
        legacy_db_paths=_legacy_db_paths(),
    )


def _write_store(store: dict[str, Any]) -> None:
    _ensure_data_dir()
    sessions = {}
    for key, value in (store.get("sessions") or {}).items():
        history = list((value or {}).get("history", []))[-32:]
        sessions[key] = {
            **value,
            "history": history,
        }
    write_document(
        db_path=_db_path(),
        module="claw_research",
        payload={"version": 1, "sessions": sessions},
    )


def _number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _extract_plan(result: dict[str, Any]) -> dict[str, Any]:
    base = result.get("brain_output", result)
    return base.get("plan") or base.get("conditional_plan") or {}


def _healthy_rr_floor(style: str | None, execution_profile: str | None) -> float:
    normalized_style = str(style or "intraday").strip().lower()
    normalized_profile = str(execution_profile or "balanced").strip().lower()
    if normalized_profile == "precision":
        return 1.15 if normalized_style in {"scalp", "scalping"} else 1.4
    return 1.0 if normalized_style in {"scalp", "scalping"} else 1.2


def _headline_tags(news_snapshot: dict[str, Any] | None) -> list[str]:
    articles = news_snapshot.get("articles", []) if isinstance(news_snapshot, dict) else []
    titles = " ".join(str(article.get("title") or "") for article in articles).lower()
    tags: list[str] = []
    for keyword in (
        "listing",
        "launch",
        "unlock",
        "ecosystem",
        "funding",
        "roadmap",
        "volume",
        "partnership",
        "airdrop",
        "cpi",
        "fed",
        "opec",
        "tariff",
        "war",
    ):
        if keyword in titles and keyword not in tags:
            tags.append(keyword)
    return tags[:4]


def _focus_from_context(
    verdict: str,
    setup_type: str,
    blockers: list[str],
    reasons: list[str],
    warnings: list[str],
    news_summary: dict[str, Any],
    live_context: dict[str, Any],
) -> str:
    blocker_text = " ".join(blockers).lower()
    reason_text = " ".join(reasons).lower()
    warning_text = " ".join(warnings).lower()
    setup_text = str(setup_type or "").lower()
    mood = str(news_summary.get("mood") or "").lower()
    listing_profile = str(live_context.get("listing_profile") or "").lower()

    if verdict == "NO TRADE" or "drawdown" in blocker_text or "loss" in blocker_text:
        return "defense"
    if "snr" in setup_text or "support zone" in setup_text or "resistance zone" in setup_text:
        return "snr"
    if "breakout" in setup_text or "continuation" in setup_text or "momentum" in reason_text:
        return "breakout"
    if "pullback" in setup_text or "retest" in setup_text:
        return "pullback"
    if "range" in setup_text or "mean reversion" in setup_text:
        return "range"
    if news_summary.get("headline_risk") or news_summary.get("macro_risk") or mood in {"headwind", "risk_off"}:
        return "macro"
    if "fresh_listing" in listing_profile or "young_market" in listing_profile:
        return "discovery"
    if "liquidity" in warning_text or "spread" in warning_text:
        return "execution"
    return "structure"


def _questions_for_focus(focus: str, verdict: str) -> list[str]:
    if focus == "snr":
        return ["Tunggu reaksi bersih di support atau resistance terdekat.", "Hanya entry kalau zona SnR tetap dihormati price action."]
    if focus == "breakout":
        return ["Tunggu retest bersih sebelum entry.", "Pastikan volume tetap hidup saat break lanjut."]
    if focus == "pullback":
        return ["Cari reclaim area entry yang rapi.", "Jangan kejar candle yang sudah terlalu jauh."]
    if focus == "macro":
        return ["Periksa headline berikutnya sebelum commit size.", "Kurangi agresi saat risk headline masih hidup."]
    if focus == "discovery":
        return ["Pantau volume, unlock, dan headline listing.", "Butuh asymmetry besar sebelum dianggap prime."]
    if focus == "defense":
        return ["Prioritaskan proteksi modal.", "Biarkan market membuktikan arah dulu."]
    if verdict in {"LONG", "SHORT"}:
        return ["Entry hanya saat area plan benar-benar disentuh.", "Batalkan jika spread atau slip memburuk."]
    return ["Tunggu alignment yang lebih bersih.", "Biarkan setup matang sebelum dikunci."]


def _research_mode(payload: dict[str, Any], live_context: dict[str, Any]) -> str:
    raw = str(
        payload.get("research_mode")
        or (payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}).get("research_mode")
        or ""
    ).strip().lower()
    if raw in {"deep", "deep_fresh", "fresh", "listing", "new_listing"}:
        return "deep_fresh"
    if raw in {"standard", "normal", "default"}:
        return "standard"
    if live_context.get("fresh_listing_candidate"):
        return "deep_fresh"
    return "standard"


def _deep_research_checklist(
    *,
    fresh_context: bool,
    headline_tags: list[str],
    headline_risk: bool,
    macro_risk: bool,
) -> list[str]:
    checks = [
        "Pantau volume, unlock, dan headline yang paling baru.",
        "Cari asymmetry besar sebelum menganggap setup ini prime.",
    ]
    if fresh_context:
        checks.insert(0, "Fresh listing butuh spread, slip, dan liquidity check sebelum entry.")
    if headline_tags:
        checks.append(f"Tag headline dominan: {', '.join(headline_tags[:3])}.")
    if headline_risk or macro_risk:
        checks.append("Headline risk aktif, jadi size dan timing harus lebih konservatif.")
    return checks[:4]


def _compose_summary(
    *,
    symbol: str,
    verdict: str,
    focus: str,
    risk_posture: str,
    rr: float | None,
    rr_floor: float,
    tags: list[str],
) -> str:
    verdict_label = {
        "LONG": "BUY / LONG",
        "SHORT": "SELL / SHORT",
        "WAIT": "WAIT / STANDBY",
        "NO TRADE": "NO TRADE / DEFENSE",
    }.get(verdict, verdict or "WAIT")
    rr_note = f"RR {rr:.2f}" if rr is not None else "RR belum penuh"
    tag_note = f" | headline: {', '.join(tags[:2])}" if tags else ""
    return f"{symbol} {verdict_label} | focus {focus} | posture {risk_posture} | {rr_note} vs floor {rr_floor:.2f}{tag_note}"


def build_claw_research(
    payload: dict[str, Any],
    analysis_result: dict[str, Any],
    learning_context: dict[str, Any] | None,
    *,
    news_snapshot: dict[str, Any] | None = None,
    source: str = "analysis",
) -> dict[str, Any]:
    result = analysis_result.get("brain_output", analysis_result)
    summary = {
        **(result.get("summary") or {}),
        **(analysis_result.get("summary") or {}),
    }
    strategic = analysis_result.get("strategic_brief", {}) if isinstance(analysis_result.get("strategic_brief"), dict) else {}
    plan = _extract_plan(analysis_result)
    training = result.get("training", {}) if isinstance(result.get("training"), dict) else {}
    learning = deepcopy(learning_context or {})
    news_summary = news_snapshot.get("summary", {}) if isinstance(news_snapshot, dict) else {}
    live_context = payload.get("live_context", {}) if isinstance(payload.get("live_context"), dict) else {}

    symbol = str(summary.get("symbol") or payload.get("symbol") or "").strip().upper()
    timeframe = str(summary.get("timeframe") or payload.get("timeframe") or "15m").strip().lower()
    style = str(payload.get("style") or summary.get("style") or "intraday").strip().lower()
    market_type = str(summary.get("market_type") or payload.get("market_type") or "auto").strip().lower()
    verdict = str(summary.get("verdict") or "WAIT").strip().upper()
    confidence = float(summary.get("confidence") or result.get("summary", {}).get("confidence") or 0.0)
    setup_type = str(plan.get("setup_type") or "").strip().lower()
    blockers = list(result.get("blockers") or [])
    warnings = list(result.get("warnings") or [])
    reasons = list(result.get("reasons") or [])
    execution_profile = str(payload.get("execution_profile") or "balanced").strip().lower()
    rr = _number(plan.get("risk_reward"))
    rr_floor = _healthy_rr_floor(style, execution_profile)
    win_rate = _number(learning.get("win_rate"))
    loss_streak = int(learning.get("loss_streak") or 0)
    learning_state = str(learning.get("state") or "warming").strip().lower()
    trainer_state = str(training.get("trainer_state") or "").strip().lower()
    tags = _headline_tags(news_snapshot)
    research_mode = _research_mode(payload, live_context)
    focus = _focus_from_context(verdict, setup_type, blockers, reasons, warnings, news_summary, live_context)

    score_delta = 0.0
    caution_score = 12.0

    if verdict in {"LONG", "SHORT"}:
        score_delta += 3.5
    elif verdict == "WAIT":
        score_delta -= 1.5
        caution_score += 8.0
    else:
        score_delta -= 4.5
        caution_score += 18.0

    if rr is not None:
        if rr >= rr_floor + 0.3:
            score_delta += 2.6
        elif rr >= rr_floor:
            score_delta += 1.2
        else:
            score_delta -= 3.4
            caution_score += 14.0
    else:
        caution_score += 6.0

    if confidence >= 0.82:
        score_delta += 1.6
    elif confidence >= 0.72:
        score_delta += 0.8
    elif confidence <= 0.58:
        score_delta -= 1.4
        caution_score += 8.0

    sentiment_score = _number(news_summary.get("score")) or _number((payload.get("sentiment") or {}).get("score")) or 0.0
    if sentiment_score >= 0.18:
        score_delta += 1.0
    elif sentiment_score <= -0.18:
        score_delta -= 1.0
        caution_score += 6.0

    if news_summary.get("headline_risk"):
        score_delta -= 2.8
        caution_score += 16.0
    if news_summary.get("macro_risk"):
        score_delta -= 1.8
        caution_score += 10.0

    if learning_state in {"in_sync", "improving", "compounding"}:
        score_delta += 1.8
    elif learning_state in {"cooldown", "defensive", "protect"}:
        score_delta -= 2.6
        caution_score += 14.0

    if trainer_state in {"compounding", "improving"}:
        score_delta += 1.2
    elif trainer_state in {"defensive", "cooldown"}:
        score_delta -= 1.2
        caution_score += 8.0

    if win_rate is not None and win_rate >= 60.0:
        score_delta += 1.0
    elif win_rate is not None and win_rate <= 42.0:
        score_delta -= 1.2
        caution_score += 7.0

    if loss_streak >= 2:
        score_delta -= 2.0
        caution_score += 12.0

    if len(blockers) >= 1:
        score_delta -= min(5.5, len(blockers) * 2.0)
        caution_score += min(28.0, len(blockers) * 8.0)
    if len(warnings) >= 2:
        score_delta -= 1.0
        caution_score += 4.0

    listing_profile = str(live_context.get("listing_profile") or "").strip().lower()
    history_age_hours = live_context.get("history_age_hours")
    fresh_context = bool(live_context.get("fresh_listing_candidate")) or listing_profile in {"new_listing", "fresh_listing", "young_market"}
    if live_context.get("fresh_listing_candidate"):
        caution_score += 10.0
        if rr is not None and rr >= max(2.6, rr_floor + 1.0) and verdict in {"LONG", "SHORT"}:
            score_delta += 1.4
        else:
            score_delta -= 1.6
    if listing_profile in {"new_listing", "fresh_listing", "young_market"} and tags:
        score_delta += 0.8

    deep_checklist = _deep_research_checklist(
        fresh_context=fresh_context,
        headline_tags=tags,
        headline_risk=bool(news_summary.get("headline_risk")),
        macro_risk=bool(news_summary.get("macro_risk")),
    )
    deep_conviction = "standard"
    deep_note = None
    asymmetry_floor = round(max(rr_floor + 1.0, 2.6), 2)
    if research_mode == "deep_fresh":
        focus = "discovery" if fresh_context else focus
        caution_score += 6.0 if fresh_context else 2.0
        if fresh_context:
            if verdict in {"LONG", "SHORT"} and rr is not None and rr >= asymmetry_floor and confidence >= 0.74 and not news_summary.get("headline_risk"):
                score_delta += 1.8
                deep_conviction = "asymmetric_ready"
                deep_note = "Mode deep fresh melihat asymmetry dan confidence sudah cukup untuk coin baru."
            else:
                score_delta -= 2.4
                caution_score += 10.0
                deep_conviction = "needs_more_proof"
                deep_note = "Mode deep fresh tetap curiga karena coin baru butuh RR besar dan headline yang lebih bersih."
        else:
            deep_conviction = "watching"
            deep_note = "Mode deep fresh aktif, tapi symbol ini belum terdeteksi sebagai fresh/new listing."

    score_delta = round(_clamp(score_delta, -12.0, 12.0), 2)
    confidence_delta = round(_clamp(score_delta / 220.0, -0.06, 0.06), 3)
    caution_score = round(_clamp(caution_score, 0.0, 100.0), 1)

    if caution_score >= 62.0:
        risk_posture = "defense"
    elif caution_score >= 36.0:
        risk_posture = "controlled"
    else:
        risk_posture = "offense"

    narrative = str(strategic.get("primary_thesis") or reasons[0] if reasons else "").strip()
    if not narrative:
        narrative = f"{symbol} sedang dibaca lewat fokus {focus} dengan posture {risk_posture}."

    session_key = "|".join([market_type or "auto", symbol or "-", timeframe, style])
    with STORE_LOCK:
        store = _read_store()
        session = deepcopy((store.get("sessions") or {}).get(session_key) or {})
        history = list(session.get("history") or [])
        turn_count = int(session.get("turn_count") or 0) + 1
        verdict_counts = dict(session.get("verdict_counts") or {})
        verdict_counts[verdict] = int(verdict_counts.get(verdict) or 0) + 1
        focus_counts = dict(session.get("focus_counts") or {})
        focus_counts[focus] = int(focus_counts.get(focus) or 0) + 1
        score_total = float(session.get("score_total") or 0.0) + score_delta

        history.append(
            {
                "id": uuid4().hex,
                "timestamp": _utc_now_iso(),
                "source": source,
                "verdict": verdict,
                "focus": focus,
                "risk_posture": risk_posture,
                "score_delta": score_delta,
                "confidence_delta": confidence_delta,
                "summary": _compose_summary(
                    symbol=symbol,
                    verdict=verdict,
                    focus=focus,
                    risk_posture=risk_posture,
                    rr=rr,
                    rr_floor=rr_floor,
                    tags=tags,
                ),
            }
        )

        updated_session = {
            "session_id": session.get("session_id") or uuid4().hex,
            "symbol": symbol,
            "timeframe": timeframe,
            "style": style,
            "market_type": market_type,
            "turn_count": turn_count,
            "score_total": round(score_total, 2),
            "avg_score_delta": round(score_total / max(1, turn_count), 2),
            "verdict_counts": verdict_counts,
            "focus_counts": focus_counts,
            "last_focus": focus,
            "last_risk_posture": risk_posture,
            "last_updated_at": _utc_now_iso(),
            "history": history[-32:],
        }
        store.setdefault("sessions", {})[session_key] = updated_session
        _write_store(store)

    summary_text = _compose_summary(
        symbol=symbol,
        verdict=verdict,
        focus=focus,
        risk_posture=risk_posture,
        rr=rr,
        rr_floor=rr_floor,
        tags=tags,
    )
    return {
        "session": {
            "id": updated_session["session_id"],
            "key": session_key,
            "turn_count": updated_session["turn_count"],
            "avg_score_delta": updated_session["avg_score_delta"],
            "last_focus": updated_session["last_focus"],
        },
        "focus": focus,
        "risk_posture": risk_posture,
        "score_delta": score_delta,
        "confidence_delta": confidence_delta,
        "caution_score": caution_score,
        "summary": summary_text,
        "narrative": narrative,
        "questions": _questions_for_focus(focus, verdict),
        "headline_tags": tags,
        "learning_state": learning_state,
        "trainer_state": trainer_state or None,
        "mode": research_mode,
        "listing_profile": listing_profile or None,
        "history_age_hours": history_age_hours,
        "deep_research": {
            "active": research_mode == "deep_fresh",
            "fresh_context": fresh_context,
            "conviction": deep_conviction,
            "asymmetry_floor": asymmetry_floor,
            "checklist": deep_checklist,
            "note": deep_note,
        },
    }


def merge_claw_research(result: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(result)
    summary_text = str(research.get("summary") or "").strip()
    narrative = str(research.get("narrative") or "").strip()
    focus = str(research.get("focus") or "").strip()
    risk_posture = str(research.get("risk_posture") or "").strip()

    strategic = deepcopy(merged.get("strategic_brief") or {})
    if summary_text:
        strategic["claw_summary"] = summary_text
        if strategic.get("edge_summary"):
            existing = str(strategic["edge_summary"])
            strategic["edge_summary"] = existing if summary_text in existing else f"{existing} | {summary_text}"
        else:
            strategic["edge_summary"] = summary_text
    if narrative:
        strategic["primary_thesis"] = strategic.get("primary_thesis") or narrative
    if focus:
        strategic["research_focus"] = focus
    if risk_posture:
        strategic["research_risk_posture"] = risk_posture
    if research.get("mode"):
        strategic["research_mode"] = research.get("mode")
    if research.get("questions"):
        strategic["research_questions"] = list(research.get("questions") or [])[:2]
    if (research.get("deep_research") or {}).get("checklist"):
        strategic["research_checklist"] = list((research.get("deep_research") or {}).get("checklist") or [])[:3]
    merged["strategic_brief"] = strategic

    summary = merged.get("summary")
    if isinstance(summary, dict):
        summary["research_focus"] = focus
        summary["research_posture"] = risk_posture
        summary["research_mode"] = research.get("mode")
    merged["claw_research"] = deepcopy(research)
    return merged


def apply_claw_research_bias(item: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
    adjusted = deepcopy(item)
    score_delta = float(research.get("score_delta") or 0.0)
    adjusted["claw_focus"] = research.get("focus")
    adjusted["claw_risk_posture"] = research.get("risk_posture")
    adjusted["claw_summary"] = research.get("summary")
    adjusted["claw_questions"] = list(research.get("questions") or [])[:2]
    adjusted["claw_headline_tags"] = list(research.get("headline_tags") or [])[:3]
    adjusted["claw_session_turns"] = int((research.get("session") or {}).get("turn_count") or 0)
    adjusted["claw_score_delta"] = round(score_delta, 2)
    adjusted["claw_mode"] = research.get("mode")
    adjusted["claw_listing_profile"] = research.get("listing_profile")
    adjusted["claw_checklist"] = list((research.get("deep_research") or {}).get("checklist") or [])[:2]
    adjusted["claw_deep_active"] = bool((research.get("deep_research") or {}).get("active"))
    adjusted["score"] = round(_clamp(float(adjusted.get("score") or 0.0) + score_delta, 0.0, 100.0), 2)
    if str(research.get("risk_posture") or "").lower() == "defense":
        adjusted["prime_setup"] = False
    return adjusted
