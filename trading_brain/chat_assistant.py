from __future__ import annotations

from datetime import datetime
from typing import Any


def _normalize(value: Any) -> str:
    return str(value or "").strip()


def _normalize_lower(value: Any) -> str:
    return _normalize(value).lower()


def _format_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"

    if abs(numeric) >= 1000:
        return f"{numeric:,.2f}"
    if abs(numeric) >= 1:
        return f"{numeric:.2f}"
    if abs(numeric) >= 0.01:
        return f"{numeric:.4f}"
    return f"{numeric:.6f}"


def _format_percent(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{numeric:.1f}%"


def _display_verdict(value: Any) -> str:
    token = _normalize(value).upper()
    if token == "LONG":
        return "BUY / LONG"
    if token == "SHORT":
        return "SELL / SHORT"
    if token == "NO TRADE":
        return "NO TRADE / DEFENSE"
    if token == "WAIT":
        return "WAIT / STANDBY"
    return token or "-"


def _extract_plan(result: dict[str, Any]) -> dict[str, Any]:
    brain_output = result.get("brain_output") or result
    return brain_output.get("plan") or result.get("plan") or {}


def _current_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    result = context.get("result") or {}
    payload = context.get("payload") or {}
    summary = result.get("summary") or {}
    plan = _extract_plan(result)
    return {
        "symbol": _normalize(summary.get("symbol") or payload.get("symbol")),
        "timeframe": _normalize(summary.get("timeframe") or payload.get("timeframe") or "15m"),
        "verdict": _normalize(summary.get("verdict")),
        "confidence": summary.get("confidence"),
        "entry": plan.get("entry_mid") or (plan.get("entry_zone") or [None])[0],
        "stop_loss": plan.get("stop_loss"),
        "tp1": plan.get("take_profit_1"),
        "tp2": plan.get("take_profit_2"),
        "rr": plan.get("risk_reward"),
        "thesis": result.get("strategic_brief", {}).get("primary_thesis")
        or result.get("summary", {}).get("edge_summary")
        or result.get("brain_output", {}).get("summary", {}).get("edge_summary")
        or result.get("summary", {}).get("note")
        or "",
        "reasons": (result.get("brain_output", {}).get("reasons") or result.get("reasons") or [])[:3],
        "warnings": (result.get("brain_output", {}).get("warnings") or result.get("warnings") or [])[:3],
        "blockers": (result.get("brain_output", {}).get("blockers") or result.get("blockers") or [])[:3],
    }


def _best_setup(context: dict[str, Any]) -> dict[str, Any]:
    recommendations = context.get("recommendations") or {}
    items = recommendations.get("items") or recommendations.get("fallback_items") or []
    best = items[0] if items else {}
    return {
        "symbol": _normalize(best.get("symbol")),
        "verdict": _normalize(best.get("verdict") or best.get("display_verdict")),
        "score": best.get("score"),
        "confidence": best.get("confidence"),
        "entry": best.get("entry_zone"),
        "stop_loss": best.get("stop_loss"),
        "tp1": best.get("take_profit_1"),
        "rr": best.get("risk_reward"),
        "reason": _normalize(best.get("reason") or best.get("claw_summary") or best.get("profile_note")),
        "headline_mood": _normalize(best.get("headline_mood")),
        "fresh_listing": bool(best.get("fresh_listing")),
        "rr_healthy": best.get("rr_healthy") is not False,
    }


def _readiness_profile(context: dict[str, Any]) -> dict[str, Any]:
    memory_dashboard = context.get("memory_dashboard") or {}
    training_dashboard = context.get("training_dashboard") or {}
    global_summary = memory_dashboard.get("global_summary") or {}
    training_current = training_dashboard.get("current") or {}
    training_overview = training_dashboard.get("overview") or {}
    pair_summary = memory_dashboard.get("pair_summary") or {}

    closed_signals = int(global_summary.get("closed_signals") or 0)
    tracked_pairs = int(global_summary.get("tracked_pairs") or 0)
    sample_size = int(training_current.get("sample_size") or pair_summary.get("sample_size") or pair_summary.get("scored_total") or 0)
    training_days = int(training_current.get("training_days") or 0)
    active_contexts = int(training_overview.get("active_contexts") or 0)
    win_rate = global_summary.get("win_rate")
    trainer_state = _normalize_lower(training_current.get("trainer_state"))
    learning_state = _normalize_lower(pair_summary.get("state"))

    def clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    closed_score = clamp(closed_signals / 40, 0, 1) * 40
    sample_score = clamp(sample_size / 24, 0, 1) * 18
    days_score = clamp(training_days / 14, 0, 1) * 14
    coverage_score = clamp(max(tracked_pairs, active_contexts) / 10, 0, 1) * 12
    consistency_score = 0.0
    if closed_signals >= 8 and win_rate is not None:
        consistency_score = clamp((float(win_rate) - 40) / 25, 0, 1) * 10

    adjustment = 0.0
    if trainer_state == "compounding":
        adjustment += 6
    elif trainer_state == "improving":
        adjustment += 4
    elif trainer_state == "calibrating":
        adjustment += 1
    elif trainer_state == "defensive":
        adjustment -= 10

    if learning_state == "in_sync":
        adjustment += 4
    elif learning_state == "cooldown":
        adjustment -= 8
    elif learning_state == "overheat":
        adjustment -= 5

    score = round(clamp(closed_score + sample_score + days_score + coverage_score + consistency_score + adjustment, 0, 100))
    phase = "Warming"
    note = f"Baru {closed_signals} closed signal, jadi sistem masih mengumpulkan pola dasar."
    if trainer_state == "defensive" or learning_state == "cooldown":
        phase = "Defensive"
        note = "Trainer sedang menahan agresi karena performa terakhir belum stabil."
    elif score >= 80:
        phase = "Live Ready"
        note = "Histori, training, dan cakupan pair sudah cukup matang untuk dipakai lebih percaya diri."
    elif score >= 62:
        phase = "Ready"
        note = "Belajar sudah cukup stabil dan pair yang sinkron mulai kebaca."
    elif score >= 42:
        phase = "Building"
        note = "Edge mulai kebentuk, tapi sistem masih butuh sample tambahan."
    elif score >= 20:
        phase = "Learning"
        note = "Arah belajar sudah ada, tapi model belum cukup tebal untuk agresif."
    return {
        "score": score,
        "phase": phase,
        "note": note,
        "closed_signals": closed_signals,
        "sample_size": sample_size,
        "training_days": training_days,
        "tracked_pairs": tracked_pairs,
        "win_rate": win_rate,
    }


def _open_signals(context: dict[str, Any]) -> list[dict[str, Any]]:
    memory_dashboard = context.get("memory_dashboard") or {}
    return memory_dashboard.get("open_breakdown") or []


def _news_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    news = context.get("news") or {}
    articles = news.get("articles") or []
    return {
        "mood": _normalize(news.get("headline_mood")),
        "score": news.get("sentiment_score"),
        "risk": _normalize(news.get("headline_risk")),
        "headline": _normalize(articles[0].get("title") if articles else ""),
        "source": _normalize(articles[0].get("source") if articles else ""),
    }


def _reply_current_signal(snapshot: dict[str, Any]) -> tuple[str, list[str]]:
    symbol = snapshot["symbol"] or "pair ini"
    verdict = _display_verdict(snapshot["verdict"])
    entry = _format_number(snapshot["entry"])
    stop_loss = _format_number(snapshot["stop_loss"])
    tp1 = _format_number(snapshot["tp1"])
    tp2 = _format_number(snapshot["tp2"])
    confidence = _format_percent(snapshot["confidence"])
    rr = _format_number(snapshot["rr"])
    thesis = snapshot["thesis"] or "Masih menunggu thesis yang lebih lengkap dari analyze terakhir."
    reply = (
        f"Untuk {symbol} {snapshot['timeframe']}, call sekarang {verdict}. "
        f"Entry {entry}, SL {stop_loss}, TP1 {tp1}, TP2 {tp2}, confidence {confidence}, dan RR {rr} : 1. "
        f"{thesis}"
    )
    chips = [symbol, snapshot["timeframe"], verdict, f"RR {rr} : 1"]
    return reply, [chip for chip in chips if chip and chip != "-"]


def _reply_best_setup(best: dict[str, Any]) -> tuple[str, list[str]]:
    if not best["symbol"]:
        return (
            "Belum ada best setup live yang layak. Scanner masih menunggu pair yang benar-benar bersih.",
            ["Best Setup", "Standby"],
        )
    verdict = _display_verdict(best["verdict"])
    rr_text = _format_number(best["rr"])
    reply = (
        f"Best setup saat ini {best['symbol']} dengan bias {verdict}. "
        f"Score { _format_number(best['score']) }, confidence {_format_percent(best['confidence'])}, "
        f"dan RR {rr_text} : 1. "
        f"{best['reason'] or 'Alasan detail belum ada, tapi pair ini sedang memimpin ranking live.'}"
    )
    chips = [best["symbol"], verdict, f"Score {_format_number(best['score'])}", f"RR {rr_text} : 1"]
    if best["fresh_listing"]:
        chips.append("Fresh Listing")
    if best["headline_mood"]:
        chips.append(best["headline_mood"])
    return reply, [chip for chip in chips if chip and chip != "-"]


def _reply_why_wait(snapshot: dict[str, Any]) -> tuple[str, list[str]]:
    blockers = snapshot["blockers"] or snapshot["warnings"] or snapshot["reasons"]
    if snapshot["verdict"].upper() not in {"WAIT", "NO TRADE"}:
        return (
            f"Karena verdict sekarang bukan wait, sistem justru sedang condong ke {_display_verdict(snapshot['verdict'])}.",
            [_display_verdict(snapshot["verdict"])],
        )
    reason = blockers[0] if blockers else "Setup belum cukup rapi untuk dieksekusi."
    return (
        f"Sistem masih {_display_verdict(snapshot['verdict'])} karena {reason}",
        [_display_verdict(snapshot["verdict"]), snapshot["symbol"] or "Current Pair"],
    )


def _reply_open_signals(open_items: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if not open_items:
        return (
            "Saat ini belum ada signal open yang sedang dipantau di memory.",
            ["No Open", "Signal Memory"],
        )
    top = open_items[:3]
    parts = []
    for item in top:
        active = item.get("active_signal") or (item.get("open_signal_details") or [{}])[0]
        verdict = _display_verdict(active.get("verdict"))
        parts.append(f"{item.get('symbol')} {verdict} ({item.get('open_signals') or 1} open)")
    return (
        f"Ada {len(open_items)} pair dengan signal open. Yang paling aktif sekarang: " + ", ".join(parts) + ".",
        [f"{len(open_items)} Open", "Memory"],
    )


def _reply_readiness(profile: dict[str, Any]) -> tuple[str, list[str]]:
    reply = (
        f"Readiness sistem sekarang {profile['score']}% dan masuk fase {profile['phase']}. "
        f"{profile['note']} Histori yang sudah dipakai: {profile['closed_signals']} closed signal, "
        f"{profile['sample_size']} sample, day {profile['training_days']}, dan {profile['tracked_pairs']} pair terlacak."
    )
    chips = [f"{profile['score']}%", profile["phase"], f"{profile['closed_signals']} closed", f"{profile['tracked_pairs']} pair"]
    return reply, chips


def _reply_news(news: dict[str, Any]) -> tuple[str, list[str]]:
    if not news["headline"]:
        return (
            "Belum ada headline aktif yang cukup kuat untuk dijadikan konteks news saat ini.",
            ["News", "Standby"],
        )
    reply = (
        f"Mood news sekarang {news['mood'] or 'neutral'} dengan headline risk {news['risk'] or 'normal'}. "
        f"Headline teratas: {news['headline']}"
        + (f" dari {news['source']}." if news["source"] else ".")
    )
    chips = [news["mood"] or "Neutral", news["risk"] or "Normal"]
    return reply, [chip for chip in chips if chip]


def _reply_default(snapshot: dict[str, Any], best: dict[str, Any], readiness: dict[str, Any], open_items: list[dict[str, Any]], news: dict[str, Any]) -> tuple[str, list[str]]:
    symbol = snapshot["symbol"] or best["symbol"] or "market"
    verdict = _display_verdict(snapshot["verdict"] or best["verdict"] or "WAIT")
    best_label = best["symbol"] or "belum ada best setup live"
    reply = (
        f"Ringkas sekarang: fokus pair {symbol}, verdict aktif {verdict}, readiness {readiness['score']}% ({readiness['phase']}), "
        f"best setup {best_label}, dan signal open {len(open_items)} pair. "
        f"{news['headline'] or 'News masih netral.'}"
    )
    chips = [symbol, verdict, f"Readiness {readiness['score']}%", f"{len(open_items)} Open"]
    return reply, chips


def answer_chat(message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    prompt = _normalize_lower(message)
    snapshot = _current_snapshot(context)
    best = _best_setup(context)
    readiness = _readiness_profile(context)
    open_items = _open_signals(context)
    news = _news_snapshot(context)

    if not prompt:
      prompt = "ringkas kondisi sekarang"

    if any(token in prompt for token in ["best", "setup", "rekom", "pair bagus", "pair terbaik"]):
        reply, chips = _reply_best_setup(best)
    elif any(token in prompt for token in ["kenapa", "wait", "no trade", "defense"]):
        reply, chips = _reply_why_wait(snapshot)
    elif any(token in prompt for token in ["open", "floating", "memory"]):
        reply, chips = _reply_open_signals(open_items)
    elif any(token in prompt for token in ["readiness", "siap", "training", "belajar"]):
        reply, chips = _reply_readiness(readiness)
    elif any(token in prompt for token in ["news", "headline", "berita"]):
        reply, chips = _reply_news(news)
    elif any(token in prompt for token in ["signal", "entry", "sl", "tp", "buy", "sell", "call"]):
        reply, chips = _reply_current_signal(snapshot)
    else:
        reply, chips = _reply_default(snapshot, best, readiness, open_items, news)

    suggestions = [
        "Apa call sekarang?",
        "Best setup mana?",
        "Open signal apa aja?",
        "Readiness sistem berapa?",
    ]
    return {
        "reply": reply,
        "chips": chips[:6],
        "suggestions": suggestions,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
