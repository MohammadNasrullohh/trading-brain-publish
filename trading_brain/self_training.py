from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .signal_memory import (
    LOSS_STATUSES,
    STORE_LOCK,
    WIN_STATUSES,
    _parse_dt,
    _read_store as _read_signal_store,
    learning_context as signal_learning_context,
)
from .storage_db import DB_FILENAME, default_data_dir, read_document, write_document
from .utils import clamp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = default_data_dir(ROOT)
DATA_PATH = DATA_DIR / "self_training.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _db_path() -> Path:
    return DATA_DIR / DB_FILENAME


def _legacy_paths() -> tuple[Path, ...]:
    paths = [DATA_PATH]
    fallback = ROOT / "logs" / "self_training.json"
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
        "profiles": {},
        "daily_reports": {},
    }


def _read_store() -> dict[str, Any]:
    _ensure_data_dir()
    return read_document(
        db_path=_db_path(),
        module="self_training",
        default_factory=_default_store,
        legacy_paths=_legacy_paths(),
        legacy_db_paths=_legacy_db_paths(),
    )


def _write_store(store: dict[str, Any]) -> None:
    _ensure_data_dir()
    profiles = store.get("profiles", {})
    if isinstance(profiles, dict) and len(profiles) > 320:
        ordered = sorted(
            profiles.items(),
            key=lambda item: str(item[1].get("updated_at") or ""),
            reverse=True,
        )[:320]
        profiles = dict(reversed(ordered))

    daily_reports = store.get("daily_reports", {})
    if isinstance(daily_reports, dict) and len(daily_reports) > 45:
        ordered_days = sorted(daily_reports.items(), key=lambda item: item[0], reverse=True)[:45]
        daily_reports = dict(reversed(ordered_days))

    write_document(
        db_path=_db_path(),
        module="self_training",
        payload={
            "version": 1,
            "profiles": profiles,
            "daily_reports": daily_reports,
        },
    )


def _normalize_text(value: Any, default: str = "") -> str:
    raw = str(value or "").strip().lower()
    return raw or default


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _context_key(symbol: str, timeframe: str, style: str, market_type: str) -> str:
    return f"{symbol}|{timeframe}|{style}|{market_type}"


def _same_context(signal: dict[str, Any], symbol: str, timeframe: str, style: str) -> bool:
    return (
        str(signal.get("symbol") or "").strip().upper() == symbol
        and str(signal.get("timeframe") or "").strip().lower() == timeframe
        and str(signal.get("style") or "").strip().lower() == style
    )


def _closed_at(signal: dict[str, Any]) -> str:
    return str(signal.get("closed_at") or signal.get("updated_at") or signal.get("opened_at") or "")


def _sorted_scored(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [signal for signal in signals if str(signal.get("status") or "") in WIN_STATUSES | LOSS_STATUSES]
    return sorted(filtered, key=_closed_at, reverse=True)


def _merge_unique(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for signal in group:
            signal_id = str(signal.get("id") or "")
            if signal_id and signal_id in seen:
                continue
            if signal_id:
                seen.add(signal_id)
            merged.append(signal)
    return merged


def _avg_rr(signals: list[dict[str, Any]]) -> float | None:
    values = []
    for signal in signals:
        value = signal.get("risk_reward")
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _direction_stats(signals: list[dict[str, Any]], verdict: str) -> dict[str, Any]:
    bucket = [signal for signal in signals if str(signal.get("verdict") or "").strip().upper() == verdict]
    wins = sum(1 for signal in bucket if str(signal.get("status") or "") in WIN_STATUSES)
    losses = sum(1 for signal in bucket if str(signal.get("status") or "") in LOSS_STATUSES)
    total = wins + losses
    win_rate = round((wins / total) * 100.0, 1) if total else None
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_rr": _avg_rr(bucket),
    }


def _daily_rollup(signals: list[dict[str, Any]], limit: int = 7) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for signal in signals:
        parsed = _parse_dt(_closed_at(signal))
        if parsed is None:
            continue
        day_key = parsed.date().isoformat()
        bucket = buckets.setdefault(day_key, {"day": day_key, "wins": 0, "losses": 0, "total": 0})
        status = str(signal.get("status") or "")
        if status in WIN_STATUSES:
            bucket["wins"] += 1
        elif status in LOSS_STATUSES:
            bucket["losses"] += 1
        bucket["total"] += 1

    ordered = sorted(buckets.values(), key=lambda item: item["day"], reverse=True)[:limit]
    for item in ordered:
        item["net"] = item["wins"] - item["losses"]
    return ordered


def _direction_edge(stats: dict[str, Any], scope: str) -> float:
    total = int(stats.get("total") or 0)
    win_rate = stats.get("win_rate")
    if total <= 0 or win_rate is None:
        return 0.0
    scope_weight = 1.0 if scope == "pair" else 0.78 if scope == "mixed" else 0.58
    maturity = clamp(total / (5.0 if scope == "pair" else 8.0), 0.24, 1.0)
    raw = ((float(win_rate) - 50.0) / 100.0) * scope_weight * maturity * 1.55
    return round(clamp(raw, -0.18, 0.18), 3)


def derive_training_profile(
    symbol: str | None,
    timeframe: str | None,
    style: str | None,
    market_type: str | None,
    *,
    learning_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_symbol = _normalize_symbol(symbol)
    normalized_timeframe = _normalize_text(timeframe, "15m")
    normalized_style = _normalize_text(style, "intraday")
    normalized_market = _normalize_text(market_type, "market")
    context = dict(
        learning_context
        or signal_learning_context(normalized_symbol, normalized_timeframe, normalized_style, normalized_market)
    )

    with STORE_LOCK:
        signal_store = _read_signal_store()
        training_store = _read_store()

    exact_closed = _sorted_scored(
        [
            signal
            for signal in signal_store.get("signals", [])
            if _same_context(signal, normalized_symbol, normalized_timeframe, normalized_style)
        ]
    )
    market_closed = _sorted_scored(
        [
            signal
            for signal in signal_store.get("signals", [])
            if str(signal.get("market_type") or "").strip().lower() == normalized_market
            and str(signal.get("style") or "").strip().lower() == normalized_style
        ]
    )
    market_only_closed = [
        signal for signal in market_closed if not _same_context(signal, normalized_symbol, normalized_timeframe, normalized_style)
    ]

    pair_scored_total = len(exact_closed)
    market_scored_total = len(market_only_closed)
    if pair_scored_total >= 3:
        sample = exact_closed[:18]
        source_scope = "pair"
    elif pair_scored_total >= 1 and market_scored_total >= 3:
        sample = _merge_unique(exact_closed[:4], market_only_closed)[:18]
        source_scope = "mixed"
    elif market_scored_total >= 4:
        sample = market_only_closed[:18]
        source_scope = "market"
    else:
        sample = _merge_unique(exact_closed, market_only_closed)[:18]
        source_scope = "warming"

    observed_days = len({_parse_dt(_closed_at(signal)).date().isoformat() for signal in sample if _parse_dt(_closed_at(signal))})
    daily_rollup = _daily_rollup(sample, limit=7)
    profitable_days = sum(1 for day in daily_rollup if day["net"] > 0)
    losing_days = sum(1 for day in daily_rollup if day["net"] < 0)

    wins = int(context.get("wins") or 0)
    losses = int(context.get("losses") or 0)
    scored_total = int(context.get("scored_total") or 0)
    win_rate = float(context.get("win_rate") or 0.0) if context.get("win_rate") is not None else None
    loss_streak = int(context.get("loss_streak") or 0)
    memory_scope = _normalize_text(context.get("memory_scope"), "warming")

    long_stats = _direction_stats(sample, "LONG")
    short_stats = _direction_stats(sample, "SHORT")
    long_edge = _direction_edge(long_stats, source_scope)
    short_edge = _direction_edge(short_stats, source_scope)
    preferred_direction = None
    if long_edge >= short_edge + 0.045:
        preferred_direction = "long"
    elif short_edge >= long_edge + 0.045:
        preferred_direction = "short"

    is_scalp = normalized_style in {"scalp", "scalping"}
    rr_threshold = 1.15 if is_scalp else 1.4
    recent_losses = [signal for signal in sample[:8] if str(signal.get("status") or "") in LOSS_STATUSES]
    low_rr_losses = 0
    overconfident_losses = 0
    for signal in recent_losses:
        try:
            rr_value = float(signal.get("risk_reward") or 0.0)
        except (TypeError, ValueError):
            rr_value = 0.0
        try:
            confidence = float(signal.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if rr_value and rr_value < rr_threshold:
            low_rr_losses += 1
        if confidence >= 0.8:
            overconfident_losses += 1

    avg_rr = _avg_rr(sample)
    lesson_notes: list[str] = []
    trainer_state = "warming"
    confidence_shift = 0.0
    rr_floor_delta = 0.0
    risk_cap_delta = 0.0

    if scored_total < 2:
        trainer_state = "warming"
        lesson_notes.append("Trainer masih mengumpulkan sample closed signal pertama.")
    elif loss_streak >= 3 or (win_rate is not None and win_rate <= 38.0) or losing_days >= profitable_days + 2:
        trainer_state = "defensive"
        confidence_shift -= 0.03
        rr_floor_delta += 0.14
        risk_cap_delta -= 0.14
        lesson_notes.append("Loss masih dominan, jadi trainer masuk mode proteksi harian.")
    elif win_rate is not None and win_rate >= 68.0 and observed_days >= 3 and wins >= max(losses + 2, 3):
        trainer_state = "compounding"
        confidence_shift += 0.02
        rr_floor_delta -= 0.05
        risk_cap_delta += 0.06
        lesson_notes.append("Histori profit cukup konsisten, jadi trainer memberi ruang compounding ringan.")
    elif win_rate is not None and win_rate >= 56.0 and wins >= max(losses, 2):
        trainer_state = "improving"
        confidence_shift += 0.012
        risk_cap_delta += 0.03
        lesson_notes.append("Trainer melihat ritme pair mulai membaik dari hari-hari terakhir.")
    else:
        trainer_state = "calibrating"
        lesson_notes.append("Trainer masih menyesuaikan bias terbaik dari sample yang ada.")

    if low_rr_losses >= 2:
        rr_floor_delta += 0.12
        lesson_notes.append("Loss terbaru banyak datang dari RR yang terlalu tipis.")
    if overconfident_losses >= 2:
        confidence_shift -= 0.015
        lesson_notes.append("Confidence tinggi belum layak diberi ruang agresif.")
    if preferred_direction == "long" and long_stats["total"] >= 2:
        lesson_notes.append("Replay profit/loss menunjukkan sisi long lebih sinkron untuk context ini.")
    elif preferred_direction == "short" and short_stats["total"] >= 2:
        lesson_notes.append("Replay profit/loss menunjukkan sisi short lebih sinkron untuk context ini.")

    confidence_shift = round(clamp(confidence_shift, -0.08, 0.05), 3)
    rr_floor_delta = round(clamp(rr_floor_delta, -0.08, 0.22), 3)
    risk_cap_delta = round(clamp(risk_cap_delta, -0.22, 0.08), 3)
    daily_progress = profitable_days - losing_days

    coach_note = lesson_notes[0] if lesson_notes else "Trainer harian siap membaca hasil profit/loss baru."
    profile = {
        "symbol": normalized_symbol,
        "timeframe": normalized_timeframe,
        "style": normalized_style,
        "market_type": normalized_market,
        "trainer_state": trainer_state,
        "source_scope": source_scope if source_scope != "warming" else memory_scope,
        "observed_days": observed_days,
        "daily_progress": daily_progress,
        "profitable_days": profitable_days,
        "losing_days": losing_days,
        "sample_size": len(sample),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1) if win_rate is not None else None,
        "loss_streak": loss_streak,
        "long_sample": long_stats["total"],
        "short_sample": short_stats["total"],
        "long_win_rate": long_stats["win_rate"],
        "short_win_rate": short_stats["win_rate"],
        "long_edge": long_edge,
        "short_edge": short_edge,
        "preferred_direction": preferred_direction,
        "avg_rr": avg_rr,
        "low_rr_losses": low_rr_losses,
        "overconfident_losses": overconfident_losses,
        "rr_floor_delta": rr_floor_delta,
        "confidence_shift": confidence_shift,
        "risk_cap_delta": risk_cap_delta,
        "lesson_notes": lesson_notes[:4],
        "note": coach_note,
        "coach_note": coach_note,
        "updated_at": _utc_now_iso(),
    }

    if not normalized_symbol:
        return profile

    day_key = _utc_now().date().isoformat()
    context_key = _context_key(normalized_symbol, normalized_timeframe, normalized_style, normalized_market)
    with STORE_LOCK:
        store = training_store
        profiles = store.setdefault("profiles", {})
        daily_reports = store.setdefault("daily_reports", {})
        reports_for_day = daily_reports.setdefault(day_key, {})
        reports_for_day[context_key] = {
            "trainer_state": trainer_state,
            "sample_size": len(sample),
            "win_rate": profile.get("win_rate"),
            "preferred_direction": preferred_direction,
            "market_type": normalized_market,
            "note": coach_note,
            "updated_at": profile["updated_at"],
        }
        training_days = sum(1 for _, bucket in daily_reports.items() if context_key in bucket)
        profile["training_days"] = training_days
        previous = profiles.get(context_key, {})
        profile["sessions"] = int(previous.get("sessions") or 0) + 1
        profiles[context_key] = profile
        _write_store(store)

    return profile


def load_training_profile(
    symbol: str | None,
    timeframe: str | None,
    style: str | None,
    market_type: str | None,
) -> dict[str, Any] | None:
    normalized_symbol = _normalize_symbol(symbol)
    if not normalized_symbol:
        return None

    normalized_timeframe = _normalize_text(timeframe, "15m")
    normalized_style = _normalize_text(style, "intraday")
    normalized_market = _normalize_text(market_type, "market")
    with STORE_LOCK:
        store = _read_store()
    return store.get("profiles", {}).get(_context_key(normalized_symbol, normalized_timeframe, normalized_style, normalized_market))


def get_training_dashboard(
    symbol: str | None,
    timeframe: str | None,
    style: str | None,
    market_type: str | None,
    *,
    limit: int = 7,
) -> dict[str, Any]:
    current_profile = derive_training_profile(symbol, timeframe, style, market_type)
    normalized_market = _normalize_text(market_type, "")

    with STORE_LOCK:
        store = _read_store()

    profiles = list((store.get("profiles") or {}).values())
    daily_reports = store.get("daily_reports") or {}

    relevant_profiles = [
        profile for profile in profiles
        if not normalized_market or _normalize_text(profile.get("market_type")) == normalized_market
    ]
    state_counts = {
        "warming": 0,
        "calibrating": 0,
        "improving": 0,
        "compounding": 0,
        "defensive": 0,
    }
    for profile in relevant_profiles:
        state_key = _normalize_text(profile.get("trainer_state"), "warming")
        state_counts[state_key] = int(state_counts.get(state_key) or 0) + 1

    avg_win_rates = [
        float(profile.get("win_rate"))
        for profile in relevant_profiles
        if profile.get("win_rate") is not None
    ]
    overview = {
        "active_contexts": len(relevant_profiles),
        "training_days_total": sum(int(profile.get("training_days") or 0) for profile in relevant_profiles),
        "avg_win_rate": round(sum(avg_win_rates) / len(avg_win_rates), 1) if avg_win_rates else None,
        "state_counts": state_counts,
    }

    daily_curve: list[dict[str, Any]] = []
    for day, bucket in sorted(daily_reports.items(), key=lambda item: item[0], reverse=True)[:limit]:
        reports = list((bucket or {}).values())
        if normalized_market:
            reports = [
                report for report in reports
                if _normalize_text((report or {}).get("market_type")) == normalized_market
            ]
        if not reports:
            continue
        avg_day_win_rates = [
            float(report.get("win_rate"))
            for report in reports
            if report.get("win_rate") is not None
        ]
        day_counts = {
            "warming": 0,
            "calibrating": 0,
            "improving": 0,
            "compounding": 0,
            "defensive": 0,
        }
        for report in reports:
            state_key = _normalize_text(report.get("trainer_state"), "warming")
            day_counts[state_key] = int(day_counts.get(state_key) or 0) + 1
        daily_curve.append(
            {
                "day": day,
                "contexts": len(reports),
                "avg_win_rate": round(sum(avg_day_win_rates) / len(avg_day_win_rates), 1) if avg_day_win_rates else None,
                "compounding": day_counts.get("compounding", 0),
                "improving": day_counts.get("improving", 0),
                "defensive": day_counts.get("defensive", 0),
                "warming": day_counts.get("warming", 0),
            }
        )
    daily_curve.reverse()

    leaderboard = sorted(
        relevant_profiles,
        key=lambda item: (
            float(item.get("sample_size") or 0),
            float(item.get("win_rate") or 0),
            float(item.get("training_days") or 0),
        ),
        reverse=True,
    )[:6]

    return {
        "current": current_profile,
        "overview": overview,
        "daily_curve": daily_curve,
        "leaderboard": [
            {
                "symbol": profile.get("symbol"),
                "timeframe": profile.get("timeframe"),
                "style": profile.get("style"),
                "trainer_state": profile.get("trainer_state"),
                "win_rate": profile.get("win_rate"),
                "sample_size": profile.get("sample_size"),
                "training_days": profile.get("training_days"),
                "preferred_direction": profile.get("preferred_direction"),
                "source_scope": profile.get("source_scope"),
                "updated_at": profile.get("updated_at"),
                "note": profile.get("note"),
            }
            for profile in leaderboard
        ],
    }
