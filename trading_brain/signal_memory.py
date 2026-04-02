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
DATA_PATH = DATA_DIR / "signal_memory.json"
STORE_LOCK = threading.RLock()

TIMEFRAME_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

WIN_STATUSES = {"win", "soft_win"}
LOSS_STATUSES = {"loss", "soft_loss"}
NEUTRAL_STATUSES = {"mixed", "stale", "open"}
MARKET_SCOPE_LABELS = {
    "crypto": "crypto",
    "forex": "forex",
    "commodity": "commodity",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _db_path() -> Path:
    return DATA_DIR / DB_FILENAME


def _legacy_paths() -> tuple[Path, ...]:
    paths = [DATA_PATH]
    fallback = ROOT / "logs" / "signal_memory.json"
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
        "signals": [],
        "events": [],
    }


def _read_store() -> dict[str, Any]:
    _ensure_data_dir()
    return read_document(
        db_path=_db_path(),
        module="signal_memory",
        default_factory=_default_store,
        legacy_paths=_legacy_paths(),
        legacy_db_paths=_legacy_db_paths(),
    )


def _write_store(store: dict[str, Any]) -> None:
    _ensure_data_dir()
    trimmed = {
        **store,
        "signals": list(store.get("signals", []))[-240:],
        "events": list(store.get("events", []))[-480:],
    }
    write_document(
        db_path=_db_path(),
        module="signal_memory",
        payload=trimmed,
    )


def _safe_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _timeframe_minutes(timeframe: str | None) -> int:
    return TIMEFRAME_MINUTES.get(str(timeframe or "").strip().lower(), 15)


def _round_price(value: Any) -> float | None:
    numeric = _safe_number(value)
    if numeric is None:
        return None
    if abs(numeric) >= 1000:
        return round(numeric, 2)
    if abs(numeric) >= 10:
        return round(numeric, 3)
    if abs(numeric) >= 1:
        return round(numeric, 4)
    return round(numeric, 5)


def _entry_mid(plan: dict[str, Any], fallback_price: Any = None) -> float | None:
    entry_zone = plan.get("entry_zone")
    if isinstance(entry_zone, list):
        values = [_safe_number(value) for value in entry_zone]
        values = [value for value in values if value is not None]
        if values:
            if len(values) == 1:
                return values[0]
            return sum(values[:2]) / min(2, len(values))
    return _safe_number(fallback_price)


def _extract_plan(result: dict[str, Any]) -> dict[str, Any]:
    base = result.get("brain_output", result)
    return base.get("plan") or base.get("conditional_plan") or {}


def _extract_summary(result: dict[str, Any]) -> dict[str, Any]:
    base = result.get("brain_output", result)
    return {
        **(base.get("summary") or {}),
        **(result.get("summary") or {}),
    }


def _direction_multiplier(verdict: str) -> int:
    return 1 if verdict == "LONG" else -1


def _signal_ttl_minutes(timeframe: str | None, style: str | None) -> int:
    tf_minutes = _timeframe_minutes(timeframe)
    style_name = str(style or "").strip().lower()
    multiplier = 18 if style_name in {"scalp", "scalping"} else 36
    return max(60, min(tf_minutes * multiplier, 72 * 60))


def _signal_fingerprint(
    symbol: str,
    timeframe: str,
    style: str,
    verdict: str,
    entry_mid: float | None,
    stop_loss: float | None,
    take_profit_1: float | None,
) -> str:
    parts = [
        symbol,
        timeframe,
        style,
        verdict,
        _round_price(entry_mid),
        _round_price(stop_loss),
        _round_price(take_profit_1),
    ]
    return "|".join("" if part is None else str(part) for part in parts)


def _append_event(store: dict[str, Any], event: dict[str, Any]) -> None:
    store.setdefault("events", []).append(event)


def _build_event(
    *,
    signal_id: str,
    event_type: str,
    symbol: str,
    timeframe: str,
    style: str,
    verdict: str,
    status: str,
    title: str,
    note: str,
) -> dict[str, Any]:
    return {
        "id": uuid4().hex,
        "signal_id": signal_id,
        "event_type": event_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "style": style,
        "verdict": verdict,
        "status": status,
        "title": title,
        "note": note,
        "timestamp": _utc_now_iso(),
    }


def _status_label(status: str | None) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in WIN_STATUSES:
        return "Win"
    if normalized in LOSS_STATUSES:
        return "Loss"
    if normalized == "mixed":
        return "Mixed"
    if normalized == "stale":
        return "Stale"
    if normalized == "standby":
        return "Standby"
    if normalized == "defense":
        return "Defense"
    return "Open"


def _signal_risk_distance(signal: dict[str, Any]) -> float | None:
    entry_mid = _safe_number(signal.get("entry_mid"))
    stop_loss = _safe_number(signal.get("stop_loss"))
    if entry_mid is None or stop_loss is None:
        return None
    distance = abs(entry_mid - stop_loss)
    return distance if distance > 0 else None


def _signal_outcome_r(signal: dict[str, Any]) -> float:
    status = str(signal.get("status") or "").strip().lower()
    rr_value = _safe_number(signal.get("risk_reward")) or 1.0
    risk_distance = _signal_risk_distance(signal)
    mfe = _safe_number(signal.get("mfe"))
    mae = _safe_number(signal.get("mae"))

    if status == "win":
        return round(max(rr_value, 1.0), 2)
    if status == "loss":
        return -1.0
    if status == "soft_win":
        if risk_distance and mfe is not None:
            return round(max(0.35, min(max(rr_value, 0.55), mfe / risk_distance)), 2)
        return round(min(max(rr_value * 0.45, 0.35), max(rr_value, 0.65)), 2)
    if status == "soft_loss":
        if risk_distance and mae is not None:
            return round(min(-0.35, max(-1.0, mae / risk_distance)), 2)
        return -0.45
    return 0.0


def _signal_outcome_pct(signal: dict[str, Any]) -> float | None:
    risk_percent = _safe_number(signal.get("risk_percent"))
    if risk_percent is None:
        return None
    return round(_signal_outcome_r(signal) * risk_percent, 3)


def _format_reason_list(values: Any, limit: int = 2) -> str | None:
    if not isinstance(values, list):
        return None
    items = [str(value).strip() for value in values if str(value or "").strip()]
    if not items:
        return None
    return "; ".join(items[:limit])


def _pair_key(signal: dict[str, Any]) -> str:
    symbol = str(signal.get("symbol") or "").strip().upper()
    timeframe = str(signal.get("timeframe") or "").strip().lower()
    style = str(signal.get("style") or "").strip().lower()
    return "|".join([symbol, timeframe, style])


def _same_context(signal: dict[str, Any], symbol: str, timeframe: str, style: str) -> bool:
    return (
        signal.get("symbol") == symbol
        and signal.get("timeframe") == timeframe
        and signal.get("style") == style
    )


def _resolve_market_state(market_state: dict[str, Any] | None) -> dict[str, float | None]:
    raw = market_state or {}
    price = _safe_number(raw.get("price")) or _safe_number(raw.get("close"))
    high = _safe_number(raw.get("high"))
    low = _safe_number(raw.get("low"))
    close = _safe_number(raw.get("close")) or price
    atr = _safe_number(raw.get("atr"))
    return {
        "price": price,
        "high": high if high is not None else max([value for value in [price, close] if value is not None], default=None),
        "low": low if low is not None else min([value for value in [price, close] if value is not None], default=None),
        "close": close,
        "atr": atr,
    }


def _close_signal(signal: dict[str, Any], status: str, note: str) -> None:
    signal["status"] = status
    signal["closed_at"] = _utc_now_iso()
    signal["updated_at"] = signal["closed_at"]
    signal["outcome_note"] = note


def _update_floats(signal: dict[str, Any], market: dict[str, float | None]) -> None:
    price = market.get("price")
    if price is None:
        return
    signal["last_price"] = _round_price(price)
    signal["updated_at"] = _utc_now_iso()
    entry_mid = _safe_number(signal.get("entry_mid"))
    if entry_mid is None:
        return
    direction = _direction_multiplier(str(signal.get("verdict") or "LONG"))
    move = direction * (price - entry_mid)
    mfe = _safe_number(signal.get("mfe")) or 0.0
    mae = _safe_number(signal.get("mae")) or 0.0
    signal["mfe"] = round(max(mfe, move), 5)
    signal["mae"] = round(min(mae, move), 5)


def _reconcile_open_signal(signal: dict[str, Any], market_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if signal.get("status") != "open":
        return None

    market = _resolve_market_state(market_state)
    if market.get("price") is None and market.get("high") is None and market.get("low") is None:
        return None

    _update_floats(signal, market)

    verdict = str(signal.get("verdict") or "").upper()
    entry_mid = _safe_number(signal.get("entry_mid"))
    stop_loss = _safe_number(signal.get("stop_loss"))
    take_profit_1 = _safe_number(signal.get("take_profit_1"))
    take_profit_2 = _safe_number(signal.get("take_profit_2"))
    high = market.get("high")
    low = market.get("low")
    price = market.get("price") or market.get("close")
    if entry_mid is None or verdict not in {"LONG", "SHORT"}:
        return None

    if verdict == "LONG":
        stop_hit = stop_loss is not None and low is not None and low <= stop_loss
        tp_hit = (
            (take_profit_2 is not None and high is not None and high >= take_profit_2)
            or (take_profit_1 is not None and high is not None and high >= take_profit_1)
        )
    else:
        stop_hit = stop_loss is not None and high is not None and high >= stop_loss
        tp_hit = (
            (take_profit_2 is not None and low is not None and low <= take_profit_2)
            or (take_profit_1 is not None and low is not None and low <= take_profit_1)
        )

    if stop_hit and tp_hit:
        note = "Bar yang sama menyentuh target dan stop, jadi hasilnya ambigu."
        _close_signal(signal, "mixed", note)
        return {"status": "mixed", "note": note}
    if tp_hit:
        note = "Target pertama atau target utama sudah tersentuh dari harga live terbaru."
        _close_signal(signal, "win", note)
        return {"status": "win", "note": note}
    if stop_hit:
        note = "Stop loss sudah tersentuh dari harga live terbaru."
        _close_signal(signal, "loss", note)
        return {"status": "loss", "note": note}

    opened_at = _parse_dt(signal.get("opened_at"))
    if opened_at is None or price is None:
        return None

    age_minutes = (_utc_now() - opened_at).total_seconds() / 60
    if age_minutes < _signal_ttl_minutes(signal.get("timeframe"), signal.get("style")):
        return None

    risk_distance = abs(entry_mid - stop_loss) if stop_loss is not None else None
    if not risk_distance or risk_distance <= 0:
        atr = market.get("atr")
        risk_distance = atr if atr and atr > 0 else max(abs(entry_mid) * 0.0025, 0.0005)

    direction = _direction_multiplier(verdict)
    move = direction * (price - entry_mid)
    if move >= risk_distance * 0.45:
        note = "Setup tidak menyentuh target penuh, tapi bergerak cukup jauh searah sebelum kadaluarsa."
        _close_signal(signal, "soft_win", note)
        return {"status": "soft_win", "note": note}
    if move <= -(risk_distance * 0.45):
        note = "Setup tidak menyentuh stop penuh, tapi bergerak cukup jauh melawan arah sebelum kadaluarsa."
        _close_signal(signal, "soft_loss", note)
        return {"status": "soft_loss", "note": note}

    note = "Setup kedaluwarsa tanpa dorongan yang cukup jelas."
    _close_signal(signal, "stale", note)
    return {"status": "stale", "note": note}


def reconcile_market_memory(symbol: str, timeframe: str, style: str, market_state: dict[str, Any] | None) -> dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_timeframe = str(timeframe or "15m").strip().lower()
    normalized_style = str(style or "intraday").strip().lower()

    with STORE_LOCK:
        store = _read_store()
        recent_events: list[dict[str, Any]] = []
        dirty = False
        for signal in store.get("signals", []):
            if signal.get("status") != "open":
                continue
            if not _same_context(signal, normalized_symbol, normalized_timeframe, normalized_style):
                continue
            before_updated = signal.get("updated_at")
            outcome = _reconcile_open_signal(signal, market_state)
            if signal.get("updated_at") != before_updated:
                dirty = True
            if not outcome:
                continue
            recent_events.append(
                _build_event(
                    signal_id=signal["id"],
                    event_type="closed",
                    symbol=normalized_symbol,
                    timeframe=normalized_timeframe,
                    style=normalized_style,
                    verdict=str(signal.get("verdict") or "WAIT"),
                    status=outcome["status"],
                    title=f"{normalized_symbol} {outcome['status'].replace('_', ' ').upper()}",
                    note=outcome["note"],
                )
            )
        if recent_events:
            for event in recent_events:
                _append_event(store, event)
            dirty = True
        if dirty:
            _write_store(store)
        return {
            "closed_events": recent_events,
            "count": len(recent_events),
        }


def _scored_closed_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [signal for signal in signals if signal.get("status") in WIN_STATUSES | LOSS_STATUSES]


def _merge_unique_signals(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _loss_streak(signals: list[dict[str, Any]]) -> int:
    streak = 0
    for signal in signals:
        status = str(signal.get("status") or "")
        if status in LOSS_STATUSES:
            streak += 1
            continue
        if status in WIN_STATUSES:
            break
    return streak


def learning_context(
    symbol: str,
    timeframe: str,
    style: str,
    market_type: str | None = None,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_timeframe = str(timeframe or "15m").strip().lower()
    normalized_style = str(style or "intraday").strip().lower()
    normalized_market = str(market_type or "").strip().lower()

    with STORE_LOCK:
        store = _read_store()

    exact_signals = [
        signal for signal in store.get("signals", [])
        if _same_context(signal, normalized_symbol, normalized_timeframe, normalized_style)
    ]
    market_signals = [
        signal for signal in store.get("signals", [])
        if str(signal.get("market_type") or "").strip().lower() == normalized_market
        and str(signal.get("style") or "").strip().lower() == normalized_style
    ]

    exact_closed = sorted(
        [signal for signal in exact_signals if signal.get("status") != "open"],
        key=lambda item: item.get("closed_at") or item.get("updated_at") or "",
        reverse=True,
    )
    market_closed = sorted(
        [signal for signal in market_signals if signal.get("status") != "open"],
        key=lambda item: item.get("closed_at") or item.get("updated_at") or "",
        reverse=True,
    )
    market_only_closed = [
        signal for signal in market_closed
        if not _same_context(signal, normalized_symbol, normalized_timeframe, normalized_style)
    ]
    combined = _merge_unique_signals(exact_closed, market_only_closed)

    exact_scored = _scored_closed_signals(exact_closed)
    market_scored = _scored_closed_signals(market_only_closed)
    combined_scored = _scored_closed_signals(combined)

    memory_scope = "warming"
    if len(exact_scored) >= 2:
        sample = exact_closed[:limit]
        memory_scope = "pair"
    elif len(exact_scored) >= 1 and len(market_scored) >= 2:
        sample = _merge_unique_signals(exact_closed[: max(2, min(limit, 4))], market_only_closed)[:limit]
        memory_scope = "mixed"
    elif len(market_scored) >= 3:
        sample = market_only_closed[:limit]
        memory_scope = "market"
    elif len(combined_scored) >= 2:
        sample = combined[:limit]
        memory_scope = "mixed"
    else:
        sample = combined[:limit]
    scored = _scored_closed_signals(sample)
    wins = sum(1 for signal in scored if signal.get("status") in WIN_STATUSES)
    losses = sum(1 for signal in scored if signal.get("status") in LOSS_STATUSES)
    scored_total = len(scored)
    win_rate = round((wins / scored_total) * 100, 1) if scored_total else None

    pair_loss_streak = _loss_streak(exact_closed)
    market_loss_streak = _loss_streak(market_only_closed)
    loss_streak = pair_loss_streak or (market_loss_streak if memory_scope == "market" else 0)

    open_same_context = sum(1 for signal in exact_signals if signal.get("status") == "open")
    state = "warming"
    score_bias = 0.0
    confidence_bias = 0.0
    note = "Histori pair ini masih tipis, jadi engine baru mengumpulkan konteks awal."
    prime_penalty = False
    scope_multiplier = {
        "pair": 1.0,
        "mixed": 0.84,
        "market": 0.72,
        "warming": 0.0,
    }.get(memory_scope, 0.0)
    market_label = MARKET_SCOPE_LABELS.get(normalized_market, normalized_market or "market")
    effective_loss_streak = pair_loss_streak if memory_scope in {"pair", "mixed"} and pair_loss_streak else market_loss_streak

    if scored_total >= 2 and win_rate is not None and memory_scope == "pair":
        state = "neutral"
        note = "Histori pair ini sudah cukup untuk dipakai sebagai rem atau boost ringan."
    elif scored_total >= 3 and win_rate is not None and memory_scope == "mixed":
        state = "neutral"
        note = f"Histori pair mulai terbentuk; engine ikut meminjam konteks {market_label} untuk adaptasi ringan."
    elif scored_total >= 3 and win_rate is not None and memory_scope == "market":
        state = "neutral"
        note = f"Histori pair masih tipis, jadi engine sementara belajar dari sektor {market_label}."

    if state != "warming" and win_rate is not None:
        if effective_loss_streak >= 2 or win_rate <= (35.0 if memory_scope == "pair" else 32.0):
            state = "cooldown"
            score_bias -= (8.0 + min(effective_loss_streak, 3) * 2.0) * max(scope_multiplier, 0.72)
            confidence_bias -= 0.05 * max(scope_multiplier, 0.72)
            note = (
                "Pair ini baru saja masuk fase sulit, jadi engine menahan agresivitas setup."
                if memory_scope == "pair"
                else f"Tone sektor {market_label} sedang dingin, jadi ranking ditahan sementara."
            )
            prime_penalty = memory_scope == "pair" and pair_loss_streak >= 4
        elif win_rate >= 60.0 and wins >= 2:
            state = "in_sync"
            score_bias += 5.0 * max(scope_multiplier, 0.72)
            confidence_bias += 0.03 * max(scope_multiplier, 0.72)
            note = (
                "Pair ini sedang sinkron dengan engine, jadi setup rapi mendapat sedikit prioritas."
                if memory_scope == "pair"
                else f"Sektor {market_label} sedang cukup sinkron, jadi setup bersih dapat sedikit prioritas."
            )
        elif win_rate >= 55.0 and wins >= 1:
            score_bias += 2.0 * max(scope_multiplier, 0.72)
            confidence_bias += 0.012 * max(scope_multiplier, 0.72)
            note = (
                "Pair ini mulai sinkron, jadi ada boost ringan."
                if memory_scope == "pair"
                else f"Belajar awal dari sektor {market_label} memberi boost ringan."
            )

    if open_same_context >= 2:
        state = "overheat" if state in {"neutral", "in_sync"} else state
        score_bias -= 4.0
        confidence_bias -= 0.02
        note = "Masih ada exposure aktif di pair yang sama, jadi ranking ditahan agar tidak overtrade."

    recent_events = sorted(
        [
            event for event in store.get("events", [])
            if event.get("symbol") == normalized_symbol
            and event.get("timeframe") == normalized_timeframe
            and event.get("style") == normalized_style
        ],
        key=lambda item: item.get("timestamp") or "",
        reverse=True,
    )[:8]

    recent_history = [
        {
            "title": event.get("title") or normalized_symbol,
            "status": event.get("status") or "open",
            "verdict": event.get("verdict") or "WAIT",
            "note": event.get("note") or "",
            "timestamp": event.get("timestamp"),
        }
        for event in recent_events
    ]

    return {
        "symbol": normalized_symbol,
        "timeframe": normalized_timeframe,
        "style": normalized_style,
        "market_type": normalized_market or None,
        "state": state,
        "score_bias": round(score_bias, 2),
        "confidence_bias": round(confidence_bias, 3),
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "scored_total": scored_total,
        "sample_size": len(sample),
        "pair_scored_total": len(exact_scored),
        "market_scored_total": len(market_scored),
        "memory_scope": memory_scope,
        "market_fallback_active": memory_scope in {"mixed", "market"},
        "loss_streak": loss_streak,
        "open_signals": open_same_context,
        "prime_penalty": prime_penalty,
        "note": note,
        "recent_history": recent_history,
    }


def _closed_signals(
    signals: list[dict[str, Any]],
    *,
    market_type: str | None = None,
    style: str | None = None,
) -> list[dict[str, Any]]:
    normalized_market = str(market_type or "").strip().lower()
    normalized_style = str(style or "").strip().lower()
    closed = []
    for signal in signals:
        status = str(signal.get("status") or "").strip().lower()
        if status == "open":
            continue
        if normalized_market and str(signal.get("market_type") or "").strip().lower() != normalized_market:
            continue
        if normalized_style and str(signal.get("style") or "").strip().lower() != normalized_style:
            continue
        closed.append(signal)
    return sorted(
        closed,
        key=lambda item: item.get("closed_at") or item.get("updated_at") or item.get("opened_at") or "",
        reverse=True,
    )


def _aggregate_bucket(signals: list[dict[str, Any]]) -> dict[str, Any]:
    closed = _closed_signals(signals)
    scored = _scored_closed_signals(closed)
    wins = sum(1 for signal in scored if str(signal.get("status") or "") in WIN_STATUSES)
    losses = sum(1 for signal in scored if str(signal.get("status") or "") in LOSS_STATUSES)
    win_rate = round((wins / len(scored)) * 100.0, 1) if scored else None
    outcomes_r = [_signal_outcome_r(signal) for signal in closed]
    gross_profit_r = round(sum(value for value in outcomes_r if value > 0), 2)
    gross_loss_r = round(sum(value for value in outcomes_r if value < 0), 2)
    net_r = round(sum(outcomes_r), 2)
    net_pct_values = [_signal_outcome_pct(signal) for signal in closed]
    net_pct_values = [value for value in net_pct_values if value is not None]
    gross_profit_pct = round(sum(value for value in net_pct_values if value > 0), 3) if net_pct_values else None
    gross_loss_pct = round(sum(value for value in net_pct_values if value < 0), 3) if net_pct_values else None
    net_pct = round(sum(net_pct_values), 3) if net_pct_values else None
    tracked_pairs = len({str(signal.get("symbol") or "").strip().upper() for signal in signals if str(signal.get("symbol") or "").strip()})
    open_signals = sum(1 for signal in signals if str(signal.get("status") or "").strip().lower() == "open")
    return {
        "tracked_pairs": tracked_pairs,
        "closed_signals": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "gross_profit_r": gross_profit_r,
        "gross_loss_r": gross_loss_r,
        "net_r": net_r,
        "gross_profit_pct": gross_profit_pct,
        "gross_loss_pct": gross_loss_pct,
        "net_pct": net_pct,
        "open_signals": open_signals,
    }


def _pair_breakdown(signals: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for signal in signals:
        symbol = str(signal.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        buckets.setdefault(symbol, []).append(signal)

    pairs = []
    for symbol, bucket in buckets.items():
        pairs.append(_pair_detail(symbol, bucket))

    pairs.sort(
        key=lambda item: (
            float(item.get("top_score") or item.get("last_score") or 0),
            float(item.get("closed_signals") or 0),
            float(item.get("net_r") or 0),
            float(item.get("win_rate") or 0),
        ),
        reverse=True,
    )
    return pairs[:limit]


def _open_breakdown(signals: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for signal in signals:
        symbol = str(signal.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if str(signal.get("status") or "").strip().lower() != "open":
            continue
        buckets.setdefault(symbol, []).append(signal)

    pairs = []
    for symbol, bucket in buckets.items():
        pairs.append(_pair_detail(symbol, bucket))

    pairs.sort(
        key=lambda item: (
            float(item.get("open_signals") or 0),
            float(item.get("top_score") or item.get("last_score") or 0),
            float(item.get("last_score") or 0),
        ),
        reverse=True,
    )
    return pairs[:limit]


def _performance_curve(signals: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for signal in _closed_signals(signals):
        parsed = _parse_dt(signal.get("closed_at") or signal.get("updated_at") or signal.get("opened_at"))
        if parsed is None:
            continue
        day = parsed.date().isoformat()
        bucket = buckets.setdefault(
            day,
            {
                "day": day,
                "wins": 0,
                "losses": 0,
                "closed": 0,
                "net_r": 0.0,
                "net_pct": 0.0,
            },
        )
        status = str(signal.get("status") or "").strip().lower()
        if status in WIN_STATUSES:
            bucket["wins"] += 1
        elif status in LOSS_STATUSES:
            bucket["losses"] += 1
        bucket["closed"] += 1
        bucket["net_r"] += _signal_outcome_r(signal)
        outcome_pct = _signal_outcome_pct(signal)
        if outcome_pct is not None:
            bucket["net_pct"] += outcome_pct

    ordered = sorted(buckets.values(), key=lambda item: item["day"], reverse=True)[:limit]
    ordered.reverse()
    for item in ordered:
        item["net_r"] = round(item["net_r"], 2)
        item["net_pct"] = round(item["net_pct"], 3)
    return ordered


def _performance_signal_curve(signals: list[dict[str, Any]], limit: int = 28) -> list[dict[str, Any]]:
    ordered = list(reversed(_closed_signals(signals)))
    if limit > 0:
        ordered = ordered[-limit:]

    points: list[dict[str, Any]] = []
    cumulative_r = 0.0
    cumulative_pct = 0.0
    for index, signal in enumerate(ordered, start=1):
        outcome_r = _signal_outcome_r(signal)
        outcome_pct = _signal_outcome_pct(signal)
        cumulative_r += outcome_r
        if outcome_pct is not None:
            cumulative_pct += outcome_pct
        points.append(
            {
                "seq": index,
                "symbol": str(signal.get("symbol") or "").strip().upper(),
                "status": str(signal.get("status") or "").strip().lower(),
                "timestamp": signal.get("closed_at") or signal.get("updated_at") or signal.get("opened_at"),
                "outcome_r": round(outcome_r, 2),
                "outcome_pct": round(outcome_pct, 3) if outcome_pct is not None else None,
                "net_r": round(cumulative_r, 2),
                "net_pct": round(cumulative_pct, 3),
            }
        )
    return points


def _journal_entry(signal: dict[str, Any]) -> dict[str, Any]:
    symbol = str(signal.get("symbol") or "").strip().upper()
    timeframe = str(signal.get("timeframe") or "").strip().lower()
    style = str(signal.get("style") or "").strip().lower()
    verdict = str(signal.get("verdict") or "WAIT").strip().upper()
    status = str(signal.get("status") or "open").strip().lower()
    setup_type = str(signal.get("setup_type") or "").strip().replace("_", " ")
    outcome_label = _status_label(status)
    primary_reason = _format_reason_list(signal.get("reasons"))
    warning_note = _format_reason_list(signal.get("warnings"))
    blocker_note = _format_reason_list(signal.get("blockers"))
    research_focus = str(signal.get("research_focus") or "").strip().replace("_", " ")
    listing_profile = str(signal.get("listing_profile") or "").strip().replace("_", " ")
    outcome_note = str(signal.get("outcome_note") or "").strip()
    review_parts = []

    if setup_type:
        review_parts.append(f"Setup {setup_type}.")
    if primary_reason:
        review_parts.append(f"Alasan awal: {primary_reason}.")
    if status in LOSS_STATUSES and warning_note:
        review_parts.append(f"Warning aktif: {warning_note}.")
    if status in {"mixed", "stale"} and blocker_note:
        review_parts.append(f"Filter aktif: {blocker_note}.")
    if research_focus:
        review_parts.append(f"Fokus riset: {research_focus}.")
    if listing_profile:
        review_parts.append(f"Profil market: {listing_profile}.")
    if outcome_note:
        review_parts.append(outcome_note)

    if not review_parts:
        review_parts.append("Belum ada catatan detail tambahan untuk signal ini.")

    return {
        "id": str(signal.get("id") or ""),
        "symbol": symbol,
        "timeframe": timeframe,
        "style": style,
        "verdict": verdict,
        "status": status,
        "status_label": outcome_label,
        "title": f"{symbol} {outcome_label}",
        "setup_type": setup_type or None,
        "review": " ".join(review_parts[:4]),
        "outcome_r": _signal_outcome_r(signal),
        "outcome_pct": _signal_outcome_pct(signal),
        "timestamp": signal.get("closed_at") or signal.get("updated_at") or signal.get("opened_at"),
    }


def _signal_detail(signal: dict[str, Any]) -> dict[str, Any]:
    confidence = _safe_number(signal.get("confidence"))
    return {
        "id": str(signal.get("id") or ""),
        "symbol": str(signal.get("symbol") or "").strip().upper(),
        "timeframe": str(signal.get("timeframe") or "").strip().lower(),
        "style": str(signal.get("style") or "").strip().lower(),
        "market_type": str(signal.get("market_type") or "").strip().lower() or None,
        "verdict": str(signal.get("verdict") or "").strip().upper() or None,
        "status": str(signal.get("status") or "").strip().lower() or None,
        "confidence": round(confidence * 100.0, 1) if confidence is not None else None,
        "score": _safe_number(signal.get("score")),
        "score_rank": signal.get("score_rank"),
        "source": str(signal.get("source") or "").strip() or None,
        "entry_mid": _round_price(signal.get("entry_mid")),
        "stop_loss": _round_price(signal.get("stop_loss")),
        "take_profit_1": _round_price(signal.get("take_profit_1")),
        "take_profit_2": _round_price(signal.get("take_profit_2")),
        "risk_reward": _round_price(signal.get("risk_reward")),
        "risk_percent": _safe_number(signal.get("risk_percent")),
        "price": _round_price(signal.get("price")),
        "last_price": _round_price(signal.get("last_price")),
        "mfe": _round_price(signal.get("mfe")),
        "mae": _round_price(signal.get("mae")),
        "setup_type": str(signal.get("setup_type") or "").strip() or None,
        "primary_thesis": str(signal.get("primary_thesis") or "").strip() or None,
        "edge_summary": str(signal.get("edge_summary") or "").strip() or None,
        "listing_profile": str(signal.get("listing_profile") or "").strip() or None,
        "fresh_listing_candidate": bool(signal.get("fresh_listing_candidate")),
        "reasons": deepcopy((signal.get("reasons") or [])[:3]),
        "warnings": deepcopy((signal.get("warnings") or [])[:3]),
        "blockers": deepcopy((signal.get("blockers") or [])[:3]),
        "opened_at": signal.get("opened_at"),
        "updated_at": signal.get("updated_at"),
        "closed_at": signal.get("closed_at"),
        "outcome_r": _signal_outcome_r(signal),
        "outcome_pct": _signal_outcome_pct(signal),
        "outcome_note": str(signal.get("outcome_note") or "").strip() or None,
    }


def _pair_detail(symbol: str, bucket: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate_bucket(bucket)
    ordered = sorted(
        bucket,
        key=lambda item: item.get("updated_at") or item.get("closed_at") or item.get("opened_at") or "",
        reverse=True,
    )
    latest = ordered[0] if ordered else {}
    open_signals = [signal for signal in ordered if str(signal.get("status") or "").strip().lower() == "open"]
    closed_signals = [signal for signal in ordered if str(signal.get("status") or "").strip().lower() != "open"]
    active_signal = open_signals[0] if open_signals else (latest if latest else None)
    score_values = [_safe_number(signal.get("score")) for signal in ordered]
    score_values = [value for value in score_values if value is not None]
    timeframes = list(dict.fromkeys(str(signal.get("timeframe") or "").strip().lower() for signal in ordered if str(signal.get("timeframe") or "").strip()))
    styles = list(dict.fromkeys(str(signal.get("style") or "").strip().lower() for signal in ordered if str(signal.get("style") or "").strip()))
    market_type = str((latest or {}).get("market_type") or "").strip().lower() or None
    return {
        "symbol": symbol,
        "market_type": market_type,
        "tracked_signals": len(bucket),
        "closed_signals": aggregate["closed_signals"],
        "open_signals": aggregate["open_signals"],
        "wins": aggregate["wins"],
        "losses": aggregate["losses"],
        "win_rate": aggregate["win_rate"],
        "gross_profit_r": aggregate["gross_profit_r"],
        "gross_loss_r": aggregate["gross_loss_r"],
        "net_r": aggregate["net_r"],
        "gross_profit_pct": aggregate["gross_profit_pct"],
        "gross_loss_pct": aggregate["gross_loss_pct"],
        "net_pct": aggregate["net_pct"],
        "last_status": latest.get("status") or "open",
        "last_updated_at": latest.get("updated_at") or latest.get("closed_at") or latest.get("opened_at"),
        "last_verdict": str(latest.get("verdict") or "").strip().upper() or None,
        "last_score": _safe_number(latest.get("score")),
        "top_score": round(max(score_values), 2) if score_values else None,
        "timeframes": timeframes[:4],
        "styles": styles[:3],
        "latest_signal": _signal_detail(latest) if latest else None,
        "active_signal": _signal_detail(active_signal) if active_signal else None,
        "open_signal_details": [_signal_detail(signal) for signal in open_signals[:4]],
        "recent_closed_details": [_signal_detail(signal) for signal in closed_signals[:4]],
        "curve": _performance_signal_curve(bucket, limit=14),
    }


def build_memory_dashboard(
    symbol: str,
    timeframe: str,
    style: str,
    market_type: str | None = None,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    normalized_market = str(market_type or "").strip().lower()
    context = learning_context(symbol, timeframe, style, normalized_market, limit=limit)
    with STORE_LOCK:
        store = _read_store()
    signals = list(store.get("signals", []))
    all_closed = _closed_signals(signals)
    normalized_symbol = str(symbol or "").strip().upper()
    market_signals = [
        signal for signal in signals if str(signal.get("market_type") or "").strip().lower() == normalized_market
    ] if normalized_market else []

    journal_signals = all_closed[:12]
    if normalized_market:
        market_closed = _closed_signals(market_signals)
    else:
        market_closed = []

    return {
        "pair_summary": {key: value for key, value in context.items() if key != "recent_history"},
        "pair_recent": deepcopy(context.get("recent_history", [])),
        "global_summary": _aggregate_bucket(signals),
        "market_summary": _aggregate_bucket(market_signals) if normalized_market else None,
        "pair_breakdown": _pair_breakdown(signals),
        "open_breakdown": _open_breakdown(signals),
        "selected_pair_detail": _pair_detail(
            normalized_symbol,
            [signal for signal in signals if str(signal.get("symbol") or "").strip().upper() == normalized_symbol],
        ) if normalized_symbol else None,
        "performance_curve": _performance_curve(signals),
        "performance_curve_signal": _performance_signal_curve(signals),
        "market_curve": _performance_curve(market_signals) if normalized_market else [],
        "market_curve_signal": _performance_signal_curve(market_signals) if normalized_market else [],
        "journal": [_journal_entry(signal) for signal in journal_signals],
        "focus_journal": [
            _journal_entry(signal)
            for signal in all_closed
            if _same_context(signal, str(symbol or "").strip().upper(), str(timeframe or "15m").strip().lower(), str(style or "intraday").strip().lower())
        ][:6],
    }


def register_signal(
    payload: dict[str, Any],
    result: dict[str, Any],
    *,
    source: str = "web",
) -> dict[str, Any]:
    summary = _extract_summary(result)
    plan = _extract_plan(result)
    symbol = str(summary.get("symbol") or payload.get("symbol") or "").strip().upper()
    timeframe = str(summary.get("timeframe") or payload.get("timeframe") or "15m").strip().lower()
    style = str(payload.get("style") or summary.get("style") or "intraday").strip().lower()
    market_type = str(summary.get("market_type") or payload.get("market_type") or "").strip().lower()
    verdict = str(summary.get("verdict") or "WAIT").strip().upper()
    confidence = _safe_number(summary.get("confidence")) or 0.0
    score = (
        _safe_number(summary.get("recommendation_score"))
        or _safe_number(summary.get("score"))
        or _safe_number(result.get("score"))
        or _safe_number(payload.get("_signal_score"))
    )
    price = _safe_number(payload.get("price")) or _safe_number(payload.get("close"))
    entry_mid = _entry_mid(plan, price)
    stop_loss = _safe_number(plan.get("stop_loss"))
    take_profit_1 = _safe_number(plan.get("take_profit_1"))
    take_profit_2 = _safe_number(plan.get("take_profit_2"))
    risk_reward = _safe_number(plan.get("risk_reward"))
    base = result.get("brain_output", result)
    risk_block = base.get("risk") if isinstance(base.get("risk"), dict) else {}
    risk_percent = _safe_number(risk_block.get("effective_risk_percent"))
    if risk_percent is None:
        risk_percent = _safe_number((payload.get("risk") or {}).get("max_risk_percent"))
    strategic = result.get("strategic_brief") if isinstance(result.get("strategic_brief"), dict) else {}
    live_context = payload.get("live_context") if isinstance(payload.get("live_context"), dict) else {}
    if not symbol:
        return learning_context(symbol, timeframe, style, market_type)

    with STORE_LOCK:
        store = _read_store()
        reconcile_market_memory(symbol, timeframe, style, payload)
        store = _read_store()

        if verdict in {"LONG", "SHORT"} and entry_mid is not None:
            fingerprint = _signal_fingerprint(symbol, timeframe, style, verdict, entry_mid, stop_loss, take_profit_1)
            existing = next(
                (
                    signal for signal in reversed(store.get("signals", []))
                    if signal.get("status") == "open" and signal.get("fingerprint") == fingerprint
                ),
                None,
            )

            if existing is None:
                signal_id = uuid4().hex
                signal = {
                    "id": signal_id,
                    "fingerprint": fingerprint,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "style": style,
                    "market_type": market_type or None,
                    "verdict": verdict,
                    "status": "open",
                    "confidence": round(confidence, 4),
                    "score": round(score, 2) if score is not None else None,
                    "score_rank": payload.get("_signal_rank"),
                    "price": _round_price(price),
                    "entry_mid": _round_price(entry_mid),
                    "stop_loss": _round_price(stop_loss),
                    "take_profit_1": _round_price(take_profit_1),
                    "take_profit_2": _round_price(take_profit_2),
                    "risk_reward": _round_price(risk_reward),
                    "risk_percent": risk_percent,
                    "setup_type": plan.get("setup_type"),
                    "reasons": deepcopy((base.get("reasons") or [])[:3]),
                    "warnings": deepcopy((base.get("warnings") or [])[:3]),
                    "blockers": deepcopy((base.get("blockers") or [])[:3]),
                    "primary_thesis": strategic.get("primary_thesis"),
                    "edge_summary": strategic.get("edge_summary"),
                    "listing_profile": live_context.get("listing_profile"),
                    "fresh_listing_candidate": bool(live_context.get("fresh_listing_candidate")),
                    "opened_at": _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                    "source": source,
                    "mfe": 0.0,
                    "mae": 0.0,
                }
                store.setdefault("signals", []).append(signal)
                _append_event(
                    store,
                    _build_event(
                        signal_id=signal_id,
                        event_type="opened",
                        symbol=symbol,
                        timeframe=timeframe,
                        style=style,
                        verdict=verdict,
                        status="open",
                        title=f"{symbol} {verdict} armed",
                        note="Signal baru masuk ke memori untuk dipantau hasilnya.",
                    ),
                )
            else:
                existing["confidence"] = round(confidence, 4)
                existing["score"] = round(score, 2) if score is not None else existing.get("score")
                existing["score_rank"] = payload.get("_signal_rank") or existing.get("score_rank")
                existing["price"] = _round_price(price)
                existing["entry_mid"] = _round_price(entry_mid)
                existing["stop_loss"] = _round_price(stop_loss)
                existing["take_profit_1"] = _round_price(take_profit_1)
                existing["take_profit_2"] = _round_price(take_profit_2)
                existing["risk_reward"] = _round_price(risk_reward)
                existing["risk_percent"] = risk_percent
                existing["setup_type"] = plan.get("setup_type")
                existing["reasons"] = deepcopy((base.get("reasons") or [])[:3])
                existing["warnings"] = deepcopy((base.get("warnings") or [])[:3])
                existing["blockers"] = deepcopy((base.get("blockers") or [])[:3])
                existing["primary_thesis"] = strategic.get("primary_thesis")
                existing["edge_summary"] = strategic.get("edge_summary")
                existing["listing_profile"] = live_context.get("listing_profile")
                existing["fresh_listing_candidate"] = bool(live_context.get("fresh_listing_candidate"))
                existing["updated_at"] = _utc_now_iso()
                _update_floats(existing, _resolve_market_state(payload))
        else:
            context_key = f"{symbol}|{timeframe}|{style}|{verdict}"
            recent_defense = next(
                (
                    event for event in reversed(store.get("events", []))
                    if event.get("event_type") == "context"
                    and f"{event.get('symbol')}|{event.get('timeframe')}|{event.get('style')}|{event.get('verdict')}" == context_key
                ),
                None,
            )
            recent_dt = _parse_dt(recent_defense.get("timestamp")) if recent_defense else None
            age_minutes = ((_utc_now() - recent_dt).total_seconds() / 60) if recent_dt else None
            if age_minutes is None or age_minutes >= 30:
                _append_event(
                    store,
                    _build_event(
                        signal_id="context",
                        event_type="context",
                        symbol=symbol,
                        timeframe=timeframe,
                        style=style,
                        verdict=verdict,
                        status="standby" if verdict == "WAIT" else "defense",
                        title=f"{symbol} {verdict.replace('_', ' ')}",
                        note="Engine memilih menunggu atau bertahan sambil mengumpulkan konfirmasi baru.",
                    ),
                )

        _write_store(store)

    return learning_context(symbol, timeframe, style, market_type)


def get_signal_history(
    symbol: str,
    timeframe: str,
    style: str,
    market_type: str | None = None,
) -> dict[str, Any]:
    context = learning_context(symbol, timeframe, style, market_type)
    dashboard = build_memory_dashboard(symbol, timeframe, style, market_type)
    return {
        "summary": {key: value for key, value in context.items() if key != "recent_history"},
        "recent": deepcopy(context.get("recent_history", [])),
        "dashboard": dashboard,
    }
