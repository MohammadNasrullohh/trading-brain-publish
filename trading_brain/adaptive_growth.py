from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .signal_memory import learning_context as signal_learning_context
from .storage_db import DB_FILENAME, default_data_dir, read_document, write_document
from .utils import clamp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = default_data_dir(ROOT)
DATA_PATH = DATA_DIR / "adaptive_growth.json"
STORE_LOCK = threading.RLock()

WIN_STATUSES = {"win", "soft_win"}
LOSS_STATUSES = {"loss", "soft_loss"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _db_path() -> Path:
    return DATA_DIR / DB_FILENAME


def _legacy_paths() -> tuple[Path, ...]:
    paths = [DATA_PATH]
    fallback = ROOT / "logs" / "adaptive_growth.json"
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
    }


def _read_store() -> dict[str, Any]:
    _ensure_data_dir()
    return read_document(
        db_path=_db_path(),
        module="adaptive_growth",
        default_factory=_default_store,
        legacy_paths=_legacy_paths(),
        legacy_db_paths=_legacy_db_paths(),
    )


def _write_store(store: dict[str, Any]) -> None:
    _ensure_data_dir()
    profiles = store.get("profiles", {})
    if isinstance(profiles, dict) and len(profiles) > 240:
        ordered_items = sorted(
            profiles.items(),
            key=lambda item: str(item[1].get("updated_at") or ""),
            reverse=True,
        )[:240]
        profiles = dict(reversed(ordered_items))
    write_document(
        db_path=_db_path(),
        module="adaptive_growth",
        payload={"version": 1, "profiles": profiles},
    )


def _normalize_text(value: Any, default: str = "") -> str:
    raw = str(value or "").strip().lower()
    return raw or default


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _context_key(symbol: str, timeframe: str, style: str, market_type: str) -> str:
    return f"{symbol}|{timeframe}|{style}|{market_type}"


def _stage_biases(growth_cycle: str) -> dict[str, float]:
    mapping = {
        "bootstrap": {
            "core": 0.08,
            "market": 0.10,
            "quality": 0.10,
            "adaptive": 0.18,
            "pro": 0.04,
            "setup": 0.04,
            "risk": 0.06,
            "output": 0.04,
        },
        "explore": {
            "core": 0.05,
            "market": 0.09,
            "signal": 0.04,
            "quality": 0.08,
            "adaptive": 0.18,
            "pro": 0.07,
            "setup": 0.07,
            "risk": 0.05,
            "output": 0.04,
        },
        "stabilize": {
            "market": 0.08,
            "signal": 0.06,
            "quality": 0.06,
            "adaptive": 0.16,
            "pro": 0.10,
            "setup": 0.10,
            "risk": 0.06,
            "output": 0.05,
        },
        "compound": {
            "market": 0.06,
            "signal": 0.08,
            "quality": 0.04,
            "adaptive": 0.14,
            "pro": 0.12,
            "setup": 0.15,
            "plan_micro": 0.10,
            "risk": 0.04,
            "output": 0.08,
        },
        "protect": {
            "market": 0.04,
            "quality": 0.10,
            "adaptive": 0.18,
            "risk_micro": 0.12,
            "risk": 0.16,
            "output": 0.10,
        },
    }
    return mapping.get(growth_cycle, mapping["bootstrap"])


def _focus_titles(growth_cycle: str) -> list[str]:
    mapping = {
        "bootstrap": ["Adaptive Bootstrap", "Exploration Drive", "Market Memory Bridge"],
        "explore": ["Exploration Drive", "Market Memory Bridge", "Regime Mutation"],
        "stabilize": ["Pair Memory Resonance", "Regime Mutation", "Confidence Memory"],
        "compound": ["Compounding Drive", "Pair Memory Resonance", "Opportunity Compression"],
        "protect": ["Survival Instinct", "Anti Overfit Filter", "Confidence Memory"],
    }
    return mapping.get(growth_cycle, mapping["bootstrap"])


def _growth_cycle(context: dict[str, Any]) -> str:
    state = _normalize_text(context.get("state"), "warming")
    memory_scope = _normalize_text(context.get("memory_scope"), "warming")
    scored_total = int(context.get("scored_total") or 0)
    pair_total = int(context.get("pair_scored_total") or 0)
    win_rate = context.get("win_rate")
    loss_streak = int(context.get("loss_streak") or 0)
    prime_penalty = bool(context.get("prime_penalty"))

    if scored_total <= 0:
        return "bootstrap"
    if state in {"cooldown", "overheat"} or prime_penalty or loss_streak >= 2:
        return "protect"
    if state == "in_sync" and win_rate is not None and float(win_rate) >= 63.0 and pair_total >= 4:
        return "compound"
    if state in {"neutral", "in_sync"} and pair_total >= 2:
        return "stabilize"
    if memory_scope in {"mixed", "market"} or scored_total < 4:
        return "explore"
    return "stabilize"


def _adaptation_mode(growth_cycle: str) -> str:
    mapping = {
        "bootstrap": "calibrating",
        "explore": "expanding",
        "stabilize": "stabilizing",
        "compound": "compounding",
        "protect": "capital_protection",
    }
    return mapping.get(growth_cycle, "calibrating")


def _memory_weights(memory_scope: str, pair_total: int, market_total: int) -> tuple[float, float]:
    if memory_scope == "pair":
        return 1.0, 0.0
    if memory_scope == "mixed":
        return 0.62 if pair_total else 0.28, 0.38 if pair_total else 0.72
    if memory_scope == "market":
        return 0.14 if pair_total else 0.0, 0.86 if market_total else 0.25
    return 0.0, 0.15 if market_total else 0.0


def _maturity(pair_total: int, market_total: int, memory_scope: str) -> int:
    pair_score = min(pair_total * 14.0, 70.0)
    market_score = min(market_total * (4.5 if memory_scope == "market" else 3.0), 30.0)
    return int(round(clamp(pair_score + market_score, 0.0, 100.0)))


def _directional_memory(recent_history: list[dict[str, Any]]) -> dict[str, float]:
    long_score = 0.0
    short_score = 0.0

    for item in recent_history:
        verdict = str(item.get("verdict") or "").strip().upper()
        status = str(item.get("status") or "").strip().lower()
        if verdict not in {"LONG", "SHORT"}:
            continue
        if status in WIN_STATUSES:
            if verdict == "LONG":
                long_score += 1.0
            else:
                short_score += 1.0
        elif status in LOSS_STATUSES:
            if verdict == "LONG":
                long_score -= 0.7
            else:
                short_score -= 0.7

    total = abs(long_score) + abs(short_score)
    if total <= 0:
        return {
            "directional_tilt": 0.0,
            "long_memory": 0.0,
            "short_memory": 0.0,
        }

    long_memory = clamp((long_score / total), -1.0, 1.0)
    short_memory = clamp((short_score / total), -1.0, 1.0)
    directional_tilt = clamp(long_memory - short_memory, -1.0, 1.0)
    return {
        "directional_tilt": round(directional_tilt, 3),
        "long_memory": round(long_memory, 3),
        "short_memory": round(short_memory, 3),
    }


def _cycle_note(growth_cycle: str, context: dict[str, Any], market_type: str) -> str:
    memory_scope = _normalize_text(context.get("memory_scope"), "warming")
    if growth_cycle == "bootstrap":
        return "Belum ada histori tertutup yang cukup, jadi sistem masih membangun baseline pair ini."
    if growth_cycle == "explore":
        if memory_scope in {"market", "mixed"}:
            return f"Histori pair masih tipis, jadi sistem meminjam konteks sektor {market_type or 'market'} untuk eksplorasi awal."
        return "Sistem mulai membuka jalur eksplorasi ringan sambil mengumpulkan respon baru dari pair ini."
    if growth_cycle == "stabilize":
        return "Histori pair mulai konsisten, jadi sistem menstabilkan bobot dan conviction secara bertahap."
    if growth_cycle == "compound":
        return "Pair sedang cukup sinkron dengan histori sebelumnya, jadi setup bersih mendapat prioritas sedikit lebih tinggi."
    return "Sistem masuk mode proteksi karena ritme pair/market belum aman untuk agresivitas normal."


def derive_growth_profile(
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

    scored_total = int(context.get("scored_total") or 0)
    pair_total = int(context.get("pair_scored_total") or 0)
    market_total = int(context.get("market_scored_total") or 0)
    wins = int(context.get("wins") or 0)
    losses = int(context.get("losses") or 0)
    loss_streak = int(context.get("loss_streak") or 0)
    open_signals = int(context.get("open_signals") or 0)
    win_rate = float(context.get("win_rate") or 0.0) if context.get("win_rate") is not None else None
    memory_scope = _normalize_text(context.get("memory_scope"), "warming")
    memory_state = _normalize_text(context.get("state"), "warming")
    pair_weight, market_weight = _memory_weights(memory_scope, pair_total, market_total)
    maturity = _maturity(pair_total, market_total, memory_scope)
    growth_cycle = _growth_cycle(context)
    adaptation_mode = _adaptation_mode(growth_cycle)
    directional_memory = _directional_memory(list(context.get("recent_history") or []))

    exploration_bias = {
        "bootstrap": 0.16,
        "explore": 0.14,
        "stabilize": 0.06,
        "compound": 0.02,
        "protect": 0.0,
    }[growth_cycle]
    compounding_bias = {
        "bootstrap": 0.0,
        "explore": 0.03,
        "stabilize": 0.08,
        "compound": 0.18,
        "protect": 0.0,
    }[growth_cycle]
    protection_bias = {
        "bootstrap": 0.03,
        "explore": 0.05,
        "stabilize": 0.06,
        "compound": 0.04,
        "protect": 0.22,
    }[growth_cycle]

    base_score_shift = float(context.get("score_bias") or 0.0) / 10.0
    directional_shift = directional_memory["directional_tilt"] * (0.25 + (pair_weight * 0.2))
    score_shift = round(clamp(base_score_shift + directional_shift, -1.2, 0.9), 3)
    confidence_shift = round(
        clamp(float(context.get("confidence_bias") or 0.0) + directional_memory["directional_tilt"] * 0.018, -0.12, 0.09),
        3,
    )
    aggression_bias = round(
        clamp((compounding_bias * 1.2) + (exploration_bias * 0.55) - (protection_bias * 0.95), -0.22, 0.34),
        3,
    )

    rr_floor = {
        "bootstrap": 1.55,
        "explore": 1.45,
        "stabilize": 1.5,
        "compound": 1.38,
        "protect": 1.7,
    }[growth_cycle]
    score_floor = {
        "bootstrap": 4.85,
        "explore": 4.7,
        "stabilize": 4.95,
        "compound": 4.75,
        "protect": 5.25,
    }[growth_cycle]
    score_gap_floor = {
        "bootstrap": 1.8,
        "explore": 1.7,
        "stabilize": 1.85,
        "compound": 1.7,
        "protect": 2.1,
    }[growth_cycle]

    stage_biases = _stage_biases(growth_cycle)
    focus_titles = _focus_titles(growth_cycle)
    evolution_score = round(
        clamp(
            (maturity * 0.56)
            + (wins * 4.5)
            - (losses * 2.5)
            + (pair_weight * 12.0)
            + (market_weight * 6.0)
            + (max(score_shift, 0.0) * 18.0)
            - (protection_bias * 18.0),
            0.0,
            100.0,
        ),
        1,
    )

    signature = {
        "memory_state": memory_state,
        "memory_scope": memory_scope,
        "scored_total": scored_total,
        "pair_total": pair_total,
        "market_total": market_total,
        "wins": wins,
        "losses": losses,
        "loss_streak": loss_streak,
        "growth_cycle": growth_cycle,
        "open_signals": open_signals,
    }

    profile = {
        "symbol": normalized_symbol,
        "timeframe": normalized_timeframe,
        "style": normalized_style,
        "market_type": normalized_market,
        "growth_cycle": growth_cycle,
        "adaptation_mode": adaptation_mode,
        "maturity": maturity,
        "evolution_score": evolution_score,
        "memory_state": memory_state,
        "memory_scope": memory_scope,
        "pair_weight": round(pair_weight, 3),
        "market_weight": round(market_weight, 3),
        "pair_sample": pair_total,
        "market_sample": market_total,
        "scored_total": scored_total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "loss_streak": loss_streak,
        "open_signals": open_signals,
        "market_fallback_active": bool(context.get("market_fallback_active")),
        "exploration_bias": round(exploration_bias, 3),
        "compounding_bias": round(compounding_bias, 3),
        "protection_bias": round(protection_bias, 3),
        "aggression_bias": aggression_bias,
        "score_shift": score_shift,
        "confidence_shift": confidence_shift,
        "directional_tilt": directional_memory["directional_tilt"],
        "long_memory": directional_memory["long_memory"],
        "short_memory": directional_memory["short_memory"],
        "rr_floor": rr_floor,
        "score_floor": score_floor,
        "score_gap_floor": score_gap_floor,
        "stage_biases": stage_biases,
        "focus_titles": focus_titles,
        "note": _cycle_note(growth_cycle, context, normalized_market),
        "signature": signature,
        "updated_at": _utc_now_iso(),
    }

    if not normalized_symbol:
        return profile

    with STORE_LOCK:
        store = _read_store()
        profiles = store.setdefault("profiles", {})
        key = _context_key(normalized_symbol, normalized_timeframe, normalized_style, normalized_market)
        previous = profiles.get(key, {})
        previous_signature = previous.get("signature")
        previous_cycle = previous.get("growth_cycle")
        cycles = int(previous.get("cycles") or 0)
        transitions = int(previous.get("transitions") or 0)

        if previous_signature != signature:
            cycles += 1
        if previous_cycle and previous_cycle != growth_cycle:
            transitions += 1

        profile["cycles"] = cycles
        profile["transitions"] = transitions
        profiles[key] = profile
        _write_store(store)

    return profile


def load_growth_profile(
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
    key = _context_key(normalized_symbol, normalized_timeframe, normalized_style, normalized_market)
    with STORE_LOCK:
        store = _read_store()
        profile = store.get("profiles", {}).get(key)
    return dict(profile) if isinstance(profile, dict) else None
