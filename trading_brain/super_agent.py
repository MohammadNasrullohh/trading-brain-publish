from __future__ import annotations

from typing import Any

from .engine import TradingBrain
from .models import MarketInput


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _level_pair(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    return values[0], values[-1]


def _active_plan(base: dict[str, Any]) -> dict[str, Any]:
    return base.get("plan", {}) or base.get("conditional_plan", {})


def _grade_setup(verdict: str, confidence: float, blockers: list[str], warnings: list[str]) -> str:
    score = (confidence * 100.0) - (len(blockers) * 18.0) - (len(warnings) * 3.5)
    if verdict in {"LONG", "SHORT"} and not blockers and score >= 85:
        return "A"
    if verdict in {"LONG", "SHORT"} and score >= 72:
        return "B"
    if verdict == "WAIT" and score >= 58:
        return "C"
    if verdict == "NO TRADE" and blockers:
        return "DEFENSIVE"
    return "D"


def _agent_state(verdict: str, blockers: list[str]) -> str:
    joined = " | ".join(blockers).lower()
    if "loss streak" in joined or "batas loss harian" in joined or "drawdown" in joined:
        return "RECOVER"
    if verdict in {"LONG", "SHORT"}:
        return "ENGAGE"
    if verdict == "WAIT":
        return "STALK"
    return "DEFEND"


def _execution_mode(state: str) -> str:
    mapping = {
        "ENGAGE": "active_execution",
        "STALK": "conditional_execution",
        "DEFEND": "capital_preservation",
        "RECOVER": "risk_shutdown",
    }
    return mapping[state]


def _primary_axis(verdict: str) -> str:
    if verdict == "LONG":
        return "bullish"
    if verdict == "SHORT":
        return "bearish"
    return "neutral"


def _risk_reward(base: dict[str, Any]) -> float:
    plan = _active_plan(base)
    return _safe_float(plan.get("risk_reward"))


def _readiness_score(base: dict[str, Any], market_input: MarketInput, state: str) -> int:
    summary = base["summary"]
    context = base["context"]
    risk = base["risk"]
    blockers = base.get("blockers", [])
    warnings = base.get("warnings", [])

    score = _safe_float(summary.get("confidence")) * 100.0
    score += min(12.0, _risk_reward(base) * 3.0)

    session_quality = context.get("session_quality")
    if session_quality == "high":
        score += 6.0
    elif session_quality == "low":
        score -= 6.0

    regime = context.get("regime")
    if regime == "trending" and summary["verdict"] in {"LONG", "SHORT"}:
        score += 4.0
    elif regime == "choppy":
        score -= 5.0

    if market_input.sentiment.headline_risk:
        score -= 12.0
    if market_input.sentiment.macro_risk:
        score -= 10.0

    score -= len(blockers) * 16.0
    score -= len(warnings) * 1.6
    score -= _safe_float(risk.get("account_heat")) * 18.0
    score -= max(0.0, _safe_float(risk.get("current_drawdown_percent")) - 2.0) * 2.5
    score -= max(0.0, _safe_float(risk.get("friction_bps")) - 10.0) * 0.6

    leverage = _safe_float(risk.get("leverage"))
    if leverage >= 10.0:
        score -= 8.0
    elif leverage >= 5.0:
        score -= 3.0

    if state == "RECOVER":
        score = min(score, 28.0)
    elif state == "DEFEND":
        score = min(score, 46.0)

    return int(round(_clamp(score, 5.0, 99.0)))


def _mission_posture(state: str, readiness_score: int, grade: str, verdict: str) -> str:
    if state == "RECOVER":
        return "LOCKDOWN"
    if state == "DEFEND":
        return "SHIELD"
    if state == "STALK":
        return "AMBUSH"
    if verdict in {"LONG", "SHORT"} and readiness_score >= 88 and grade == "A":
        return "ATTACK"
    if verdict in {"LONG", "SHORT"} and readiness_score >= 72:
        return "PRESS"
    return "PROBE"


def _operating_doctrine(posture: str) -> str:
    mapping = {
        "ATTACK": "Tekan edge kuat, tapi tetap disiplin pada invalidation dan tidak mengejar candle.",
        "PRESS": "Eksekusi terukur pada edge valid, lalu pertahankan kontrol risiko sepanjang trade.",
        "AMBUSH": "Tunggu lokasi terbaik, biarkan market datang ke trigger sebelum commit modal.",
        "SHIELD": "Utamakan perlindungan modal sambil tetap memonitor perubahan konteks.",
        "LOCKDOWN": "Hentikan agresi, pulihkan kondisi akun, dan reset kualitas keputusan.",
        "PROBE": "Masuk sangat selektif dengan ukuran kecil hanya bila bukti bertambah kuat.",
    }
    return mapping[posture]


def _dominant_playbook(base: dict[str, Any], market_input: MarketInput, state: str) -> str:
    summary = base["summary"]
    context = base["context"]
    verdict = summary["verdict"]
    regime = context.get("regime")
    market_type = summary["market_type"]
    style = context.get("style")

    if state == "RECOVER":
        return "account_recovery_lockdown"
    if verdict == "NO TRADE":
        return "capital_preservation_scan"
    if verdict == "WAIT":
        if regime == "ranging":
            return "range_observation_grid"
        return "conditional_stalk_retest"
    if verdict == "LONG":
        if market_type == "crypto" and regime == "trending":
            return "crypto_momentum_press"
        if market_type == "forex" and market_input.session in {"london", "new_york", "overlap"}:
            return "forex_session_continuation"
        if style == "swing":
            return "higher_timeframe_trend_hold"
        return "directional_long_continuation"
    if market_type == "crypto":
        return "crypto_rejection_breakdown"
    if market_type == "forex":
        return "forex_session_reversal"
    return "directional_short_continuation"


def _build_primary_thesis(base: dict[str, Any]) -> str:
    summary = base["summary"]
    context = base["context"]
    scores = base["scores"]
    return (
        f"Bias {summary['bias']} pada {summary['symbol']} {summary['timeframe']} didukung regime "
        f"{context['regime']}, structure {context['structure']}, dan skor "
        f"long/short {scores['long']}/{scores['short']}."
    )


def _build_counter_thesis(base: dict[str, Any]) -> str:
    warnings = base.get("warnings", [])
    blockers = base.get("blockers", [])
    if blockers:
        return "Counter-thesis utama berasal dari blocker: " + "; ".join(blockers[:3]) + "."
    if warnings:
        return "Counter-thesis utama berasal dari warning: " + "; ".join(warnings[:3]) + "."
    return "Counter-thesis lemah; belum ada faktor penyanggah yang dominan."


def _build_edge_summary(base: dict[str, Any]) -> str:
    rr = _risk_reward(base)
    if rr:
        return f"Edge utama ada pada asimetri risk-reward {rr}:1 dengan level invalidation yang jelas."
    return "Edge belum cukup tegas; agent harus lebih selektif dan menunggu struktur yang lebih bersih."


def _build_entry_triggers(base: dict[str, Any]) -> list[str]:
    summary = base["summary"]
    levels = base["levels"]
    plan = _active_plan(base)
    entry_zone = plan.get("entry_zone") or []
    low, high = _level_pair(entry_zone)
    verdict = summary["verdict"]
    triggers: list[str] = []

    if verdict == "LONG":
        if low is not None and high is not None:
            triggers.append(f"Harga bertahan di area entry {low} sampai {high} tanpa breakdown bersih.")
        if levels.get("nearest_support") is not None:
            triggers.append(f"Support {levels['nearest_support']} tetap dihormati oleh price action.")
        triggers.append("Momentum tidak melemah drastis dan candle follow-through tetap sehat.")
    elif verdict == "SHORT":
        if low is not None and high is not None:
            triggers.append(f"Harga gagal menembus area entry {low} sampai {high} ke atas secara bersih.")
        if levels.get("nearest_resistance") is not None:
            triggers.append(f"Resistance {levels['nearest_resistance']} tetap menahan pantulan harga.")
        triggers.append("Momentum turun tetap dominan dan breakdown tidak langsung diserap.")
    elif verdict == "WAIT":
        triggers.append("Masuk hanya jika salah satu sisi mendapat konfirmasi struktur dan volume yang lebih kuat.")
        if low is not None and high is not None:
            triggers.append(f"Pantau reaksi baru di sekitar area {low} sampai {high}.")
    else:
        triggers.append("Tidak ada trigger eksekusi sekarang; agent fokus ke observasi dan perlindungan modal.")

    return triggers


def _build_invalidation_checks(base: dict[str, Any]) -> list[str]:
    plan = _active_plan(base)
    levels = base["levels"]
    checks: list[str] = []

    if plan.get("stop_loss") is not None:
        checks.append(f"Stop loss teknikal berada di {plan['stop_loss']}.")
    if levels.get("nearest_support") is not None:
        checks.append(f"Kehilangan support {levels['nearest_support']} melemahkan skenario bullish.")
    if levels.get("nearest_resistance") is not None:
        checks.append(f"Rejection kuat dari resistance {levels['nearest_resistance']} melemahkan continuation naik.")
    if base.get("blockers"):
        checks.append("Jika blocker baru bertambah, agent otomatis menurunkan agresivitas.")
    return checks[:4]


def _build_execution_steps(base: dict[str, Any], state: str) -> list[str]:
    summary = base["summary"]
    risk = base["risk"]
    plan = _active_plan(base)
    verdict = summary["verdict"]

    if state == "RECOVER":
        return [
            "Jangan buka posisi baru sampai kondisi akun kembali stabil.",
            "Review blocker akun lebih dulu sebelum mengevaluasi chart lagi.",
            "Aktifkan mode observasi dan tunggu setup kelas A berikutnya.",
        ]

    if verdict == "LONG":
        return [
            "Masuk hanya di zona entry yang sudah ditentukan, jangan kejar candle memanjang.",
            f"Gunakan effective risk {risk.get('effective_risk_percent')}% dan size multiplier {risk.get('size_multiplier')}.",
            f"Tempatkan invalidation keras di {plan.get('stop_loss')}.",
            "Kurangi risiko jika muncul rejection tajam atau headline risk baru.",
        ]

    if verdict == "SHORT":
        return [
            "Eksekusi short hanya jika area entry tetap menahan harga.",
            f"Gunakan effective risk {risk.get('effective_risk_percent')}% dan size multiplier {risk.get('size_multiplier')}.",
            f"Pertahankan invalidation di {plan.get('stop_loss')}.",
            "Jangan tambah posisi jika breakdown tidak punya follow-through.",
        ]

    if verdict == "WAIT":
        return [
            "Belum ada eksekusi. Agent masuk mode stalking.",
            "Pantau perubahan structure, volume, dan spread sebelum commit modal.",
            "Prioritaskan alert di area entry kondisional daripada entry manual emosional.",
        ]

    return [
        "Mode defensif aktif. Tidak ada trade baru.",
        "Fokus pada capital preservation dan tunggu blocker hilang.",
        "Reset bias bila market context berubah total.",
    ]


def _build_take_profit_logic(base: dict[str, Any]) -> list[str]:
    plan = _active_plan(base)
    targets: list[str] = []
    if plan.get("take_profit_1") is not None:
        targets.append(f"Target pertama di {plan['take_profit_1']}.")
    if plan.get("take_profit_2") is not None:
        targets.append(f"Target kedua di {plan['take_profit_2']}.")
    rr = plan.get("risk_reward")
    if rr is not None:
        targets.append(f"Asimetri setup saat ini sekitar {rr}:1.")
    if not targets:
        targets.append("Belum ada target aktif karena agent belum menyalakan mode eksekusi.")
    return targets


def _build_skip_conditions(base: dict[str, Any]) -> list[str]:
    warnings = base.get("warnings", [])
    blockers = base.get("blockers", [])
    skip = blockers[:4]
    if len(skip) < 4:
        skip.extend(warnings[: max(0, 4 - len(skip))])
    if not skip:
        skip.append("Skip bila structure dan momentum tidak lagi searah.")
    return skip


def _build_watch_items(base: dict[str, Any]) -> list[str]:
    items = [
        f"Pantau bias {base['summary']['bias']} terhadap perubahan structure.",
        f"Pantau quality session {base['context'].get('session_quality')}.",
    ]
    items.extend(base.get("warnings", [])[:3])
    return items[:5]


def _build_escalation_rules(base: dict[str, Any], state: str) -> list[str]:
    risk = base["risk"]
    rules = []
    if risk.get("account_heat") is not None and risk["account_heat"] >= 0.75:
        rules.append("Jika account heat tetap tinggi, agent mempertahankan mode defensif.")
    if risk.get("leverage") is not None and risk["leverage"] >= 10:
        rules.append("Leverage tinggi memaksa eksekusi limit-only dan ukuran posisi tetap kecil.")
    if state == "RECOVER":
        rules.append("Recovery mode aktif sampai loss streak dan blocker akun turun.")
    if not rules:
        rules.append("Jika warning berubah menjadi blocker, turunkan mode dari ENGAGE ke STALK atau DEFEND.")
    return rules


def _build_if_then_map(base: dict[str, Any], state: str) -> list[dict[str, str]]:
    summary = base["summary"]
    plan = _active_plan(base)
    verdict = summary["verdict"]
    mapping: list[dict[str, str]] = []

    if verdict == "LONG":
        mapping.append({"if": "entry zone tetap bertahan", "then": "aktifkan eksekusi long terukur"})
        mapping.append({"if": f"stop {plan.get('stop_loss')} tersentuh", "then": "keluar penuh tanpa negosiasi"})
    elif verdict == "SHORT":
        mapping.append({"if": "rejection di resistance tetap valid", "then": "aktifkan eksekusi short terukur"})
        mapping.append({"if": f"stop {plan.get('stop_loss')} tersentuh", "then": "batalkan ide short"})
    elif state == "RECOVER":
        mapping.append({"if": "akun masih di recovery mode", "then": "tidak ada posisi baru"})
        mapping.append({"if": "blocker akun hilang", "then": "turun ke mode STALK dulu, bukan langsung ENGAGE"})
    else:
        mapping.append({"if": "konfirmasi baru muncul", "then": "re-score setup dan evaluasi ulang"})
        mapping.append({"if": "warning bertambah", "then": "pertahankan no-trade"})

    return mapping


def _capital_allocation(base: dict[str, Any], state: str, readiness_score: int) -> dict[str, Any]:
    risk = base["risk"]
    max_risk = _safe_float(risk.get("max_risk_percent"), 0.0)
    effective_risk = _safe_float(risk.get("effective_risk_percent"), 0.0)
    denominator = max(max_risk, effective_risk, 0.01)
    utilization = 0.0 if state == "RECOVER" else _clamp(effective_risk / denominator, 0.0, 1.0)

    if state == "RECOVER":
        band = "flat"
    elif readiness_score >= 85:
        band = "full_risk"
    elif readiness_score >= 70:
        band = "reduced_risk"
    elif readiness_score >= 55:
        band = "probe_risk"
    else:
        band = "minimal_risk"

    if state == "RECOVER":
        sizing_style = "no_new_positions"
    elif utilization >= 0.95:
        sizing_style = "standard_deployment"
    elif utilization >= 0.6:
        sizing_style = "scaled_in"
    else:
        sizing_style = "pilot_position"

    return {
        "capital_mode": band,
        "risk_budget_utilization": round(utilization, 2),
        "sizing_style": sizing_style,
        "effective_risk_percent": risk.get("effective_risk_percent"),
        "max_risk_percent": risk.get("max_risk_percent"),
    }


def _priority_stack(base: dict[str, Any], state: str) -> list[str]:
    verdict = base["summary"]["verdict"]
    levels = base["levels"]

    if state == "RECOVER":
        return [
            "Pulihkan disiplin akun sebelum membuka risk baru.",
            "Kurangi exposure sampai blocker akun hilang.",
            "Kembalikan proses review setup ke standar A-grade.",
        ]

    items = [
        "Jaga kualitas lokasi entry lebih penting daripada frekuensi trade.",
        "Pastikan invalidation tetap jelas sebelum eksekusi.",
    ]
    if verdict in {"LONG", "SHORT"}:
        items.append("Pertahankan reward-to-risk lebih besar dari noise market.")
    if levels.get("nearest_support") is not None or levels.get("nearest_resistance") is not None:
        items.append("Pantau level terdekat karena respons harga di sana menentukan agresivitas berikutnya.")
    return items[:4]


def _desk_consensus(base: dict[str, Any], market_input: MarketInput, state: str, readiness_score: int) -> dict[str, Any]:
    summary = base["summary"]
    context = base["context"]
    risk = base["risk"]
    scores = base["scores"]
    blockers = base.get("blockers", [])

    sentiment_score = market_input.sentiment.score
    macro_score = 72.0 + (sentiment_score * 35.0)
    if market_input.sentiment.headline_risk:
        macro_score -= 16.0
    if market_input.sentiment.macro_risk:
        macro_score -= 12.0
    macro_score = _clamp(macro_score, 10.0, 95.0)
    macro_stance = "tailwind" if macro_score >= 68.0 else "neutral" if macro_score >= 45.0 else "headwind"

    directional_gap = abs(_safe_float(scores.get("long")) - _safe_float(scores.get("short")))
    structure_score = 58.0 + min(22.0, directional_gap * 4.5)
    if summary["bias"] == context.get("higher_timeframe_bias") and summary["bias"] != "netral":
        structure_score += 8.0
    structure_score = _clamp(structure_score, 20.0, 96.0)
    structure_stance = "aligned" if structure_score >= 74.0 else "mixed" if structure_score >= 52.0 else "fragile"

    friction = _safe_float(risk.get("friction_bps"))
    execution_score = readiness_score - max(0.0, friction - 8.0) * 0.8
    if not _active_plan(base):
        execution_score -= 10.0
    execution_score = _clamp(execution_score, 10.0, 95.0)
    if summary["verdict"] in {"LONG", "SHORT"} and execution_score >= 72.0:
        execution_stance = "green_light"
    elif summary["verdict"] == "WAIT":
        execution_stance = "conditional"
    else:
        execution_stance = "hold"

    risk_score = 92.0 - (len(blockers) * 24.0)
    risk_score -= _safe_float(risk.get("account_heat")) * 25.0
    risk_score -= max(0.0, _safe_float(risk.get("current_drawdown_percent")) - 2.0) * 4.0
    risk_score -= max(0.0, _safe_float(risk.get("leverage")) - 5.0) * 1.8
    risk_score = _clamp(risk_score, 1.0, 94.0)
    if state == "RECOVER" or risk_score < 32.0:
        risk_stance = "blocked"
    elif risk_score < 60.0:
        risk_stance = "reduced"
    else:
        risk_stance = "approved"

    structure_axis = summary["bias"] if summary["bias"] != "netral" else _primary_axis(summary["verdict"])

    return {
        "macro_desk": {
            "stance": macro_stance,
            "confidence": int(round(macro_score)),
            "note": "Sentimen, headline risk, dan macro risk dipakai untuk menilai angin pendorong pasar.",
        },
        "structure_desk": {
            "stance": structure_stance,
            "confidence": int(round(structure_score)),
            "note": (
                f"Structure {context.get('structure')} dan higher timeframe {context.get('higher_timeframe_bias')} "
                f"mengarahkan bias {structure_axis}."
            ),
        },
        "execution_desk": {
            "stance": execution_stance,
            "confidence": int(round(execution_score)),
            "note": (
                f"Session {context.get('session_quality')} dan friction {risk.get('friction_bps')} bps "
                "menentukan kualitas timing eksekusi."
            ),
        },
        "risk_desk": {
            "stance": risk_stance,
            "confidence": int(round(risk_score)),
            "note": (
                f"Drawdown {risk.get('current_drawdown_percent')}%, leverage {risk.get('leverage')}x, "
                f"dan blocker aktif {len(blockers)} menjadi veto terakhir."
            ),
        },
    }


def _scenario_map(base: dict[str, Any], state: str, posture: str) -> dict[str, Any]:
    summary = base["summary"]
    plan = _active_plan(base)
    verdict = summary["verdict"]
    entry_zone = plan.get("entry_zone") or []
    low, high = _level_pair(entry_zone)
    entry_text = f"{low} sampai {high}" if low is not None and high is not None else "zona teknikal terdekat"
    stop = plan.get("stop_loss")

    if state == "RECOVER":
        return {
            "primary_case": {
                "name": "recover_first",
                "trigger": "blocker akun belum hilang",
                "response": "tetap flat dan fokus pulihkan stabilitas keputusan",
            },
            "contingency_case": {
                "name": "stabilizing",
                "trigger": "drawdown dan loss streak turun",
                "response": "naik ke mode STALK, belum langsung eksekusi penuh",
            },
            "abort_case": {
                "name": "re_damage",
                "trigger": "emosi trading kembali naik atau akun makin panas",
                "response": "hentikan trading dan review proses sebelum lanjut",
            },
        }

    if verdict == "LONG":
        return {
            "primary_case": {
                "name": "trend_continuation",
                "trigger": f"harga mempertahankan area {entry_text}",
                "response": "eksekusi long bertahap sesuai risk budget",
            },
            "contingency_case": {
                "name": "shallow_pullback",
                "trigger": "follow-through melambat tapi structure belum rusak",
                "response": "kurangi agresivitas dan tunggu reclaim konfirmasi",
            },
            "abort_case": {
                "name": "bull_failure",
                "trigger": f"stop {stop} tersentuh atau blocker baru muncul",
                "response": f"keluar penuh dan turun posture dari {posture} ke SHIELD",
            },
        }

    if verdict == "SHORT":
        return {
            "primary_case": {
                "name": "breakdown_follow_through",
                "trigger": f"harga gagal merebut kembali area {entry_text}",
                "response": "eksekusi short disiplin dengan invalidation jelas",
            },
            "contingency_case": {
                "name": "weak_breakdown",
                "trigger": "breakdown terjadi tapi langsung diserap",
                "response": "turunkan size atau batal masuk sampai ada konfirmasi ulang",
            },
            "abort_case": {
                "name": "short_squeeze",
                "trigger": f"stop {stop} tersentuh atau reclaim resistance terlalu bersih",
                "response": "batalkan ide short dan pindah ke posture AMBUSH",
            },
        }

    if verdict == "WAIT":
        return {
            "primary_case": {
                "name": "information_gathering",
                "trigger": "market belum memberi kualitas konfirmasi yang cukup",
                "response": "tetap stalking dan biarkan alert bekerja",
            },
            "contingency_case": {
                "name": "clean_confirmation",
                "trigger": "structure, volume, dan timing session tiba-tiba sinkron",
                "response": "re-score setup dan aktifkan execution brief baru",
            },
            "abort_case": {
                "name": "quality_decay",
                "trigger": "noise, spread, atau blocker bertambah",
                "response": "turun ke no-trade penuh",
            },
        }

    return {
        "primary_case": {
            "name": "capital_preservation",
            "trigger": "tidak ada edge yang layak",
            "response": "pertahankan modal dan lanjutkan scan market",
        },
        "contingency_case": {
            "name": "fresh_signal",
            "trigger": "konteks berubah dan level baru terbentuk",
            "response": "evaluasi ulang dari awal",
        },
        "abort_case": {
            "name": "discipline_break",
            "trigger": "ada dorongan entry emosional tanpa setup",
            "response": "jangan ambil trade",
        },
    }


class SuperTradingAgent:
    def __init__(self, brain: TradingBrain | None = None) -> None:
        self.brain = brain or TradingBrain()

    def analyze(self, market_input: MarketInput) -> dict[str, Any]:
        base = self.brain.analyze(market_input)
        summary = base["summary"]
        adaptive = base.get("adaptive", {})
        warnings = base.get("warnings", [])
        blockers = base.get("blockers", [])
        confidence = float(summary.get("confidence", 0.0))
        grade = _grade_setup(summary["verdict"], confidence, blockers, warnings)
        state = _agent_state(summary["verdict"], blockers)
        readiness_score = _readiness_score(base, market_input, state)
        posture = _mission_posture(state, readiness_score, grade, summary["verdict"])
        playbook = _dominant_playbook(base, market_input, state)

        result = {
            "meta": {
                "agent": "SuperTradingAgent",
                "version": "2.0.0",
                "base_brain": base["meta"]["brain"],
                "base_neuron_count": base["meta"]["neuron_count"],
                "operating_mode": _execution_mode(state),
            },
            "summary": {
                "symbol": summary["symbol"],
                "timeframe": summary["timeframe"],
                "market_type": summary["market_type"],
                "verdict": summary["verdict"],
                "agent_state": state,
                "setup_grade": grade,
                "confidence": summary["confidence"],
                "readiness_score": readiness_score,
                "mission_posture": posture,
                "dominant_playbook": playbook,
                "adaptive_mode": adaptive.get("adaptation_mode"),
                "growth_cycle": adaptive.get("growth_cycle"),
                "adaptive_maturity": adaptive.get("maturity"),
            },
            "mission_control": {
                "mission_posture": posture,
                "dominant_playbook": playbook,
                "adaptive_mode": adaptive.get("adaptation_mode"),
                "growth_cycle": adaptive.get("growth_cycle"),
                "operating_doctrine": _operating_doctrine(posture),
                "capital_allocation": _capital_allocation(base, state, readiness_score),
                "priority_stack": _priority_stack(base, state),
                "constraint_stack": blockers[:4] or warnings[:4],
            },
            "strategic_brief": {
                "primary_thesis": _build_primary_thesis(base),
                "counter_thesis": _build_counter_thesis(base),
                "edge_summary": _build_edge_summary(base),
            },
            "desk_consensus": _desk_consensus(base, market_input, state, readiness_score),
            "scenario_map": _scenario_map(base, state, posture),
            "action_plan": {
                "primary_action": summary["verdict"],
                "execution_mode": _execution_mode(state),
                "entry_triggers": _build_entry_triggers(base),
                "invalidation_checks": _build_invalidation_checks(base),
                "execution_steps": _build_execution_steps(base, state),
                "take_profit_logic": _build_take_profit_logic(base),
                "skip_conditions": _build_skip_conditions(base),
                "if_then_map": _build_if_then_map(base, state),
            },
            "monitoring": {
                "watch_levels": base["levels"],
                "watch_items": _build_watch_items(base),
                "escalation_rules": _build_escalation_rules(base, state),
                "next_review": "review ulang saat price menyentuh entry zone, invalidation, atau target utama",
            },
            "risk_protocol": {
                "risk_per_trade_percent": base["risk"].get("max_risk_percent"),
                "effective_risk_percent": base["risk"].get("effective_risk_percent"),
                "size_multiplier": base["risk"].get("size_multiplier"),
                "leverage": base["risk"].get("leverage"),
                "account_heat": base["risk"].get("account_heat"),
                "drawdown_percent": base["risk"].get("current_drawdown_percent"),
                "loss_streak": base["risk"].get("loss_streak"),
                "hard_stop_conditions": blockers[:5],
            },
            "brain_output": base,
        }
        return result

    def analyze_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        market_input = MarketInput.from_dict(payload)
        return self.analyze(market_input)
