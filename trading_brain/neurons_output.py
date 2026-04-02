from __future__ import annotations

from .models import BrainState
from .neuron_base import Neuron
from .utils import clamp, round_price


class OutputFormatterNeuron(Neuron):
    name = "neuron21_output_formatter"

    def run(self, state: BrainState) -> None:
        state.features["output_sections"] = [
            "meta",
            "summary",
            "context",
            "levels",
            "scores",
            "adaptive",
            "training",
            "plan",
            "reasons",
            "warnings",
            "blockers",
        ]


class VerdictGateNeuron(Neuron):
    name = "neuron22_verdict_gate"

    def run(self, state: BrainState) -> None:
        adaptive = state.features.get("adaptive_summary") or state.features.get("adaptive_profile") or {}
        long_rr = state.candidate_long.risk_reward or 0.0
        short_rr = state.candidate_short.risk_reward or 0.0
        rr_floor = float(state.features.get("adaptive_rr_floor") or adaptive.get("rr_floor") or 1.5)
        score_floor = float(state.features.get("adaptive_score_floor") or adaptive.get("score_floor") or 5.0)
        score_gap_floor = float(state.features.get("adaptive_score_gap") or adaptive.get("score_gap_floor") or 2.0)
        wait_floor = max(4.0, score_floor - 0.8)
        hard_blocker_markers = (
            "batas loss harian",
            "drawdown akun",
            "loss streak tinggi",
            "leverage crypto terlalu agresif",
            "likuiditas terlalu tipis",
            "insting survival aktif",
        )
        has_hard_blocker = any(any(marker in blocker for marker in hard_blocker_markers) for blocker in state.blockers)

        if state.long_score - state.short_score >= 1.5:
            state.bias = "bullish"
        elif state.short_score - state.long_score >= 1.5:
            state.bias = "bearish"
        else:
            structure = state.features.get("structure")
            htf_bias = state.features.get("higher_timeframe_bias")
            state.bias = structure if structure == htf_bias else "netral"

        if has_hard_blocker:
            state.verdict = "NO TRADE"
        elif state.blockers and max(state.long_score, state.short_score) < max(score_floor + 0.6, 6.0):
            state.verdict = "NO TRADE"
        elif state.long_score >= score_floor and state.long_score >= state.short_score + score_gap_floor and long_rr >= rr_floor:
            state.verdict = "LONG"
            state.selected_plan = state.candidate_long
        elif state.short_score >= score_floor and state.short_score >= state.long_score + score_gap_floor and short_rr >= rr_floor:
            state.verdict = "SHORT"
            state.selected_plan = state.candidate_short
        elif max(state.long_score, state.short_score) >= wait_floor:
            state.verdict = "WAIT"
            state.selected_plan = state.candidate_long if state.long_score >= state.short_score else state.candidate_short
        else:
            state.verdict = "NO TRADE"

        max_score = max(state.long_score, state.short_score)
        confidence = (max_score / 8.0) - (len(state.blockers) * 0.08) + float(state.features.get("adaptive_confidence_shift") or 0.0)
        state.confidence = round(clamp(confidence, 0.05, 0.95), 2)

        chosen_reasons = state.reasons_long if state.long_score >= state.short_score else state.reasons_short
        context = {
            "market_type": state.features.get("market_type"),
            "asset_profile": state.features.get("asset_profile"),
            "session": state.features.get("session"),
            "session_quality": state.features.get("session_quality"),
            "timeframe_minutes": state.features.get("timeframe_minutes"),
            "regime": state.features.get("regime"),
            "higher_timeframe_bias": state.features.get("higher_timeframe_bias"),
            "structure": state.features.get("structure"),
            "volatility": state.features.get("volatility"),
            "style": state.market.style,
        }
        levels = {
            "nearest_support": round_price(state.features.get("nearest_support")),
            "nearest_resistance": round_price(state.features.get("nearest_resistance")),
            "broken_resistance": round_price(state.features.get("broken_resistance")),
            "broken_support": round_price(state.features.get("broken_support")),
        }
        scores = {
            "long": round(state.long_score, 2),
            "short": round(state.short_score, 2),
            "confluence_long": state.features.get("confluence_long"),
            "confluence_short": state.features.get("confluence_short"),
        }
        adaptive_output = {
            "adaptation_mode": adaptive.get("adaptation_mode"),
            "growth_cycle": adaptive.get("growth_cycle"),
            "maturity": adaptive.get("maturity"),
            "evolution_score": adaptive.get("evolution_score"),
            "memory_state": adaptive.get("memory_state"),
            "memory_scope": adaptive.get("memory_scope"),
            "pair_weight": adaptive.get("pair_weight"),
            "market_weight": adaptive.get("market_weight"),
            "pair_sample": adaptive.get("pair_sample"),
            "market_sample": adaptive.get("market_sample"),
            "wins": adaptive.get("wins"),
            "losses": adaptive.get("losses"),
            "loss_streak": adaptive.get("loss_streak"),
            "win_rate": adaptive.get("win_rate"),
            "rr_floor": adaptive.get("rr_floor"),
            "score_floor": adaptive.get("score_floor"),
            "score_gap_floor": adaptive.get("score_gap_floor"),
            "score_shift": adaptive.get("score_shift"),
            "confidence_shift": adaptive.get("confidence_shift"),
            "aggression_bias": adaptive.get("aggression_bias"),
            "exploration_bias": adaptive.get("exploration_bias"),
            "compounding_bias": adaptive.get("compounding_bias"),
            "protection_bias": adaptive.get("protection_bias"),
            "focus_titles": adaptive.get("focus_titles"),
            "stage_biases": adaptive.get("stage_biases"),
            "cycles": adaptive.get("cycles"),
            "transitions": adaptive.get("transitions"),
            "note": adaptive.get("note"),
        }
        training = state.features.get("training_summary") or state.features.get("training_profile") or {}
        training_output = {
            "trainer_state": training.get("trainer_state"),
            "source_scope": training.get("source_scope"),
            "observed_days": training.get("observed_days"),
            "training_days": training.get("training_days"),
            "sample_size": training.get("sample_size"),
            "wins": training.get("wins"),
            "losses": training.get("losses"),
            "win_rate": training.get("win_rate"),
            "preferred_direction": training.get("preferred_direction"),
            "long_edge": training.get("long_edge"),
            "short_edge": training.get("short_edge"),
            "rr_floor_delta": training.get("rr_floor_delta"),
            "confidence_shift": training.get("confidence_shift"),
            "risk_cap_delta": training.get("risk_cap_delta"),
            "daily_progress": training.get("daily_progress"),
            "lesson_notes": training.get("lesson_notes"),
            "note": training.get("note"),
        }

        output_plan = state.selected_plan.as_dict() if state.selected_plan.valid else {}
        conditional_plan = {}
        if state.verdict == "WAIT":
            conditional_plan = output_plan

        if state.verdict not in {"LONG", "SHORT"}:
            output_plan = {}

        state.output = {
            "meta": {
                "brain": state.features.get("brain_name"),
                "version": state.features.get("brain_version"),
                "neuron_count": state.features.get("neuron_count"),
            },
            "summary": {
                "symbol": state.market.symbol,
                "timeframe": state.market.timeframe,
                "market_type": state.features.get("market_type"),
                "bias": state.bias,
                "verdict": state.verdict,
                "confidence": state.confidence,
                "adaptive_mode": adaptive.get("adaptation_mode"),
                "growth_cycle": adaptive.get("growth_cycle"),
                "adaptive_maturity": adaptive.get("maturity"),
                "trainer_state": training.get("trainer_state"),
            },
            "context": context,
            "levels": levels,
            "scores": scores,
            "adaptive": adaptive_output,
            "training": training_output,
            "risk": {
                "max_risk_percent": state.features.get("risk_percent"),
                "effective_risk_percent": state.features.get("effective_risk_percent"),
                "size_multiplier": state.features.get("size_multiplier"),
                "leverage": state.market.risk.leverage,
                "leverage_profile": state.features.get("leverage_profile"),
                "current_drawdown_percent": state.features.get("current_drawdown_percent"),
                "account_heat": state.features.get("account_heat"),
                "loss_streak": state.features.get("loss_streak"),
                "friction_bps": state.features.get("friction_bps"),
                "rr_floor": rr_floor,
            },
            "plan": output_plan,
            "conditional_plan": conditional_plan,
            "reasons": chosen_reasons[:5],
            "warnings": state.warnings,
            "blockers": state.blockers,
            "notes": state.notes[-6:],
        }
