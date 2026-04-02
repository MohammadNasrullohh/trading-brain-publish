from __future__ import annotations

from .adaptive_growth import derive_growth_profile
from .models import BrainState
from .neuron_base import Neuron, price_anchor
from .self_training import derive_training_profile
from .utils import clamp, distance_pct


def _profile(state: BrainState) -> dict:
    return state.features.get("adaptive_profile") or {}


def _training(state: BrainState) -> dict:
    return state.features.get("training_profile") or {}


def _structural_direction(state: BrainState, profile: dict | None = None) -> str | None:
    structure = str(state.features.get("structure") or "").strip().lower()
    htf_bias = str(state.features.get("higher_timeframe_bias") or "").strip().lower()

    if structure == "bullish" and htf_bias in {"", "bullish", "netral"}:
        return "long"
    if structure == "bearish" and htf_bias in {"", "bearish", "netral"}:
        return "short"
    if htf_bias == "bullish":
        return "long"
    if htf_bias == "bearish":
        return "short"

    tilt = float((profile or {}).get("directional_tilt") or 0.0)
    if tilt >= 0.14:
        return "long"
    if tilt <= -0.14:
        return "short"
    return None


def _dampen_direction(state: BrainState, direction: str, amount: float) -> None:
    if direction == "long":
        state.long_score = max(0.0, state.long_score - amount)
    elif direction == "short":
        state.short_score = max(0.0, state.short_score - amount)


class AdaptiveGrowthBootstrapNeuron(Neuron):
    name = "neuron47_adaptive_growth_bootstrap"

    def run(self, state: BrainState) -> None:
        market_type = state.features.get("market_type") or state.market.market_type
        profile = derive_growth_profile(
            state.market.symbol,
            state.market.timeframe,
            state.market.style,
            market_type,
        )
        state.features["adaptive_profile"] = profile
        state.features["adaptive_rr_floor"] = profile.get("rr_floor", 1.5)
        state.features["adaptive_score_floor"] = profile.get("score_floor", 5.0)
        state.features["adaptive_score_gap"] = profile.get("score_gap_floor", 2.0)
        state.features["adaptive_confidence_shift"] = profile.get("confidence_shift", 0.0)
        state.features["adaptive_growth_cycle"] = profile.get("growth_cycle", "bootstrap")
        state.features["adaptive_mode"] = profile.get("adaptation_mode", "calibrating")
        state.features["risk_cap_multiplier"] = 1.0
        note = profile.get("note")
        if note and note not in state.notes:
            state.notes.append(note)


class SelfTrainingLoopNeuron(Neuron):
    name = "neuron73_self_training_loop"

    def run(self, state: BrainState) -> None:
        market_type = state.features.get("market_type") or state.market.market_type
        profile = derive_training_profile(
            state.market.symbol,
            state.market.timeframe,
            state.market.style,
            market_type,
        )
        state.features["training_profile"] = profile
        note = profile.get("note")
        if note and note not in state.notes:
            state.notes.append(note)


class PairMemoryResonanceNeuron(Neuron):
    name = "neuron48_pair_memory_resonance"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        pair_weight = float(profile.get("pair_weight") or 0.0)
        if pair_weight <= 0.0:
            return

        direction = _structural_direction(state, profile)
        shift = float(profile.get("score_shift") or 0.0)
        maturity = float(profile.get("maturity") or 0.0)

        if direction and shift > 0.04:
            boost = clamp((shift * 0.45) + (pair_weight * 0.18) + (maturity / 400.0), 0.12, 0.78)
            self.boost(state, direction, boost, "memori pair sedang sinkron dengan struktur saat ini")
        elif direction and shift < -0.18:
            _dampen_direction(state, direction, abs(shift) * 0.45)
            state.add_warning("memori pair belum sinkron, jadi conviction ditahan")


class MarketMemoryBridgeNeuron(Neuron):
    name = "neuron49_market_memory_bridge"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        market_weight = float(profile.get("market_weight") or 0.0)
        if market_weight <= 0.0:
            return

        direction = _structural_direction(state, profile)
        scope = str(profile.get("memory_scope") or "").strip().lower()
        fallback_active = bool(profile.get("market_fallback_active"))

        if fallback_active and direction:
            boost = clamp((market_weight * 0.26) + 0.08, 0.08, 0.32)
            self.boost(state, direction, boost, "konteks sektor membantu memberi referensi awal untuk pair ini")

        if scope in {"mixed", "market"} and float(profile.get("maturity") or 0.0) < 35.0:
            state.add_warning("adaptasi masih banyak bertumpu pada memori sektor, bukan pair ini saja")


class ProfitReplayNeuron(Neuron):
    name = "neuron74_profit_replay"

    def run(self, state: BrainState) -> None:
        trainer = _training(state)
        if not trainer:
            return

        long_edge = float(trainer.get("long_edge") or 0.0)
        short_edge = float(trainer.get("short_edge") or 0.0)

        if long_edge >= 0.04:
            self.boost(
                state,
                "long",
                clamp(long_edge * 2.2, 0.08, 0.36),
                "trainer harian melihat replay profit long lebih sehat untuk context ini",
            )
        elif long_edge <= -0.06:
            _dampen_direction(state, "long", clamp(abs(long_edge) * 1.7, 0.08, 0.24))
            state.add_warning("review loss harian menahan sisi long karena historinya belum sehat")

        if short_edge >= 0.04:
            self.boost(
                state,
                "short",
                clamp(short_edge * 2.2, 0.08, 0.36),
                "trainer harian melihat replay profit short lebih sehat untuk context ini",
            )
        elif short_edge <= -0.06:
            _dampen_direction(state, "short", clamp(abs(short_edge) * 1.7, 0.08, 0.24))
            state.add_warning("review loss harian menahan sisi short karena historinya belum sehat")


class RegimeMutationNeuron(Neuron):
    name = "neuron50_regime_mutation"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        cycle = str(profile.get("growth_cycle") or "bootstrap").strip().lower()
        regime = str(state.features.get("regime") or "").strip().lower()
        direction = _structural_direction(state, profile)

        if cycle == "compound" and direction and regime in {"trending", "expansion"}:
            self.boost(state, direction, 0.38, "mode compound aktif dan regime mendukung continuation")
        elif cycle == "protect" and regime in {"expansion", "compression", "ranging"}:
            current = float(state.features.get("risk_cap_multiplier") or 1.0)
            state.features["risk_cap_multiplier"] = min(current, 0.78)
            state.add_warning("regime sekarang membuat mode proteksi semakin dominan")


class ExplorationDriveNeuron(Neuron):
    name = "neuron51_exploration_drive"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        cycle = str(profile.get("growth_cycle") or "bootstrap").strip().lower()
        if cycle not in {"bootstrap", "explore"}:
            return

        direction = _structural_direction(state, profile)
        volatility = str(state.features.get("volatility") or "").strip().lower()
        headline_risk = bool(state.market.sentiment.headline_risk or state.market.sentiment.macro_risk)

        if direction and volatility != "high" and not headline_risk:
            boost = clamp(float(profile.get("exploration_bias") or 0.0) * 2.1, 0.12, 0.34)
            self.boost(state, direction, boost, "mode eksplorasi aktif, jadi setup rapi diberi ruang berkembang")


class OpportunityCompressionNeuron(Neuron):
    name = "neuron52_opportunity_compression"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        price = price_anchor(state)
        if not price:
            return

        direction = _structural_direction(state, profile)
        support = state.features.get("nearest_support")
        resistance = state.features.get("nearest_resistance")
        cycle = str(profile.get("growth_cycle") or "").strip().lower()

        if direction == "long" and support is not None:
            distance = distance_pct(price, support)
            if distance is not None and distance <= 0.45:
                boost = 0.46 if cycle == "compound" else 0.24
                self.boost(state, "long", boost, "lokasi entry long cukup rapat terhadap support aktif")
            elif cycle == "protect" and distance is not None and distance >= 1.1:
                state.add_warning("lokasi long terlalu longgar untuk mode proteksi")

        if direction == "short" and resistance is not None:
            distance = distance_pct(price, resistance)
            if distance is not None and distance <= 0.45:
                boost = 0.46 if cycle == "compound" else 0.24
                self.boost(state, "short", boost, "lokasi entry short cukup rapat terhadap resistance aktif")
            elif cycle == "protect" and distance is not None and distance >= 1.1:
                state.add_warning("lokasi short terlalu longgar untuk mode proteksi")


class CompoundingDriveNeuron(Neuron):
    name = "neuron53_compounding_drive"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        if str(profile.get("growth_cycle") or "").strip().lower() != "compound":
            return

        direction = _structural_direction(state, profile)
        max_score = max(state.long_score, state.short_score)
        if not direction or max_score < 4.6 or state.blockers:
            return

        multiplier = 1.0 + clamp(float(profile.get("compounding_bias") or 0.0) * 0.55, 0.02, 0.1)
        current = float(state.features.get("risk_cap_multiplier") or 1.0)
        state.features["risk_cap_multiplier"] = min(1.12, current * multiplier)
        self.boost(state, direction, 0.26, "histori pair sedang sinkron, jadi continuation bersih diprioritaskan")


class SurvivalInstinctNeuron(Neuron):
    name = "neuron54_survival_instinct"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        protection_bias = float(profile.get("protection_bias") or 0.0)
        if protection_bias <= 0.0:
            return

        current = float(state.features.get("risk_cap_multiplier") or 1.0)
        state.features["risk_cap_multiplier"] = min(current, clamp(1.0 - (protection_bias * 1.9), 0.52, 0.88))

        if int(profile.get("loss_streak") or 0) >= 3:
            state.add_blocker("insting survival aktif karena loss streak masih berat")
        elif protection_bias >= 0.18:
            state.add_warning("insting survival menahan agresivitas sambil menunggu setup lebih sehat")


class AntiOverfitFilterNeuron(Neuron):
    name = "neuron55_anti_overfit_filter"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        scope = str(profile.get("memory_scope") or "").strip().lower()
        pair_sample = int(profile.get("pair_sample") or 0)
        rr_floor = float(profile.get("rr_floor") or 1.5)
        long_rr = float(state.candidate_long.risk_reward or 0.0)
        short_rr = float(state.candidate_short.risk_reward or 0.0)

        if scope in {"mixed", "market"} and pair_sample <= 1:
            if long_rr < rr_floor + 0.1 and short_rr < rr_floor + 0.1:
                state.add_blocker("adaptasi pair belum matang, jadi reward-to-risk harus ekstra sehat")
            else:
                state.add_warning("sinyal masih banyak bergantung pada memori sektor, jadi validasi pair tetap utama")


class ConfidenceMemoryNeuron(Neuron):
    name = "neuron56_confidence_memory"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        shift = float(profile.get("confidence_shift") or 0.0)
        maturity = float(profile.get("maturity") or 0.0)
        if maturity >= 60.0 and profile.get("growth_cycle") == "compound":
            shift += 0.01
        elif maturity <= 18.0 and profile.get("growth_cycle") in {"bootstrap", "explore"}:
            shift -= 0.01
        state.features["adaptive_confidence_shift"] = round(clamp(shift, -0.12, 0.1), 3)


class LossReviewNeuron(Neuron):
    name = "neuron75_loss_review"

    def run(self, state: BrainState) -> None:
        trainer = _training(state)
        if not trainer:
            return

        rr_floor = float(state.features.get("adaptive_rr_floor") or 1.5)
        rr_delta = float(trainer.get("rr_floor_delta") or 0.0)
        if rr_delta:
            state.features["adaptive_rr_floor"] = round(clamp(rr_floor + rr_delta, 1.2, 2.3), 2)

        if str(trainer.get("trainer_state") or "").strip().lower() == "defensive":
            state.add_warning("trainer harian sedang protektif karena replay loss masih dominan")
            if int(trainer.get("low_rr_losses") or 0) >= 2:
                state.add_warning("banyak loss datang dari RR tipis, jadi standar entry dinaikkan")


class DailyCoachNeuron(Neuron):
    name = "neuron76_daily_coach"

    def run(self, state: BrainState) -> None:
        trainer = _training(state)
        if not trainer:
            return

        existing_confidence = float(state.features.get("adaptive_confidence_shift") or 0.0)
        trainer_shift = float(trainer.get("confidence_shift") or 0.0)
        state.features["adaptive_confidence_shift"] = round(
            clamp(existing_confidence + trainer_shift, -0.16, 0.12),
            3,
        )

        current_risk_cap = float(state.features.get("risk_cap_multiplier") or 1.0)
        risk_delta = float(trainer.get("risk_cap_delta") or 0.0)
        state.features["risk_cap_multiplier"] = clamp(current_risk_cap * (1.0 + risk_delta), 0.48, 1.14)

        state.features["training_summary"] = {
            "trainer_state": trainer.get("trainer_state"),
            "source_scope": trainer.get("source_scope"),
            "observed_days": trainer.get("observed_days"),
            "training_days": trainer.get("training_days"),
            "sample_size": trainer.get("sample_size"),
            "wins": trainer.get("wins"),
            "losses": trainer.get("losses"),
            "win_rate": trainer.get("win_rate"),
            "preferred_direction": trainer.get("preferred_direction"),
            "long_edge": trainer.get("long_edge"),
            "short_edge": trainer.get("short_edge"),
            "rr_floor_delta": trainer.get("rr_floor_delta"),
            "confidence_shift": trainer.get("confidence_shift"),
            "risk_cap_delta": trainer.get("risk_cap_delta"),
            "daily_progress": trainer.get("daily_progress"),
            "lesson_notes": trainer.get("lesson_notes"),
            "note": trainer.get("coach_note") or trainer.get("note"),
        }
        note = trainer.get("coach_note") or trainer.get("note")
        if note and note not in state.notes:
            state.notes.append(note)


class SelfReflectionNeuron(Neuron):
    name = "neuron57_self_reflection"

    def run(self, state: BrainState) -> None:
        profile = _profile(state)
        if not profile:
            return

        state.features["adaptive_summary"] = {
            "growth_cycle": profile.get("growth_cycle"),
            "adaptation_mode": profile.get("adaptation_mode"),
            "maturity": profile.get("maturity"),
            "evolution_score": profile.get("evolution_score"),
            "memory_state": profile.get("memory_state"),
            "memory_scope": profile.get("memory_scope"),
            "pair_weight": profile.get("pair_weight"),
            "market_weight": profile.get("market_weight"),
            "pair_sample": profile.get("pair_sample"),
            "market_sample": profile.get("market_sample"),
            "wins": profile.get("wins"),
            "losses": profile.get("losses"),
            "loss_streak": profile.get("loss_streak"),
            "win_rate": profile.get("win_rate"),
            "rr_floor": profile.get("rr_floor"),
            "score_floor": profile.get("score_floor"),
            "score_gap_floor": profile.get("score_gap_floor"),
            "score_shift": profile.get("score_shift"),
            "confidence_shift": profile.get("confidence_shift"),
            "aggression_bias": profile.get("aggression_bias"),
            "exploration_bias": profile.get("exploration_bias"),
            "compounding_bias": profile.get("compounding_bias"),
            "protection_bias": profile.get("protection_bias"),
            "focus_titles": profile.get("focus_titles"),
            "stage_biases": profile.get("stage_biases"),
            "cycles": profile.get("cycles", 0),
            "transitions": profile.get("transitions", 0),
            "training": state.features.get("training_summary") or {},
            "note": profile.get("note"),
        }
        note = profile.get("note")
        if note and note not in state.notes:
            state.notes.append(note)
