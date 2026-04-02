from __future__ import annotations

from .models import BrainState
from .neuron_base import Neuron, price_anchor
from .utils import distance_pct


def _bias(state: BrainState) -> str:
    return str(state.features.get("higher_timeframe_bias") or "netral")


def _structure(state: BrainState) -> str:
    return str(state.features.get("structure") or "netral")


def _regime(state: BrainState) -> str:
    return str(state.features.get("regime") or "")


def _session_quality(state: BrainState) -> str:
    return str(state.features.get("session_quality") or "")


def _risk_is_clean(state: BrainState) -> bool:
    return not state.blockers and len(state.warnings) <= 3


class NarrativePressureNeuron(Neuron):
    name = "neuron58_narrative_pressure"
    title = "Narrative Pressure"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Mengikat sentimen, headline risk, dan bias utama menjadi tekanan naratif."

    def run(self, state: BrainState) -> None:
        sentiment = state.market.sentiment.score
        bias = _bias(state)
        if sentiment is None:
            return
        if sentiment >= 0.18 and bias == "bullish":
            self.boost(state, "long", 0.36, "narasi market ikut mengalir ke skenario bullish")
        elif sentiment <= -0.18 and bias == "bearish":
            self.boost(state, "short", 0.36, "narasi market ikut mengalir ke skenario bearish")


class SessionRotationMapNeuron(Neuron):
    name = "neuron59_session_rotation_map"
    title = "Session Rotation"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Membaca kapan session aktif benar-benar mendukung impuls market."

    def run(self, state: BrainState) -> None:
        session_quality = _session_quality(state)
        market_type = str(state.features.get("market_type") or "")
        style = state.market.style

        if session_quality == "high" and style in {"scalping", "intraday"}:
            self.boost(state, "long", 0.12)
            self.boost(state, "short", 0.12)
        elif session_quality == "low" and market_type in {"forex", "commodity"} and style in {"scalping", "intraday"}:
            state.add_warning("rotasi session belum memberi impuls yang cukup sehat")


class StochasticPhaseNeuron(Neuron):
    name = "neuron60_stochastic_phase"
    title = "Stochastic Phase"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Membaca fase dorong stochastic tanpa terlalu mengejar kondisi ekstrem."

    def run(self, state: BrainState) -> None:
        stochastic = state.market.indicators.stochastic
        structure = _structure(state)
        if stochastic is None:
            return

        if 56 <= stochastic <= 82 and structure == "bullish":
            self.boost(state, "long", 0.28, "fase stochastic mendukung continuation bullish")
        elif 18 <= stochastic <= 44 and structure == "bearish":
            self.boost(state, "short", 0.28, "fase stochastic mendukung continuation bearish")


class BollingerElasticityNeuron(Neuron):
    name = "neuron61_bollinger_elasticity"
    title = "Bollinger Elasticity"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Membaca elastisitas harga terhadap band untuk continuation atau fade."

    def run(self, state: BrainState) -> None:
        position = state.market.indicators.bollinger_position
        regime = _regime(state)
        structure = _structure(state)
        if position is None:
            return

        if position >= 0.72 and structure == "bullish" and regime in {"trending", "expansion"}:
            self.boost(state, "long", 0.34, "ekspansi band mendukung continuation bullish")
        elif position <= 0.28 and structure == "bearish" and regime in {"trending", "expansion"}:
            self.boost(state, "short", 0.34, "ekspansi band mendukung continuation bearish")
        elif 0.42 <= position <= 0.58 and regime == "compression":
            state.add_warning("harga masih tertahan di tengah band, dorongan belum pecah")


class CCIImpulseNeuron(Neuron):
    name = "neuron62_cci_impulse"
    title = "CCI Impulse"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Menguji apakah impuls CCI sejalan dengan arah struktur yang aktif."

    def run(self, state: BrainState) -> None:
        cci = state.market.indicators.cci
        structure = _structure(state)
        if cci is None:
            return

        if cci >= 90 and structure == "bullish":
            self.boost(state, "long", 0.26, "CCI mendorong impuls bullish")
        elif cci <= -90 and structure == "bearish":
            self.boost(state, "short", 0.26, "CCI mendorong impuls bearish")


class DeltaFlowNeuron(Neuron):
    name = "neuron63_delta_flow"
    title = "Delta Flow"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Membaca delta volume sebagai petunjuk siapa yang benar-benar menekan tape."

    def run(self, state: BrainState) -> None:
        delta = state.market.indicators.delta_volume
        bias = _bias(state)
        if delta is None:
            return

        if delta >= 0.18 and bias == "bullish":
            self.boost(state, "long", 0.32, "delta volume mendukung buyer yang aktif")
        elif delta <= -0.18 and bias == "bearish":
            self.boost(state, "short", 0.32, "delta volume mendukung seller yang aktif")


class VWAPElasticityNeuron(Neuron):
    name = "neuron64_vwap_elasticity"
    title = "VWAP Elasticity"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Menguji apakah harga masih cukup dekat ke VWAP untuk entry yang efisien."

    def run(self, state: BrainState) -> None:
        price = price_anchor(state)
        vwap = state.market.indicators.vwap
        bias = _bias(state)
        if not price or vwap is None:
            return

        stretch = abs(price - vwap) / price * 100.0
        if 0.08 <= stretch <= 0.65:
            if price > vwap and bias == "bullish":
                self.boost(state, "long", 0.22, "harga masih elastis di atas VWAP untuk continuation")
            elif price < vwap and bias == "bearish":
                self.boost(state, "short", 0.22, "harga masih elastis di bawah VWAP untuk continuation")


class LevelCompressionNeuron(Neuron):
    name = "neuron65_level_compression"
    title = "Level Compression"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Membaca tekanan saat support dan resistance mulai merapat ke area ledakan."

    def run(self, state: BrainState) -> None:
        price = price_anchor(state)
        support = state.features.get("nearest_support")
        resistance = state.features.get("nearest_resistance")
        regime = _regime(state)
        if not price or support is None or resistance is None or resistance <= support:
            return

        width_pct = (resistance - support) / price * 100.0
        state.features["level_width_pct"] = round(width_pct, 3)
        if width_pct <= 0.85 and regime in {"compression", "ranging"}:
            state.add_warning("level mulai rapat, market siap bergerak tapi belum pilih arah bersih")
        elif width_pct <= 1.2 and regime == "trending":
            self.boost(state, "long" if _structure(state) == "bullish" else "short", 0.18, "kompresi level menjaga continuation tetap rapat")


class FailedAuctionNeuron(Neuron):
    name = "neuron66_failed_auction"
    title = "Failed Auction"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Menilai tanda gagal lelang dari wick dan posisi range."

    def run(self, state: BrainState) -> None:
        range_position = state.features.get("range_position")
        upper_wick = state.features.get("upper_wick_ratio")
        lower_wick = state.features.get("lower_wick_ratio")

        if range_position is None:
            return
        if range_position >= 0.72 and upper_wick is not None and upper_wick >= 0.34:
            self.boost(state, "short", 0.26, "failed auction terlihat di bagian atas range")
        if range_position <= 0.28 and lower_wick is not None and lower_wick >= 0.34:
            self.boost(state, "long", 0.26, "failed auction terlihat di bagian bawah range")


class LiquidityPocketNeuron(Neuron):
    name = "neuron67_liquidity_pocket"
    title = "Liquidity Pocket"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Mencari entry dekat pocket likuiditas yang tetap sehat."

    def run(self, state: BrainState) -> None:
        price = price_anchor(state)
        support = state.features.get("nearest_support")
        resistance = state.features.get("nearest_resistance")
        liquidity = state.features.get("liquidity_score")

        if not price or liquidity is None or liquidity < 0.58:
            return

        if support is not None:
            distance = distance_pct(price, support)
            if distance is not None and distance <= 0.42:
                self.boost(state, "long", 0.22, "pocket likuiditas bawah cukup sehat untuk long terukur")
        if resistance is not None:
            distance = distance_pct(price, resistance)
            if distance is not None and distance <= 0.42:
                self.boost(state, "short", 0.22, "pocket likuiditas atas cukup sehat untuk short terukur")


class RegimeCoherenceNeuron(Neuron):
    name = "neuron68_regime_coherence"
    title = "Regime Coherence"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Memastikan bias, struktur, dan regime sedang bicara bahasa yang sama."

    def run(self, state: BrainState) -> None:
        bias = _bias(state)
        structure = _structure(state)
        regime = _regime(state)

        if bias == "bullish" and structure == "bullish" and regime in {"trending", "expansion"}:
            self.boost(state, "long", 0.42, "bias, struktur, dan regime bullish sedang sinkron")
        elif bias == "bearish" and structure == "bearish" and regime in {"trending", "expansion"}:
            self.boost(state, "short", 0.42, "bias, struktur, dan regime bearish sedang sinkron")


class PrimeWindowNeuron(Neuron):
    name = "neuron69_prime_window"
    title = "Prime Window"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Menilai apakah kombinasi session, volatility, dan noise sedang ideal."

    def run(self, state: BrainState) -> None:
        if _session_quality(state) != "high":
            return
        if str(state.features.get("volatility") or "") != "normal":
            return
        if state.market.sentiment.headline_risk or state.market.sentiment.macro_risk:
            return

        direction = "long" if _structure(state) == "bullish" else "short" if _structure(state) == "bearish" else None
        if direction:
            self.boost(state, direction, 0.24, "prime window aktif: session dan volatility sedang ideal")


class CorrelationImpulseNeuron(Neuron):
    name = "neuron70_correlation_impulse"
    title = "Correlation Impulse"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Membaca tailwind atau headwind dari korelasi market lintas aset."

    def run(self, state: BrainState) -> None:
        correlation = state.market.sentiment.correlation_bias
        bias = _bias(state)
        if correlation is None:
            return

        if correlation >= 0.14 and bias == "bullish":
            self.boost(state, "long", 0.18, "korelasi lintas aset memberi tailwind bullish")
        elif correlation <= -0.14 and bias == "bearish":
            self.boost(state, "short", 0.18, "korelasi lintas aset memberi tailwind bearish")


class RiskCompressionNeuron(Neuron):
    name = "neuron71_risk_compression"
    title = "Risk Compression"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Mengecek apakah risk aktif cukup rapat untuk setup kelas atas."

    def run(self, state: BrainState) -> None:
        effective_risk = float(state.features.get("effective_risk_percent") or 0.0)
        long_rr = float(state.candidate_long.risk_reward or 0.0)
        short_rr = float(state.candidate_short.risk_reward or 0.0)

        if effective_risk <= 0.4 and long_rr >= 1.9 and _risk_is_clean(state):
            self.boost(state, "long", 0.16, "risk aktif cukup rapat untuk continuation long")
        if effective_risk <= 0.4 and short_rr >= 1.9 and _risk_is_clean(state):
            self.boost(state, "short", 0.16, "risk aktif cukup rapat untuk continuation short")


class DecisionLatencyNeuron(Neuron):
    name = "neuron72_decision_latency"
    title = "Decision Latency"
    visual_group = "cortex"
    visual_stage = "cortex"
    description = "Menahan trade saat konteks terlalu lambat atau terlalu bertabrakan."

    def run(self, state: BrainState) -> None:
        gap = abs(state.long_score - state.short_score)
        if gap <= 0.65 and len(state.warnings) >= 3:
            state.add_warning("keputusan masih lambat terbentuk, market belum memberi arah yang tegas")
