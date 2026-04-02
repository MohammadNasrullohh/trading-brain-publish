from __future__ import annotations

from .models import BrainState
from .neuron_base import Neuron, price_anchor
from .utils import distance_pct, farthest_above, farthest_below, nearest_above, nearest_below, normalize_label


class PersonaCoreNeuron(Neuron):
    name = "neuron01_persona_core"

    def run(self, state: BrainState) -> None:
        state.features["brain_name"] = "TradingBrain"
        state.features["brain_version"] = "1.3.0"
        state.features["persona"] = "risk_first_trading_analyst"


class MissionGuardNeuron(Neuron):
    name = "neuron02_mission_guard"

    def run(self, state: BrainState) -> None:
        state.features["mission"] = "protect_capital_before_chasing_profit"


class InputReaderNeuron(Neuron):
    name = "neuron03_input_reader"

    def run(self, state: BrainState) -> None:
        market = state.market
        if not market.symbol:
            state.add_blocker("symbol belum diberikan")
        if not market.timeframe:
            state.add_blocker("timeframe belum diberikan")
        if market.price is None and market.close is None:
            state.add_blocker("price atau close belum diberikan")
        if not market.levels.support:
            state.add_warning("support belum diberikan")
        if not market.levels.resistance:
            state.add_warning("resistance belum diberikan")
        state.features["style"] = market.style


class MarketRegimeNeuron(Neuron):
    name = "neuron04_market_regime"

    def run(self, state: BrainState) -> None:
        market = state.market
        hint = normalize_label(market.context.regime_hint, "")
        price = price_anchor(state)
        range_pct = None
        ema_spread_pct = None
        if market.high is not None and market.low is not None and price:
            range_pct = (market.high - market.low) / price * 100.0
        if market.indicators.ema_fast is not None and market.indicators.ema_slow is not None and price:
            ema_spread_pct = abs(market.indicators.ema_fast - market.indicators.ema_slow) / price * 100.0
        atr_pct = (market.atr / price * 100.0) if market.atr and price else None

        if hint:
            regime = hint
        elif range_pct is not None and range_pct < 0.35:
            regime = "compression"
        elif (atr_pct is not None and atr_pct > 1.6) or (range_pct is not None and range_pct > 2.2):
            regime = "expansion"
        elif ema_spread_pct is not None and ema_spread_pct >= 0.18:
            regime = "trending"
        else:
            regime = "ranging"

        state.features["regime"] = regime
        if regime in {"compression", "ranging"}:
            state.add_warning("market belum menunjukkan dorongan yang bersih")
        if regime == "expansion":
            state.add_warning("market sedang ekspansif, hindari entry terburu-buru")


class HigherTimeframeBiasNeuron(Neuron):
    name = "neuron05_higher_timeframe_bias"

    def run(self, state: BrainState) -> None:
        market = state.market
        bias_hint = normalize_label(market.context.bias_hint, "")
        price = price_anchor(state)

        if bias_hint in {"bullish", "bearish", "netral", "neutral"}:
            bias = "netral" if bias_hint == "neutral" else bias_hint
            state.features["higher_timeframe_bias"] = bias
            if bias == "bullish":
                self.boost(state, "long", 1.5, "bias timeframe besar cenderung bullish")
            elif bias == "bearish":
                self.boost(state, "short", 1.5, "bias timeframe besar cenderung bearish")
            return

        if market.indicators.ema_fast is not None and market.indicators.ema_slow is not None:
            if market.indicators.ema_fast > market.indicators.ema_slow:
                self.boost(state, "long", 1.0, "EMA cepat berada di atas EMA lambat")
            elif market.indicators.ema_fast < market.indicators.ema_slow:
                self.boost(state, "short", 1.0, "EMA cepat berada di bawah EMA lambat")

        if market.indicators.vwap is not None and price:
            if price > market.indicators.vwap:
                self.boost(state, "long", 0.5, "harga berada di atas VWAP")
            elif price < market.indicators.vwap:
                self.boost(state, "short", 0.5, "harga berada di bawah VWAP")

        if market.close is not None and market.open is not None:
            if market.close > market.open:
                self.boost(state, "long", 0.5, "close berada di atas open")
            elif market.close < market.open:
                self.boost(state, "short", 0.5, "close berada di bawah open")

        if state.long_score > state.short_score:
            state.features["higher_timeframe_bias"] = "bullish"
        elif state.short_score > state.long_score:
            state.features["higher_timeframe_bias"] = "bearish"
        else:
            state.features["higher_timeframe_bias"] = "netral"


class StructureMapperNeuron(Neuron):
    name = "neuron06_structure_mapper"

    def run(self, state: BrainState) -> None:
        market = state.market
        hint = normalize_label(market.context.structure_hint, "")
        if hint in {"bullish", "bearish", "netral", "neutral"}:
            structure = "netral" if hint == "neutral" else hint
        elif None not in (market.high, market.low, market.close) and market.high != market.low:
            close_position = (market.close - market.low) / (market.high - market.low)
            if close_position >= 0.65:
                structure = "bullish"
            elif close_position <= 0.35:
                structure = "bearish"
            else:
                structure = "netral"
        else:
            structure = "netral"

        state.features["structure"] = structure
        if structure == "bullish":
            self.boost(state, "long", 1.0, "struktur candle mendukung arah bullish")
        elif structure == "bearish":
            self.boost(state, "short", 1.0, "struktur candle mendukung arah bearish")
        else:
            state.add_warning("struktur market belum cukup tegas")


class KeyLevelsNeuron(Neuron):
    name = "neuron07_key_levels"

    def run(self, state: BrainState) -> None:
        levels = state.market.levels
        price = price_anchor(state)

        nearest_support = nearest_below(levels.support, price)
        nearest_resistance = nearest_above(levels.resistance, price)
        if nearest_support is None and levels.support:
            nearest_support = max(levels.support)
        if nearest_resistance is None and levels.resistance:
            nearest_resistance = min(levels.resistance)

        state.features["nearest_support"] = nearest_support
        state.features["nearest_resistance"] = nearest_resistance
        state.features["broken_resistance"] = farthest_below(levels.resistance, price)
        state.features["broken_support"] = farthest_above(levels.support, price)

        if nearest_support is None:
            state.add_warning("tidak ada support terdekat yang jelas")
        if nearest_resistance is None:
            state.add_warning("tidak ada resistance terdekat yang jelas")


class LiquidityContextNeuron(Neuron):
    name = "neuron08_liquidity_context"

    def run(self, state: BrainState) -> None:
        price = price_anchor(state)
        broken_resistance = state.features.get("broken_resistance")
        broken_support = state.features.get("broken_support")
        nearest_resistance = state.features.get("nearest_resistance")
        nearest_support = state.features.get("nearest_support")

        if broken_resistance is not None:
            broken_distance = distance_pct(price, broken_resistance)
            if broken_distance is not None and broken_distance <= 0.35:
                self.boost(state, "long", 0.5, "harga bertahan di atas resistance yang sudah pecah")

        if broken_support is not None:
            broken_distance = distance_pct(price, broken_support)
            if broken_distance is not None and broken_distance <= 0.35:
                self.boost(state, "short", 0.5, "harga bertahan di bawah support yang sudah pecah")

        if nearest_resistance is not None:
            distance_to_resistance = distance_pct(price, nearest_resistance)
            if distance_to_resistance is not None and distance_to_resistance <= 0.2:
                state.add_warning("resistance terlalu dekat, upside bisa terbatas")

        if nearest_support is not None:
            distance_to_support = distance_pct(price, nearest_support)
            if distance_to_support is not None and distance_to_support <= 0.2:
                state.add_warning("support terlalu dekat, downside bisa memicu sweep")


class VolumeMomentumNeuron(Neuron):
    name = "neuron09_volume_momentum"

    def run(self, state: BrainState) -> None:
        market = state.market
        volume_trend = normalize_label(market.indicators.volume_trend, "")
        rsi = market.indicators.rsi
        macd = market.indicators.macd_histogram

        if rsi is not None:
            if rsi >= 55:
                self.boost(state, "long", 1.0, "RSI berada di area bullish")
            elif rsi <= 45:
                self.boost(state, "short", 1.0, "RSI berada di area bearish")

            if rsi >= 70:
                state.add_warning("RSI tinggi, waspadai kondisi overbought")
            if rsi <= 30:
                state.add_warning("RSI rendah, waspadai kondisi oversold")

        if macd is not None:
            if macd > 0:
                self.boost(state, "long", 0.5, "MACD histogram berada di atas nol")
            elif macd < 0:
                self.boost(state, "short", 0.5, "MACD histogram berada di bawah nol")

        if volume_trend in {"rising", "strong", "expanding"}:
            if state.market.close is not None and state.market.open is not None and state.market.close >= state.market.open:
                self.boost(state, "long", 0.5, "volume mendukung pergerakan naik")
            elif state.market.close is not None and state.market.open is not None:
                self.boost(state, "short", 0.5, "volume mendukung pergerakan turun")
        elif volume_trend in {"falling", "weak", "thin"}:
            state.add_warning("volume belum mendukung pergerakan dengan kuat")


class VolatilityEngineNeuron(Neuron):
    name = "neuron10_volatility_engine"

    def run(self, state: BrainState) -> None:
        price = price_anchor(state)
        market = state.market

        range_pct = None
        if None not in (market.high, market.low) and price:
            range_pct = (market.high - market.low) / price * 100.0
        atr_pct = (market.atr / price * 100.0) if market.atr and price else None

        if (atr_pct is not None and atr_pct >= 2.0) or (range_pct is not None and range_pct >= 3.0):
            volatility = "high"
            state.add_warning("volatilitas tinggi, pertimbangkan size lebih kecil")
        elif (atr_pct is not None and atr_pct <= 0.35) or (range_pct is not None and range_pct <= 0.35):
            volatility = "low"
            state.add_warning("range sempit, market rawan noise")
        else:
            volatility = "normal"

        state.features["volatility"] = volatility
