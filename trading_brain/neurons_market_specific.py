from __future__ import annotations

from .models import BrainState
from .neuron_base import Neuron, price_anchor
from .utils import distance_pct, normalize_label


def _infer_market_type(symbol: str | None) -> str:
    if not symbol:
        return "unknown"
    upper = symbol.upper()
    forex_quotes = ("USD", "JPY", "EUR", "GBP", "AUD", "NZD", "CHF", "CAD")
    if upper in {"WTI", "USOIL", "CL.F"}:
        return "commodity"
    if upper.endswith("USDT") or upper.endswith("USDTPERP") or upper.endswith("PERP"):
        return "crypto"
    if any(upper.endswith(quote) for quote in forex_quotes) and len(upper) <= 8:
        return "forex"
    return "unknown"


class AssetClassifierNeuron(Neuron):
    name = "neuron04_asset_classifier"

    def run(self, state: BrainState) -> None:
        hint = normalize_label(state.market.context.market_type_hint, "")
        market_type = normalize_label(state.market.market_type, "")

        if market_type in {"crypto", "forex", "commodity"}:
            detected = market_type
        elif hint in {"crypto", "forex", "commodity"}:
            detected = hint
        else:
            detected = _infer_market_type(state.market.symbol)

        if detected == "unknown":
            symbol = (state.market.symbol or "").upper()
            if symbol in {"WTI", "USOIL", "CL.F"}:
                detected = "commodity"
            else:
                detected = "crypto" if "USDT" in symbol else "forex"
            state.add_warning("market type tidak jelas, engine membuat asumsi otomatis")

        state.features["market_type"] = detected
        if detected == "crypto":
            asset_profile = "twenty_four_seven"
        elif detected == "commodity":
            asset_profile = "macro_driven"
        else:
            asset_profile = "session_driven"
        state.features["asset_profile"] = asset_profile


class SessionContextNeuron(Neuron):
    name = "neuron05_session_context"

    def run(self, state: BrainState) -> None:
        market_type = state.features.get("market_type", "unknown")
        session = normalize_label(state.market.session, "")
        if not session:
            session = normalize_label(state.market.context.session_hint, "")

        if not session:
            session = "rolling" if market_type == "crypto" else "unknown"

        if market_type == "forex":
            if session in {"london", "new_york", "overlap"}:
                session_quality = "high"
            elif session in {"asia"}:
                session_quality = "medium"
            else:
                session_quality = "low"
        elif market_type == "commodity":
            if session in {"us", "new_york", "overlap", "london"}:
                session_quality = "high"
            elif session in {"asia"}:
                session_quality = "medium"
            else:
                session_quality = "low"
        else:
            if state.market.microstructure.weekend:
                session_quality = "low"
            elif session in {"us", "europe", "london", "new_york"}:
                session_quality = "high"
            else:
                session_quality = "medium"

        hint = normalize_label(state.market.context.session_quality_hint, "")
        if hint in {"high", "medium", "low"}:
            session_quality = hint

        state.features["session"] = session
        state.features["session_quality"] = session_quality


class TrendStrengthNeuron(Neuron):
    name = "neuron13_trend_strength"

    def run(self, state: BrainState) -> None:
        market = state.market
        price = price_anchor(state)
        adx = market.indicators.adx
        ema_fast = market.indicators.ema_fast
        ema_slow = market.indicators.ema_slow
        structure = state.features.get("structure")

        if adx is not None:
            if adx >= 25 and structure == "bullish":
                self.boost(state, "long", 0.8, "ADX menunjukkan trend bullish cukup kuat")
            elif adx >= 25 and structure == "bearish":
                self.boost(state, "short", 0.8, "ADX menunjukkan trend bearish cukup kuat")
            elif adx < 18:
                state.add_warning("trend strength rendah, market bisa mudah whipsaw")

        if price and ema_fast is not None and ema_slow is not None:
            spread_pct = abs(ema_fast - ema_slow) / price * 100.0
            state.features["ema_spread_pct"] = round(spread_pct, 3)
            if spread_pct >= 0.2 and ema_fast > ema_slow:
                self.boost(state, "long", 0.4, "jarak EMA mendukung continuation bullish")
            elif spread_pct >= 0.2 and ema_fast < ema_slow:
                self.boost(state, "short", 0.4, "jarak EMA mendukung continuation bearish")


class MeanReversionPressureNeuron(Neuron):
    name = "neuron14_mean_reversion_pressure"

    def run(self, state: BrainState) -> None:
        market = state.market
        price = price_anchor(state)
        rsi = market.indicators.rsi
        vwap = market.indicators.vwap
        stochastic = market.indicators.stochastic

        stretch_from_vwap = None
        if price and vwap is not None:
            stretch_from_vwap = abs(price - vwap) / price * 100.0
            state.features["stretch_from_vwap_pct"] = round(stretch_from_vwap, 3)

        if rsi is not None and rsi >= 72:
            state.add_warning("tekanan mean reversion bearish meningkat karena RSI terlalu tinggi")
        if rsi is not None and rsi <= 28:
            state.add_warning("tekanan mean reversion bullish meningkat karena RSI terlalu rendah")

        if stochastic is not None:
            if stochastic >= 85:
                state.add_warning("stochastic tinggi, follow-through naik bisa melambat")
            elif stochastic <= 15:
                state.add_warning("stochastic rendah, follow-through turun bisa melambat")

        if stretch_from_vwap is not None and stretch_from_vwap >= 1.2:
            state.add_warning("harga terlalu jauh dari VWAP, entry kejar harga lebih berisiko")


class ForexSessionQualityNeuron(Neuron):
    name = "neuron15_forex_session_quality"

    def run(self, state: BrainState) -> None:
        if state.features.get("market_type") != "forex":
            return

        session = state.features.get("session")
        quality = state.features.get("session_quality")
        style = state.market.style

        if quality == "high":
            if style in {"scalping", "intraday"}:
                self.boost(state, "long", 0.2)
                self.boost(state, "short", 0.2)
        elif quality == "low":
            state.add_warning("session forex sepi, breakout cenderung kurang bersih")
            if style in {"scalping", "intraday"}:
                state.add_blocker("intraday forex di luar session aktif lebih rawan false move")

        if session == "overlap":
            self.boost(state, "long", 0.2, "session overlap memberi likuiditas yang sehat")
            self.boost(state, "short", 0.2, "session overlap memberi likuiditas yang sehat")


class CryptoMicrostructureNeuron(Neuron):
    name = "neuron16_crypto_microstructure"

    def run(self, state: BrainState) -> None:
        if state.features.get("market_type") != "crypto":
            return

        indicators = state.market.indicators
        weekend = state.market.microstructure.weekend

        if weekend:
            state.add_warning("weekend crypto sering lebih tipis dan rawan spike")

        if indicators.funding_rate is not None:
            if indicators.funding_rate >= 0.04:
                state.add_warning("funding rate cukup panas, long crowded perlu diwaspadai")
            elif indicators.funding_rate <= -0.04:
                state.add_warning("funding rate negatif ekstrem, short crowded perlu diwaspadai")

        if indicators.open_interest_delta is not None:
            if indicators.open_interest_delta > 0 and state.features.get("higher_timeframe_bias") == "bullish":
                self.boost(state, "long", 0.4, "open interest ikut naik bersama bias bullish")
            elif indicators.open_interest_delta > 0 and state.features.get("higher_timeframe_bias") == "bearish":
                self.boost(state, "short", 0.4, "open interest ikut naik bersama bias bearish")
            elif indicators.open_interest_delta < 0:
                state.add_warning("open interest menurun, tenaga continuation bisa melemah")


class SpreadSlippageGuardNeuron(Neuron):
    name = "neuron17_spread_slippage_guard"

    def run(self, state: BrainState) -> None:
        micro = state.market.microstructure
        atr = state.market.atr
        price = price_anchor(state)
        spread = micro.spread

        spread_pct = None
        if price and spread is not None:
            spread_pct = spread / price * 100.0
            state.features["spread_pct"] = round(spread_pct, 4)

        friction_bps = (micro.fee_bps or 0.0) + (micro.slippage_bps or 0.0)
        state.features["friction_bps"] = round(friction_bps, 2)

        if spread_pct is not None and spread_pct >= 0.08:
            state.add_warning("spread cukup lebar untuk timeframe cepat")
        if atr is not None and spread is not None and atr > 0 and (spread / atr) >= 0.2:
            state.add_blocker("spread terlalu besar dibanding range ATR")
        if friction_bps >= 20:
            state.add_warning("biaya eksekusi cukup berat, target dekat jadi kurang menarik")


class TrendContinuationNeuron(Neuron):
    name = "neuron22_trend_continuation"

    def run(self, state: BrainState) -> None:
        regime = state.features.get("regime")
        bias = state.features.get("higher_timeframe_bias")
        structure = state.features.get("structure")
        volume = normalize_label(state.market.indicators.volume_trend, "")

        if regime == "trending" and bias == "bullish" and structure == "bullish" and volume in {"rising", "strong", "expanding"}:
            self.boost(state, "long", 0.8, "trend continuation bullish didukung regime dan volume")
        if regime == "trending" and bias == "bearish" and structure == "bearish" and volume in {"rising", "strong", "expanding"}:
            self.boost(state, "short", 0.8, "trend continuation bearish didukung regime dan volume")


class RangeFadeNeuron(Neuron):
    name = "neuron23_range_fade"

    def run(self, state: BrainState) -> None:
        if state.features.get("regime") != "ranging":
            return

        price = price_anchor(state)
        resistance = state.features.get("nearest_resistance")
        support = state.features.get("nearest_support")
        rsi = state.market.indicators.rsi

        if resistance is not None:
            distance_to_resistance = distance_pct(price, resistance)
            if distance_to_resistance is not None and distance_to_resistance <= 0.3 and rsi is not None and rsi >= 60:
                state.features["setup_type_short"] = "snr resistance zone"
                self.boost(state, "short", 0.6, "range fade short terbuka di dekat resistance")

        if support is not None:
            distance_to_support = distance_pct(price, support)
            if distance_to_support is not None and distance_to_support <= 0.3 and rsi is not None and rsi <= 40:
                state.features["setup_type_long"] = "snr support zone"
                self.boost(state, "long", 0.6, "range fade long terbuka di dekat support")


class PositionSizingNeuron(Neuron):
    name = "neuron26_position_sizing"

    def run(self, state: BrainState) -> None:
        risk_percent = state.features.get("risk_percent", 0.5)
        market_type = state.features.get("market_type")
        style = state.market.style

        sizing = 1.0
        if style == "scalping":
            sizing *= 0.85
        if state.features.get("volatility") == "high":
            sizing *= 0.7
        if state.features.get("session_quality") == "low":
            sizing *= 0.8
        if market_type == "crypto" and state.market.microstructure.weekend:
            sizing *= 0.75

        sizing = round(max(0.25, min(sizing, 1.0)), 2)
        state.features["size_multiplier"] = sizing
        state.features["effective_risk_percent"] = round(risk_percent * sizing, 2)


class LeverageGuardNeuron(Neuron):
    name = "neuron27_leverage_guard"

    def run(self, state: BrainState) -> None:
        leverage = state.market.risk.leverage
        market_type = state.features.get("market_type")
        if leverage is None:
            state.features["leverage_profile"] = "spot_or_unlevered"
            return

        if leverage >= 10:
            state.add_warning("leverage tinggi mempersempit ruang salah")
        if market_type == "crypto" and leverage >= 15:
            state.add_blocker("leverage crypto terlalu agresif untuk brain konservatif ini")
        if market_type == "forex" and leverage >= 30:
            state.add_warning("leverage forex tinggi, noise kecil bisa jadi mahal")
        if market_type == "commodity" and leverage >= 15:
            state.add_warning("leverage commodity tinggi, headline energi bisa memicu spike tajam")

        state.features["leverage_profile"] = "leveraged"


class CorrelationContextNeuron(Neuron):
    name = "neuron31_correlation_context"

    def run(self, state: BrainState) -> None:
        correlation_bias = state.market.sentiment.correlation_bias
        bias = state.features.get("higher_timeframe_bias")
        if correlation_bias is None:
            return

        if correlation_bias >= 0.6 and bias == "bullish":
            self.boost(state, "long", 0.4, "konteks korelasi mendukung bias bullish")
        elif correlation_bias <= -0.6 and bias == "bearish":
            self.boost(state, "short", 0.4, "konteks korelasi mendukung bias bearish")
        elif abs(correlation_bias) >= 0.7 and bias == "netral":
            state.add_warning("korelasi market kuat tetapi bias chart lokal belum bersih")
