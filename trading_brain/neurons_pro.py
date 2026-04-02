from __future__ import annotations

from .models import BrainState
from .neuron_base import Neuron, price_anchor
from .utils import distance_pct, normalize_label


def _parse_timeframe_to_minutes(timeframe: str | None) -> int | None:
    if not timeframe:
        return None
    raw = timeframe.strip().lower()
    if raw.endswith("m"):
        return int(float(raw[:-1]))
    if raw.endswith("h"):
        return int(float(raw[:-1]) * 60)
    if raw.endswith("d"):
        return int(float(raw[:-1]) * 1440)
    if raw.endswith("w"):
        return int(float(raw[:-1]) * 10080)
    return None


def _candle_metrics(state: BrainState) -> dict[str, float] | None:
    market = state.market
    if None in (market.open, market.high, market.low, market.close):
        return None
    high = float(market.high)
    low = float(market.low)
    if high <= low:
        return None
    open_ = float(market.open)
    close = float(market.close)
    total_range = high - low
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    return {
        "body_ratio": body / total_range,
        "upper_wick_ratio": upper_wick / total_range,
        "lower_wick_ratio": lower_wick / total_range,
        "close_position": (close - low) / total_range,
    }


class TimeframeStyleCompatibilityNeuron(Neuron):
    name = "neuron35_timeframe_style_compatibility"

    def run(self, state: BrainState) -> None:
        minutes = _parse_timeframe_to_minutes(state.market.timeframe)
        style = state.market.style
        state.features["timeframe_minutes"] = minutes
        if minutes is None:
            return

        if style == "scalping" and minutes > 15:
            state.add_warning("timeframe terlalu besar untuk scalping agresif")
        elif style == "swing" and minutes < 60:
            state.add_warning("timeframe terlalu kecil untuk swing yang tenang")
        elif style == "position" and minutes < 240:
            state.add_blocker("timeframe terlalu kecil untuk gaya position trading")


class ReferenceLevelMemoryNeuron(Neuron):
    name = "neuron36_reference_level_memory"

    def run(self, state: BrainState) -> None:
        levels = state.market.levels
        price = price_anchor(state)

        reference_map = {
            "previous_high": levels.previous_high,
            "previous_low": levels.previous_low,
            "session_high": levels.session_high,
            "session_low": levels.session_low,
        }
        state.features["reference_levels"] = reference_map

        if levels.previous_high is not None and distance_pct(price, levels.previous_high) is not None:
            if distance_pct(price, levels.previous_high) <= 0.35:
                state.add_warning("harga dekat previous high, breakout perlu follow-through yang rapi")

        if levels.previous_low is not None and distance_pct(price, levels.previous_low) is not None:
            if distance_pct(price, levels.previous_low) <= 0.35:
                state.add_warning("harga dekat previous low, breakdown perlu follow-through yang rapi")


class OrderbookPressureNeuron(Neuron):
    name = "neuron37_orderbook_pressure"

    def run(self, state: BrainState) -> None:
        micro = state.market.microstructure
        imbalance = micro.orderbook_imbalance
        liquidity_score = micro.liquidity_score
        bias = state.features.get("higher_timeframe_bias")

        if liquidity_score is not None:
            state.features["liquidity_score"] = round(liquidity_score, 2)
            if liquidity_score < 0.35:
                state.add_warning("likuiditas tipis, eksekusi bisa lebih kasar")
            if liquidity_score < 0.2 and state.market.style in {"scalping", "intraday"}:
                state.add_blocker("likuiditas terlalu tipis untuk eksekusi cepat")

        if imbalance is not None:
            state.features["orderbook_imbalance"] = round(imbalance, 3)
            if imbalance >= 0.2 and bias == "bullish":
                self.boost(state, "long", 0.4, "orderbook imbalance condong ke buyer")
            elif imbalance <= -0.2 and bias == "bearish":
                self.boost(state, "short", 0.4, "orderbook imbalance condong ke seller")


class CandleAnatomyNeuron(Neuron):
    name = "neuron38_candle_anatomy"

    def run(self, state: BrainState) -> None:
        metrics = _candle_metrics(state)
        if metrics is None:
            return

        state.features["candle_body_ratio"] = round(metrics["body_ratio"], 3)
        state.features["upper_wick_ratio"] = round(metrics["upper_wick_ratio"], 3)
        state.features["lower_wick_ratio"] = round(metrics["lower_wick_ratio"], 3)

        if metrics["body_ratio"] >= 0.55 and metrics["close_position"] >= 0.72:
            self.boost(state, "long", 0.6, "anatomi candle menunjukkan dorongan bullish bersih")
        if metrics["body_ratio"] >= 0.55 and metrics["close_position"] <= 0.28:
            self.boost(state, "short", 0.6, "anatomi candle menunjukkan dorongan bearish bersih")

        if metrics["upper_wick_ratio"] >= 0.42:
            state.add_warning("upper wick panjang menandakan rejection dari atas")
        if metrics["lower_wick_ratio"] >= 0.42:
            state.add_warning("lower wick panjang menandakan rejection dari bawah")


class BreakoutQualityNeuron(Neuron):
    name = "neuron39_breakout_quality"

    def run(self, state: BrainState) -> None:
        metrics = _candle_metrics(state)
        if metrics is None:
            return

        volume = normalize_label(state.market.indicators.volume_trend, "")
        broken_resistance = state.features.get("broken_resistance")
        broken_support = state.features.get("broken_support")
        close = state.market.close

        if broken_resistance is not None and close is not None and close > broken_resistance:
            if metrics["body_ratio"] >= 0.5 and metrics["close_position"] >= 0.7 and volume in {"rising", "strong", "expanding"}:
                self.boost(state, "long", 0.7, "breakout bullish berkualitas tinggi")
            elif metrics["upper_wick_ratio"] >= 0.35:
                state.add_warning("breakout bullish belum rapi karena ada wick atas yang besar")

        if broken_support is not None and close is not None and close < broken_support:
            if metrics["body_ratio"] >= 0.5 and metrics["close_position"] <= 0.3 and volume in {"rising", "strong", "expanding"}:
                self.boost(state, "short", 0.7, "breakdown bearish berkualitas tinggi")
            elif metrics["lower_wick_ratio"] >= 0.35:
                state.add_warning("breakdown bearish belum rapi karena ada wick bawah yang besar")


class TrapDetectionNeuron(Neuron):
    name = "neuron40_trap_detection"

    def run(self, state: BrainState) -> None:
        metrics = _candle_metrics(state)
        if metrics is None:
            return

        price = price_anchor(state)
        resistance = state.features.get("nearest_resistance")
        support = state.features.get("nearest_support")

        if resistance is not None and distance_pct(price, resistance) is not None and distance_pct(price, resistance) <= 0.3:
            if metrics["upper_wick_ratio"] >= 0.38:
                self.boost(state, "short", 0.4, "ada tanda bull trap di dekat resistance")

        if support is not None and distance_pct(price, support) is not None and distance_pct(price, support) <= 0.3:
            if metrics["lower_wick_ratio"] >= 0.38:
                self.boost(state, "long", 0.4, "ada tanda bear trap di dekat support")


class ExecutionLocationNeuron(Neuron):
    name = "neuron41_execution_location"

    def run(self, state: BrainState) -> None:
        price = price_anchor(state)
        support = state.features.get("nearest_support")
        resistance = state.features.get("nearest_resistance")
        if support is None or resistance is None or resistance <= support:
            return

        position = (price - support) / (resistance - support)
        state.features["range_position"] = round(position, 3)

        if 0.35 <= position <= 0.65:
            state.add_warning("harga berada di tengah range, lokasi entry kurang efisien")
        elif position < 0.25:
            self.boost(state, "long", 0.3, "lokasi entry dekat dasar range relatif lebih efisien")
        elif position > 0.75:
            self.boost(state, "short", 0.3, "lokasi entry dekat atas range relatif lebih efisien")


class DrawdownGuardNeuron(Neuron):
    name = "neuron42_drawdown_guard"

    def run(self, state: BrainState) -> None:
        risk = state.market.risk
        cap_multiplier = 1.0

        if risk.current_drawdown_percent is not None:
            state.features["current_drawdown_percent"] = round(risk.current_drawdown_percent, 2)
            if risk.current_drawdown_percent >= 4:
                cap_multiplier *= 0.7
                state.add_warning("akun sedang drawdown, mode risk harus lebih defensif")
            if risk.current_drawdown_percent >= 7:
                cap_multiplier *= 0.55
                state.add_blocker("drawdown akun terlalu dalam untuk setup biasa")

        state.features["risk_cap_multiplier"] = round(cap_multiplier, 2)


class RecoveryModeNeuron(Neuron):
    name = "neuron43_recovery_mode"

    def run(self, state: BrainState) -> None:
        loss_streak = state.market.risk.loss_streak
        state.features["loss_streak"] = loss_streak
        if loss_streak >= 2:
            state.add_warning("loss streak meningkat, hanya A-setup yang layak diambil")
        if loss_streak >= 3:
            state.add_blocker("loss streak tinggi, brain masuk mode recovery")


class AccountHeatNeuron(Neuron):
    name = "neuron44_account_heat"

    def run(self, state: BrainState) -> None:
        risk = state.market.risk
        if risk.current_drawdown_percent is None or risk.max_daily_loss_percent is None:
            return

        heat = risk.current_drawdown_percent / risk.max_daily_loss_percent if risk.max_daily_loss_percent > 0 else 0.0
        state.features["account_heat"] = round(heat, 2)
        if heat >= 0.75:
            state.add_warning("akun sudah mendekati batas loss harian")
        if heat >= 1.0:
            state.add_blocker("batas loss harian sudah tercapai")


class AsymmetryNeuron(Neuron):
    name = "neuron45_asymmetry"

    def run(self, state: BrainState) -> None:
        long_rr = state.candidate_long.risk_reward or 0.0
        short_rr = state.candidate_short.risk_reward or 0.0
        state.features["rr_long"] = round(long_rr, 2)
        state.features["rr_short"] = round(short_rr, 2)

        if long_rr >= 2.2:
            self.boost(state, "long", 0.5, "asimetri risk-reward long sangat sehat")
        elif 0 < long_rr < 1.4:
            state.add_warning("asimetri long belum menarik")

        if short_rr >= 2.2:
            self.boost(state, "short", 0.5, "asimetri risk-reward short sangat sehat")
        elif 0 < short_rr < 1.4:
            state.add_warning("asimetri short belum menarik")


class ConvictionCalibrationNeuron(Neuron):
    name = "neuron46_conviction_calibration"

    def run(self, state: BrainState) -> None:
        warnings = len(state.warnings)
        blockers = len(state.blockers)
        if warnings >= 4:
            state.long_score = max(0.0, state.long_score - 0.3)
            state.short_score = max(0.0, state.short_score - 0.3)
        if blockers >= 2:
            state.long_score = max(0.0, state.long_score - 0.5)
            state.short_score = max(0.0, state.short_score - 0.5)
