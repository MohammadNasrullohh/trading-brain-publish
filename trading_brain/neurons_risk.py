from __future__ import annotations

from .models import BrainState
from .neuron_base import Neuron, build_long_candidate, build_short_candidate, price_anchor
from .utils import clamp, risk_reward, round_price


class RiskBudgetNeuron(Neuron):
    name = "neuron16_risk_budget"

    def run(self, state: BrainState) -> None:
        style_defaults = {
            "scalping": 0.25,
            "intraday": 0.5,
            "swing": 1.0,
            "position": 1.0,
        }
        risk_percent = state.market.risk.max_risk_percent or style_defaults.get(state.market.style, 0.5)
        market_type = state.features.get("market_type")
        if market_type == "crypto" and risk_percent > 0.75:
            risk_percent -= 0.15
        if market_type == "forex" and state.features.get("session_quality") == "low":
            risk_percent -= 0.1
        if state.features.get("volatility") == "high" and risk_percent > 0.25:
            risk_percent -= 0.25
        if state.market.sentiment.headline_risk and risk_percent > 0.25:
            risk_percent -= 0.25
        risk_percent *= state.features.get("risk_cap_multiplier", 1.0)
        state.features["risk_percent"] = round(clamp(risk_percent, 0.1, 2.0), 2)


class StopLossArchitectNeuron(Neuron):
    name = "neuron17_stop_loss_architect"

    def run(self, state: BrainState) -> None:
        build_long_candidate(state)
        build_short_candidate(state)


class TargetEngineNeuron(Neuron):
    name = "neuron18_target_engine"

    def run(self, state: BrainState) -> None:
        price = price_anchor(state)
        support = state.features.get("nearest_support")
        resistance = state.features.get("nearest_resistance")
        long_plan = state.candidate_long
        short_plan = state.candidate_short

        if long_plan.valid:
            entry_mid = state.features.get("entry_mid_long", price)
            risk_per_unit = entry_mid - float(long_plan.stop_loss)
            target_1 = resistance if resistance is not None and resistance > entry_mid else entry_mid + (risk_per_unit * 1.6)
            target_2 = max(target_1 + (risk_per_unit * 0.8), entry_mid + (risk_per_unit * 2.2))
            long_plan.take_profit_1 = round_price(target_1)
            long_plan.take_profit_2 = round_price(target_2)
            long_plan.risk_reward = round(risk_reward(entry_mid, long_plan.stop_loss, long_plan.take_profit_1, "long") or 0.0, 2)

        if short_plan.valid:
            entry_mid = state.features.get("entry_mid_short", price)
            risk_per_unit = float(short_plan.stop_loss) - entry_mid
            target_1 = support if support is not None and support < entry_mid else entry_mid - (risk_per_unit * 1.6)
            target_2 = min(target_1 - (risk_per_unit * 0.8), entry_mid - (risk_per_unit * 2.2))
            short_plan.take_profit_1 = round_price(target_1)
            short_plan.take_profit_2 = round_price(target_2)
            short_plan.risk_reward = round(risk_reward(entry_mid, short_plan.stop_loss, short_plan.take_profit_1, "short") or 0.0, 2)


class NoTradeFilterNeuron(Neuron):
    name = "neuron19_no_trade_filter"

    def run(self, state: BrainState) -> None:
        max_score = max(state.long_score, state.short_score)
        score_gap = abs(state.long_score - state.short_score)
        long_rr = state.candidate_long.risk_reward or 0.0
        short_rr = state.candidate_short.risk_reward or 0.0

        if max_score < 4.0 and score_gap < 1.0:
            state.add_blocker("bias terlalu lemah untuk entry berkualitas")

        if long_rr < 1.2 and short_rr < 1.2:
            state.add_blocker("reward-to-risk kedua sisi masih kurang menarik")

        if state.features.get("volatility") == "high" and state.market.sentiment.headline_risk:
            state.add_blocker("headline risk dan volatilitas tinggi membuat setup rapuh")

        if state.features.get("session_quality") == "low" and state.market.style in {"scalping", "intraday"}:
            state.add_warning("quality session rendah, entry cepat perlu lebih selektif")

        leverage = state.market.risk.leverage or 0.0
        if leverage >= 20 and max_score < 7.0:
            state.add_blocker("leverage tinggi butuh kualitas setup yang lebih tinggi")


class EventSentimentGuardNeuron(Neuron):
    name = "neuron20_event_sentiment_guard"

    def run(self, state: BrainState) -> None:
        sentiment = state.market.sentiment
        price = price_anchor(state)
        ema_fast = state.market.indicators.ema_fast

        if sentiment.headline_risk:
            state.add_warning("ada headline risk, waspadai slippage dan spike")
        if sentiment.macro_risk:
            state.add_warning("ada macro risk, validasi setup perlu lebih ketat")

        if sentiment.score is not None:
            if sentiment.score >= 0.6 and ema_fast is not None and price >= ema_fast:
                self.boost(state, "long", 0.5, "sentimen pasar ikut mendukung skenario bullish")
            elif sentiment.score <= -0.6 and ema_fast is not None and price <= ema_fast:
                self.boost(state, "short", 0.5, "sentimen pasar ikut mendukung skenario bearish")
