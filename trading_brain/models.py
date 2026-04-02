from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _float_list(values: Any) -> list[float]:
    if not values:
        return []
    cleaned = [float(value) for value in values if value not in (None, "")]
    return sorted(cleaned)


@dataclass
class LevelData:
    support: list[float] = field(default_factory=list)
    resistance: list[float] = field(default_factory=list)
    supply: list[float] = field(default_factory=list)
    demand: list[float] = field(default_factory=list)
    previous_high: float | None = None
    previous_low: float | None = None
    session_high: float | None = None
    session_low: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LevelData":
        payload = payload or {}
        return cls(
            support=_float_list(payload.get("support")),
            resistance=_float_list(payload.get("resistance")),
            supply=_float_list(payload.get("supply")),
            demand=_float_list(payload.get("demand")),
            previous_high=_float_or_none(payload.get("previous_high")),
            previous_low=_float_or_none(payload.get("previous_low")),
            session_high=_float_or_none(payload.get("session_high")),
            session_low=_float_or_none(payload.get("session_low")),
        )


@dataclass
class IndicatorData:
    ema_fast: float | None = None
    ema_slow: float | None = None
    rsi: float | None = None
    macd_histogram: float | None = None
    volume_trend: str | None = None
    vwap: float | None = None
    adx: float | None = None
    stochastic: float | None = None
    open_interest_delta: float | None = None
    funding_rate: float | None = None
    delta_volume: float | None = None
    bollinger_position: float | None = None
    cci: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "IndicatorData":
        payload = payload or {}
        return cls(
            ema_fast=_float_or_none(payload.get("ema_fast")),
            ema_slow=_float_or_none(payload.get("ema_slow")),
            rsi=_float_or_none(payload.get("rsi")),
            macd_histogram=_float_or_none(payload.get("macd_histogram")),
            volume_trend=payload.get("volume_trend"),
            vwap=_float_or_none(payload.get("vwap")),
            adx=_float_or_none(payload.get("adx")),
            stochastic=_float_or_none(payload.get("stochastic")),
            open_interest_delta=_float_or_none(payload.get("open_interest_delta")),
            funding_rate=_float_or_none(payload.get("funding_rate")),
            delta_volume=_float_or_none(payload.get("delta_volume")),
            bollinger_position=_float_or_none(payload.get("bollinger_position")),
            cci=_float_or_none(payload.get("cci")),
        )


@dataclass
class RiskConfig:
    max_risk_percent: float | None = None
    leverage: float | None = None
    current_drawdown_percent: float | None = None
    max_daily_loss_percent: float | None = None
    loss_streak: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RiskConfig":
        payload = payload or {}
        return cls(
            max_risk_percent=_float_or_none(payload.get("max_risk_percent")),
            leverage=_float_or_none(payload.get("leverage")),
            current_drawdown_percent=_float_or_none(payload.get("current_drawdown_percent")),
            max_daily_loss_percent=_float_or_none(payload.get("max_daily_loss_percent")),
            loss_streak=int(payload.get("loss_streak", 0) or 0),
        )


@dataclass
class SentimentData:
    score: float | None = None
    headline_risk: bool = False
    correlation_bias: float | None = None
    macro_risk: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SentimentData":
        payload = payload or {}
        return cls(
            score=_float_or_none(payload.get("score")),
            headline_risk=bool(payload.get("headline_risk", False)),
            correlation_bias=_float_or_none(payload.get("correlation_bias")),
            macro_risk=bool(payload.get("macro_risk", False)),
        )


@dataclass
class ContextHints:
    regime_hint: str | None = None
    structure_hint: str | None = None
    bias_hint: str | None = None
    session_hint: str | None = None
    market_type_hint: str | None = None
    session_quality_hint: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ContextHints":
        payload = payload or {}
        return cls(
            regime_hint=payload.get("regime_hint"),
            structure_hint=payload.get("structure_hint"),
            bias_hint=payload.get("bias_hint"),
            session_hint=payload.get("session_hint"),
            market_type_hint=payload.get("market_type_hint"),
            session_quality_hint=payload.get("session_quality_hint"),
        )


@dataclass
class MicrostructureData:
    spread: float | None = None
    fee_bps: float | None = None
    slippage_bps: float | None = None
    weekend: bool = False
    liquidity_score: float | None = None
    orderbook_imbalance: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MicrostructureData":
        payload = payload or {}
        return cls(
            spread=_float_or_none(payload.get("spread")),
            fee_bps=_float_or_none(payload.get("fee_bps")),
            slippage_bps=_float_or_none(payload.get("slippage_bps")),
            weekend=bool(payload.get("weekend", False)),
            liquidity_score=_float_or_none(payload.get("liquidity_score")),
            orderbook_imbalance=_float_or_none(payload.get("orderbook_imbalance")),
        )


@dataclass
class MarketInput:
    symbol: str | None = None
    timeframe: str | None = None
    style: str = "intraday"
    market_type: str = "auto"
    session: str | None = None
    price: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    atr: float | None = None
    levels: LevelData = field(default_factory=LevelData)
    indicators: IndicatorData = field(default_factory=IndicatorData)
    risk: RiskConfig = field(default_factory=RiskConfig)
    sentiment: SentimentData = field(default_factory=SentimentData)
    context: ContextHints = field(default_factory=ContextHints)
    microstructure: MicrostructureData = field(default_factory=MicrostructureData)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarketInput":
        return cls(
            symbol=payload.get("symbol"),
            timeframe=payload.get("timeframe"),
            style=(payload.get("style") or "intraday").lower(),
            market_type=(payload.get("market_type") or "auto").lower(),
            session=(payload.get("session") or None),
            price=_float_or_none(payload.get("price")),
            open=_float_or_none(payload.get("open")),
            high=_float_or_none(payload.get("high")),
            low=_float_or_none(payload.get("low")),
            close=_float_or_none(payload.get("close")),
            atr=_float_or_none(payload.get("atr")),
            levels=LevelData.from_dict(payload.get("levels")),
            indicators=IndicatorData.from_dict(payload.get("indicators")),
            risk=RiskConfig.from_dict(payload.get("risk")),
            sentiment=SentimentData.from_dict(payload.get("sentiment")),
            context=ContextHints.from_dict(payload.get("context")),
            microstructure=MicrostructureData.from_dict(payload.get("microstructure")),
        )


@dataclass
class TradePlan:
    direction: str = "none"
    setup_type: str = "none"
    entry_zone: list[float] = field(default_factory=list)
    stop_loss: float | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    risk_reward: float | None = None
    valid: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "setup_type": self.setup_type,
            "entry_zone": self.entry_zone,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "risk_reward": self.risk_reward,
            "valid": self.valid,
        }


@dataclass
class BrainState:
    market: MarketInput
    features: dict[str, Any] = field(default_factory=dict)
    long_score: float = 0.0
    short_score: float = 0.0
    reasons_long: list[str] = field(default_factory=list)
    reasons_short: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    candidate_long: TradePlan = field(default_factory=lambda: TradePlan(direction="long"))
    candidate_short: TradePlan = field(default_factory=lambda: TradePlan(direction="short"))
    selected_plan: TradePlan = field(default_factory=TradePlan)
    bias: str = "netral"
    verdict: str = "WAIT"
    confidence: float = 0.0
    output: dict[str, Any] = field(default_factory=dict)

    def add_reason(self, direction: str, message: str) -> None:
        if direction == "long" and message not in self.reasons_long:
            self.reasons_long.append(message)
        if direction == "short" and message not in self.reasons_short:
            self.reasons_short.append(message)

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def add_blocker(self, message: str) -> None:
        if message not in self.blockers:
            self.blockers.append(message)
