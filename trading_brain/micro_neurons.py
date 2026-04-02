from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import BrainState
from .neuron_base import Neuron


Condition = Callable[[BrainState], bool]
ValueGetter = Callable[[BrainState], float | int | None]


class RuleNeuron(Neuron):
    def __init__(
        self,
        name: str,
        title: str,
        visual_group: str,
        visual_stage: str,
        description: str,
        condition: Condition,
        effect: str,
        message: str,
        amount: float = 0.0,
        direction: str | None = None,
    ) -> None:
        self.name = name
        self.title = title
        self.visual_group = visual_group
        self.visual_stage = visual_stage
        self.description = description
        self.condition = condition
        self.effect = effect
        self.message = message
        self.amount = amount
        self.direction = direction

    def run(self, state: BrainState) -> None:
        if not self.condition(state):
            return
        if self.effect == "boost" and self.direction:
            self.boost(state, self.direction, self.amount, self.message)
        elif self.effect == "warn":
            state.add_warning(self.message)
        elif self.effect == "block":
            state.add_blocker(self.message)


def _bias(state: BrainState) -> str:
    return str(state.features.get("higher_timeframe_bias", "netral"))


def _structure(state: BrainState) -> str:
    return str(state.features.get("structure", "netral"))


def _bullish_context(state: BrainState) -> bool:
    return _bias(state) == "bullish" or _structure(state) == "bullish"


def _bearish_context(state: BrainState) -> bool:
    return _bias(state) == "bearish" or _structure(state) == "bearish"


def _get_rsi(state: BrainState) -> float | None:
    return state.market.indicators.rsi


def _get_adx(state: BrainState) -> float | None:
    return state.market.indicators.adx


def _get_ema_spread(state: BrainState) -> float | None:
    value = state.features.get("ema_spread_pct")
    return float(value) if value is not None else None


def _get_open_interest(state: BrainState) -> float | None:
    return state.market.indicators.open_interest_delta


def _get_funding(state: BrainState) -> float | None:
    return state.market.indicators.funding_rate


def _get_stochastic(state: BrainState) -> float | None:
    return state.market.indicators.stochastic


def _get_delta_volume(state: BrainState) -> float | None:
    return state.market.indicators.delta_volume


def _get_bollinger_position(state: BrainState) -> float | None:
    return state.market.indicators.bollinger_position


def _get_cci(state: BrainState) -> float | None:
    return state.market.indicators.cci


def _get_spread_pct(state: BrainState) -> float | None:
    value = state.features.get("spread_pct")
    return float(value) if value is not None else None


def _get_liquidity_score(state: BrainState) -> float | None:
    value = state.features.get("liquidity_score")
    if value is not None:
        return float(value)
    return state.market.microstructure.liquidity_score


def _get_orderbook_imbalance(state: BrainState) -> float | None:
    value = state.features.get("orderbook_imbalance")
    if value is not None:
        return float(value)
    return state.market.microstructure.orderbook_imbalance


def _get_correlation_bias(state: BrainState) -> float | None:
    return state.market.sentiment.correlation_bias


def _get_account_heat(state: BrainState) -> float | None:
    value = state.features.get("account_heat")
    return float(value) if value is not None else None


def _get_drawdown(state: BrainState) -> float | None:
    value = state.features.get("current_drawdown_percent")
    return float(value) if value is not None else None


def _get_loss_streak(state: BrainState) -> int | None:
    value = state.features.get("loss_streak")
    return int(value) if value is not None else None


def _get_leverage(state: BrainState) -> float | None:
    return state.market.risk.leverage


def _get_rr_long(state: BrainState) -> float | None:
    return state.candidate_long.risk_reward


def _get_rr_short(state: BrainState) -> float | None:
    return state.candidate_short.risk_reward


def _get_range_position(state: BrainState) -> float | None:
    value = state.features.get("range_position")
    return float(value) if value is not None else None


def _get_body_ratio(state: BrainState) -> float | None:
    value = state.features.get("candle_body_ratio")
    return float(value) if value is not None else None


def _get_upper_wick(state: BrainState) -> float | None:
    value = state.features.get("upper_wick_ratio")
    return float(value) if value is not None else None


def _get_lower_wick(state: BrainState) -> float | None:
    value = state.features.get("lower_wick_ratio")
    return float(value) if value is not None else None


def _numeric_min_condition(
    getter: ValueGetter,
    threshold: float,
    context: Condition | None = None,
) -> Condition:
    def condition(state: BrainState) -> bool:
        value = getter(state)
        if value is None:
            return False
        if context is not None and not context(state):
            return False
        return float(value) >= threshold

    return condition


def _numeric_max_condition(
    getter: ValueGetter,
    threshold: float,
    context: Condition | None = None,
) -> Condition:
    def condition(state: BrainState) -> bool:
        value = getter(state)
        if value is None:
            return False
        if context is not None and not context(state):
            return False
        return float(value) <= threshold

    return condition


def _range_condition(getter: ValueGetter, minimum: float, maximum: float) -> Condition:
    def condition(state: BrainState) -> bool:
        value = getter(state)
        if value is None:
            return False
        numeric = float(value)
        return minimum <= numeric <= maximum

    return condition


def _build_micro_neurons() -> list[RuleNeuron]:
    neurons: list[RuleNeuron] = []

    for threshold, amount in [(52, 0.05), (55, 0.06), (58, 0.07), (60, 0.08), (63, 0.09)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_rsi_bull_{threshold}",
                title=f"RSI Bull {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description=f"Micro check RSI bullish di atas {threshold}.",
                condition=_numeric_min_condition(_get_rsi, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"RSI micro bullish melewati {threshold}",
            )
        )

    for threshold, amount in [(48, 0.05), (45, 0.06), (42, 0.07), (40, 0.08), (37, 0.09)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_rsi_bear_{threshold}",
                title=f"RSI Bear {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description=f"Micro check RSI bearish di bawah {threshold}.",
                condition=_numeric_max_condition(_get_rsi, threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"RSI micro bearish turun ke {threshold}",
            )
        )

    for threshold, amount in [(18, 0.04), (22, 0.05), (25, 0.06), (30, 0.08)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_adx_bull_{threshold}",
                title=f"ADX Bull {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description=f"Micro check kekuatan trend bullish dengan ADX di atas {threshold}.",
                condition=_numeric_min_condition(_get_adx, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"ADX micro menguatkan trend bullish di atas {threshold}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_adx_bear_{threshold}",
                title=f"ADX Bear {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description=f"Micro check kekuatan trend bearish dengan ADX di atas {threshold}.",
                condition=_numeric_min_condition(_get_adx, threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"ADX micro menguatkan trend bearish di atas {threshold}",
            )
        )

    for threshold, amount in [(0.08, 0.04), (0.12, 0.05), (0.18, 0.06), (0.25, 0.08)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_ema_spread_bull_{str(threshold).replace('.', '_')}",
                title=f"EMA Spread Bull {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro check pelebaran EMA untuk continuation bullish.",
                condition=_numeric_min_condition(_get_ema_spread, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"EMA spread micro bullish di atas {threshold:.2f}%",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_ema_spread_bear_{str(threshold).replace('.', '_')}",
                title=f"EMA Spread Bear {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro check pelebaran EMA untuk continuation bearish.",
                condition=_numeric_min_condition(_get_ema_spread, threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"EMA spread micro bearish di atas {threshold:.2f}%",
            )
        )

    for threshold, amount in [(0.3, 0.04), (1.0, 0.05), (2.0, 0.06)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_oi_bull_{str(threshold).replace('.', '_')}",
                title=f"OI Bull {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro check open interest yang naik bersama bias bullish.",
                condition=_numeric_min_condition(_get_open_interest, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"Open interest micro mendukung bullish di atas {threshold}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_oi_bear_{str(threshold).replace('.', '_')}",
                title=f"OI Bear {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro check open interest yang naik bersama bias bearish.",
                condition=_numeric_min_condition(_get_open_interest, threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"Open interest micro mendukung bearish di atas {threshold}",
            )
        )

    for threshold, amount in [(56, 0.04), (64, 0.05), (72, 0.06)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_stoch_bull_{threshold}",
                title=f"Stoch Bull {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro check stochastic saat continuation bullish mulai hidup.",
                condition=_numeric_min_condition(_get_stochastic, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"Stochastic micro bullish melewati {threshold}",
            )
        )

    for threshold, amount in [(44, 0.04), (36, 0.05), (28, 0.06)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_stoch_bear_{threshold}",
                title=f"Stoch Bear {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro check stochastic saat continuation bearish mulai hidup.",
                condition=_numeric_max_condition(_get_stochastic, threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"Stochastic micro bearish turun ke {threshold}",
            )
        )

    for threshold in [88, 12]:
        neurons.append(
            RuleNeuron(
                name=f"micro_stoch_extreme_{str(threshold).replace('.', '_')}",
                title=f"Stoch Extreme {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro warning saat stochastic masuk ke area ekstrem.",
                condition=_numeric_min_condition(_get_stochastic, threshold) if threshold > 50 else _numeric_max_condition(_get_stochastic, threshold),
                effect="warn",
                message=f"Stochastic micro berada di area ekstrem {threshold}",
            )
        )

    for threshold in [0.02, 0.04, 0.06]:
        neurons.append(
            RuleNeuron(
                name=f"micro_funding_hot_pos_{str(threshold).replace('.', '_')}",
                title=f"Funding Hot +{threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro warning saat funding rate positif mulai terlalu panas.",
                condition=_numeric_min_condition(_get_funding, threshold),
                effect="warn",
                message=f"Funding micro panas di atas +{threshold:.2f}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_funding_hot_neg_{str(threshold).replace('.', '_')}",
                title=f"Funding Hot -{threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro warning saat funding rate negatif mulai terlalu panas.",
                condition=_numeric_max_condition(_get_funding, -threshold),
                effect="warn",
                message=f"Funding micro panas di bawah -{threshold:.2f}",
            )
        )

    for threshold in [0.02, 0.04, 0.08]:
        neurons.append(
            RuleNeuron(
                name=f"micro_spread_warn_{str(threshold).replace('.', '_')}",
                title=f"Spread Warn {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro warning ketika spread mulai terlalu lebar untuk precision entry.",
                condition=_numeric_min_condition(_get_spread_pct, threshold),
                effect="warn",
                message=f"Spread micro melebar di atas {threshold:.2f}%",
            )
        )

    for threshold in [0.5, 0.35, 0.2]:
        neurons.append(
            RuleNeuron(
                name=f"micro_liquidity_low_{str(threshold).replace('.', '_')}",
                title=f"Liquidity Low {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro warning saat skor likuiditas mulai menipis.",
                condition=_numeric_max_condition(_get_liquidity_score, threshold),
                effect="warn",
                message=f"Likuiditas micro turun ke bawah {threshold:.2f}",
            )
        )

    for threshold, amount in [(0.55, 0.03), (0.7, 0.04), (0.85, 0.05)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_liquidity_high_bull_{str(threshold).replace('.', '_')}",
                title=f"Liquidity Bull {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro boost saat likuiditas sehat mendukung eksekusi bullish.",
                condition=_numeric_min_condition(_get_liquidity_score, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"Likuiditas micro sehat untuk bullish di atas {threshold:.2f}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_liquidity_high_bear_{str(threshold).replace('.', '_')}",
                title=f"Liquidity Bear {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro boost saat likuiditas sehat mendukung eksekusi bearish.",
                condition=_numeric_min_condition(_get_liquidity_score, threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"Likuiditas micro sehat untuk bearish di atas {threshold:.2f}",
            )
        )

    for threshold, amount in [(0.12, 0.04), (0.22, 0.05)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_orderbook_bull_{str(threshold).replace('.', '_')}",
                title=f"Book Bull {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro boost saat orderbook condong ke buyer.",
                condition=_numeric_min_condition(_get_orderbook_imbalance, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"Orderbook micro condong ke buyer di atas {threshold:.2f}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_orderbook_bear_{str(threshold).replace('.', '_')}",
                title=f"Book Bear {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro boost saat orderbook condong ke seller.",
                condition=_numeric_max_condition(_get_orderbook_imbalance, -threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"Orderbook micro condong ke seller di bawah -{threshold:.2f}",
            )
        )

    for threshold, amount in [(0.62, 0.04), (0.72, 0.05), (0.82, 0.06)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_boll_bull_{str(threshold).replace('.', '_')}",
                title=f"Boll Bull {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro boost saat posisi band mendukung continuation bullish.",
                condition=_numeric_min_condition(_get_bollinger_position, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"Bollinger micro mendukung bullish di atas {threshold:.2f}",
            )
        )

    for threshold, amount in [(0.38, 0.04), (0.28, 0.05), (0.18, 0.06)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_boll_bear_{str(threshold).replace('.', '_')}",
                title=f"Boll Bear {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro boost saat posisi band mendukung continuation bearish.",
                condition=_numeric_max_condition(_get_bollinger_position, threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"Bollinger micro mendukung bearish di bawah {threshold:.2f}",
            )
        )

    for threshold, amount in [(80, 0.04), (120, 0.05), (160, 0.06)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_cci_bull_{threshold}",
                title=f"CCI Bull {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro boost saat impuls CCI bullish makin nyata.",
                condition=_numeric_min_condition(_get_cci, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"CCI micro bullish melewati {threshold}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_cci_bear_{threshold}",
                title=f"CCI Bear {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro boost saat impuls CCI bearish makin nyata.",
                condition=_numeric_max_condition(_get_cci, -threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"CCI micro bearish turun ke -{threshold}",
            )
        )

    for threshold, amount in [(0.12, 0.04), (0.25, 0.05), (0.4, 0.06)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_delta_bull_{str(threshold).replace('.', '_')}",
                title=f"Delta Bull {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro boost saat delta volume buyer makin dominan.",
                condition=_numeric_min_condition(_get_delta_volume, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"Delta volume micro bullish di atas {threshold:.2f}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_delta_bear_{str(threshold).replace('.', '_')}",
                title=f"Delta Bear {threshold}",
                visual_group="signal",
                visual_stage="signal",
                description="Micro boost saat delta volume seller makin dominan.",
                condition=_numeric_max_condition(_get_delta_volume, -threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"Delta volume micro bearish di bawah -{threshold:.2f}",
            )
        )

    for threshold, amount in [(0.12, 0.03), (0.24, 0.04)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_corr_bull_{str(threshold).replace('.', '_')}",
                title=f"Corr Bull {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro boost saat korelasi lintas aset memberi tailwind bullish.",
                condition=_numeric_min_condition(_get_correlation_bias, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"Korelasi micro memberi tailwind bullish di atas {threshold:.2f}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_corr_bear_{str(threshold).replace('.', '_')}",
                title=f"Corr Bear {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro boost saat korelasi lintas aset memberi tailwind bearish.",
                condition=_numeric_max_condition(_get_correlation_bias, -threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"Korelasi micro memberi tailwind bearish di bawah -{threshold:.2f}",
            )
        )

    for threshold in [0.5, 0.75, 1.0]:
        neurons.append(
            RuleNeuron(
                name=f"micro_account_heat_{str(threshold).replace('.', '_')}",
                title=f"Account Heat {threshold}",
                visual_group="risk_micro",
                visual_stage="risk_micro",
                description="Micro warning saat akun mulai panas terhadap batas loss harian.",
                condition=_numeric_min_condition(_get_account_heat, threshold),
                effect="warn",
                message=f"Account heat micro mencapai {threshold:.2f}",
            )
        )

    for threshold in [2.0, 4.0, 6.0]:
        neurons.append(
            RuleNeuron(
                name=f"micro_drawdown_{str(threshold).replace('.', '_')}",
                title=f"Drawdown {threshold}",
                visual_group="risk_micro",
                visual_stage="risk_micro",
                description="Micro warning saat drawdown akun makin dalam.",
                condition=_numeric_min_condition(_get_drawdown, threshold),
                effect="warn",
                message=f"Drawdown micro melewati {threshold:.1f}%",
            )
        )

    for threshold in [1, 2, 3]:
        neurons.append(
            RuleNeuron(
                name=f"micro_loss_streak_{threshold}",
                title=f"Loss Streak {threshold}",
                visual_group="risk_micro",
                visual_stage="risk_micro",
                description="Micro warning saat loss streak bertambah.",
                condition=_numeric_min_condition(_get_loss_streak, threshold),
                effect="warn",
                message=f"Loss streak micro mencapai {threshold}",
            )
        )

    for threshold in [5.0, 10.0, 15.0, 20.0]:
        neurons.append(
            RuleNeuron(
                name=f"micro_leverage_{str(threshold).replace('.', '_')}",
                title=f"Leverage {threshold}",
                visual_group="risk_micro",
                visual_stage="risk_micro",
                description="Micro warning saat leverage makin menuntut presisi yang lebih tinggi.",
                condition=_numeric_min_condition(_get_leverage, threshold),
                effect="warn",
                message=f"Leverage micro melewati {threshold:.0f}x",
            )
        )

    for threshold, amount in [(1.4, 0.04), (1.8, 0.05), (2.2, 0.06), (2.8, 0.07)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_rr_long_{str(threshold).replace('.', '_')}",
                title=f"RR Long {threshold}",
                visual_group="plan_micro",
                visual_stage="plan_micro",
                description="Micro boost saat reward-to-risk long cukup sehat.",
                condition=_numeric_min_condition(_get_rr_long, threshold),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"RR long micro sehat di atas {threshold:.1f}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_rr_short_{str(threshold).replace('.', '_')}",
                title=f"RR Short {threshold}",
                visual_group="plan_micro",
                visual_stage="plan_micro",
                description="Micro boost saat reward-to-risk short cukup sehat.",
                condition=_numeric_min_condition(_get_rr_short, threshold),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"RR short micro sehat di atas {threshold:.1f}",
            )
        )

    for threshold, amount in [(0.2, 0.04), (0.15, 0.05), (0.1, 0.06)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_range_long_{str(threshold).replace('.', '_')}",
                title=f"Range Long {threshold}",
                visual_group="plan_micro",
                visual_stage="plan_micro",
                description="Micro boost saat harga dekat dasar range dan cocok untuk long.",
                condition=_numeric_max_condition(_get_range_position, threshold),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"Posisi range micro efisien untuk long di bawah {threshold:.2f}",
            )
        )

    for threshold, amount in [(0.8, 0.04), (0.85, 0.05), (0.9, 0.06)]:
        def condition_factory(limit: float) -> Condition:
            return _numeric_min_condition(_get_range_position, limit)

        neurons.append(
            RuleNeuron(
                name=f"micro_range_short_{str(threshold).replace('.', '_')}",
                title=f"Range Short {threshold}",
                visual_group="plan_micro",
                visual_stage="plan_micro",
                description="Micro boost saat harga dekat puncak range dan cocok untuk short.",
                condition=condition_factory(threshold),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"Posisi range micro efisien untuk short di atas {threshold:.2f}",
            )
        )

    for minimum, maximum in [(0.45, 0.55), (0.4, 0.6)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_midrange_warn_{str(minimum).replace('.', '_')}_{str(maximum).replace('.', '_')}",
                title=f"Midrange {minimum}-{maximum}",
                visual_group="plan_micro",
                visual_stage="plan_micro",
                description="Micro warning saat harga terlalu tengah dan lokasi entry jadi kurang efisien.",
                condition=_range_condition(_get_range_position, minimum, maximum),
                effect="warn",
                message=f"Range position micro terlalu tengah di area {minimum:.2f}-{maximum:.2f}",
            )
        )

    for threshold, amount in [(0.45, 0.04), (0.55, 0.05), (0.65, 0.06)]:
        neurons.append(
            RuleNeuron(
                name=f"micro_body_bull_{str(threshold).replace('.', '_')}",
                title=f"Body Bull {threshold}",
                visual_group="plan_micro",
                visual_stage="plan_micro",
                description="Micro boost saat body candle tebal mendukung bullish continuation.",
                condition=_numeric_min_condition(_get_body_ratio, threshold, _bullish_context),
                effect="boost",
                direction="long",
                amount=amount,
                message=f"Body candle micro mendukung bullish di atas {threshold:.2f}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_body_bear_{str(threshold).replace('.', '_')}",
                title=f"Body Bear {threshold}",
                visual_group="plan_micro",
                visual_stage="plan_micro",
                description="Micro boost saat body candle tebal mendukung bearish continuation.",
                condition=_numeric_min_condition(_get_body_ratio, threshold, _bearish_context),
                effect="boost",
                direction="short",
                amount=amount,
                message=f"Body candle micro mendukung bearish di atas {threshold:.2f}",
            )
        )

    for threshold in [0.25, 0.35, 0.45]:
        neurons.append(
            RuleNeuron(
                name=f"micro_upper_wick_{str(threshold).replace('.', '_')}",
                title=f"Upper Wick {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro warning saat upper wick membesar dan menunjukkan rejection.",
                condition=_numeric_min_condition(_get_upper_wick, threshold),
                effect="warn",
                message=f"Upper wick micro membesar di atas {threshold:.2f}",
            )
        )
        neurons.append(
            RuleNeuron(
                name=f"micro_lower_wick_{str(threshold).replace('.', '_')}",
                title=f"Lower Wick {threshold}",
                visual_group="quality",
                visual_stage="quality",
                description="Micro warning saat lower wick membesar dan menunjukkan rejection.",
                condition=_numeric_min_condition(_get_lower_wick, threshold),
                effect="warn",
                message=f"Lower wick micro membesar di atas {threshold:.2f}",
            )
        )

    return neurons


MICRO_NEURONS = _build_micro_neurons()
