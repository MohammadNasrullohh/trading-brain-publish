from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from trading_brain import claw_research
from trading_brain.live_news import _NEWS_CACHE, _NEWS_PROVIDER_BACKOFF
from trading_brain.recommendations import scan_recommendations


DEFAULT_LEARNING = {
    "state": "warming",
    "score_bias": 0.0,
    "confidence_bias": 0.0,
    "win_rate": None,
    "wins": 0,
    "losses": 0,
    "scored_total": 0,
    "sample_size": 0,
    "loss_streak": 0,
    "open_signals": 0,
    "prime_penalty": False,
    "note": "No memory bias.",
    "recent_history": [],
}


class RecommendationScanTest(unittest.TestCase):
    def setUp(self) -> None:
        _NEWS_CACHE.clear()
        _NEWS_PROVIDER_BACKOFF.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.claw_path_patch = patch.object(claw_research, "DATA_DIR", self.data_dir)
        self.claw_file_patch = patch.object(claw_research, "DATA_PATH", self.data_dir / "claw_research_memory.json")
        self.claw_path_patch.start()
        self.claw_file_patch.start()

    def tearDown(self) -> None:
        self.claw_file_patch.stop()
        self.claw_path_patch.stop()
        self.temp_dir.cleanup()

    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.discover_market_symbols", return_value=["BTCUSDT", "XAUUSD"])
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    def test_scan_recommendations_ranks_directional_setup_first(self, mock_analyze, mock_snapshot, mock_news, mock_discover, mock_learning) -> None:
        def analyze_side_effect(payload: dict, mode: str = "super") -> dict:
            if payload["symbol"] == "BTCUSDT":
                return {
                    "result": {
                        "summary": {
                            "symbol": "BTCUSDT",
                            "timeframe": payload["timeframe"],
                            "market_type": "crypto",
                            "verdict": "LONG",
                            "confidence": 0.84,
                            "dominant_playbook": "crypto_momentum_press",
                        },
                        "brain_output": {
                            "summary": {
                                "symbol": "BTCUSDT",
                                "timeframe": payload["timeframe"],
                                "market_type": "crypto",
                                "verdict": "LONG",
                                "confidence": 0.84,
                            },
                            "reasons": ["Momentum clean"],
                            "warnings": [],
                            "blockers": [],
                            "plan": {
                                "setup_type": "continuation",
                                "entry_zone": [66500, 66540],
                                "stop_loss": 66380,
                                "take_profit_1": 66780,
                                "take_profit_2": 66940,
                                "risk_reward": 2.6,
                            },
                        },
                        "strategic_brief": {
                            "edge_summary": "BTC shows clean breakout continuation.",
                            "primary_thesis": "Momentum remains constructive.",
                        },
                    }
                }

            return {
                "result": {
                    "summary": {
                        "symbol": "XAUUSD",
                        "timeframe": payload["timeframe"],
                        "market_type": "forex",
                        "verdict": "WAIT",
                        "confidence": 0.51,
                        "dominant_playbook": "gold_reaction_wait",
                    },
                    "brain_output": {
                        "summary": {
                            "symbol": "XAUUSD",
                            "timeframe": payload["timeframe"],
                            "market_type": "forex",
                            "verdict": "WAIT",
                            "confidence": 0.51,
                        },
                        "reasons": ["Gold still noisy"],
                        "warnings": ["headline risk"],
                        "blockers": [],
                        "plan": {
                            "setup_type": "mean_reversion",
                            "entry_zone": [3000.2, 3001.1],
                            "stop_loss": 2996.8,
                            "take_profit_1": 3004.0,
                            "take_profit_2": 3008.6,
                            "risk_reward": 1.2,
                        },
                    },
                    "strategic_brief": {
                        "edge_summary": "Gold is still waiting for clearer reaction.",
                    },
                }
            }

        def snapshot_side_effect(symbol: str, timeframe: str, market_type: str | None = None) -> dict:
            if symbol == "BTCUSDT":
                return {
                    "provider": "binance",
                    "price": 66520,
                    "open": 66380,
                    "high": 66610,
                    "low": 66340,
                    "close": 66520,
                    "atr": 120,
                    "levels": {"support": [66380], "resistance": [66780]},
                    "indicators": {"ema_fast": 66510, "ema_slow": 66460, "rsi": 58, "macd_histogram": 6.2, "stochastic": 61, "vwap": 66490, "bollinger_position": 0.71},
                    "context": {"regime_hint": "trending"},
                }
            return {
                "provider": "stooq",
                "price": 3001.2,
                "open": 2998.0,
                "high": 3005.0,
                "low": 2996.5,
                "close": 3001.2,
                "atr": 8.4,
                "levels": {"support": [2997.2], "resistance": [3005.4]},
                "indicators": {"ema_fast": 3000.8, "ema_slow": 2999.9, "rsi": 51, "macd_histogram": 0.4, "stochastic": 50, "vwap": 3000.2, "bollinger_position": 0.52},
                "context": {"regime_hint": "ranging"},
            }

        def news_side_effect(symbol: str, market_type: str | None = None, limit: int = 4, force: bool = False) -> dict:
            if symbol == "BTCUSDT":
                return {
                    "provider": "google_news_rss",
                    "summary": {"score": 0.26, "headline_risk": False, "macro_risk": False, "mood": "tailwind"},
                    "articles": [{"title": "Bitcoin momentum holds", "link": "https://example.com/btc"}],
                }
            return {
                "provider": "google_news_rss",
                "summary": {"score": -0.08, "headline_risk": True, "macro_risk": True, "mood": "neutral"},
                "articles": [{"title": "Gold traders await macro catalyst", "link": "https://example.com/xau"}],
            }

        mock_analyze.side_effect = analyze_side_effect
        mock_snapshot.side_effect = snapshot_side_effect
        mock_news.side_effect = news_side_effect

        result = scan_recommendations(
            symbols=["BTCUSDT", "XAUUSD"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"risk": {"max_risk_percent": 0.4, "leverage": 3}},
            scope="manual",
        )

        self.assertEqual(result["best"]["symbol"], "BTCUSDT")
        self.assertEqual(result["items"][0]["display_verdict"], "BUY / LONG")
        self.assertEqual(result["items"][0]["claw_focus"], "breakout")
        self.assertIn("BTCUSDT", result["items"][0]["claw_summary"])
        self.assertGreater(result["items"][0]["score"], result["items"][1]["score"])
        self.assertEqual(result["items"][1]["display_verdict"], "WAIT / STANDBY")

    @patch("trading_brain.recommendations.fetch_live_snapshot", side_effect=RuntimeError("feed down"))
    def test_scan_recommendations_collects_skipped_symbols_on_error(self, mock_snapshot) -> None:
        result = scan_recommendations(symbols=["WTI"], timeframe="15m", style="intraday", mode="super", base_payload={}, scope="manual")
        self.assertEqual(result["items"], [])
        self.assertEqual(result["skipped"][0]["symbol"], "WTI")

    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.live_news._http_get_text", side_effect=TimeoutError("news provider timeout"))
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    def test_scan_recommendations_keeps_candidates_when_news_provider_fails(self, mock_analyze, mock_snapshot, mock_news_http, mock_learning) -> None:
        mock_snapshot.return_value = {
            "provider": "binance",
            "price": 66520,
            "open": 66380,
            "high": 66610,
            "low": 66340,
            "close": 66520,
            "atr": 120,
            "levels": {"support": [66380], "resistance": [66780]},
            "indicators": {"ema_fast": 66510, "ema_slow": 66460, "rsi": 58, "macd_histogram": 6.2, "stochastic": 61, "vwap": 66490, "bollinger_position": 0.71},
            "context": {"regime_hint": "trending"},
        }
        mock_analyze.return_value = {
            "result": {
                "summary": {
                    "symbol": "BTCUSDT",
                    "timeframe": "15m",
                    "market_type": "crypto",
                    "verdict": "LONG",
                    "confidence": 0.84,
                    "dominant_playbook": "crypto_momentum_press",
                },
                "brain_output": {
                    "summary": {
                        "symbol": "BTCUSDT",
                        "timeframe": "15m",
                        "market_type": "crypto",
                        "verdict": "LONG",
                        "confidence": 0.84,
                    },
                    "reasons": ["Momentum clean"],
                    "warnings": [],
                    "blockers": [],
                    "plan": {
                        "setup_type": "continuation",
                        "entry_zone": [66500, 66540],
                        "stop_loss": 66380,
                        "take_profit_1": 66780,
                        "take_profit_2": 66940,
                        "risk_reward": 2.6,
                    },
                },
                "strategic_brief": {
                    "edge_summary": "BTC shows clean breakout continuation.",
                },
            }
        }

        result = scan_recommendations(
            symbols=["BTCUSDT"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"symbol": "BTCUSDT", "market_type": "crypto"},
            scope="manual",
            prime_only=False,
        )

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["symbol"], "BTCUSDT")
        self.assertEqual(result["items"][0]["news_provider"], "news_unavailable")
        self.assertEqual(result["items"][0]["headline_mood"], "neutral")

    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    @patch("trading_brain.recommendations.discover_market_symbols")
    def test_scan_recommendations_cross_market_keeps_symbol_specific_market_type(self, mock_discover, mock_analyze, mock_snapshot, mock_news, mock_learning) -> None:
        def discover_side_effect(market_type: str | None, **kwargs) -> list[str]:
            if market_type == "crypto":
                return ["BTCUSDT", "ETHUSDT"]
            if market_type == "forex":
                return ["EURUSD", "GBPUSD", "XAUUSD"]
            if market_type == "commodity":
                return ["WTI", "XAUUSD"]
            return []

        seen_market_types: dict[str, str | None] = {}

        def snapshot_side_effect(symbol: str, timeframe: str, market_type: str | None = None) -> dict:
            seen_market_types[symbol] = market_type
            base_price = {
                "BTCUSDT": 66520.0,
                "ETHUSDT": 3420.0,
                "EURUSD": 1.082,
                "GBPUSD": 1.294,
                "XAUUSD": 3010.0,
                "WTI": 82.4,
            }.get(symbol, 100.0)
            return {
                "provider": "binance" if market_type == "crypto" else "stooq" if symbol in {"XAUUSD", "WTI"} else "frankfurter",
                "price": base_price,
                "open": base_price,
                "high": base_price * 1.001,
                "low": base_price * 0.999,
                "close": base_price,
                "atr": max(base_price * 0.003, 0.001),
                "levels": {"support": [base_price * 0.998], "resistance": [base_price * 1.002]},
                "indicators": {"ema_fast": base_price, "ema_slow": base_price * 0.999, "rsi": 55, "macd_histogram": 0.2, "stochastic": 57, "vwap": base_price},
                "context": {"regime_hint": "normal"},
                "live_context": {},
            }

        def analyze_side_effect(payload: dict, mode: str = "super") -> dict:
            symbol = payload["symbol"]
            return {
                "result": {
                    "summary": {
                        "symbol": symbol,
                        "timeframe": payload["timeframe"],
                        "market_type": payload["market_type"],
                        "verdict": "WAIT",
                        "confidence": 0.64,
                        "dominant_playbook": "market_scan",
                    },
                    "brain_output": {
                        "summary": {
                            "symbol": symbol,
                            "timeframe": payload["timeframe"],
                            "market_type": payload["market_type"],
                            "verdict": "WAIT",
                            "confidence": 0.64,
                        },
                        "reasons": [f"{symbol} setup live"],
                        "warnings": [],
                        "blockers": [],
                        "plan": {
                            "setup_type": "rotation",
                            "entry_zone": [1, 2],
                            "stop_loss": 0.5,
                            "take_profit_1": 3,
                            "take_profit_2": 4,
                            "risk_reward": 2.1,
                        },
                    },
                    "strategic_brief": {
                        "edge_summary": f"{symbol} edge",
                    },
                }
            }

        mock_discover.side_effect = discover_side_effect
        mock_snapshot.side_effect = snapshot_side_effect
        mock_news.return_value = {
            "provider": "google_news_rss",
            "summary": {"score": 0.0, "headline_risk": False, "macro_risk": False, "mood": "neutral"},
            "articles": [],
        }
        mock_analyze.side_effect = analyze_side_effect

        scan_recommendations(
            symbols=["BTCUSDT"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"symbol": "BTCUSDT", "market_type": "crypto"},
            scope="cross",
            discover_limit=6,
            prime_only=False,
        )

        self.assertEqual(seen_market_types["BTCUSDT"], "crypto")
        self.assertEqual(seen_market_types["EURUSD"], "forex")
        self.assertEqual(seen_market_types["GBPUSD"], "forex")
        self.assertEqual(seen_market_types["XAUUSD"], "forex")
        self.assertEqual(seen_market_types["WTI"], "commodity")

    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    @patch("trading_brain.recommendations.discover_market_symbols")
    def test_cross_market_scan_does_not_force_btc_as_crypto_seed_for_non_crypto_context(self, mock_discover, mock_analyze, mock_snapshot, mock_news, mock_learning) -> None:
        crypto_seed_calls: list[str | None] = []

        def discover_side_effect(market_type: str | None, **kwargs) -> list[str]:
            if market_type == "crypto":
                crypto_seed_calls.append(kwargs.get("base_symbol"))
                return ["ETHUSDT", "SOLUSDT", "BNBUSDT"]
            if market_type == "forex":
                return ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
            if market_type == "commodity":
                return ["WTI", "XAUUSD"]
            return []

        def snapshot_side_effect(symbol: str, timeframe: str, market_type: str | None = None) -> dict:
            base_price = {
                "ETHUSDT": 3410.0,
                "SOLUSDT": 188.0,
                "BNBUSDT": 640.0,
                "EURUSD": 1.0832,
                "GBPUSD": 1.2954,
                "USDJPY": 151.3,
                "XAUUSD": 3012.0,
                "WTI": 82.6,
            }.get(symbol, 100.0)
            return {
                "provider": "binance" if market_type == "crypto" else "oanda" if market_type == "forex" else "stooq",
                "price": base_price,
                "open": base_price,
                "high": base_price * 1.001,
                "low": base_price * 0.999,
                "close": base_price,
                "atr": max(base_price * 0.003, 0.001),
                "levels": {"support": [base_price * 0.998], "resistance": [base_price * 1.002]},
                "indicators": {"ema_fast": base_price, "ema_slow": base_price * 0.999, "rsi": 55, "macd_histogram": 0.2, "stochastic": 57, "vwap": base_price},
                "context": {"regime_hint": "normal"},
                "live_context": {},
            }

        def analyze_side_effect(payload: dict, mode: str = "super") -> dict:
            symbol = payload["symbol"]
            return {
                "result": {
                    "summary": {
                        "symbol": symbol,
                        "timeframe": payload["timeframe"],
                        "market_type": payload["market_type"],
                        "verdict": "WAIT",
                        "confidence": 0.64,
                        "dominant_playbook": "market_scan",
                    },
                    "brain_output": {
                        "summary": {
                            "symbol": symbol,
                            "timeframe": payload["timeframe"],
                            "market_type": payload["market_type"],
                            "verdict": "WAIT",
                            "confidence": 0.64,
                        },
                        "reasons": [f"{symbol} setup live"],
                        "warnings": [],
                        "blockers": [],
                        "plan": {
                            "setup_type": "rotation",
                            "entry_zone": [1, 2],
                            "stop_loss": 0.5,
                            "take_profit_1": 3,
                            "take_profit_2": 4,
                            "risk_reward": 2.1,
                        },
                    },
                    "strategic_brief": {
                        "edge_summary": f"{symbol} edge",
                    },
                }
            }

        mock_discover.side_effect = discover_side_effect
        mock_snapshot.side_effect = snapshot_side_effect
        mock_news.return_value = {
            "provider": "google_news_rss",
            "summary": {"score": 0.0, "headline_risk": False, "macro_risk": False, "mood": "neutral"},
            "articles": [],
        }
        mock_analyze.side_effect = analyze_side_effect

        result = scan_recommendations(
            symbols=["EURUSD"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"symbol": "EURUSD", "market_type": "forex"},
            scope="cross",
            discover_limit=8,
            prime_only=False,
        )

        self.assertIn("ETHUSDT", result["symbols"])
        self.assertIn(None, crypto_seed_calls)
        self.assertNotIn("BTCUSDT", crypto_seed_calls)

    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    @patch("trading_brain.recommendations.discover_market_symbols")
    def test_scan_recommendations_macro_scope_combines_fx_gold_and_oil(self, mock_discover, mock_analyze, mock_snapshot, mock_news, mock_learning) -> None:
        def discover_side_effect(market_type: str | None, **kwargs) -> list[str]:
            if market_type == "forex":
                return ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
            if market_type == "commodity":
                return ["WTI", "XAUUSD"]
            return []

        def snapshot_side_effect(symbol: str, timeframe: str, market_type: str | None = None) -> dict:
            base_price = {
                "EURUSD": 1.0842,
                "GBPUSD": 1.2964,
                "USDJPY": 151.7,
                "XAUUSD": 3012.5,
                "WTI": 82.4,
            }.get(symbol, 100.0)
            return {
                "provider": "oanda" if market_type == "forex" else "stooq",
                "price": base_price,
                "open": base_price,
                "high": base_price * 1.001,
                "low": base_price * 0.999,
                "close": base_price,
                "atr": max(base_price * 0.003, 0.001),
                "levels": {"support": [base_price * 0.998], "resistance": [base_price * 1.002]},
                "indicators": {"ema_fast": base_price, "ema_slow": base_price * 0.999, "rsi": 55, "macd_histogram": 0.2, "stochastic": 57, "vwap": base_price},
                "context": {"regime_hint": "normal"},
                "live_context": {},
            }

        def news_side_effect(symbol: str, market_type: str | None = None, limit: int = 4, force: bool = False) -> dict:
            return {
                "provider": "google_news_rss",
                "summary": {"score": 0.08, "headline_risk": symbol in {"XAUUSD", "WTI"}, "macro_risk": symbol in {"XAUUSD", "WTI"}, "mood": "neutral"},
                "articles": [{"title": f"{symbol} headline", "source": "Reuters", "link": "https://example.com"}],
            }

        def analyze_side_effect(payload: dict, mode: str = "super") -> dict:
            symbol = payload["symbol"]
            verdict = "WAIT"
            confidence = 0.64
            if symbol == "EURUSD":
                verdict = "LONG"
                confidence = 0.79
            elif symbol == "WTI":
                verdict = "SHORT"
                confidence = 0.74
            return {
                "result": {
                    "summary": {
                        "symbol": symbol,
                        "timeframe": payload["timeframe"],
                        "market_type": payload["market_type"],
                        "verdict": verdict,
                        "confidence": confidence,
                        "dominant_playbook": "macro_rotation",
                    },
                    "brain_output": {
                        "summary": {
                            "symbol": symbol,
                            "timeframe": payload["timeframe"],
                            "market_type": payload["market_type"],
                            "verdict": verdict,
                            "confidence": confidence,
                        },
                        "reasons": [f"{symbol} setup live"],
                        "warnings": [],
                        "blockers": [],
                        "plan": {
                            "setup_type": "rotation",
                            "entry_zone": [1, 2],
                            "stop_loss": 0.5,
                            "take_profit_1": 3,
                            "take_profit_2": 4,
                            "risk_reward": 2.1,
                        },
                    },
                    "strategic_brief": {
                        "edge_summary": f"{symbol} edge",
                    },
                }
            }

        mock_discover.side_effect = discover_side_effect
        mock_snapshot.side_effect = snapshot_side_effect
        mock_news.side_effect = news_side_effect
        mock_analyze.side_effect = analyze_side_effect

        result = scan_recommendations(
            symbols=["EURUSD"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"symbol": "EURUSD", "market_type": "forex"},
            scope="macro",
            discover_limit=8,
            prime_only=False,
        )

        self.assertEqual(result["scope"], "macro")
        self.assertEqual(result["scope_label"], "FX + Gold + Oil")
        self.assertIn("EURUSD", result["symbols"])
        self.assertIn("XAUUSD", result["symbols"])
        self.assertIn("WTI", result["symbols"])
        self.assertGreaterEqual(len(result["items"]), 3)
        self.assertTrue(any(item["news_provider"] == "google_news_rss" for item in result["items"]))

    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    @patch("trading_brain.recommendations.discover_market_symbols", return_value=["ETHUSDT", "SOLUSDT", "BTCUSDT"])
    def test_scan_recommendations_can_expand_with_market_discovery(self, mock_discover, mock_analyze, mock_snapshot, mock_news, mock_learning) -> None:
        mock_snapshot.return_value = {
            "provider": "binance",
            "price": 100.0,
            "open": 98.0,
            "high": 101.0,
            "low": 97.5,
            "close": 100.0,
            "atr": 2.5,
            "levels": {"support": [98.4], "resistance": [101.8]},
            "indicators": {"ema_fast": 99.8, "ema_slow": 99.1, "rsi": 57, "macd_histogram": 0.8, "stochastic": 61, "vwap": 99.4, "bollinger_position": 0.67},
            "context": {"regime_hint": "trending"},
        }
        mock_news.return_value = {
            "provider": "google_news_rss",
            "summary": {"score": 0.12, "headline_risk": False, "macro_risk": False, "mood": "tailwind"},
            "articles": [],
        }
        mock_analyze.return_value = {
            "result": {
                "summary": {
                    "symbol": "ETHUSDT",
                    "timeframe": "15m",
                    "market_type": "crypto",
                    "verdict": "LONG",
                    "confidence": 0.73,
                    "dominant_playbook": "momentum_rotation",
                },
                "brain_output": {
                    "summary": {
                        "symbol": "ETHUSDT",
                        "timeframe": "15m",
                        "market_type": "crypto",
                        "verdict": "LONG",
                        "confidence": 0.73,
                    },
                    "reasons": ["Rotation momentum clean"],
                    "warnings": [],
                    "blockers": [],
                    "plan": {
                        "setup_type": "rotation",
                        "entry_zone": [99.5, 100.2],
                        "stop_loss": 98.4,
                        "take_profit_1": 101.8,
                        "take_profit_2": 103.0,
                        "risk_reward": 2.1,
                    },
                },
                "strategic_brief": {
                    "edge_summary": "Leaders rotate cleanly into ETH.",
                },
            }
        }

        result = scan_recommendations(
            symbols=["BTCUSDT"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"symbol": "BTCUSDT", "market_type": "crypto"},
            scope="hybrid",
            discover_limit=3,
        )

        self.assertEqual(result["scope"], "hybrid")
        self.assertEqual(result["discovered_symbols"], ["ETHUSDT", "SOLUSDT", "BTCUSDT"])
        self.assertIn("ETHUSDT", result["symbols"])
        self.assertEqual(len(result["items"]), 3)

    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    def test_scan_recommendations_precision_mode_keeps_only_prime_setups(self, mock_analyze, mock_snapshot, mock_news, mock_learning) -> None:
        mock_snapshot.side_effect = [
            {
                "provider": "binance",
                "price": 101.0,
                "open": 99.0,
                "high": 101.5,
                "low": 98.8,
                "close": 101.0,
                "atr": 1.8,
                "levels": {"support": [99.4], "resistance": [102.6]},
                "indicators": {"ema_fast": 100.8, "ema_slow": 100.1, "rsi": 59, "macd_histogram": 0.6},
                "context": {"regime_hint": "trending"},
                "live_context": {"listing_profile": "established", "fresh_listing_candidate": False},
            },
            {
                "provider": "binance",
                "price": 45.0,
                "open": 44.4,
                "high": 45.4,
                "low": 44.0,
                "close": 45.0,
                "atr": 1.9,
                "levels": {"support": [44.1], "resistance": [46.8]},
                "indicators": {"ema_fast": 44.8, "ema_slow": 44.2, "rsi": 58, "macd_histogram": 0.7},
                "context": {"regime_hint": "trending"},
                "live_context": {"listing_profile": "fresh_listing", "fresh_listing_candidate": True},
            },
        ]
        mock_news.return_value = {
            "provider": "google_news_rss",
            "summary": {"score": 0.15, "headline_risk": False, "macro_risk": False, "mood": "tailwind"},
            "articles": [],
        }

        def analyze_side_effect(payload: dict, mode: str = "super") -> dict:
            if payload["symbol"] == "AAAUSDT":
                return {
                    "result": {
                        "summary": {
                            "symbol": "AAAUSDT",
                            "timeframe": payload["timeframe"],
                            "market_type": "crypto",
                            "verdict": "LONG",
                            "confidence": 0.88,
                            "dominant_playbook": "clean_breakout",
                            "execution_profile": "precision",
                        },
                        "brain_output": {
                            "summary": {
                                "symbol": "AAAUSDT",
                                "timeframe": payload["timeframe"],
                                "market_type": "crypto",
                                "verdict": "LONG",
                                "confidence": 0.88,
                                "execution_profile": "precision",
                            },
                            "reasons": ["AAA clean continuation"],
                            "warnings": [],
                            "blockers": [],
                            "plan": {
                                "setup_type": "breakout",
                                "entry_zone": [100.7, 101.1],
                                "stop_loss": 99.9,
                                "take_profit_1": 102.2,
                                "take_profit_2": 103.0,
                                "risk_reward": 1.7,
                            },
                        },
                        "profile_guard": {
                            "status": "pass",
                            "floating_risk": "low",
                            "notes": ["Precision mode pass."],
                        },
                        "strategic_brief": {
                            "edge_summary": "AAA is clean enough for precision execution.",
                        },
                    }
                }

            return {
                "result": {
                    "summary": {
                        "symbol": "BBBUSDT",
                        "timeframe": payload["timeframe"],
                        "market_type": "crypto",
                        "verdict": "WAIT",
                        "confidence": 0.61,
                        "dominant_playbook": "fresh_rotation_watch",
                        "execution_profile": "precision",
                    },
                    "brain_output": {
                        "summary": {
                            "symbol": "BBBUSDT",
                            "timeframe": payload["timeframe"],
                            "market_type": "crypto",
                            "verdict": "WAIT",
                            "confidence": 0.61,
                            "execution_profile": "precision",
                        },
                        "reasons": ["Fresh listing but still noisy"],
                        "warnings": ["target masih jauh"],
                        "blockers": [],
                        "conditional_plan": {
                            "setup_type": "watch",
                            "entry_zone": [44.8, 45.1],
                            "stop_loss": 43.5,
                            "take_profit_1": 47.8,
                            "take_profit_2": 49.2,
                            "risk_reward": 1.1,
                        },
                    },
                    "profile_guard": {
                        "status": "blocked",
                        "floating_risk": "elevated",
                        "notes": ["TP terlalu jauh untuk precision mode."],
                    },
                    "strategic_brief": {
                        "edge_summary": "BBB still needs cleaner timing.",
                    },
                }
            }

        mock_analyze.side_effect = analyze_side_effect

        result = scan_recommendations(
            symbols=["AAAUSDT", "BBBUSDT"],
            timeframe="15m",
            style="scalping",
            mode="super",
            base_payload={"symbol": "BTCUSDT", "market_type": "crypto", "execution_profile": "precision"},
            scope="manual",
            prime_only=True,
        )

        self.assertEqual(result["execution_profile"], "precision")
        self.assertTrue(result["prime_only"])
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["symbol"], "AAAUSDT")
        self.assertEqual(result["filtered_out"][0]["symbol"], "BBBUSDT")

    @patch("trading_brain.recommendations.register_signal")
    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    def test_best_setup_can_be_promoted_into_signal_memory(self, mock_analyze, mock_snapshot, mock_news, mock_learning, mock_register_signal) -> None:
        mock_snapshot.side_effect = [
            {
                "provider": "binance",
                "price": 3420.0,
                "open": 3398.0,
                "high": 3436.0,
                "low": 3388.0,
                "close": 3420.0,
                "atr": 44.0,
                "levels": {"support": [3392.0], "resistance": [3468.0]},
                "indicators": {"ema_fast": 3418.0, "ema_slow": 3399.0, "rsi": 59, "macd_histogram": 1.2, "stochastic": 63, "vwap": 3410.0},
                "context": {"regime_hint": "trending"},
                "live_context": {},
            },
            {
                "provider": "oanda",
                "price": 1.0832,
                "open": 1.0821,
                "high": 1.0844,
                "low": 1.0814,
                "close": 1.0832,
                "atr": 0.0041,
                "levels": {"support": [1.0818], "resistance": [1.0852]},
                "indicators": {"ema_fast": 1.0831, "ema_slow": 1.0824, "rsi": 54, "macd_histogram": 0.2, "stochastic": 56, "vwap": 1.0828},
                "context": {"regime_hint": "normal"},
                "live_context": {},
            },
        ]
        mock_news.return_value = {
            "provider": "google_news_rss",
            "summary": {"score": 0.14, "headline_risk": False, "macro_risk": False, "mood": "tailwind"},
            "articles": [],
        }

        def analyze_side_effect(payload: dict, mode: str = "super") -> dict:
            if payload["symbol"] == "ETHUSDT":
                return {
                    "result": {
                        "summary": {
                            "symbol": "ETHUSDT",
                            "timeframe": payload["timeframe"],
                            "market_type": "crypto",
                            "verdict": "LONG",
                            "confidence": 0.82,
                            "dominant_playbook": "rotation_press",
                        },
                        "brain_output": {
                            "summary": {
                                "symbol": "ETHUSDT",
                                "timeframe": payload["timeframe"],
                                "market_type": "crypto",
                                "verdict": "LONG",
                                "confidence": 0.82,
                            },
                            "reasons": ["ETH leadership clean"],
                            "warnings": [],
                            "blockers": [],
                            "plan": {
                                "setup_type": "continuation",
                                "entry_zone": [3412.0, 3422.0],
                                "stop_loss": 3388.0,
                                "take_profit_1": 3468.0,
                                "take_profit_2": 3495.0,
                                "risk_reward": 2.2,
                            },
                        },
                        "strategic_brief": {
                            "edge_summary": "ETH now leads the live crypto rotation.",
                        },
                    }
                }

            return {
                "result": {
                    "summary": {
                        "symbol": "EURUSD",
                        "timeframe": payload["timeframe"],
                        "market_type": "forex",
                        "verdict": "WAIT",
                        "confidence": 0.58,
                        "dominant_playbook": "range_wait",
                    },
                    "brain_output": {
                        "summary": {
                            "symbol": "EURUSD",
                            "timeframe": payload["timeframe"],
                            "market_type": "forex",
                            "verdict": "WAIT",
                            "confidence": 0.58,
                        },
                        "reasons": ["EURUSD still compressing"],
                        "warnings": [],
                        "blockers": [],
                        "plan": {
                            "setup_type": "wait",
                            "entry_zone": [1.0828, 1.0835],
                            "stop_loss": 1.0810,
                            "take_profit_1": 1.0854,
                            "take_profit_2": 1.0868,
                            "risk_reward": 1.1,
                        },
                    },
                    "strategic_brief": {
                        "edge_summary": "EURUSD still needs confirmation.",
                    },
                }
            }

        mock_analyze.side_effect = analyze_side_effect
        mock_register_signal.return_value = {
            **DEFAULT_LEARNING,
            "state": "in_sync",
            "score_bias": 6.0,
            "confidence_bias": 0.03,
            "win_rate": 66.7,
            "scored_total": 6,
            "sample_size": 6,
            "recent_history": [{"symbol": "ETHUSDT", "status": "win"}],
        }

        result = scan_recommendations(
            symbols=["ETHUSDT", "EURUSD"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"symbol": "ETHUSDT", "market_type": "crypto"},
            scope="manual",
            prime_only=False,
            track_best_setup=True,
        )

        self.assertEqual(result["tracked_best"]["symbol"], "ETHUSDT")
        self.assertTrue(result["tracked_best"]["tracked_in_signal_memory"])
        self.assertEqual(result["tracked_best"]["signal_source"], "best_setups")
        self.assertEqual(result["tracked_best"]["recent_sample_size"], 6)
        mock_register_signal.assert_called_once()

    @patch("trading_brain.recommendations.register_signal")
    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    def test_top_scored_directional_candidates_from_multiple_pairs_can_enter_signal_memory(self, mock_analyze, mock_snapshot, mock_news, mock_learning, mock_register_signal) -> None:
        mock_snapshot.side_effect = [
            {
                "provider": "binance",
                "price": 3420.0,
                "open": 3398.0,
                "high": 3436.0,
                "low": 3388.0,
                "close": 3420.0,
                "atr": 44.0,
                "levels": {"support": [3392.0], "resistance": [3468.0]},
                "indicators": {"ema_fast": 3418.0, "ema_slow": 3399.0, "rsi": 59, "macd_histogram": 1.2},
                "context": {"regime_hint": "trending"},
                "live_context": {},
            },
            {
                "provider": "binance",
                "price": 188.0,
                "open": 184.0,
                "high": 189.0,
                "low": 183.0,
                "close": 188.0,
                "atr": 4.2,
                "levels": {"support": [183.4], "resistance": [191.2]},
                "indicators": {"ema_fast": 187.2, "ema_slow": 184.4, "rsi": 61, "macd_histogram": 1.0},
                "context": {"regime_hint": "trending"},
                "live_context": {},
            },
        ]
        mock_news.return_value = {
            "provider": "google_news_rss",
            "summary": {"score": 0.14, "headline_risk": False, "macro_risk": False, "mood": "tailwind"},
            "articles": [],
        }

        def analyze_side_effect(payload: dict, mode: str = "super") -> dict:
            if payload["symbol"] == "ETHUSDT":
                return {
                    "result": {
                        "summary": {
                            "symbol": "ETHUSDT",
                            "timeframe": payload["timeframe"],
                            "market_type": "crypto",
                            "verdict": "LONG",
                            "confidence": 0.82,
                            "dominant_playbook": "rotation_press",
                        },
                        "brain_output": {
                            "summary": {
                                "symbol": "ETHUSDT",
                                "timeframe": payload["timeframe"],
                                "market_type": "crypto",
                                "verdict": "LONG",
                                "confidence": 0.82,
                            },
                            "reasons": ["ETH leadership clean"],
                            "warnings": [],
                            "blockers": [],
                            "plan": {
                                "setup_type": "continuation",
                                "entry_zone": [3412.0, 3422.0],
                                "stop_loss": 3388.0,
                                "take_profit_1": 3468.0,
                                "take_profit_2": 3495.0,
                                "risk_reward": 2.2,
                            },
                        },
                        "strategic_brief": {
                            "edge_summary": "ETH now leads the live crypto rotation.",
                        },
                    }
                }

            return {
                "result": {
                    "summary": {
                        "symbol": "SOLUSDT",
                        "timeframe": payload["timeframe"],
                        "market_type": "crypto",
                        "verdict": "LONG",
                        "confidence": 0.81,
                        "dominant_playbook": "continuation_press",
                    },
                    "brain_output": {
                        "summary": {
                            "symbol": "SOLUSDT",
                            "timeframe": payload["timeframe"],
                            "market_type": "crypto",
                            "verdict": "LONG",
                            "confidence": 0.81,
                        },
                        "reasons": ["SOL breakout stayed clean"],
                        "warnings": [],
                        "blockers": [],
                        "plan": {
                            "setup_type": "continuation",
                            "entry_zone": [186.2, 187.1],
                            "stop_loss": 183.0,
                            "take_profit_1": 191.0,
                            "take_profit_2": 194.4,
                            "risk_reward": 2.1,
                        },
                    },
                    "strategic_brief": {
                        "edge_summary": "SOL keeps pace with the rotation.",
                    },
                }
            }

        mock_analyze.side_effect = analyze_side_effect
        mock_register_signal.return_value = {
            **DEFAULT_LEARNING,
            "state": "in_sync",
            "score_bias": 5.0,
            "confidence_bias": 0.02,
            "win_rate": 63.0,
            "scored_total": 8,
            "sample_size": 8,
            "recent_history": [{"symbol": "ETHUSDT", "status": "win"}],
        }

        result = scan_recommendations(
            symbols=["ETHUSDT", "SOLUSDT"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"symbol": "ETHUSDT", "market_type": "crypto"},
            scope="manual",
            prime_only=False,
            track_best_setup=True,
        )

        self.assertEqual(len(result["tracked_memory"]), 2)
        self.assertEqual(result["tracked_memory"][0]["memory_track_rank"], 1)
        self.assertEqual(result["tracked_memory"][1]["memory_track_rank"], 2)
        self.assertEqual(mock_register_signal.call_count, 2)
        first_call_payload = mock_register_signal.call_args_list[0].args[0]
        self.assertGreaterEqual(first_call_payload["_signal_score"], 100.0)

    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    def test_scan_recommendations_returns_fallback_candidates_when_no_prime_setup(self, mock_analyze, mock_snapshot, mock_news, mock_learning) -> None:
        mock_snapshot.return_value = {
            "provider": "binance",
            "price": 101.0,
            "open": 100.0,
            "high": 101.4,
            "low": 99.8,
            "close": 101.0,
            "atr": 1.8,
            "levels": {"support": [99.4], "resistance": [102.4]},
            "indicators": {"ema_fast": 100.8, "ema_slow": 100.1, "rsi": 57, "macd_histogram": 0.5},
            "context": {"regime_hint": "normal"},
            "live_context": {"listing_profile": "fresh_listing", "fresh_listing_candidate": True, "history_age_hours": 18},
        }
        mock_news.return_value = {
            "provider": "google_news_rss",
            "summary": {"score": 0.05, "headline_risk": False, "macro_risk": False, "mood": "neutral"},
            "articles": [{"title": "ABC listing launch roadmap volume grows", "source": "CoinDesk", "link": "https://example.com"}],
        }
        mock_analyze.return_value = {
            "result": {
                "summary": {
                    "symbol": "ABCUSDT",
                    "timeframe": "15m",
                    "market_type": "crypto",
                    "verdict": "LONG",
                    "confidence": 0.74,
                    "dominant_playbook": "fresh_rotation",
                    "execution_profile": "precision",
                },
                "brain_output": {
                    "summary": {
                        "symbol": "ABCUSDT",
                        "timeframe": "15m",
                        "market_type": "crypto",
                        "verdict": "LONG",
                        "confidence": 0.74,
                        "execution_profile": "precision",
                    },
                    "reasons": ["Fresh listing momentum"],
                    "warnings": [],
                    "blockers": [],
                    "plan": {
                        "setup_type": "breakout",
                        "entry_zone": [100.9, 101.1],
                        "stop_loss": 99.7,
                        "take_profit_1": 102.2,
                        "take_profit_2": 103.2,
                        "risk_reward": 1.2,
                    },
                },
                "profile_guard": {
                    "status": "pass",
                    "floating_risk": "low",
                    "notes": ["Setup clean but RR not big enough for fresh coin precision mode."],
                },
                "strategic_brief": {
                    "edge_summary": "Fresh coin moves, but asymmetry is still too small.",
                },
            }
        }

        result = scan_recommendations(
            symbols=["ABCUSDT"],
            timeframe="15m",
            style="scalping",
            mode="super",
            base_payload={"symbol": "BTCUSDT", "market_type": "crypto", "execution_profile": "precision"},
            scope="manual",
            prime_only=True,
        )

        self.assertEqual(result["items"], [])
        self.assertTrue(result["fallback_items"])
        self.assertEqual(result["fallback_items"][0]["symbol"], "ABCUSDT")
        self.assertIn("Fresh coin", result["fallback_items"][0]["survival_note"])

    @patch("trading_brain.recommendations.signal_learning_context", return_value=DEFAULT_LEARNING)
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    def test_scan_recommendations_marks_low_rr_directional_setup_as_not_prime(self, mock_analyze, mock_snapshot, mock_news, mock_learning) -> None:
        mock_snapshot.return_value = {
            "provider": "frankfurter",
            "price": 1.0842,
            "open": 1.0838,
            "high": 1.0847,
            "low": 1.0832,
            "close": 1.0842,
            "atr": 0.0042,
            "levels": {"support": [1.0831], "resistance": [1.0850]},
            "indicators": {"ema_fast": 1.0840, "ema_slow": 1.0834, "rsi": 56, "macd_histogram": 0.0004, "stochastic": 58, "vwap": 1.0839},
            "context": {"regime_hint": "trending"},
        }
        mock_news.return_value = {
            "provider": "google_news_rss",
            "summary": {"score": 0.08, "headline_risk": False, "macro_risk": False, "mood": "neutral"},
            "articles": [],
        }
        mock_analyze.return_value = {
            "result": {
                "summary": {
                    "symbol": "EURUSD",
                    "timeframe": "15m",
                    "market_type": "forex",
                    "verdict": "LONG",
                    "confidence": 0.79,
                    "execution_profile": "balanced",
                },
                "brain_output": {
                    "summary": {
                        "symbol": "EURUSD",
                        "timeframe": "15m",
                        "market_type": "forex",
                        "verdict": "LONG",
                        "confidence": 0.79,
                        "execution_profile": "balanced",
                    },
                    "reasons": ["Structure clean but target still close."],
                    "warnings": [],
                    "blockers": [],
                    "plan": {
                        "setup_type": "pullback",
                        "entry_zone": [1.0840, 1.0842],
                        "stop_loss": 1.0830,
                        "take_profit_1": 1.0851,
                        "take_profit_2": 1.0854,
                        "risk_reward": 1.1,
                    },
                },
                "strategic_brief": {
                    "edge_summary": "Directional structure exists, but asymmetry is still thin.",
                },
            }
        }

        result = scan_recommendations(
            symbols=["EURUSD"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"symbol": "EURUSD", "market_type": "forex", "execution_profile": "balanced"},
            scope="manual",
            prime_only=False,
        )

        self.assertEqual(len(result["items"]), 1)
        self.assertFalse(result["items"][0]["prime_setup"])
        self.assertFalse(result["items"][0]["rr_healthy"])
        self.assertEqual(result["items"][0]["rr_floor"], 1.2)

    @patch("trading_brain.recommendations.signal_learning_context")
    @patch("trading_brain.recommendations.fetch_live_news")
    @patch("trading_brain.recommendations.fetch_live_snapshot")
    @patch("trading_brain.recommendations.analyze_for_web")
    def test_scan_recommendations_respects_learning_cooldown(self, mock_analyze, mock_snapshot, mock_news, mock_learning) -> None:
        mock_snapshot.return_value = {
            "provider": "frankfurter",
            "price": 1.08,
            "open": 1.079,
            "high": 1.081,
            "low": 1.078,
            "close": 1.08,
            "atr": 0.002,
            "levels": {"support": [1.078], "resistance": [1.082]},
            "indicators": {"ema_fast": 1.08, "ema_slow": 1.079, "rsi": 55, "macd_histogram": 0.2},
            "context": {"regime_hint": "normal"},
        }
        mock_news.return_value = {
            "provider": "google_news_rss",
            "summary": {"score": 0.0, "headline_risk": False, "macro_risk": False, "mood": "neutral"},
            "articles": [],
        }
        mock_analyze.return_value = {
            "result": {
                "summary": {
                    "symbol": "EURUSD",
                    "timeframe": "15m",
                    "market_type": "forex",
                    "verdict": "LONG",
                    "confidence": 0.84,
                    "dominant_playbook": "session_break",
                },
                "brain_output": {
                    "summary": {
                        "symbol": "EURUSD",
                        "timeframe": "15m",
                        "market_type": "forex",
                        "verdict": "LONG",
                        "confidence": 0.84,
                    },
                    "reasons": ["Trend clean"],
                    "warnings": [],
                    "blockers": [],
                    "plan": {
                        "setup_type": "breakout",
                        "entry_zone": [1.0798, 1.0802],
                        "stop_loss": 1.0788,
                        "take_profit_1": 1.0825,
                        "take_profit_2": 1.084,
                        "risk_reward": 2.3,
                    },
                },
                "strategic_brief": {
                    "edge_summary": "EURUSD structure is clean.",
                },
            }
        }
        mock_learning.return_value = {
            **DEFAULT_LEARNING,
            "state": "cooldown",
            "score_bias": -12.0,
            "confidence_bias": -0.06,
            "loss_streak": 3,
            "win_rate": 25.0,
            "losses": 3,
            "scored_total": 4,
            "prime_penalty": True,
            "note": "Recent losses trigger cooldown.",
        }

        result = scan_recommendations(
            symbols=["EURUSD"],
            timeframe="15m",
            style="intraday",
            mode="super",
            base_payload={"symbol": "EURUSD", "market_type": "forex"},
            scope="manual",
            prime_only=False,
        )

        self.assertEqual(result["items"][0]["learning_state"], "cooldown")
        self.assertFalse(result["items"][0]["prime_setup"])
        self.assertLess(result["items"][0]["score"], 100)


if __name__ == "__main__":
    unittest.main()
