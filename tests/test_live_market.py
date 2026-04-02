from __future__ import annotations

import unittest
from unittest.mock import patch

from trading_brain.live_market import (
    _parse_oanda_stream_payload,
    build_live_config,
    discover_market_symbols,
    fetch_live_snapshot,
    infer_market_type,
    parse_forex_pair,
)


def _build_binance_klines(count: int = 80, start_price: float = 66000.0) -> list[list[object]]:
    rows: list[list[object]] = []
    price = start_price
    for index in range(count):
        drift = 18 if index % 7 in {1, 2, 3} else -9
        open_price = price
        close_price = price + drift
        high_price = max(open_price, close_price) + 24 + (index % 5)
        low_price = min(open_price, close_price) - 19 - (index % 4)
        volume = 420 + (index * 9)
        rows.append([
            index * 60000,
            f"{open_price:.2f}",
            f"{high_price:.2f}",
            f"{low_price:.2f}",
            f"{close_price:.2f}",
            f"{volume:.2f}",
            (index + 1) * 60000,
        ])
        price = close_price + (6 if index % 4 == 0 else -3)
    return rows


def _build_oanda_candles(count: int = 80, start_price: float = 1.0810) -> dict[str, object]:
    candles = []
    price = start_price
    for index in range(count):
        drift = 0.00042 if index % 6 in {1, 2, 3} else -0.00018
        open_price = price
        close_price = price + drift
        high_price = max(open_price, close_price) + 0.00024 + ((index % 3) * 0.00003)
        low_price = min(open_price, close_price) - 0.00021 - ((index % 4) * 0.00002)
        candles.append(
            {
                "time": f"2026-03-29T{index:02d}:00:00.000000000Z",
                "volume": 100 + index,
                "mid": {
                    "o": f"{open_price:.5f}",
                    "h": f"{high_price:.5f}",
                    "l": f"{low_price:.5f}",
                    "c": f"{close_price:.5f}",
                },
            }
        )
        price = close_price + (0.00008 if index % 5 == 0 else -0.00003)
    return {"candles": candles}


class LiveMarketTest(unittest.TestCase):
    def test_build_live_config_for_crypto_exposes_websocket_stream(self) -> None:
        config = build_live_config("BTCUSDT", "15m", "crypto")
        self.assertTrue(config["supported"])
        self.assertEqual(config["provider"], "binance")
        self.assertEqual(config["transport"], "websocket")
        self.assertIn("btcusdt@miniTicker", config["websocket"]["url"])
        self.assertIn("btcusdt@kline_15m", config["websocket"]["url"])

    def test_build_live_config_for_forex_uses_reference_poll(self) -> None:
        config = build_live_config("EURUSD", "1h", "forex")
        self.assertTrue(config["supported"])
        self.assertEqual(config["provider"], "frankfurter")
        self.assertEqual(config["transport"], "poll")
        self.assertFalse(config["realtime"])
        self.assertEqual(config["pair"]["base"], "EUR")
        self.assertEqual(config["pair"]["quote"], "USD")

    def test_build_live_config_for_xauusd_uses_stooq_without_oanda(self) -> None:
        config = build_live_config("XAUUSD", "15m", "forex")
        self.assertTrue(config["supported"])
        self.assertEqual(config["provider"], "stooq")
        self.assertEqual(config["transport"], "poll")
        self.assertEqual(config["stooq_symbol"], "xauusd")

    def test_build_live_config_for_wti_uses_stooq_reference_feed(self) -> None:
        config = build_live_config("WTI", "15m", "commodity")
        self.assertTrue(config["supported"])
        self.assertEqual(config["provider"], "stooq")
        self.assertEqual(config["market_type"], "commodity")
        self.assertEqual(config["stooq_symbol"], "cl.f")

    def test_market_type_and_forex_pair_inference(self) -> None:
        self.assertEqual(infer_market_type("EURUSD", "auto"), "forex")
        self.assertEqual(infer_market_type("BTCUSDT", "auto"), "crypto")
        self.assertEqual(infer_market_type("WTI", "auto"), "commodity")
        self.assertEqual(parse_forex_pair("EUR/USD"), ("EUR", "USD"))

    @patch("trading_brain.live_market._http_get_json")
    def test_discover_market_symbols_returns_ranked_crypto_leaders(self, http_get_json) -> None:
        http_get_json.return_value = [
            {"symbol": "BTCUSDT", "quoteVolume": "1800000000", "priceChangePercent": "2.4", "lastPrice": "66500"},
            {"symbol": "ETHUSDT", "quoteVolume": "950000000", "priceChangePercent": "4.8", "lastPrice": "1990"},
            {"symbol": "SOLUSDT", "quoteVolume": "640000000", "priceChangePercent": "7.5", "lastPrice": "182"},
            {"symbol": "USDCUSDT", "quoteVolume": "220000000", "priceChangePercent": "0.0", "lastPrice": "1"},
            {"symbol": "BTCDOWNUSDT", "quoteVolume": "54000000", "priceChangePercent": "16.0", "lastPrice": "0.12"},
        ]

        discovered = discover_market_symbols("crypto", base_symbol="BTCUSDT", limit=3)
        self.assertEqual(discovered[0], "BTCUSDT")
        self.assertIn("SOLUSDT", discovered)
        self.assertIn("ETHUSDT", discovered)
        self.assertNotIn("USDCUSDT", discovered)
        self.assertNotIn("BTCDOWNUSDT", discovered)

    @patch("trading_brain.live_market._binance_candles")
    @patch("trading_brain.live_market._http_get_json")
    def test_discover_market_symbols_can_prioritize_fresh_listings(self, http_get_json, mock_candles) -> None:
        http_get_json.return_value = [
            {"symbol": "AAAUSDT", "quoteVolume": "125000000", "priceChangePercent": "9.2", "lastPrice": "1.24", "count": 12450},
            {"symbol": "BBBUSDT", "quoteVolume": "98000000", "priceChangePercent": "7.1", "lastPrice": "0.84", "count": 8540},
            {"symbol": "BTCUSDT", "quoteVolume": "1800000000", "priceChangePercent": "2.4", "lastPrice": "66500", "count": 321450},
        ]

        def candle_side_effect(symbol: str, timeframe: str, limit: int = 160) -> list[dict]:
            if symbol == "AAAUSDT":
                return _build_binance_klines(count=42, start_price=1.1)
            if symbol == "BBBUSDT":
                return _build_binance_klines(count=74, start_price=0.8)
            return _build_binance_klines(count=160, start_price=66000)

        mock_candles.side_effect = candle_side_effect

        discovered = discover_market_symbols("crypto", base_symbol="BTCUSDT", timeframe="15m", limit=3, discovery_mode="fresh")
        self.assertEqual(discovered[0], "BTCUSDT")
        self.assertIn("AAAUSDT", discovered)
        self.assertIn("BBBUSDT", discovered)

    def test_discover_market_symbols_returns_extended_forex_universe(self) -> None:
        discovered = discover_market_symbols("forex", base_symbol="EURUSD", limit=16)
        self.assertEqual(discovered[0], "EURUSD")
        self.assertIn("GBPUSD", discovered)
        self.assertIn("USDJPY", discovered)
        self.assertIn("EURJPY", discovered)
        self.assertIn("XAUUSD", discovered)

    @patch.dict("os.environ", {"OANDA_API_TOKEN": "demo-token", "OANDA_ACCOUNT_ID": "101-001-1234567-001"}, clear=False)
    def test_build_live_config_for_forex_prefers_oanda_when_credentials_exist(self) -> None:
        config = build_live_config("EURUSD", "1h", "forex")
        self.assertTrue(config["supported"])
        self.assertEqual(config["provider"], "oanda")
        self.assertEqual(config["transport"], "sse")
        self.assertTrue(config["realtime"])
        self.assertIn("/api/live/stream", config["stream_url"])
        self.assertEqual(config["instrument"], "EUR_USD")

    @patch("trading_brain.live_market._http_get_json")
    def test_fetch_live_snapshot_parses_crypto_payload(self, http_get_json) -> None:
        http_get_json.side_effect = [
            {
                "lastPrice": "68321.40",
                "priceChangePercent": "2.54",
                "openPrice": "66600.00",
                "highPrice": "68920.00",
                "lowPrice": "66010.00",
                "volume": "1234.56",
                "quoteVolume": "84321000.12",
                "closeTime": 1710000000000,
            },
            _build_binance_klines(),
        ]

        snapshot = fetch_live_snapshot("BTCUSDT", "15m", "crypto")
        self.assertTrue(snapshot["supported"])
        self.assertEqual(snapshot["provider"], "binance")
        self.assertEqual(snapshot["price"], 68321.40)
        self.assertIn("levels", snapshot)
        self.assertIn("indicators", snapshot)
        self.assertIn("context", snapshot)
        self.assertIn("support", snapshot["levels"])
        self.assertIn("resistance", snapshot["levels"])
        self.assertIsNotNone(snapshot["atr"])
        self.assertIsNotNone(snapshot["indicators"]["ema_fast"])
        self.assertIsNotNone(snapshot["indicators"]["rsi"])
        self.assertEqual(snapshot["live_context"]["listing_profile"], "fresh_listing")
        self.assertTrue(snapshot["live_context"]["fresh_listing_candidate"])

    @patch("trading_brain.live_market._http_get_json")
    def test_fetch_live_snapshot_parses_forex_reference_payload(self, http_get_json) -> None:
        http_get_json.return_value = {
            "date": "2026-03-29",
            "base": "EUR",
            "quote": "USD",
            "rate": 1.0845,
        }

        snapshot = fetch_live_snapshot("EURUSD", "1h", "forex")
        self.assertTrue(snapshot["supported"])
        self.assertEqual(snapshot["provider"], "frankfurter")
        self.assertEqual(snapshot["price"], 1.0845)
        self.assertEqual(snapshot["reference_date"], "2026-03-29")

    @patch("trading_brain.live_market._stooq_csv_row")
    def test_fetch_live_snapshot_parses_stooq_payload(self, stooq_csv_row) -> None:
        stooq_csv_row.return_value = {
            "Symbol": "CL.F",
            "Date": "2026-03-29",
            "Time": "14:35:00",
            "Open": "92.40",
            "High": "94.10",
            "Low": "91.88",
            "Close": "93.55",
            "Volume": "",
        }

        snapshot = fetch_live_snapshot("WTI", "15m", "commodity")
        self.assertTrue(snapshot["supported"])
        self.assertEqual(snapshot["provider"], "stooq")
        self.assertEqual(snapshot["price"], 93.55)
        self.assertIn("levels", snapshot)
        self.assertGreater(len(snapshot["levels"]["support"]), 0)
        self.assertGreater(len(snapshot["levels"]["resistance"]), 0)
        self.assertEqual(snapshot["live_context"]["source"], "stooq_quote_derived")

    @patch.dict("os.environ", {"OANDA_API_TOKEN": "demo-token", "OANDA_ACCOUNT_ID": "101-001-1234567-001"}, clear=False)
    @patch("trading_brain.live_market._http_get_json")
    def test_fetch_live_snapshot_parses_oanda_payload(self, http_get_json) -> None:
        http_get_json.side_effect = [
            {
                "prices": [
                    {
                        "instrument": "EUR_USD",
                        "time": "2026-03-29T01:02:03.000000000Z",
                        "closeoutBid": "1.08240",
                        "closeoutAsk": "1.08260",
                    }
                ]
            },
            _build_oanda_candles(),
        ]

        snapshot = fetch_live_snapshot("EURUSD", "1h", "forex")
        self.assertTrue(snapshot["supported"])
        self.assertEqual(snapshot["provider"], "oanda")
        self.assertEqual(snapshot["price"], 1.0825)
        self.assertIn("levels", snapshot)
        self.assertIn("indicators", snapshot)
        self.assertIn("context", snapshot)
        self.assertIsNotNone(snapshot["levels"]["session_high"])
        self.assertIsNotNone(snapshot["indicators"]["ema_fast"])
        self.assertIsNotNone(snapshot["indicators"]["adx"])

    def test_parse_oanda_stream_payload_returns_price_and_heartbeat(self) -> None:
        config = {
            "provider": "oanda",
            "transport": "sse",
            "symbol": "EURUSD",
            "instrument": "EUR_USD",
        }
        price_event = _parse_oanda_stream_payload(
            config,
            {
                "type": "PRICE",
                "time": "2026-03-29T01:02:03.000000000Z",
                "closeoutBid": "1.08240",
                "closeoutAsk": "1.08260",
            },
        )
        heartbeat_event = _parse_oanda_stream_payload(
            config,
            {
                "type": "HEARTBEAT",
                "time": "2026-03-29T01:02:08.000000000Z",
            },
        )
        self.assertEqual(price_event["price"], 1.0825)
        self.assertTrue(heartbeat_event["heartbeat"])


if __name__ == "__main__":
    unittest.main()
