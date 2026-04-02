from __future__ import annotations

import unittest
from unittest.mock import patch

from trading_brain.live_news import _NEWS_CACHE, _NEWS_PROVIDER_BACKOFF, build_news_query, fetch_live_news


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Sample Feed</title>
    <item>
      <title>Bitcoin rally continues as ETF demand stays strong</title>
      <link>https://news.google.com/rss/articles/one</link>
      <pubDate>Sun, 29 Mar 2026 12:00:00 GMT</pubDate>
      <source url="https://example.com">CoinDesk</source>
    </item>
    <item>
      <title>Fed minutes and CPI risk keep crypto traders cautious</title>
      <link>https://news.google.com/rss/articles/two</link>
      <pubDate>Sun, 29 Mar 2026 11:00:00 GMT</pubDate>
      <source url="https://example.com">Reuters</source>
    </item>
  </channel>
</rss>
"""


class LiveNewsTest(unittest.TestCase):
    def setUp(self) -> None:
        _NEWS_CACHE.clear()
        _NEWS_PROVIDER_BACKOFF.clear()

    def test_build_news_query_for_crypto(self) -> None:
        query = build_news_query("BTCUSDT", "crypto")
        self.assertEqual(query["market_type"], "crypto")
        self.assertIn("Bitcoin", query["query"])

    def test_build_news_query_for_emerging_crypto_adds_listing_research_terms(self) -> None:
        query = build_news_query("ABCUSDT", "crypto")
        self.assertIn("listing", query["query"])
        self.assertIn("unlock", query["query"])

    def test_build_news_query_for_forex(self) -> None:
        query = build_news_query("EURUSD", "forex")
        self.assertEqual(query["market_type"], "forex")
        self.assertIn("EURUSD", query["query"])
        self.assertIn("Euro", query["query"])

    def test_build_news_query_for_xauusd_and_wti(self) -> None:
        gold = build_news_query("XAUUSD", "forex")
        oil = build_news_query("WTI", "commodity")

        self.assertIn("XAUUSD", gold["query"])
        self.assertIn("Gold", gold["label"])
        self.assertIn("OPEC", oil["query"])
        self.assertIn("WTI", oil["label"])

    @patch("trading_brain.live_news._http_get_text")
    def test_fetch_live_news_parses_articles_and_summary(self, http_get_text) -> None:
        http_get_text.return_value = SAMPLE_RSS
        payload = fetch_live_news("BTCUSDT", "crypto", limit=4, force=True)

        self.assertTrue(payload["supported"])
        self.assertEqual(payload["provider"], "google_news_rss")
        self.assertEqual(payload["symbol"], "BTCUSDT")
        self.assertEqual(len(payload["articles"]), 2)
        self.assertEqual(payload["articles"][0]["source"], "CoinDesk")
        self.assertTrue(payload["summary"]["headline_risk"])
        self.assertTrue(payload["summary"]["macro_risk"])
        self.assertGreater(payload["summary"]["headline_count"], 0)

    @patch("trading_brain.live_news._http_get_text")
    def test_fetch_live_news_returns_neutral_payload_when_provider_fails(self, http_get_text) -> None:
        http_get_text.side_effect = TimeoutError("provider timeout")
        payload = fetch_live_news("BTCUSDT", "crypto", limit=4, force=True)

        self.assertFalse(payload["supported"])
        self.assertEqual(payload["provider"], "news_unavailable")
        self.assertEqual(payload["summary"]["mood"], "neutral")
        self.assertEqual(payload["summary"]["headline_count"], 0)
        self.assertIn("provider timeout", payload["error"])

    @patch("trading_brain.live_news._http_get_text")
    def test_fetch_live_news_uses_stale_cache_during_backoff(self, http_get_text) -> None:
        http_get_text.return_value = SAMPLE_RSS
        warm_payload = fetch_live_news("BTCUSDT", "crypto", limit=4, force=True)

        http_get_text.side_effect = TimeoutError("provider timeout")
        payload = fetch_live_news("BTCUSDT", "crypto", limit=4, force=True)

        self.assertTrue(payload["supported"])
        self.assertTrue(payload["stale"])
        self.assertEqual(payload["provider"], warm_payload["provider"])
        self.assertEqual(len(payload["articles"]), len(warm_payload["articles"]))


if __name__ == "__main__":
    unittest.main()
