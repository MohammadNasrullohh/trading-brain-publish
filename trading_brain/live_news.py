from __future__ import annotations

import copy
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from .live_market import infer_market_type, normalize_symbol, parse_forex_pair


GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"
HTTP_TIMEOUT_SECONDS = 6
NEWS_CACHE_TTL_SECONDS = 45
NEWS_PROVIDER_BACKOFF_SECONDS = 120

CRYPTO_NAMES = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "XRP": "Ripple",
    "DOGE": "Dogecoin",
    "BNB": "BNB",
    "ADA": "Cardano",
    "AVAX": "Avalanche",
    "LINK": "Chainlink",
    "SUI": "Sui",
}

CURRENCY_NAMES = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "AUD": "Australian Dollar",
    "NZD": "New Zealand Dollar",
    "CAD": "Canadian Dollar",
    "CHF": "Swiss Franc",
    "XAU": "Gold",
}

POSITIVE_KEYWORDS = {
    "surge",
    "rally",
    "gain",
    "gains",
    "jump",
    "soar",
    "bull",
    "bullish",
    "approval",
    "approved",
    "beat",
    "growth",
    "support",
    "record",
    "strong",
    "rebound",
    "breakout",
    "cooling",
    "eases",
    "ease",
}

NEGATIVE_KEYWORDS = {
    "drop",
    "drops",
    "fall",
    "falls",
    "selloff",
    "slump",
    "crash",
    "hack",
    "lawsuit",
    "ban",
    "bear",
    "bearish",
    "risk",
    "warning",
    "liquidation",
    "recession",
    "hotter",
    "tariff",
    "war",
    "fraud",
    "miss",
    "weak",
}

RISK_KEYWORDS = {
    "fed",
    "fomc",
    "powell",
    "ecb",
    "boj",
    "nfp",
    "nonfarm",
    "cpi",
    "ppi",
    "inflation",
    "rates",
    "rate",
    "tariff",
    "war",
    "sec",
    "lawsuit",
    "hack",
    "liquidation",
    "default",
}

MACRO_KEYWORDS = {
    "fed",
    "fomc",
    "powell",
    "ecb",
    "boj",
    "bank of japan",
    "bank of england",
    "cpi",
    "ppi",
    "inflation",
    "rates",
    "rate cut",
    "rate hike",
    "nfp",
    "jobs",
    "payrolls",
    "gdp",
    "tariff",
}

_NEWS_CACHE: dict[str, dict[str, Any]] = {}
_NEWS_PROVIDER_BACKOFF: dict[str, float] = {}


def _http_get_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "Accept": "application/rss+xml, application/xml, text/xml, text/plain",
            "User-Agent": "TradingBrain/2.0",
        },
    )
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="ignore")


def _tokenize(text: str) -> set[str]:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in text)
    return {token for token in normalized.split() if token}


def _article_score(title: str) -> float:
    tokens = _tokenize(title)
    positive = sum(1 for word in POSITIVE_KEYWORDS if word in tokens or word in title.lower())
    negative = sum(1 for word in NEGATIVE_KEYWORDS if word in tokens or word in title.lower())
    if positive == negative == 0:
        return 0.0
    raw = (positive - negative) / max(positive + negative, 1)
    return max(-1.0, min(1.0, raw))


def _article_flags(title: str) -> tuple[bool, bool]:
    lower = title.lower()
    risk = any(word in lower for word in RISK_KEYWORDS)
    macro = any(word in lower for word in MACRO_KEYWORDS)
    return risk, macro


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_pub_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        return None


def build_news_query(symbol: str | None, market_type: str | None = None) -> dict[str, str]:
    normalized_symbol = normalize_symbol(symbol)
    resolved_market = infer_market_type(normalized_symbol, market_type)

    if resolved_market == "crypto":
        base_asset = normalized_symbol
        for suffix in ("USDT", "USDC", "BUSD", "FDUSD", "BTC", "ETH"):
            if normalized_symbol.endswith(suffix) and len(normalized_symbol) > len(suffix):
                base_asset = normalized_symbol[: -len(suffix)]
                break
        asset_name = CRYPTO_NAMES.get(base_asset, base_asset)
        is_emerging = base_asset not in CRYPTO_NAMES
        query = f"{asset_name} {base_asset} crypto"
        if is_emerging:
            query += " listing launch token unlock ecosystem funding roadmap volume"
        return {
            "symbol": normalized_symbol,
            "market_type": "crypto",
            "query": query,
            "label": asset_name,
        }

    pair = parse_forex_pair(normalized_symbol)
    if resolved_market == "forex" and pair:
        base, quote = pair
        if base == "XAU":
            query = "gold XAUUSD fed yields dollar macro market"
            label = "Gold / XAUUSD"
        else:
            base_name = CURRENCY_NAMES.get(base, base)
            quote_name = CURRENCY_NAMES.get(quote, quote)
            query = f"{normalized_symbol} {base_name} {quote_name} forex central bank rates"
            label = f"{base}/{quote}"
        return {
            "symbol": normalized_symbol,
            "market_type": "forex",
            "query": query,
            "label": label,
        }

    if resolved_market == "commodity":
        if normalized_symbol in {"WTI", "USOIL", "CL.F"}:
            query = "WTI crude oil OPEC inventory energy market"
            label = "WTI Crude Oil"
        else:
            query = f"{normalized_symbol} commodity market"
            label = normalized_symbol
        return {
            "symbol": normalized_symbol,
            "market_type": "commodity",
            "query": query,
            "label": label,
        }

    query = f"{normalized_symbol or 'market'} OR trading news"
    return {
        "symbol": normalized_symbol or "MARKET",
        "market_type": resolved_market or "unknown",
        "query": query,
        "label": normalized_symbol or "Market",
    }


def _rss_url(query: str) -> str:
    encoded_query = quote_plus(f"{query} when:1d")
    return f"{GOOGLE_NEWS_RSS_BASE}?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"


def _parse_feed(xml_text: str, limit: int) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    items: list[dict[str, Any]] = []

    for item in root.findall("./channel/item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = _parse_pub_date(item.findtext("pubDate"))
        source = item.find("source")
        source_name = (source.text or "").strip() if source is not None else ""
        score = _article_score(title)
        headline_risk, macro_risk = _article_flags(title)
        items.append(
            {
                "title": title,
                "link": link,
                "source": source_name or "Google News",
                "published_at": _format_timestamp(pub_date),
                "score": round(score, 2),
                "headline_risk": headline_risk,
                "macro_risk": macro_risk,
            }
        )

    return items


def _summary_from_articles(articles: list[dict[str, Any]]) -> dict[str, Any]:
    if not articles:
        return {
            "score": 0.0,
            "headline_risk": False,
            "macro_risk": False,
            "mood": "neutral",
            "headline_count": 0,
        }

    score = sum(float(article.get("score", 0.0) or 0.0) for article in articles) / len(articles)
    headline_risk = sum(1 for article in articles if article.get("headline_risk")) >= 1
    macro_risk = sum(1 for article in articles if article.get("macro_risk")) >= 1
    if score >= 0.18:
        mood = "tailwind"
    elif score <= -0.18:
        mood = "headwind"
    else:
        mood = "neutral"

    return {
        "score": round(score, 2),
        "headline_risk": headline_risk,
        "macro_risk": macro_risk,
        "mood": mood,
        "headline_count": len(articles),
    }


def _fallback_news_payload(
    config: dict[str, Any],
    *,
    error: Exception | None = None,
    cached_payload: dict[str, Any] | None = None,
    stale: bool = False,
    backoff_active: bool = False,
) -> dict[str, Any]:
    if cached_payload:
        payload = copy.deepcopy(cached_payload)
        payload["supported"] = True
        payload["stale"] = True
        payload["backoff_active"] = backoff_active
        payload["error"] = str(error) if error else payload.get("error")
        payload["source_note"] = "News memakai cache terakhir karena provider news sedang bermasalah."
        return payload

    return {
        "supported": False,
        "provider": "news_unavailable",
        "symbol": config["symbol"],
        "market_type": config["market_type"],
        "query": config["query"],
        "label": config["label"],
        "refreshed_at": _format_timestamp(datetime.now(timezone.utc)),
        "summary": _summary_from_articles([]),
        "articles": [],
        "stale": stale,
        "backoff_active": backoff_active,
        "error": str(error) if error else None,
        "source_note": "Provider news publik sedang bermasalah, jadi sentiment diperlakukan netral sementara.",
    }


def fetch_live_news(symbol: str | None, market_type: str | None = None, limit: int = 6, force: bool = False) -> dict[str, Any]:
    config = build_news_query(symbol, market_type)
    cache_key = f"{config['market_type']}|{config['symbol']}|{limit}"
    provider_key = "google_news_rss"
    now = time.time()

    if not force and cache_key in _NEWS_CACHE:
        cached = _NEWS_CACHE[cache_key]
        if now - cached["ts"] < NEWS_CACHE_TTL_SECONDS:
            return copy.deepcopy(cached["data"])

    stale_cached = copy.deepcopy(_NEWS_CACHE[cache_key]["data"]) if cache_key in _NEWS_CACHE else None
    backoff_until = _NEWS_PROVIDER_BACKOFF.get(provider_key, 0.0)
    if not force and backoff_until > now:
        return _fallback_news_payload(
            config,
            cached_payload=stale_cached,
            stale=bool(stale_cached),
            backoff_active=True,
        )

    try:
        xml_text = _http_get_text(_rss_url(config["query"]))
        articles = _parse_feed(xml_text, limit)
    except Exception as exc:  # noqa: BLE001
        _NEWS_PROVIDER_BACKOFF[provider_key] = now + NEWS_PROVIDER_BACKOFF_SECONDS
        return _fallback_news_payload(
            config,
            error=exc,
            cached_payload=stale_cached,
            stale=bool(stale_cached),
        )

    _NEWS_PROVIDER_BACKOFF.pop(provider_key, None)
    payload = {
        "supported": True,
        "provider": "google_news_rss",
        "symbol": config["symbol"],
        "market_type": config["market_type"],
        "query": config["query"],
        "label": config["label"],
        "refreshed_at": _format_timestamp(datetime.now(timezone.utc)),
        "summary": _summary_from_articles(articles),
        "articles": articles,
        "source_note": "Headline live dari Google News RSS ikut mewarnai sentiment pair aktif.",
    }
    _NEWS_CACHE[cache_key] = {
        "ts": now,
        "data": copy.deepcopy(payload),
    }
    return payload
