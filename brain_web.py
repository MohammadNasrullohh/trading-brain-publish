from __future__ import annotations

import argparse
import json
from contextlib import suppress
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from trading_brain.chat_assistant import answer_chat
from trading_brain.live_market import build_live_config, fetch_live_snapshot, iter_oanda_live_stream
from trading_brain.live_news import fetch_live_news
from trading_brain.recommendations import scan_recommendations
from trading_brain.signal_memory import get_signal_history
from trading_brain.web_bridge import analyze_for_web, list_example_files, load_example_payload


ROOT = Path(__file__).resolve().parent


class BrainWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/web/index.html")
            self.end_headers()
            return

        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/examples":
            self._send_json({"examples": list_example_files()})
            return

        if parsed.path == "/api/live/config":
            symbol = (query.get("symbol") or [""])[0]
            timeframe = (query.get("timeframe") or ["15m"])[0]
            market_type = (query.get("market_type") or ["auto"])[0]
            self._send_json(build_live_config(symbol, timeframe, market_type))
            return

        if parsed.path == "/api/live/snapshot":
            symbol = (query.get("symbol") or [""])[0]
            timeframe = (query.get("timeframe") or ["15m"])[0]
            market_type = (query.get("market_type") or ["auto"])[0]
            try:
                self._send_json(fetch_live_snapshot(symbol, timeframe, market_type))
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    {
                        "supported": False,
                        "error": str(exc),
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "market_type": market_type,
                    },
                    status=HTTPStatus.BAD_GATEWAY,
                )
            return

        if parsed.path == "/api/news":
            symbol = (query.get("symbol") or [""])[0]
            market_type = (query.get("market_type") or ["auto"])[0]
            limit = int((query.get("limit") or ["6"])[0] or 6)
            force = str((query.get("force") or ["0"])[0]).strip().lower() in {"1", "true", "yes"}
            try:
                self._send_json(fetch_live_news(symbol, market_type, limit=limit, force=force))
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    {
                        "supported": False,
                        "error": str(exc),
                        "symbol": symbol,
                        "market_type": market_type,
                    },
                    status=HTTPStatus.BAD_GATEWAY,
                )
            return

        if parsed.path == "/api/history":
            symbol = (query.get("symbol") or [""])[0]
            timeframe = (query.get("timeframe") or ["15m"])[0]
            style = (query.get("style") or ["intraday"])[0]
            market_type = (query.get("market_type") or ["auto"])[0]
            try:
                self._send_json(get_signal_history(symbol, timeframe, style, market_type))
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    {
                        "supported": False,
                        "error": str(exc),
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "style": style,
                        "market_type": market_type,
                    },
                    status=HTTPStatus.BAD_GATEWAY,
                )
            return

        if parsed.path == "/api/live/stream":
            symbol = (query.get("symbol") or [""])[0]
            timeframe = (query.get("timeframe") or ["15m"])[0]
            market_type = (query.get("market_type") or ["auto"])[0]
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self.wfile.write(b"retry: 2500\n\n")
                self.wfile.flush()

                for payload in iter_oanda_live_stream(symbol, timeframe, market_type):
                    event_name = "heartbeat" if payload.get("heartbeat") else "tick"
                    raw = json.dumps(payload, ensure_ascii=False)
                    self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
                    self.wfile.write(f"data: {raw}\n\n".encode("utf-8"))
                    self.wfile.flush()
                return
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # noqa: BLE001
                with suppress(BrokenPipeError, ConnectionResetError):
                    raw = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    self.wfile.write(f"event: error\ndata: {raw}\n\n".encode("utf-8"))
                    self.wfile.flush()
                return

        if parsed.path.startswith("/api/examples/"):
            name = Path(parsed.path).name
            try:
                self._send_json({"name": name, "payload": load_example_payload(name)})
            except FileNotFoundError:
                self._send_json({"error": f"Example not found: {name}"}, status=HTTPStatus.NOT_FOUND)
            return

        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/analyze", "/api/recommendations", "/api/chat"}:
            self._send_json({"error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            body = self._read_json_body()
            if parsed.path == "/api/chat":
                self._send_json(
                    answer_chat(
                        str(body.get("message") or ""),
                        body.get("context") or {},
                    )
                )
                return

            if parsed.path == "/api/recommendations":
                track_best_setup = True if "track_best_setup" not in body else bool(body.get("track_best_setup"))
                result = scan_recommendations(
                    symbols=body.get("symbols"),
                    timeframe=str(body.get("timeframe") or "15m"),
                    style=str(body.get("style") or "intraday"),
                    mode=str(body.get("mode") or "super"),
                    base_payload=body.get("base_payload"),
                    scope=str(body.get("scope") or "hybrid"),
                    discover_limit=int(body.get("discover_limit") or 8),
                    discovery_mode=str(body.get("discovery_mode") or "leaders"),
                    research_mode=str(body.get("research_mode") or ""),
                    prime_only=body.get("prime_only"),
                    track_best_setup=track_best_setup,
                )
                self._send_json(result)
                return

            mode = body.get("mode", "super")
            track_signal = bool(body.get("track_signal")) if "track_signal" in body else False

            if "payload" in body:
                payload = body["payload"]
            elif "example" in body:
                payload = load_example_payload(str(body["example"]))
            else:
                payload = load_example_payload("bullish_btc.json")

            result = analyze_for_web(payload, mode=mode, track_signal=track_signal, signal_source="web")
            self._send_json(result)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve trading brain web viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Host untuk server lokal")
    parser.add_argument("--port", type=int, default=8765, help="Port untuk server lokal")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), BrainWebHandler)
    print(f"Trading brain web server running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
