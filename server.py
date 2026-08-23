"""Local web frontend for Counterpoint.

    python server.py            # then open http://127.0.0.1:8765

Binds to localhost only. The API key stays in this process and is never sent to
the browser. Turns are pushed to the page over Server-Sent Events as they land,
so you watch the two agents argue in real time instead of waiting for the run.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import debate
from providers import LLMClient, LLMError, load_env

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

_models_cache: dict[str, list[str]] = {}
_models_lock = threading.Lock()


def free_models() -> dict[str, list[str]]:
    """Free models per provider, fetched once per process."""
    global _models_cache
    with _models_lock:
        if not _models_cache:
            try:
                _models_cache = LLMClient().models_by_provider()
            except (LLMError, OSError) as exc:
                print(f"model list unavailable: {exc}", file=sys.stderr)
            if not _models_cache:
                # Offline or every list call failed: still offer the built-ins.
                known = sorted(
                    {debate.MODEL_A, debate.MODEL_B, debate.MODEL_JUDGE, *debate.FALLBACKS}
                )
                _models_cache = {"configured": known}
        return _models_cache


def clamp(value: str | None, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def clamp_float(value: str | None, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def args_from_query(q: dict[str, list[str]]) -> argparse.Namespace:
    one = {k: v[0] for k, v in q.items() if v}
    mode = one.get("mode", "discuss")
    if mode not in debate.ALL_MODES:
        mode = "discuss"
    council = mode == "council"
    # Council rounds are decided by the council itself; the cap is the backstop.
    default_rounds = debate.MAX_COUNCIL_ROUNDS if council else 4
    return argparse.Namespace(
        topic=(one.get("topic") or "").strip()[:500],
        mode=mode,
        council=council,
        per_provider=clamp(one.get("per_provider"), 1, 2, 1),
        rounds=clamp(one.get("rounds"), 1, debate.MAX_COUNCIL_ROUNDS, default_rounds),
        min_rounds=clamp(one.get("min_rounds"), 1, 12, 2),
        model_a=one.get("model_a") or debate.MODEL_A,
        model_b=one.get("model_b") or debate.MODEL_B,
        model_judge=one.get("model_judge") or debate.MODEL_JUDGE,
        names=one.get("names") or None,
        temperature=clamp_float(one.get("temperature"), 0.0, 2.0, 0.8),
        max_tokens=clamp(one.get("max_tokens"), 100, 2000, 500),
        no_fallback=one.get("no_fallback") == "1",
    )


def history() -> list[dict]:
    items = []
    for path in sorted(debate.TRANSCRIPTS.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items.append(
            {
                "name": path.stem,
                "topic": data.get("topic", path.stem),
                "mode": data.get("mode", ""),
                "date": data.get("date", ""),
                "turns": len(data.get("turns", [])),
            }
        )
    return items[:100]


class Handler(BaseHTTPRequestHandler):
    server_version = "counterpoint"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *fargs) -> None:  # quieter console
        if "/api/stream" in fmt % fargs:
            return
        sys.stderr.write("  %s\n" % (fmt % fargs))

    # -- helpers ---------------------------------------------------------

    def send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        route, query = url.path, parse_qs(url.query)

        if route in ("/", "/index.html"):
            self.send_file(STATIC / "index.html")
        elif route == "/api/config":
            self.send_json(
                {
                    "models": free_models(),
                    "modes": {
                        mode: [p[0] for p in pair] for mode, pair in debate.MODES.items()
                    },
                    "council": {
                        "max_rounds": debate.MAX_COUNCIL_ROUNDS,
                        "max_seats": len(debate.COUNCIL_PERSONAS),
                    },
                    "defaults": {
                        "model_a": debate.MODEL_A,
                        "model_b": debate.MODEL_B,
                        "model_judge": debate.MODEL_JUDGE,
                    },
                }
            )
        elif route == "/api/history":
            self.send_json(history())
        elif route.startswith("/api/history/"):
            name = Path(route[len("/api/history/") :]).name  # no traversal
            path = debate.TRANSCRIPTS / f"{name}.json"
            if not path.is_file():
                self.send_json({"error": "not found"}, 404)
                return
            self.send_json(json.loads(path.read_text(encoding="utf-8")))
        elif route == "/api/stream":
            self.stream_debate(args_from_query(query))
        else:
            self.send_json({"error": "not found"}, 404)

    def stream_debate(self, args: argparse.Namespace) -> None:
        if not args.topic:
            self.send_json({"error": "topic required"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def push(event: dict) -> bool:
            """Write one SSE frame. False once the browser has gone away."""
            try:
                frame = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False

        print(f"  debate: {args.topic!r} ({args.mode}, {args.rounds} rounds)")
        try:
            for event in debate.converse(args):
                if not push(event):
                    print("  client disconnected, stopping run")
                    return
        except LLMError as exc:
            push({"type": "error", "text": str(exc)})
        except Exception as exc:  # keep the socket honest about the failure
            push({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
        push({"type": "done"})

    # HTTP/1.1 keep-alive after a streamed response confuses some browsers.
    def do_HEAD(self) -> None:  # noqa: N802
        self.send_json({})


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    load_env(HERE / ".env")

    p = argparse.ArgumentParser(description="Web frontend for Counterpoint.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1", help="localhost only by default")
    p.add_argument("--no-browser", action="store_true")
    opts = p.parse_args()

    try:
        LLMClient()  # fail fast on a missing key
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    httpd = ThreadingHTTPServer((opts.host, opts.port), Handler)
    url = f"http://{opts.host}:{opts.port}"
    print(f"\n  Counterpoint -> {url}")
    print("  ctrl-c to stop\n")
    if not opts.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
