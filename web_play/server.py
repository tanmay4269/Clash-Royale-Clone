from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from web_play.checkpoints import CheckpointInfo, best_by_run, discover_checkpoints


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


class MatchStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.match = None
        self.last_error: str | None = None


STORE = MatchStore()


def checkpoint_payload(checkpoints: list[CheckpointInfo]) -> dict:
    best = best_by_run(checkpoints)
    return {
        "builtIns": [
            {
                "id": "random",
                "kind": "random",
                "label": "Random bot",
                "architectureLabel": "Built-in stochastic policy",
            },
            {
                "id": "scripted",
                "kind": "scripted",
                "label": "Scripted bot",
                "architectureLabel": "Built-in deterministic policy",
            },
            {
                "id": "skip",
                "kind": "skip",
                "label": "Skip-only bot",
                "architectureLabel": "Built-in no-play baseline",
            },
        ],
        "bestByRun": [
            {
                "id": item.id,
                "kind": "checkpoint",
                "runId": run_id,
                "runName": item.run_name,
                "label": f"{run_id} best, {item.checkpoint_name}",
                "checkpointName": item.checkpoint_name,
                "checkpointIndex": item.checkpoint_index,
                "elo": item.elo,
                "architecture": item.architecture,
                "architectureLabel": item.architecture_label,
            }
            for run_id, item in sorted(best.items())
        ],
        "checkpoints": [
            {
                "id": item.id,
                "kind": "checkpoint",
                "runId": item.run_id,
                "runName": item.run_name,
                "label": f"{item.run_id}, {item.checkpoint_name}",
                "checkpointName": item.checkpoint_name,
                "checkpointIndex": item.checkpoint_index,
                "elo": item.elo,
                "architecture": item.architecture,
                "architectureLabel": item.architecture_label,
            }
            for item in checkpoints
        ],
    }


def json_bytes(payload: dict, status: int = 200) -> tuple[int, bytes, str]:
    return status, json.dumps(payload).encode("utf-8"), "application/json"


class Handler(BaseHTTPRequestHandler):
    server_version = "ClashRoyaleWebPlay/0.1"

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path.startswith("/static/") or parsed.path.startswith("/api/"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html")
            elif parsed.path.startswith("/static/"):
                rel = parsed.path.removeprefix("/static/")
                self._send_file(STATIC_DIR / rel)
            elif parsed.path == "/api/opponents":
                self._send_json(checkpoint_payload(discover_checkpoints(ROOT)))
            elif parsed.path == "/api/state":
                with STORE.lock:
                    if STORE.match is None:
                        self._send_json({"running": False, "error": STORE.last_error})
                        return
                    STORE.match.tick()
                    self._send_json({"running": True, "state": STORE.match.state()})
            elif parsed.path == "/api/frame":
                with STORE.lock:
                    if STORE.match is None:
                        self.send_error(HTTPStatus.NOT_FOUND, "No active match")
                        return
                    STORE.match.tick()
                    state = STORE.match.state()
                    frame = STORE.match.frame_rgba()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Canvas-Width", str(state["canvas"]["width"]))
                self.send_header("X-Canvas-Height", str(state["canvas"]["height"]))
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/start":
                self._start_match(payload)
            elif parsed.path == "/api/select-card":
                with STORE.lock:
                    if STORE.match is None:
                        self.send_error(HTTPStatus.NOT_FOUND, "No active match")
                        return
                    STORE.match.select_card(int(payload.get("index", 0)))
                    self._send_json({"ok": True, "state": STORE.match.state()})
            elif parsed.path == "/api/deploy":
                with STORE.lock:
                    if STORE.match is None:
                        self.send_error(HTTPStatus.NOT_FOUND, "No active match")
                        return
                    ok = STORE.match.deploy(int(payload.get("x", 0)), int(payload.get("y", 0)))
                    self._send_json({"ok": ok, "state": STORE.match.state()})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _start_match(self, payload: dict):
        try:
            from web_play.game_session import OpponentSpec, WebMatch
        except Exception as exc:
            message = (
                "Could not import the simulator dependencies. Install the project runtime "
                f"with pygame, gymnasium, numpy, and torch. Import error: {exc}"
            )
            with STORE.lock:
                STORE.last_error = message
            self._send_json({"error": message}, status=500)
            return

        checkpoints = {item.id: item for item in discover_checkpoints(ROOT)}
        opponent_id = payload.get("opponentId", "random")

        if opponent_id in {"random", "scripted", "skip"}:
            spec = OpponentSpec(
                kind=opponent_id,
                label=f"{opponent_id.title()} bot",
                architecture="built_in",
                architecture_label="Built-in bot",
            )
        else:
            checkpoint = checkpoints.get(opponent_id)
            if checkpoint is None:
                self._send_json({"error": f"Unknown checkpoint: {opponent_id}"}, status=400)
                return
            spec = OpponentSpec(
                kind="checkpoint",
                label=f"{checkpoint.run_id}, {checkpoint.checkpoint_name}",
                checkpoint_path=checkpoint.path,
                architecture=checkpoint.architecture,
                architecture_label=checkpoint.architecture_label,
            )

        with STORE.lock:
            STORE.match = WebMatch(spec)
            STORE.last_error = None
            self._send_json({"ok": True, "state": STORE.match.state()})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200):
        status_code, body, content_type = json_bytes(payload, status)
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        resolved = path.resolve()
        if not str(resolved).startswith(str(STATIC_DIR.resolve())) or not resolved.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = resolved.read_bytes()
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description="Run local browser play for Clash Royale RL.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving Clash Royale web play at {url}")
    print("Open the URL, choose an opponent, then press Enter to play.")
    server.serve_forever()


if __name__ == "__main__":
    main()
