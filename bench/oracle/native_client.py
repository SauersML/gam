"""Client for the native experiment server (#2951): one JSON request per line, one JSON reply line back,
{"ok": ...} or {"error": "..."}. Requests are gam_mpd::oracle::Request serialized by serde (tag "op":
"info", "run", "crossed", "scan", ...). A batch {"op": "batch", "requests": [...]} returns
{"ok": [reply, ...]}, one {"ok": ...} or {"error": ...} per request.

The client opens one connection per request, which serves both a server that answers one line per
connection (examples/mpd_oracle_2951.rs) and one that keeps a connection open. It does no arithmetic.

  ADDRESS is a Unix socket path, or HOST:PORT for TCP.
"""

from __future__ import annotations

import json
import socket
import time


class ServerError(RuntimeError):
    """The server answered {"error": ...}."""


class NativeClient:
    def __init__(self, address: str):
        self.address = address
        # Every request and reply in order, with the client's wall time around each call.
        self.log: list[dict] = []

    def _connect(self) -> socket.socket:
        host, sep, port = self.address.rpartition(":")
        if sep and port.isdigit() and "/" not in self.address:
            return socket.create_connection((host, int(port)))
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.address)
        return s

    def raw(self, request: dict) -> dict:
        """Send one request, return the parsed reply line as it came ({"ok": ...} or {"error": ...})."""
        line = json.dumps(request, separators=(",", ":")) + "\n"
        start = time.time()
        with self._connect() as s:
            s.sendall(line.encode())
            with s.makefile("rb") as f:
                text = f.readline()
        if not text:
            raise ConnectionError(f"{self.address}: connection closed without a reply")
        reply = json.loads(text)
        self.log.append({"request": request, "reply": reply, "sent": start, "client_seconds": time.time() - start})
        return reply

    def call(self, request: dict):
        """The "ok" value of one request; ServerError on {"error": ...}."""
        reply = self.raw(request)
        if "error" in reply:
            raise ServerError(reply["error"])
        return reply["ok"]

    def batch(self, requests: list[dict]) -> list[dict]:
        """One reply per request, each {"ok": ...} or {"error": ...}, in order."""
        if not requests:
            return []
        replies = self.call({"op": "batch", "requests": requests})
        if len(replies) != len(requests):
            raise ServerError(f"{len(replies)} replies to a batch of {len(requests)}")
        return replies

    def info(self) -> dict:
        return self.call({"op": "info"})

    def run(self, model: str, sequences: list[list[int]], **fields) -> dict:
        return self.call({"op": "run", "model": model, "sequences": sequences, **fields})

    def server_seconds(self) -> float:
        """Server-reported seconds over the log: a reply's "seconds", else its batch members' sum."""
        total = 0.0
        for entry in self.log:
            reply = entry["reply"]
            if reply.get("seconds") is not None:
                total += reply["seconds"]
            elif isinstance(reply.get("ok"), list) and entry["request"].get("op") == "batch":
                total += sum(r.get("seconds") or 0.0 for r in reply["ok"] if isinstance(r, dict))
        return total

    def request_count(self) -> int:
        """Requests in the log, a batch counted as its members."""
        return sum(len(e["request"]["requests"]) if e["request"].get("op") == "batch" else 1 for e in self.log)
