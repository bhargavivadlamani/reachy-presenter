"""Peer HTTP server — lets Robot 2 inject messages into Robot 1's Gemini session.

Listens on port 8001. Robot 2 POSTs {"text": "..."} to /robot-message.
The message is injected directly into the running Bidi LiveRequestQueue so
Gemini hears it even if audio pickup from Robot 2's speaker fails.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)
_PORT = 8001


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/robot-message":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "").strip()
        except Exception:
            self.send_response(400)
            self.end_headers()
            return

        if text:
            from app.agent import inject_message
            ok = inject_message(text)
            self.send_response(200 if ok else 503)
        else:
            self.send_response(400)
        self.end_headers()

    def log_message(self, fmt, *args) -> None:
        pass  # suppress per-request access logs


def start() -> None:
    """Start the peer server in a background daemon thread."""
    server = HTTPServer(("", _PORT), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("[peer_server] listening on port %d", _PORT)
