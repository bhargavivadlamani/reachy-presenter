"""Lightweight HTTP server that receives slide events from the presenter robot.

Robot 1 (presenter) POSTs to http://<robot2-ip>:5001/slide with JSON:
    {"slide_number": 1, "script": "..."}

The server forwards the call to the registered AudienceAgent.
Default port: 5001 (override with AUDIENCE_SERVER_PORT env var).
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Callable

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 5001


class _Handler(BaseHTTPRequestHandler):
    callback: Optional[Callable[[int, str], None]] = None

    def do_POST(self):
        if self.path != "/slide":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            slide_number = int(data.get("slide_number", 0))
            script = str(data.get("script", ""))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            logger.info("[audience server] received slide %d", slide_number)
            if self.callback and slide_number >= 0:
                threading.Thread(
                    target=self.callback,
                    args=(slide_number, script),
                    daemon=True,
                ).start()
        except Exception as e:
            logger.error("[audience server] bad request: %s", e)
            self.send_response(400)
            self.end_headers()

    def log_message(self, fmt, *args):
        logger.debug("[audience server] " + fmt, *args)


class AudienceServer:
    """Runs a background HTTP server; call register() to wire up the agent."""

    def __init__(self) -> None:
        self._port = int(os.environ.get("AUDIENCE_SERVER_PORT", _DEFAULT_PORT))
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def register(self, callback: Callable[[int, str], None]) -> None:
        """Set the function to call when a slide event arrives."""
        _Handler.callback = callback

    def start(self) -> None:
        self._server = HTTPServer(("0.0.0.0", self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        logger.info("[audience server] listening on port %d", self._port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
