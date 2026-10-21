"""Vercel Python function: POST raw image bytes, get back the plate region edited.

`?mode=censor` (default) blurs the detected plate/sign region, `?mode=highlight`
keeps it sharp and blurs the rest. The demo page in ``public/index.html`` is the
client. Only the segmentation code is imported, no pandas or matplotlib, to keep
the cold start short.
"""

from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from censor_license_plate.segmentation import censor, highlight  # noqa: E402

MAX_BODY_BYTES = 8 * 1024 * 1024
MODES = {"censor": censor, "highlight": highlight}
DEFAULT_MODE = "censor"


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            # Read the body first: answering before draining it resets the
            # connection for the client while it is still uploading.
            body = self._read_body()
            mode = self._mode()
            image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self._send(400, b"cannot decode image data", "text/plain")
                return
            result = MODES[mode](image)
            ok, buffer = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            if not ok:
                self._send(500, b"png encoding failed", "text/plain")
                return
            self._send(200, buffer.tobytes(), "image/png")
        except ValueError as exc:
            self._send(422, str(exc).encode(), "text/plain")
        except Exception:  # pragma: no cover - defensive, never leak a traceback
            self._send(500, b"segmentation failed", "text/plain")

    def do_GET(self) -> None:
        self._send(405, b"POST an image body to this endpoint", "text/plain")

    def _mode(self) -> str:
        requested = parse_qs(urlparse(self.path).query).get("mode", [DEFAULT_MODE])[0]
        if requested not in MODES:
            raise ValueError(f"unknown mode {requested!r}, expected one of {sorted(MODES)}")
        return requested

    def _read_body(self) -> bytes:
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            raise ValueError("empty request body")
        if length > MAX_BODY_BYTES:
            raise ValueError("image larger than 8 MB")
        return self.rfile.read(length)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)
