"""Smoke: the computer tool's InfraConfig-era attach_endpoint() path, end-to-end.

Track 2 (#94) removed the legacy ``cube.vm.VM`` / ``attach_vm`` coupling. The
only live wiring is now ``ComputerBase.attach_endpoint(endpoint)`` against a
running guest agent. This smoke stands up a *real* HTTP fake guest agent
(stdlib only) and drives the migrated path with no mocks:

  1. A freshly ``make()``-d tool with no endpoint raises the new
     "No guest agent attached" error (proves the old vm=/attach_vm shim is
     gone and the failure mode is sane).
  2. ``attach_endpoint(url)`` + ``get_observation()`` round-trips over HTTP
     against /screenshot (+ /accessibility) and yields a real Observation.

Prints ``SMOKE OK|FAIL|SKIP: attach_endpoint`` and exits 0|1|2.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO


def _png_bytes() -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return b""
    buf = BytesIO()
    Image.new("RGB", (8, 8), (123, 222, 64)).save(buf, format="PNG")
    return buf.getvalue()


_PNG = _png_bytes()


class _GuestAgentStub(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # silence
        pass

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/screenshot"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_PNG)))
            self.end_headers()
            self.wfile.write(_PNG)
        elif self.path.startswith("/accessibility"):
            body = json.dumps({"AT": "<desktop><window name='Smoke'/></desktop>"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main() -> int:
    if not _PNG:
        print("SMOKE SKIP: attach_endpoint (Pillow not installed)")
        return 2

    from cube_computer_tool.computer import ComputerConfig

    server = HTTPServer(("127.0.0.1", 0), _GuestAgentStub)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        cfg = ComputerConfig(require_a11y_tree=True, require_terminal=False)

        # 1. No endpoint attached -> the new, post-attach_vm-removal error.
        tool = cfg.make()
        try:
            tool.get_observation()
            print("SMOKE FAIL: attach_endpoint (expected RuntimeError before attach)")
            return 1
        except RuntimeError as exc:
            if "No guest agent attached" not in str(exc):
                print(f"SMOKE FAIL: attach_endpoint (unexpected error: {exc!r})")
                return 1

        # 2. attach_endpoint + real HTTP round-trip.
        tool.attach_endpoint(f"http://127.0.0.1:{port}")
        obs = tool.get_observation()
        if obs is None:
            print("SMOKE FAIL: attach_endpoint (get_observation returned None)")
            return 1
        dumped = obs.model_dump() if hasattr(obs, "model_dump") else obs
        if not dumped:
            print("SMOKE FAIL: attach_endpoint (empty observation)")
            return 1

        print("SMOKE OK: attach_endpoint (no-endpoint error + HTTP screenshot/a11y round-trip)")
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
