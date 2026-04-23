"""
Cube Toolkit exec relay server — runs INSIDE the container.

Purpose: provide a reliable HTTP channel for command execution, bypassing
`eai job exec` which has a known TCP half-close bug hanging ~6% of calls.

Security posture:

  1. Bind address: 127.0.0.1 only. NEVER 0.0.0.0 — other pods sharing the
     cluster network must not reach this endpoint. `eai job port-forward`
     tunnels into pod-local loopback, so 127.0.0.1 is sufficient.
  2. Authentication: 256-bit random bearer token, passed via the
     CUBE_EXEC_RELAY_TOKEN_FILE env var (path to a file containing the token).
     Token is never on argv (avoids /proc/<pid>/cmdline leaks) and never
     logged. Comparison uses hmac.compare_digest (timing-safe).
  3. Request body cap: 1 MiB. Prevents memory exhaustion on malformed input.
  4. Per-request timeout: enforced via subprocess.run(timeout=...). Hung
     commands return exit_code=124 (GNU timeout convention).
  5. Error responses: generic — never echo internal state, exception text,
     or stack traces to the client.
  6. No persistent state on disk. Token file is read once at startup and
     its contents held only in memory.
  7. No TLS: the link is localhost-only inside a single container. TLS would
     add cost for no threat-model benefit.

Invocation (from the cube client):

    nohup python3 /tmp/_cube_exec_relay.py >/dev/null 2>&1 &
    disown

Env vars read at startup:
    CUBE_EXEC_RELAY_PORT         — port to bind on 127.0.0.1 (default 8787)
    CUBE_EXEC_RELAY_TOKEN_FILE   — path to file containing the bearer token

This file is stdlib-only (http.server, json, subprocess, secrets, hmac,
socket) so it runs on any Python 3.8+ image with no pip install.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MiB — reject oversized requests
_DEFAULT_PORT = 8787
_BIND_ADDR = "127.0.0.1"  # SECURITY: never change to 0.0.0.0


def _load_token() -> str:
    path = os.environ.get("CUBE_EXEC_RELAY_TOKEN_FILE")
    if not path:
        sys.stderr.write("CUBE_EXEC_RELAY_TOKEN_FILE not set\n")
        sys.exit(2)
    with open(path, "r", encoding="utf-8") as f:
        tok = f.read().strip()
    if len(tok) < 32:
        sys.stderr.write("Token too short\n")
        sys.exit(2)
    return tok


# Captured at startup, never logged.
_TOKEN = _load_token()


def _check_auth(header_value: str | None) -> bool:
    if not header_value or not header_value.startswith("Bearer "):
        return False
    presented = header_value[len("Bearer "):].strip()
    return hmac.compare_digest(presented, _TOKEN)


def _run_command(command: str, timeout: float, workdir: str | None, env_overrides: dict[str, str] | None) -> dict:
    env = os.environ.copy()
    # Redact our own secret from the child process environment so the
    # executed command cannot trivially exfiltrate the relay token.
    env.pop("CUBE_EXEC_RELAY_TOKEN_FILE", None)
    if env_overrides:
        for k, v in env_overrides.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("env keys/values must be strings")
            env[k] = v

    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workdir or None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        exit_code = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124  # GNU timeout convention
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr = (stderr + f"\n[exec_relay] timed out after {timeout}s\n").lstrip("\n")
    duration = time.monotonic() - start

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "duration_seconds": round(duration, 3),
    }


class _Handler(BaseHTTPRequestHandler):
    # Silence default request logging — stderr on Toolkit jobs is captured
    # and we don't want per-request lines (even without secrets) flooding logs.
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            # No auth: liveness probe used during bootstrap before the client
            # has confirmed the token is live. Returns no sensitive state.
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/exec":
            self._send_json(404, {"error": "not_found"})
            return

        if not _check_auth(self.headers.get("Authorization")):
            # Generic 401 — never hint whether header was missing vs mismatched.
            self._send_json(401, {"error": "unauthorized"})
            return

        length_raw = self.headers.get("Content-Length")
        try:
            length = int(length_raw) if length_raw is not None else -1
        except ValueError:
            self._send_json(400, {"error": "bad_content_length"})
            return
        if length < 0 or length > _MAX_BODY_BYTES:
            self._send_json(413, {"error": "body_too_large"})
            return

        try:
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "bad_json"})
            return

        command = body.get("command")
        timeout = body.get("timeout", 120)
        workdir = body.get("workdir")
        env_overrides = body.get("env")

        if not isinstance(command, str) or not command:
            self._send_json(400, {"error": "bad_command"})
            return
        if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 24 * 3600:
            self._send_json(400, {"error": "bad_timeout"})
            return
        if workdir is not None and not isinstance(workdir, str):
            self._send_json(400, {"error": "bad_workdir"})
            return
        if env_overrides is not None and not isinstance(env_overrides, dict):
            self._send_json(400, {"error": "bad_env"})
            return

        try:
            result = _run_command(command, float(timeout), workdir, env_overrides)
        except Exception:
            # SECURITY: do not return exception text — could leak token via env
            # or filesystem paths. Log to stderr (captured by job log).
            logging.exception("exec failed")
            self._send_json(500, {"error": "internal_error"})
            return

        self._send_json(200, result)


def main() -> None:
    port = int(os.environ.get("CUBE_EXEC_RELAY_PORT", _DEFAULT_PORT))
    # ThreadingHTTPServer so /health can respond while a long /exec is running.
    server = ThreadingHTTPServer((_BIND_ADDR, port), _Handler)
    # stderr banner (no token) so bootstrap can confirm startup from the log.
    sys.stderr.write(f"cube-exec-relay listening on {_BIND_ADDR}:{port}\n")
    sys.stderr.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
