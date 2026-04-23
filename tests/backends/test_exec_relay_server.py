"""Security + correctness tests for the toolkit exec relay server.

These tests run the server as a subprocess on localhost — no toolkit/eai
dependency. They cover the security surface we documented in
cube_infra_toolkit/_exec_relay_server.py.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

SERVER_PATH = (
    Path(__file__).parent.parent.parent
    / "cube-resources"
    / "cube-infra-toolkit"
    / "src"
    / "cube_infra_toolkit"
    / "_exec_relay_server.py"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for(url: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.05)
    raise TimeoutError(f"server did not come up at {url}")


@contextmanager
def _server(token: str = "x" * 48):
    port = _free_port()
    tok_path = Path(f"/tmp/_cube_exec_relay_test_token_{os.getpid()}_{port}")
    tok_path.write_text(token)
    try:
        env = {
            **os.environ,
            "CUBE_EXEC_RELAY_PORT": str(port),
            "CUBE_EXEC_RELAY_TOKEN_FILE": str(tok_path),
        }
        proc = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _wait_for(f"http://127.0.0.1:{port}/health", timeout=5.0)
            yield port, token, proc
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
    finally:
        tok_path.unlink(missing_ok=True)


def _post(port: int, token: str | None, body: dict, path: str = "/exec"):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# -------------------- Security tests --------------------


def test_binds_localhost_only():
    """SECURITY: server must refuse connections from non-loopback addresses."""
    with _server() as (port, _, _):
        # Bind to a non-loopback local interface if available — here we just
        # verify the port is not listening on 0.0.0.0 via a socket probe.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        # Loopback should succeed
        s.connect(("127.0.0.1", port))
        s.close()
    # Verify the bind constant via AST so docstring mentions don't false-positive.
    import ast

    tree = ast.parse(SERVER_PATH.read_text())
    bind_values = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_BIND_ADDR" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert bind_values == ["127.0.0.1"], bind_values


def test_rejects_missing_auth():
    with _server() as (port, _, _):
        status, body = _post(port, None, {"command": "echo hi"})
    assert status == 401
    assert body == {"error": "unauthorized"}


def test_rejects_bad_scheme():
    with _server(token="goodtoken" + "x" * 40) as (port, _, _):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/exec",
            data=b"{}",
            method="POST",
            headers={"Authorization": "Basic Zm9vOmJhcg=="},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                status = r.status
        except urllib.error.HTTPError as exc:
            status = exc.code
    assert status == 401


def test_rejects_wrong_token():
    with _server(token="correct_token_" + "x" * 40) as (port, _, _):
        status, body = _post(port, "wrong_token_" + "y" * 40, {"command": "echo hi"})
    assert status == 401
    assert body == {"error": "unauthorized"}


def test_accepts_correct_token():
    with _server() as (port, tok, _):
        status, body = _post(port, tok, {"command": "echo hello"})
    assert status == 200
    assert body["exit_code"] == 0
    assert body["stdout"].strip() == "hello"


def test_body_size_cap():
    """SECURITY: oversized requests return 413. Use a raw socket because
    urllib can error out on the broken pipe before reading the response."""
    with _server() as (port, tok, _):
        payload = b"A" * (2 * 1024 * 1024)  # 2 MiB > 1 MiB cap
        req = (
            b"POST /exec HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Authorization: Bearer " + tok.encode() + b"\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
            b"\r\n"
        )
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(("127.0.0.1", port))
        s.sendall(req)
        # Server should respond 413 before we need to send the body.
        try:
            s.sendall(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        resp = b""
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                resp += chunk
                if b"\r\n\r\n" in resp and len(resp) > 50:
                    break
        except (socket.timeout, ConnectionResetError):
            pass
        s.close()
    assert b"413" in resp.split(b"\r\n", 1)[0], resp[:200]
    assert b"body_too_large" in resp


def test_malformed_json():
    with _server() as (port, tok, _):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/exec",
            data=b"not json{",
            method="POST",
            headers={"Authorization": f"Bearer {tok}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                status = r.status
                resp = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            status = exc.code
            resp = json.loads(exc.read())
    assert status == 400
    assert resp == {"error": "bad_json"}


def test_token_file_env_redacted_from_child():
    """SECURITY: child process must NOT see CUBE_EXEC_RELAY_TOKEN_FILE env var,
    otherwise a malicious command could read the token file path and exfiltrate
    the secret."""
    with _server() as (port, tok, _):
        status, body = _post(
            port,
            tok,
            {"command": "env | grep -c CUBE_EXEC_RELAY_TOKEN_FILE || true"},
        )
    assert status == 200
    # Either grep found 0 matches (exit 1 → printed '0') or -c returned 0
    assert body["stdout"].strip() == "0"


def test_error_response_does_not_leak_internals():
    """Errors from the subprocess layer must not echo paths or exception text."""
    with _server() as (port, tok, _):
        # env with non-string value → ValueError inside _run_command → 500
        status, body = _post(port, tok, {"command": "echo hi", "env": {"K": 123}})
    assert status == 500
    assert set(body.keys()) == {"error"}
    assert body["error"] == "internal_error"


def test_short_token_startup_fails():
    """SECURITY: server refuses to start with a short token (prevents accidental
    weak-secret deployments)."""
    port = _free_port()
    tok_path = Path(f"/tmp/_cube_exec_relay_test_short_{os.getpid()}")
    tok_path.write_text("short")
    try:
        env = {
            **os.environ,
            "CUBE_EXEC_RELAY_PORT": str(port),
            "CUBE_EXEC_RELAY_TOKEN_FILE": str(tok_path),
        }
        proc = subprocess.run(
            [sys.executable, str(SERVER_PATH)],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
    finally:
        tok_path.unlink(missing_ok=True)
    assert proc.returncode == 2


def test_missing_token_file_startup_fails():
    port = _free_port()
    env = {**os.environ, "CUBE_EXEC_RELAY_PORT": str(port)}
    env.pop("CUBE_EXEC_RELAY_TOKEN_FILE", None)
    proc = subprocess.run(
        [sys.executable, str(SERVER_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode == 2


# -------------------- Functional tests --------------------


def test_exec_nonzero_exit():
    with _server() as (port, tok, _):
        status, body = _post(port, tok, {"command": "exit 7"})
    assert status == 200
    assert body["exit_code"] == 7


def test_exec_timeout():
    with _server() as (port, tok, _):
        status, body = _post(port, tok, {"command": "sleep 5", "timeout": 0.5})
    assert status == 200
    assert body["exit_code"] == 124
    assert "timed out" in body["stderr"].lower()


def test_exec_workdir(tmp_path):
    with _server() as (port, tok, _):
        status, body = _post(port, tok, {"command": "pwd", "workdir": str(tmp_path)})
    assert status == 200
    assert os.path.realpath(body["stdout"].strip()) == os.path.realpath(str(tmp_path))


def test_exec_env_override():
    with _server() as (port, tok, _):
        status, body = _post(port, tok, {"command": "echo $CUBE_TEST_VAR", "env": {"CUBE_TEST_VAR": "xyz"}})
    assert status == 200
    assert body["stdout"].strip() == "xyz"


def test_exec_shell_features():
    """Pipes + redirects must work (shell=True is intentional)."""
    with _server() as (port, tok, _):
        status, body = _post(port, tok, {"command": "echo a && echo b | tr a-z A-Z"})
    assert status == 200
    assert body["stdout"].strip().splitlines() == ["a", "B"]


def test_health_no_auth_required():
    with _server() as (port, _, _):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            assert r.status == 200
            assert json.loads(r.read()) == {"ok": True}


def test_health_during_long_exec():
    """ThreadingHTTPServer: health must not block on an in-flight exec."""
    import threading

    with _server() as (port, tok, _):
        results = {}

        def slow():
            results["exec"] = _post(port, tok, {"command": "sleep 1.5"})

        t = threading.Thread(target=slow)
        t.start()
        time.sleep(0.2)
        start = time.monotonic()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            assert r.status == 200
        assert time.monotonic() - start < 0.5, "health blocked on exec"
        t.join()
    assert results["exec"][0] == 200


def test_404_on_unknown_path():
    with _server() as (port, tok, _):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/nonexistent",
            headers={"Authorization": f"Bearer {tok}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as r:
                status = r.status
        except urllib.error.HTTPError as exc:
            status = exc.code
    assert status == 404
