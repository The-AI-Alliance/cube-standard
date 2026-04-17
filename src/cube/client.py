"""
JSON-RPC 2.0 client utilities for CUBE.

Provides low-level helpers (``rpc``, ``wait_for_server``) and high-level typed
clients (``BenchmarkClient``, ``TaskClient``) for driving CUBE benchmark and
task servers over HTTP.

These clients use the same JSON-RPC 2.0 wire protocol as ``cube.server``, so
they work against any CUBE server regardless of implementation language — a
``TaskClient`` talking to a Node.js task server behaves identically to one
talking to a Python server.

Low-level usage::

    from cube.client import rpc, wait_for_server

    wait_for_server("http://127.0.0.1:8765")
    info = rpc("http://127.0.0.1:8765", "cube/info")

High-level usage::

    from cube.client import BenchmarkClient

    bench = BenchmarkClient("http://127.0.0.1:8765")
    task = bench.spawn(task_id="count-to-3")   # polls for readiness automatically
    task.reset()
    result = task.step("increment")
    task.close()
    bench.shutdown()
"""

import time
from typing import Any

import httpx

# Defaults for wait_for_server polling — 30 × 0.5 s = 15 s total
_DEFAULT_POLL_RETRIES: int = 30
_DEFAULT_POLL_DELAY: float = 0.5  # 0.5 seconds


# ── Low-level helpers ─────────────────────────────────────────────────────────


def rpc(
    url: str,
    method: str,
    params: dict | None = None,
    req_id: int = 1,
) -> Any:
    """Send a single JSON-RPC 2.0 request and return the ``result`` field.

    Args:
        url:     Base URL of the server (``POST /`` will be called).
        method:  JSON-RPC method name (e.g. ``"cube/info"``).
        params:  Optional params dict.
        req_id:  JSON-RPC request id (for correlating responses in logs).

    Returns:
        The ``result`` value from the JSON-RPC response.

    Raises:
        RuntimeError:         The server returned a JSON-RPC error object.
        httpx.HTTPStatusError: The HTTP response status was 4xx/5xx.
    """
    body: dict = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        body["params"] = params
    resp = httpx.post(url, json=body, timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"JSON-RPC error {err['code']}: {err['message']}")
    return data["result"]


def wait_for_server(
    url: str,
    retries: int = _DEFAULT_POLL_RETRIES,
    delay: float = _DEFAULT_POLL_DELAY,
) -> None:
    """Poll *url* until the server is accepting requests, then return.

    Posts an empty JSON object ``{}`` to ``/``.  This is intentionally not a
    valid JSON-RPC request, so any CUBE server responds with HTTP 200 and a
    ``-32600 Invalid Request`` error — enough to confirm the port is bound.
    Works for both benchmark servers and task servers without requiring a
    shared method name.  Only a connection error means the port is not yet up.

    Used after ``cube/spawn`` to absorb the subprocess startup race: the
    benchmark server returns the task URL immediately, but uvicorn takes a
    moment to bind.

    Args:
        url:     Server base URL to poll.
        retries: Maximum number of attempts before raising.
        delay:   Seconds to wait between attempts.

    Raises:
        RuntimeError: Server did not become ready within the allotted attempts.
    """
    for _ in range(retries):
        try:
            if httpx.post(url, json={}, timeout=5.0).status_code == 200:
                return
        except httpx.ConnectError as e:
            print(f"Waiting for server at {url} to be ready... (error: {e})")
        time.sleep(delay)
    raise RuntimeError(f"Server at {url} did not become ready after {retries} attempts ({retries * delay:.1f}s total)")


# ── High-level typed clients ──────────────────────────────────────────────────


class BenchmarkClient:
    """Python client for a running CUBE benchmark server.

    Wraps every benchmark JSON-RPC method as a typed Python call.
    Maintains an auto-incrementing request-id counter so log correlation
    is easy without the caller needing to manage ids.

    Example::

        bench = BenchmarkClient("http://127.0.0.1:8765")
        print(bench.info())
        task = bench.spawn("count-to-3")  # returns a ready TaskClient
        ...
        bench.shutdown()
    """

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _rpc(self, method: str, params: dict | None = None) -> Any:
        return rpc(self.url, method, params, req_id=self._next_id())

    # ── Benchmark methods ─────────────────────────────────────────────────────

    def info(self) -> dict:
        """``cube/info`` — benchmark metadata (name, version, description, …)."""
        return self._rpc("cube/info")

    def tasks(
        self,
        task_id: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[dict]:
        """``cube/tasks`` — list task metadata, with optional filtering.

        Args:
            task_id: Return only the task with this id.
            offset:  Skip the first *offset* tasks.
            limit:   Return at most *limit* tasks.
        """
        params: dict = {}
        if task_id is not None:
            params["task_id"] = task_id
        if offset:
            params["offset"] = offset
        if limit is not None:
            params["limit"] = limit
        return self._rpc("cube/tasks", params or None)

    def task_configs(
        self,
        task_id: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[dict]:
        """``cube/task_configs`` — list serialized TaskConfig dicts, with optional filtering.

        Unlike ``tasks()`` which returns display metadata, this returns fully
        populated TaskConfig dicts ready to pass directly to ``spawn()``.  The
        benchmark fills in all benchmark-level fields (tool config, seeds, infra
        URLs, …) so callers never need to construct configs by hand.

        Args:
            task_id: Return only configs for the task with this id.
            offset:  Skip the first *offset* configs.
            limit:   Return at most *limit* configs.
        """
        params: dict = {}
        if task_id is not None:
            params["task_id"] = task_id
        if offset:
            params["offset"] = offset
        if limit is not None:
            params["limit"] = limit
        return self._rpc("cube/task_configs", params or None)

    def spawn(
        self,
        task_config: dict,
        host: str | None = None,
        port: int | None = None,
    ) -> "TaskClient":
        """``cube/spawn`` — start a task server subprocess and return a ready client.

        The benchmark server starts the task subprocess and returns its URL
        immediately, before uvicorn has bound the port.  This method polls the
        task URL with ``wait_for_server`` before returning, so the caller can
        start sending requests straight away.

        Args:
            task_config: Serialised TaskConfig dict.  Use ``task_configs()`` to
                         obtain a fully-populated dict from the benchmark server;
                         the benchmark fills in all benchmark-level fields (tool
                         config, seeds, infra URLs, …) automatically.
            host:        Host for the task server subprocess (default: 127.0.0.1).
            port:        Port for the task server subprocess (default: OS-assigned).

        Returns:
            A ``TaskClient`` connected to the ready task server.
        """
        params: dict = {"task_config": task_config}
        if host is not None:
            params["host"] = host
        if port is not None:
            params["port"] = port
        task_url: str = self._rpc("cube/spawn", params)
        wait_for_server(task_url)
        return TaskClient(task_url)

    def shutdown(self) -> None:
        """``cube/shutdown`` — close benchmark resources."""
        self._rpc("cube/shutdown")


class TaskClient:
    """Python client for a running CUBE task server.

    Wraps every task JSON-RPC method as a typed Python call.
    Maintains an auto-incrementing request-id counter.

    Example::

        task = TaskClient("http://127.0.0.1:54321")
        task.reset()
        while True:
            result = task.step("increment")
            if result["done"]:
                break
        reward, _ = task.evaluate()["reward"], task.evaluate()["info"]
        task.close()
    """

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _rpc(self, method: str, params: dict | None = None) -> Any:
        return rpc(self.url, method, params, req_id=self._next_id())

    # ── Task methods ──────────────────────────────────────────────────────────

    def tools_list(self) -> list[dict]:
        """``tools/list`` — available actions (name, description, input schema)."""
        return self._rpc("tools/list")

    def tools_call(
        self,
        name: str,
        arguments: dict | None = None,
        action_id: str | None = None,
    ) -> dict:
        """``tools/call`` — execute a tool action outside the episode loop.

        Returns the raw ``Observation`` dict.
        """
        params: dict = {"name": name}
        if arguments:
            params["arguments"] = arguments
        if action_id is not None:
            params["action_id"] = action_id
        return self._rpc("tools/call", params)

    def reset(self) -> dict:
        """``cube/reset`` — reset the task. Returns ``{obs, info}``."""
        return self._rpc("cube/reset")

    def step(
        self,
        name: str,
        arguments: dict | None = None,
        action_id: str | None = None,
    ) -> dict:
        """``cube/step`` — execute an action. Returns ``EnvironmentOutput`` dict.

        The returned dict has keys: ``obs``, ``reward``, ``done``,
        ``truncated``, ``info``, ``error``.
        """
        params: dict = {"name": name}
        if arguments:
            params["arguments"] = arguments
        if action_id is not None:
            params["action_id"] = action_id
        return self._rpc("cube/step", params)

    def evaluate(self, obs: dict | None = None) -> dict:
        """``cube/evaluate`` — score the current state. Returns ``{reward, info}``."""
        params = {"obs": obs} if obs is not None else None
        return self._rpc("cube/evaluate", params)

    def close(self) -> None:
        """``cube/close`` — release task resources."""
        self._rpc("cube/close")

    def status(self) -> str:
        """``cube/status`` — human-readable status string."""
        return self._rpc("cube/status")

    def privileged_info(self) -> dict:
        """``cube/privileged_info`` — ground truth / oracle information."""
        return self._rpc("cube/privileged_info")
