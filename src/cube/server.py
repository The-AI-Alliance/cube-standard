"""
JSON-RPC 2.0 server utilities for CUBE.

This module provides FastAPI app factories and server launchers for serving
benchmarks and tasks over JSON-RPC 2.0.  All requests go to a single
``POST /`` endpoint; the method name is inside the JSON body.

Protocol
--------
Request::

    {"jsonrpc": "2.0", "method": "<name>", "params": {...}, "id": 1}

Success response::

    {"jsonrpc": "2.0", "result": <value>, "id": 1}

Error response::

    {"jsonrpc": "2.0", "error": {"code": <int>, "message": "<str>"}, "id": 1}

Benchmark methods (``POST /``)
-------------------------------
- ``cube/info``      → BenchmarkMetadata
- ``cube/tasks``     → list[TaskMetadata]   (params: task_id?, offset?, limit?)
- ``cube/spawn``     → str URL              (params: task_config, host?, port?)
- ``cube/shutdown``  → null

Task methods (``POST /``)
--------------------------
- ``tools/list``           → list[ActionSchema]
- ``tools/call``           → Observation | StepError    (params: action)
- ``cube/reset``           → {obs, info}
- ``cube/step``            → EnvironmentOutput          (params: action)
- ``cube/evaluate``        → {reward, info}             (params: obs)
- ``cube/close``           → null
- ``cube/status``          → str
- ``cube/privileged_info`` → Content

Note on ``cube/spawn`` vs ``benchmark.spawn()``
------------------------------------------------
These are two separate things:

* ``cube/spawn`` (network endpoint): called by remote clients; starts a task
  server in a subprocess and returns its URL.
* ``benchmark.spawn(task_config)`` (Python API): creates the task and its
  JSON-RPC app in-process and returns ``(task, app)``.  No subprocess.  Useful
  for tests and direct Python usage.

Note on deployment
------------------
``make_benchmark_jsonrpc_app`` and ``make_task_jsonrpc_app`` return plain FastAPI
apps with no subprocess or deployment concerns — they can be used with
``starlette.testclient.TestClient`` for in-process testing, or deployed to any
ASGI-compatible host (Modal, fly.io, GCP Cloud Run, …).

``make_benchmark_rpc_server`` and ``make_task_rpc_server`` are convenience
wrappers that launch the app in a local subprocess via ``multiprocessing.Process``.
"""

import logging
import multiprocessing
import socket
from typing import Any, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from cube.benchmark import Benchmark
from cube.core import Action, Observation
from cube.task import Task

# Type alias for server return value: (app, process, url)
ServerInfo = Tuple[FastAPI, multiprocessing.Process, str]

logger = logging.getLogger(__name__)

# ── JSON-RPC 2.0 error codes ──────────────────────────────────────────────────

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


def _ok(req_id: Any, result: Any) -> dict:
    """Build a JSON-RPC 2.0 success response envelope."""
    return {"jsonrpc": "2.0", "result": result, "id": req_id}


def _err(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    """Build a JSON-RPC 2.0 error response envelope."""
    error: dict = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "error": error, "id": req_id}


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Return an available TCP port on *host* by letting the OS pick one.

    Binds a temporary socket to port 0 — the OS kernel assigns a free
    ephemeral port — reads it back via ``getsockname()[1]``, then closes the
    socket immediately.

    **Security / race-condition note (TOCTOU):** closing the socket releases
    the port.  Between this function returning and the real server binding to
    the port, another process could claim it, causing the server to fail with
    ``OSError: [Errno 98] Address already in use``.  This is unlikely on a
    development machine, but callers should handle that error gracefully and
    retry if needed.  Binding to ``127.0.0.1`` (the default) limits exposure
    to the local machine; passing ``0.0.0.0`` would make the port visible on
    all interfaces during that window.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def make_benchmark_jsonrpc_app(benchmark: Benchmark) -> FastAPI:
    """
    Return a FastAPI app that exposes benchmark-level JSON-RPC 2.0 methods.

    All requests are dispatched through ``POST /``.  The app has no subprocess
    or deployment concerns and can be used with ``TestClient`` for in-process
    testing or deployed to any ASGI-compatible host.

    Note on ``async def dispatch``: the handler is async only because
    ``await request.json()`` needs it.  All benchmark methods are synchronous.
    ``cube/spawn`` calls ``multiprocessing.Process.start()``, which forks
    immediately and returns — it does not block the event loop.  If a future
    method needs to perform genuinely long-running synchronous work, wrap it
    with ``asyncio.to_thread()``.

    Warning: results are serialized with ``model.model_dump(mode="json")``.
    Any field whose type is not JSON-serializable and has no Pydantic serializer
    will be silently excluded from the response.
    """
    app = FastAPI(title=f"CUBE Benchmark Server - {benchmark.name}")

    @app.post("/")
    async def _dispatch(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(None, _PARSE_ERROR, "Parse error"))

        req_id = body.get("id")

        if body.get("jsonrpc") != "2.0" or "method" not in body:
            return JSONResponse(_err(req_id, _INVALID_REQUEST, "Invalid Request"))

        method = body["method"]
        params = body.get("params") or {}

        try:
            if method == "cube/info":
                result = benchmark.benchmark_metadata.model_dump(mode="json")

            elif method == "cube/tasks":
                tasks_metadata = list(benchmark.task_metadata.values())
                task_id = params.get("task_id")
                offset = int(params.get("offset", 0))
                limit = int(params.get("limit", -1))
                if task_id:
                    tasks_metadata = [tm for tm in tasks_metadata if tm.id == task_id]
                tasks_metadata = tasks_metadata[offset:] if limit == -1 else tasks_metadata[offset : offset + limit]
                result = [tm.model_dump(mode="json") for tm in tasks_metadata]

            elif method == "cube/spawn":
                if "task_config" not in params:
                    return JSONResponse(_err(req_id, _INVALID_PARAMS, "Missing 'task_config' in params"))
                task_config = benchmark.task_config_class.model_validate(params["task_config"])
                host = params.get("host", "127.0.0.1")
                port = int(params.get("port", _find_free_port(host)))

                # Capture references needed inside the subprocess.
                # task_config is a Pydantic model (picklable).  runtime_context
                # and container_backend may not be picklable for all benchmarks
                # (e.g. live SSH sessions) — callers should be aware of this.
                runtime_ctx = benchmark._runtime_context
                container_be = benchmark.container_backend

                def _run_task_server() -> None:
                    task = task_config.make(
                        runtime_context=runtime_ctx,
                        container_backend=container_be,
                    )
                    task_app = make_task_jsonrpc_app(task)
                    uvicorn.run(task_app, host=host, port=port)

                p = multiprocessing.Process(target=_run_task_server)
                p.start()
                result = f"http://{host}:{port}"

            elif method == "cube/shutdown":
                benchmark.close()
                result = None

            else:
                return JSONResponse(_err(req_id, _METHOD_NOT_FOUND, f"Method not found: {method!r}"))

        except Exception as exc:
            logger.exception("Error handling benchmark method %r", method)
            return JSONResponse(_err(req_id, _INTERNAL_ERROR, str(exc)))

        return JSONResponse(_ok(req_id, result))

    return app


def make_task_jsonrpc_app(task: Task) -> FastAPI:
    """
    Return a FastAPI app that exposes task-level JSON-RPC 2.0 methods.

    All requests are dispatched through ``POST /``.  The app has no subprocess
    or deployment concerns and can be used with ``TestClient`` for in-process
    testing or deployed to any ASGI-compatible host.

    Note on ``async def dispatch``: the handler is async only because
    ``await request.json()`` needs it.  All task methods are synchronous.
    If a future method needs long-running synchronous work, wrap it with
    ``asyncio.to_thread()``.

    Warning: results are serialized with ``model.model_dump(mode="json")``.
    Any field whose type is not JSON-serializable and has no Pydantic serializer
    will be silently excluded from the response.
    """
    app = FastAPI(title=f"CUBE Task Server - {task.id}")

    @app.post("/")
    async def _dispatch(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(None, _PARSE_ERROR, "Parse error"))

        req_id = body.get("id")

        if body.get("jsonrpc") != "2.0" or "method" not in body:
            return JSONResponse(_err(req_id, _INVALID_REQUEST, "Invalid Request"))

        method = body["method"]
        params = body.get("params") or {}

        try:
            if method == "tools/list":
                result = [a.model_dump(mode="json") for a in task.action_set]

            elif method == "tools/call":
                if "action" not in params:
                    return JSONResponse(_err(req_id, _INVALID_PARAMS, "Missing 'action' in params"))
                action = Action.model_validate(params["action"])
                result = task.tool.execute_action(action).model_dump(mode="json")

            elif method == "cube/reset":
                obs, info = task.reset()
                result = {"obs": obs.model_dump(mode="json"), "info": info}

            elif method == "cube/step":
                if "action" not in params:
                    return JSONResponse(_err(req_id, _INVALID_PARAMS, "Missing 'action' in params"))
                raw = params["action"]
                action: Action | list[Action] = (
                    [Action.model_validate(a) for a in raw] if isinstance(raw, list) else Action.model_validate(raw)
                )
                result = task.step(action).model_dump(mode="json")

            elif method == "cube/evaluate":
                if "obs" not in params:
                    return JSONResponse(_err(req_id, _INVALID_PARAMS, "Missing 'obs' in params"))
                obs = Observation.model_validate(params["obs"])
                reward, info = task.evaluate(obs)
                result = {"reward": reward, "info": info}

            elif method == "cube/close":
                task.close()
                result = None

            elif method == "cube/status":
                result = task.get_status()

            elif method == "cube/privileged_info":
                result = task.get_privileged_info().model_dump(mode="json")

            else:
                return JSONResponse(_err(req_id, _METHOD_NOT_FOUND, f"Method not found: {method!r}"))

        except Exception as exc:
            logger.exception("Error handling task method %r", method)
            return JSONResponse(_err(req_id, _INTERNAL_ERROR, str(exc)))

        return JSONResponse(_ok(req_id, result))

    return app


# ── Server launchers ──────────────────────────────────────────────────────────


def make_benchmark_rpc_server(benchmark: Benchmark, host: str = "127.0.0.1", port: int = 8000) -> ServerInfo:
    """
    Spawn a JSON-RPC 2.0 benchmark server in a separate process.

    Returns:
        ServerInfo: (app, process, url)
    """
    app = make_benchmark_jsonrpc_app(benchmark)

    def _run() -> None:
        uvicorn.run(app, host=host, port=port)

    process = multiprocessing.Process(target=_run)
    process.start()
    return app, process, f"http://{host}:{port}"


def make_task_rpc_server(task: Task, host: str = "127.0.0.1", port: int = 8000) -> ServerInfo:
    """
    Spawn a JSON-RPC 2.0 task server in a separate process.

    Returns:
        ServerInfo: (app, process, url)
    """
    app = make_task_jsonrpc_app(task)

    def _run() -> None:
        uvicorn.run(app, host=host, port=port)

    process = multiprocessing.Process(target=_run)
    process.start()
    return app, process, f"http://{host}:{port}"
