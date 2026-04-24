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
- ``cube/info``         → BenchmarkMetadata
- ``cube/tasks``        → list[TaskMetadata]   (params: task_id?, offset?, limit?)
- ``cube/task_configs`` → list[TaskConfig]     (params: task_id?, offset?, limit?)
- ``cube/spawn``        → str URL              (params: task_config, host?, port?)
- ``cube/shutdown``     → null

Task methods (``POST /``)
--------------------------
- ``tools/list``           → list[ActionSchema]
- ``tools/call``           → Observation | StepError    (params: name, arguments?)
- ``cube/reset``           → {obs, info}
- ``cube/step``            → EnvironmentOutput          (params: name, arguments?)
- ``cube/evaluate``        → {reward, info}             (params: obs?)
- ``cube/close``           → null
- ``cube/status``          → str
- ``cube/privileged_info`` → Content

Note on ``tools/call`` and ``cube/step`` param shape
-----------------------------------------------------
Both methods use a flat MCP-compatible param shape::

    {"name": "click", "arguments": {"x": 100, "y": 200}, "action_id": "abc-123"}

This mirrors the MCP wire format for ``tools/call`` exactly, so a CUBE task
server can be driven by a standard MCP client with no adapter layer.  The
``Action`` type remains an internal implementation detail — clients never
construct it directly.

``action_id`` is optional.  When provided it is forwarded as ``Action.id``,
allowing clients to correlate actions to observations in logs and traces, and
anticipating the Phase 2 WebSocket async flow where the server pushes results
back with the same id.  ``action_id`` is intentionally distinct from the
JSON-RPC envelope ``id`` field (which identifies the request, not the action).

Note on ``cube/spawn`` vs ``benchmark.spawn()``
------------------------------------------------
These are two separate things:

* ``cube/spawn`` (network endpoint): called by remote clients; starts a task
  server in a subprocess and returns its URL.  The subprocess calls
  ``task_config.make()`` — TaskConfig is the serialization boundary.
* ``benchmark.spawn(task_config)`` (Python API): creates the task in-process
  and returns the ``Task`` object.  No subprocess, no server.  Useful for
  tests and direct Python usage.

Note on serialization boundaries
---------------------------------
**TaskConfig** (Pydantic model) is the unit of serialization across process
boundaries for tasks.  Task objects hold live resources (tool connections,
containers, …) and are never pickled.  All subprocess entry points receive a
TaskConfig and call ``task_config.make()`` inside the worker.

**BenchmarkConfig** (Pydantic model) is the same boundary for benchmarks.
``make_benchmark_rpc_server`` accepts a ``BenchmarkConfig`` and optional
``InfraConfig`` and passes both directly to a subprocess, which calls
``config.make(infra)`` locally.  Multiprocessing's pickle handles
``TypedBaseModel`` polymorphism across the boundary.  Live ``Benchmark``
instances never cross the process boundary — real process isolation, no
shared-memory thread fallback.

Note on deployment
------------------
``make_benchmark_jsonrpc_app`` and ``make_task_jsonrpc_app`` return plain FastAPI
apps with no subprocess or deployment concerns — they can be used with
``starlette.testclient.TestClient`` for in-process testing, or deployed to any
ASGI-compatible host (Modal, fly.io, GCP Cloud Run, …).

``make_benchmark_rpc_server`` and ``make_task_rpc_server`` are convenience
wrappers for local development and testing.
"""

import logging
import multiprocessing
import socket
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from cube.benchmark import Benchmark, BenchmarkConfig
from cube.core import Action, Observation
from cube.resource import InfraConfig
from cube.task import Task, TaskConfig

logger = logging.getLogger(__name__)


# ── Module-level subprocess target ────────────────────────────────────────────
#
# On macOS the default multiprocessing start method is "spawn".  Unlike "fork"
# (the Linux default), "spawn" starts a fresh Python interpreter and pickles
# the target function by *reference* — the child re-imports the module and
# looks up the function by name.  Local closures / nested functions are not
# reachable by name and therefore cannot be pickled.
#
# Design rule: every multiprocessing.Process target must be a module-level
# function, never a local closure.
#
# There is only one subprocess entry point for tasks because TaskConfig is the
# sole serialization unit: both ``cube/spawn`` (the network endpoint) and
# ``make_task_rpc_server`` pass a TaskConfig to this function, which calls
# ``task_config.make()`` inside the worker to create the Task.  Task objects
# themselves are never sent across process boundaries.


def _spawn_task_subprocess(
    task_config: TaskConfig,
    runtime_ctx: dict[str, Any] | None,
    host: str,
    port: int,
) -> None:
    """Subprocess entry point: materialise a task from its config, then serve it.

    Called by both the ``cube/spawn`` network endpoint and
    ``make_task_rpc_server``.  The task is created *inside* the subprocess so
    that live resources (tool instances, containers, …) are owned by the worker
    process and never need to cross a process boundary.

    ``container_backend`` is intentionally not forwarded — it is a legacy
    parameter being replaced by the ``infra`` / ``resource`` pattern.  Infra
    state is passed via ``runtime_ctx`` instead (see TaskConfig.make()).
    """
    task = task_config.make(runtime_context=runtime_ctx)
    uvicorn.run(make_task_jsonrpc_app(task), host=host, port=port)


def _spawn_benchmark_subprocess(
    config: BenchmarkConfig,
    infra: InfraConfig | None,
    host: str,
    port: int,
) -> None:
    """Subprocess entry point: call ``config.make(infra)``, then serve.

    Receives ``BenchmarkConfig`` and ``InfraConfig`` as live Pydantic models;
    multiprocessing pickles them across the spawn/fork boundary, which handles
    ``TypedBaseModel`` polymorphism correctly.  The benchmark is produced
    *inside* the subprocess so runtime handles (container backend,
    runtime_context, etc.) are owned by the worker and torn down when the
    process exits.

    ``benchmark.close()`` runs in a ``finally`` so the worker releases
    resources even if uvicorn exits via a signal or error.
    """
    benchmark = config.make(infra)
    try:
        uvicorn.run(make_benchmark_jsonrpc_app(benchmark), host=host, port=port)
    finally:
        try:
            benchmark.close()
        except Exception:
            logger.exception("Error while closing benchmark in subprocess")


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


# ── FastAPI app factories ─────────────────────────────────────────────────────


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
    app = FastAPI(title=f"CUBE Benchmark Server - {benchmark.config.name}")

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
                result = benchmark.config.benchmark_metadata.model_dump(mode="json")

            elif method == "cube/tasks":
                tasks_metadata = list(benchmark.config.tasks().values())
                task_id = params.get("task_id")
                offset = int(params.get("offset", 0))
                limit = int(params.get("limit", -1))
                if task_id:
                    tasks_metadata = [tm for tm in tasks_metadata if tm.id == task_id]
                tasks_metadata = tasks_metadata[offset:] if limit == -1 else tasks_metadata[offset : offset + limit]
                result = [tm.model_dump(mode="json") for tm in tasks_metadata]

            elif method == "cube/task_configs":
                task_id_filter = params.get("task_id")
                offset = int(params.get("offset", 0))
                limit = int(params.get("limit", -1))
                configs = list(benchmark.config.get_task_configs())
                if task_id_filter:
                    configs = [c for c in configs if c.task_id == task_id_filter]
                configs = configs[offset:] if limit == -1 else configs[offset : offset + limit]
                result = [tc.model_dump(mode="json") for tc in configs]

            elif method == "cube/spawn":
                if "task_config" not in params:
                    return JSONResponse(_err(req_id, _INVALID_PARAMS, "Missing 'task_config' in params"))
                task_config = benchmark.config.task_config_class.model_validate(params["task_config"])
                host = params.get("host", "127.0.0.1")
                port = int(params.get("port", _find_free_port(host)))

                # Pass only picklable values to the subprocess.  TaskConfig is a
                # Pydantic model and is always picklable.  runtime_context is a
                # plain dict (or None).  container_backend is intentionally not
                # forwarded — infra state lives in runtime_context instead.
                p = multiprocessing.Process(
                    target=_spawn_task_subprocess,
                    args=(task_config, benchmark._runtime_context, host, port),
                )
                p.start()
                # Returns the URL immediately; the subprocess starts uvicorn
                # asynchronously.  Clients must poll until the server is ready
                # before sending requests (readiness race — see client.py example).
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
                if "name" not in params:
                    return JSONResponse(_err(req_id, _INVALID_PARAMS, "Missing 'name' in params"))
                action = Action(name=params["name"], arguments=params.get("arguments", {}), id=params.get("action_id"))
                result = task.tool.execute_action(action).model_dump(mode="json")

            elif method == "cube/reset":
                obs, info = task.reset()
                result = {"obs": obs.model_dump(mode="json"), "info": info}

            elif method == "cube/step":
                if "name" not in params:
                    return JSONResponse(_err(req_id, _INVALID_PARAMS, "Missing 'name' in params"))
                action = Action(name=params["name"], arguments=params.get("arguments", {}), id=params.get("action_id"))
                result = task.step(action).model_dump(mode="json")

            elif method == "cube/evaluate":
                obs = Observation.model_validate(params["obs"]) if "obs" in params else None
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


def make_benchmark_rpc_server(
    config: BenchmarkConfig,
    *,
    infra: InfraConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> tuple[multiprocessing.Process, str]:
    """Spawn a benchmark JSON-RPC server in a subprocess.

    Accepts a ``BenchmarkConfig`` (the serialisation boundary for benchmarks)
    and an optional ``InfraConfig``.  Both are passed directly to the
    subprocess; multiprocessing pickles them at spawn, which handles
    ``TypedBaseModel`` polymorphism correctly.  The worker then calls
    ``config.make(infra)`` to produce the live ``Benchmark``.  Live benchmarks
    never cross the process boundary — real process isolation, symmetric with
    ``make_task_rpc_server``.

    The subprocess owns its own resources: container handles, runtime_context,
    shared servers launched inside ``_setup()``.  When the subprocess exits
    (signal, uvicorn shutdown, or parent teardown), ``benchmark.close()`` runs
    in a ``finally`` inside the worker.

    Returns:
        ``(process, url)`` — the subprocess handle and the server base URL.
        The subprocess starts uvicorn asynchronously; poll the URL until it
        responds before sending requests.
    """
    process = multiprocessing.Process(
        target=_spawn_benchmark_subprocess,
        args=(config, infra, host, port),
    )
    process.start()
    return process, f"http://{host}:{port}"


def make_task_rpc_server(
    task_config: TaskConfig,
    host: str = "127.0.0.1",
    port: int = 8000,
    runtime_context: dict[str, Any] | None = None,
) -> tuple[multiprocessing.Process, str]:
    """Spawn a task JSON-RPC server in a subprocess.

    Accepts a **TaskConfig** (not a Task) because only configs cross process
    boundaries safely.  The subprocess calls ``task_config.make()`` to create
    the Task inside the worker.  This is identical to the ``cube/spawn``
    network endpoint — both use ``_spawn_task_subprocess`` as their entry
    point.

    Use this when you already have a TaskConfig in Python and want to expose
    the task over the network without going through a benchmark server.  The
    typical remote workflow (harness → benchmark server → ``cube/spawn`` →
    task URL) does not call this function directly.

    Returns:
        ``(process, url)`` — the subprocess handle and the server base URL.
        The subprocess starts uvicorn asynchronously; poll the URL until it
        responds before sending requests.
    """
    process = multiprocessing.Process(
        target=_spawn_task_subprocess,
        args=(task_config, runtime_context, host, port),
    )
    process.start()
    return process, f"http://{host}:{port}"
